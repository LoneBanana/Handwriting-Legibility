import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { SpeakOptions } from '../../lib/speech'
import { AFTER_CANCEL_MS, MIN_SEGMENT_MS, sleep, SpeechQueue, type QueueItem, type SpeakFn } from './speechQueue'

const item = (index: number, text = 'Hello there friend'): QueueItem => ({ turnKey: 't', index, text, emotion: 'content' })

function recorder() {
  const log: string[] = []
  return {
    log,
    handlers: {
      onStart: (i: QueueItem, voiced: boolean) => log.push(`start ${i.index}${voiced ? '' : ' (silent)'}`),
      onBoundary: (i: QueueItem, c: number) => log.push(`boundary ${i.index}@${c}`),
      onEnd: (i: QueueItem) => log.push(`end ${i.index}`),
      onVoiceFailed: (i: QueueItem) => log.push(`failed ${i.index}`),
      onIdle: () => log.push('idle'),
    },
  }
}

/** A fake voice: starts, reports a boundary, and finishes after `ms`. */
const fakeVoice =
  (ms: number): SpeakFn =>
  (_text: string, options: SpeakOptions) =>
    new Promise((resolve) => {
      options.onStart?.()
      options.onBoundary?.(0)
      const timer = setTimeout(resolve, ms)
      options.signal?.addEventListener('abort', () => {
        clearTimeout(timer)
        resolve()
      })
    })

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('sleep', () => {
  it('resolves early when aborted', async () => {
    const controller = new AbortController()
    let done = false
    void sleep(10_000, controller.signal).then(() => {
      done = true
    })
    controller.abort()
    await vi.advanceTimersByTimeAsync(0)
    expect(done).toBe(true)
  })
})

describe('SpeechQueue', () => {
  it('speaks items one after another, then reports idle', async () => {
    const { log, handlers } = recorder()
    const queue = new SpeechQueue({ voice: true, wordsPerMinute: 150, voiceName: null }, handlers, { speakFn: fakeVoice(500) })
    queue.enqueue(item(0))
    queue.enqueue(item(1))
    expect(queue.busy).toBe(true)
    await vi.advanceTimersByTimeAsync(400)
    expect(log).toEqual(['start 0', 'boundary 0@0'])
    await vi.advanceTimersByTimeAsync(700)
    expect(log).toEqual(['start 0', 'boundary 0@0', 'end 0', 'start 1', 'boundary 1@0', 'end 1', 'idle'])
    expect(queue.busy).toBe(false)
  })

  it('paces items silently by their estimated length when there is no voice', async () => {
    const { log, handlers } = recorder()
    const speakFn = vi.fn()
    const queue = new SpeechQueue({ voice: false, wordsPerMinute: 60, voiceName: null }, handlers, { speakFn })
    queue.enqueue(item(0, 'one two three')) // 3 words at 60 wpm = 3 s
    await vi.advanceTimersByTimeAsync(2900)
    expect(log).toEqual(['start 0 (silent)'])
    await vi.advanceTimersByTimeAsync(200)
    expect(log).toEqual(['start 0 (silent)', 'end 0', 'idle'])
    expect(speakFn).not.toHaveBeenCalled()
  })

  it('falls back to silent pacing when the browser refuses to speak', async () => {
    const { log, handlers } = recorder()
    const refuses: SpeakFn = () => Promise.resolve() // never fires onStart
    const queue = new SpeechQueue({ voice: true, wordsPerMinute: 150, voiceName: null }, handlers, { speakFn: refuses })
    queue.enqueue(item(0, 'hi'))
    await vi.advanceTimersByTimeAsync(0)
    expect(log).toEqual(['failed 0', 'start 0 (silent)'])
    await vi.advanceTimersByTimeAsync(MIN_SEGMENT_MS)
    expect(log).toEqual(['failed 0', 'start 0 (silent)', 'end 0', 'idle'])
  })

  it('moves on if an utterance never ends', async () => {
    const { log, handlers } = recorder()
    let signal: AbortSignal | undefined
    const hangs: SpeakFn = (_text, options) => {
      signal = options.signal
      options.onStart?.()
      return new Promise(() => undefined)
    }
    const queue = new SpeechQueue({ voice: true, wordsPerMinute: 150, voiceName: null }, handlers, { speakFn: hangs })
    queue.enqueue(item(0, 'hi'))
    await vi.advanceTimersByTimeAsync(MIN_SEGMENT_MS * 2 + 4000 + 10)
    expect(log).toEqual(['start 0', 'end 0', 'idle'])
    expect(signal?.aborted).toBe(true) // the stuck utterance is cancelled
  })

  it('clears without further callbacks and can start again', async () => {
    const { log, handlers } = recorder()
    const queue = new SpeechQueue({ voice: true, wordsPerMinute: 150, voiceName: null }, handlers, { speakFn: fakeVoice(500) })
    queue.enqueue(item(0))
    queue.enqueue(item(1))
    await vi.advanceTimersByTimeAsync(100)
    queue.clear()
    expect(queue.busy).toBe(false)
    await vi.advanceTimersByTimeAsync(2000)
    expect(log).toEqual(['start 0', 'boundary 0@0'])
    queue.enqueue(item(2))
    await vi.advanceTimersByTimeAsync(600)
    expect(log.slice(2)).toEqual(['start 2', 'boundary 2@0', 'end 2', 'idle'])
  })

  it('pauses briefly after a clear before speaking again', async () => {
    const speakFn = vi.fn(fakeVoice(10))
    const queue = new SpeechQueue({ voice: true, wordsPerMinute: 150, voiceName: null }, {}, { speakFn })
    queue.clear()
    queue.enqueue(item(0))
    await vi.advanceTimersByTimeAsync(AFTER_CANCEL_MS - 50)
    expect(speakFn).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(60)
    expect(speakFn).toHaveBeenCalledTimes(1)
  })

  it('uses the current rate and voice for each item', async () => {
    const speakFn = vi.fn(fakeVoice(10))
    const queue = new SpeechQueue({ voice: true, wordsPerMinute: 170, voiceName: null }, {}, { speakFn })
    queue.configure({ wordsPerMinute: 85, voiceName: 'Samantha' })
    queue.enqueue(item(0))
    await vi.advanceTimersByTimeAsync(20)
    expect(speakFn.mock.calls[0][1]).toMatchObject({ rate: 0.5, voiceName: 'Samantha' })
  })
})
