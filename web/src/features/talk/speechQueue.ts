// Speaks answer segments one after another as they stream in. Each segment
// reports start / word boundaries / end so the avatar can change expression and
// move its lips, and captions can follow. Where speech isn't possible (no
// SpeechSynthesis, the browser blocked it, or the video avatar does the
// talking) segments are paced silently by their estimated length instead, so
// the rest of the page behaves the same.

import type { Emotion } from '../../api/types'
import { estimateSpeechMs, speak, speechSupported, wpmToRate, type SpeakOptions } from '../../lib/speech'

export type QueueItem = { turnKey: string; index: number; text: string; emotion: Emotion }

export type SpeakFn = (text: string, options: SpeakOptions) => Promise<void>
export type SleepFn = (ms: number, signal: AbortSignal) => Promise<void>

export type SpeechQueueSettings = {
  /** Speak aloud with SpeechSynthesis; false paces each item silently. */
  voice: boolean
  wordsPerMinute: number
  voiceName: string | null
}

export type SpeechQueueHandlers = {
  /** `voiced` is true when SpeechSynthesis actually started speaking it. */
  onStart?: (item: QueueItem, voiced: boolean) => void
  onBoundary?: (item: QueueItem, charIndex: number) => void
  onEnd?: (item: QueueItem) => void
  /** The voice never started (blocked or broken); the item was paced silently instead. */
  onVoiceFailed?: (item: QueueItem) => void
  /** Everything queued has been spoken. */
  onIdle?: () => void
}

/** A silently paced segment stays "speaking" at least this long. */
export const MIN_SEGMENT_MS = 700

/** Chrome can drop an utterance spoken right after cancel(), so wait this long after a clear. */
export const AFTER_CANCEL_MS = 250

/** Resolves after `ms`, or as soon as `signal` aborts. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve()
      return
    }
    const done = () => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', done)
      resolve()
    }
    const timer = setTimeout(done, Math.max(0, ms))
    signal?.addEventListener('abort', done)
  })
}

export class SpeechQueue {
  private items: QueueItem[] = []
  private current: AbortController | null = null
  private generation = 0
  private running = false
  private clearedAt = -Infinity
  private settings: SpeechQueueSettings
  private handlers: SpeechQueueHandlers
  private readonly speakFn: SpeakFn
  private readonly sleepFn: SleepFn

  constructor(
    settings: SpeechQueueSettings,
    handlers: SpeechQueueHandlers = {},
    { speakFn = speak, sleepFn = sleep }: { speakFn?: SpeakFn; sleepFn?: SleepFn } = {},
  ) {
    this.settings = settings
    this.handlers = handlers
    this.speakFn = speakFn
    this.sleepFn = sleepFn
  }

  configure(settings: Partial<SpeechQueueSettings>): void {
    this.settings = { ...this.settings, ...settings }
  }

  setHandlers(handlers: SpeechQueueHandlers): void {
    this.handlers = handlers
  }

  get busy(): boolean {
    return this.running
  }

  enqueue(item: QueueItem): void {
    this.items.push(item)
    if (!this.running) void this.run(this.generation)
  }

  /** Stops the current segment and forgets the rest. No further callbacks fire for them. */
  clear(): void {
    this.generation += 1
    this.items = []
    this.running = false
    this.current?.abort()
    this.current = null
    this.clearedAt = Date.now()
  }

  private async run(generation: number): Promise<void> {
    this.running = true
    while (generation === this.generation && this.items.length > 0) {
      const item = this.items.shift()!
      const controller = new AbortController()
      this.current = controller
      await this.play(item, controller.signal)
      if (generation !== this.generation) return
      this.handlers.onEnd?.(item)
    }
    if (generation !== this.generation) return
    this.running = false
    this.current = null
    this.handlers.onIdle?.()
  }

  private async play(item: QueueItem, signal: AbortSignal): Promise<void> {
    const { voice, wordsPerMinute, voiceName } = this.settings
    const estimate = Math.max(MIN_SEGMENT_MS, estimateSpeechMs(item.text, wordsPerMinute))
    if (!voice) {
      this.handlers.onStart?.(item, false)
      await this.sleepFn(estimate, signal)
      return
    }
    const sinceClear = Date.now() - this.clearedAt
    if (sinceClear < AFTER_CANCEL_MS) {
      await this.sleepFn(AFTER_CANCEL_MS - sinceClear, signal)
      if (signal.aborted) return
    }

    let started = false
    const attempt = new AbortController()
    const forwardAbort = () => attempt.abort()
    signal.addEventListener('abort', forwardAbort)
    try {
      await Promise.race([
        this.speakFn(item.text, {
          rate: wpmToRate(wordsPerMinute),
          voiceName,
          signal: attempt.signal,
          onStart: () => {
            started = true
            this.handlers.onStart?.(item, true)
          },
          onBoundary: (charIndex) => this.handlers.onBoundary?.(item, charIndex),
        }),
        // Watchdog: some browsers occasionally never fire "end" for an utterance.
        this.sleepFn(estimate * 2 + 4000, attempt.signal),
      ])
    } finally {
      attempt.abort() // stops the watchdog, or cancels speech if the watchdog won
      signal.removeEventListener('abort', forwardAbort)
    }

    if (!started && !signal.aborted) {
      this.handlers.onVoiceFailed?.(item)
      this.handlers.onStart?.(item, false)
      await this.sleepFn(estimate, signal)
    }
  }
}

let primed = false

/**
 * iOS only lets a page speak after a tap. Call this inside the tap or key press
 * that sends a question, so speech that starts later (when the answer streams
 * in) is allowed.
 */
export function primeSpeech(): void {
  if (primed || !speechSupported()) return
  primed = true
  try {
    const utterance = new SpeechSynthesisUtterance(' ')
    utterance.volume = 0
    window.speechSynthesis.speak(utterance)
  } catch {
    // Nothing to do: speech will simply fall back to silent pacing if blocked.
  }
}
