// Stream parsing, the Tavus message format and speech-recognition helpers.
import { describe, expect, it, vi } from 'vitest'
import { joinTranscript, recognitionErrorMessage, recognitionLanguage } from './recognition'
import { parseTalkEvent, STREAM_ERROR_MESSAGE } from './stream'
import { echoInteraction, interruptInteraction, replicaSpeakingFrom, sendEcho, sendInterrupt } from './tavus'

const sse = (event: string, data: unknown) => ({ event, data: typeof data === 'string' ? data : JSON.stringify(data), id: null })

describe('parseTalkEvent', () => {
  it('reads segments, defaulting unknown emotions to neutral', () => {
    expect(parseTalkEvent(sse('segment', { turn_id: 'a', index: 0, emotion: 'excited', text: 'Great question!' }))).toEqual({
      type: 'segment',
      segment: { turn_id: 'a', index: 0, emotion: 'excited', text: 'Great question!' },
    })
    const odd = parseTalkEvent(sse('segment', { turn_id: 'a', index: 1, emotion: 'smug', text: 'Hmm.' }))
    expect(odd).toMatchObject({ type: 'segment', segment: { emotion: 'neutral' } })
  })

  it('reads done and error events', () => {
    expect(parseTalkEvent(sse('done', { turn_id: 'a', text: 'Hi.', echo_text: '<emotion value="content"/> Hi.' }))).toEqual({
      type: 'done',
      done: { turn_id: 'a', text: 'Hi.', echo_text: '<emotion value="content"/> Hi.' },
    })
    expect(parseTalkEvent(sse('error', { detail: 'You have reached the hourly limit.' }))).toEqual({
      type: 'error',
      message: 'You have reached the hourly limit.',
    })
    expect(parseTalkEvent(sse('error', 'not json'))).toEqual({ type: 'error', message: STREAM_ERROR_MESSAGE })
  })

  it('ignores malformed or unknown events', () => {
    expect(parseTalkEvent(sse('segment', 'nope'))).toEqual({ type: 'ignored' })
    expect(parseTalkEvent(sse('segment', { index: 'x', text: 'Hi' }))).toEqual({ type: 'ignored' })
    expect(parseTalkEvent(sse('ping', {}))).toEqual({ type: 'ignored' })
  })
})

describe('Tavus interactions', () => {
  it('builds the echo message exactly as documented', () => {
    expect(echoInteraction('c123', '<emotion value="excited"/> Hi!')).toEqual({
      message_type: 'conversation',
      event_type: 'conversation.echo',
      conversation_id: 'c123',
      properties: { modality: 'text', text: '<emotion value="excited"/> Hi!', done: true },
    })
  })

  it('sends echo and interrupt messages to every participant', () => {
    const call = { sendAppMessage: vi.fn() }
    sendEcho(call, 'c123', 'Hello.')
    sendInterrupt(call, 'c123')
    expect(call.sendAppMessage).toHaveBeenNthCalledWith(1, echoInteraction('c123', 'Hello.'), '*')
    expect(call.sendAppMessage).toHaveBeenNthCalledWith(2, interruptInteraction('c123'), '*')
    expect(interruptInteraction('c123')).toEqual({
      message_type: 'conversation',
      event_type: 'conversation.interrupt',
      conversation_id: 'c123',
    })
  })

  it('recognises the replica starting and stopping speaking', () => {
    expect(replicaSpeakingFrom({ event_type: 'conversation.replica.started_speaking' })).toBe(true)
    expect(replicaSpeakingFrom({ event_type: 'conversation.replica.stopped_speaking' })).toBe(false)
    expect(replicaSpeakingFrom({ event_type: 'conversation.utterance' })).toBeNull()
    expect(replicaSpeakingFrom(null)).toBeNull()
    expect(replicaSpeakingFrom('started')).toBeNull()
  })
})

describe('speech recognition helpers', () => {
  const result = (transcript: string, isFinal: boolean) => ({ isFinal, length: 1, 0: { transcript } })

  it('joins results into one tidy transcript', () => {
    expect(joinTranscript([result('why do ', true), result(' cats purr', false)])).toEqual({ text: 'why do cats purr', final: false })
    expect(joinTranscript([result('why do cats purr', true)])).toEqual({ text: 'why do cats purr', final: true })
    expect(joinTranscript([])).toEqual({ text: '', final: false })
  })

  it('listens for English, keeping the learner’s own variety', () => {
    expect(recognitionLanguage('en-GB')).toBe('en-GB')
    expect(recognitionLanguage('fr-FR')).toBe('en-US')
    expect(recognitionLanguage(undefined)).toBe('en-US')
  })

  it('explains problems in plain words and stays quiet when cancelled', () => {
    expect(recognitionErrorMessage('aborted')).toBeNull()
    expect(recognitionErrorMessage('not-allowed')).toMatch(/microphone is blocked/i)
    expect(recognitionErrorMessage('no-speech')).toMatch(/didn’t hear/)
    expect(recognitionErrorMessage('something-new')).toMatch(/type your question/)
  })
})
