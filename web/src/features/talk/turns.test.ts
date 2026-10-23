import { describe, expect, it } from 'vitest'
import type { Emotion } from '../../api/types'
import { captionView, effectiveCaptionMode } from './captions'
import { canCheck, isTurnActive, lastSpokenSegment, speakingSegment, turnsReducer, type Turn, type TurnsAction } from './turns'

const segment = (index: number, text: string, emotion: Emotion = 'content') => ({ turn_id: 'srv-1', index, emotion, text })

function play(actions: TurnsAction[]): Turn {
  return actions.reduce(turnsReducer, [] as Turn[])[0]
}

const asked: TurnsAction = { type: 'asked', id: 't1', question: 'Why is the sky blue?' }

describe('turnsReducer', () => {
  it('collects segments in order, once each, and takes the server turn id', () => {
    const turn = play([
      asked,
      { type: 'segment', id: 't1', segment: segment(1, 'Second.') },
      { type: 'segment', id: 't1', segment: segment(0, 'First.') },
      { type: 'segment', id: 't1', segment: segment(0, 'First again?') },
    ])
    expect(turn.segments.map((s) => s.text)).toEqual(['First.', 'Second.'])
    expect(turn.segments.every((s) => s.state === 'queued')).toBe(true)
    expect(turn.turnId).toBe('srv-1')
    expect(isTurnActive(turn)).toBe(true)
  })

  it('moves segments forward only, and allows a check once everything was heard', () => {
    let turn = play([
      asked,
      { type: 'segment', id: 't1', segment: segment(0, 'First.') },
      { type: 'segmentState', id: 't1', index: 0, state: 'speaking' },
      { type: 'done', id: 't1', done: { turn_id: 'srv-1', text: 'First.', echo_text: '' } },
    ])
    expect(speakingSegment(turn)?.index).toBe(0)
    expect(canCheck(turn)).toBe(false)
    turn = turnsReducer([turn], { type: 'segmentState', id: 't1', index: 0, state: 'spoken' })[0]
    const unchanged = turnsReducer([turn], { type: 'segmentState', id: 't1', index: 0, state: 'speaking' })
    expect(unchanged[0]).toBe(turn) // no going back
    expect(lastSpokenSegment(turn)?.text).toBe('First.')
    expect(isTurnActive(turn)).toBe(false)
    expect(canCheck(turn)).toBe(true)
  })

  it('stops cleanly: what was being said counts as heard, the rest is skipped', () => {
    const turn = play([
      asked,
      { type: 'segment', id: 't1', segment: segment(0, 'One.') },
      { type: 'segment', id: 't1', segment: segment(1, 'Two.') },
      { type: 'segmentState', id: 't1', index: 0, state: 'speaking' },
      { type: 'stopped', id: 't1' },
    ])
    expect(turn.status).toBe('stopped')
    expect(turn.segments.map((s) => s.state)).toEqual(['spoken', 'skipped'])
    expect(isTurnActive(turn)).toBe(false)
    expect(canCheck(turn)).toBe(false) // the answer never finished
  })

  it('never offers a check for an answer that failed part-way', () => {
    const turn = play([
      asked,
      { type: 'segment', id: 't1', segment: segment(0, 'One.') },
      { type: 'failed', id: 't1', message: 'Hourly limit reached.' },
      { type: 'allSpoken', id: 't1' },
    ])
    expect(turn.status).toBe('error')
    expect(turn.error).toBe('Hourly limit reached.')
    expect(turn.segments[0].state).toBe('spoken')
    expect(canCheck(turn)).toBe(false)
  })

  it('reveals words and clears the conversation', () => {
    const actions: TurnsAction[] = [asked, { type: 'revealed', id: 't1' }]
    const turns = actions.reduce(turnsReducer, [] as Turn[])
    expect(turns[0].revealed).toBe(true)
    expect(turnsReducer(turns, { type: 'cleared' })).toEqual([])
    expect(turnsReducer(turns, { type: 'revealed', id: 'nope' })).toBe(turns)
  })
})

describe('captionView', () => {
  const base = play([
    asked,
    { type: 'segment', id: 't1', segment: segment(0, 'One.') },
    { type: 'segment', id: 't1', segment: segment(1, 'Two.') },
    { type: 'segment', id: 't1', segment: segment(2, 'Three.') },
    { type: 'segmentState', id: 't1', index: 0, state: 'speaking' },
    { type: 'segmentState', id: 't1', index: 0, state: 'spoken' },
    { type: 'segmentState', id: 't1', index: 1, state: 'speaking' },
  ])

  it('full: shows the sentence being spoken and those before it', () => {
    const view = captionView('full', base)
    expect(view.lines.map((l) => [l.text, l.current])).toEqual([
      ['One.', false],
      ['Two.', true],
    ])
  })

  it('delayed: shows a sentence only after it has been spoken', () => {
    const view = captionView('delayed', base)
    expect(view.lines.map((l) => l.text)).toEqual(['One.'])
    expect(view.waitingForSpeech).toBe(true)
  })

  it('off: hides the words until the turn ends, then offers to show them', () => {
    expect(captionView('off', base)).toMatchObject({ lines: [], hiddenWhileSpeaking: true, canReveal: false })
    const finishing: TurnsAction[] = [
      { type: 'done', id: 't1', done: { turn_id: 'srv-1', text: '', echo_text: '' } },
      { type: 'allSpoken', id: 't1' },
    ]
    const finished = finishing.reduce(turnsReducer, [base])[0]
    expect(captionView('off', finished)).toMatchObject({ lines: [], hiddenWhileSpeaking: false, canReveal: true })
    const revealed = turnsReducer([finished], { type: 'revealed', id: 't1' })[0]
    expect(captionView('off', revealed).lines.map((l) => l.text)).toEqual(['One.', 'Two.', 'Three.'])
  })

  it('falls back to full captions when there is no voice', () => {
    expect(effectiveCaptionMode('off', false)).toBe('full')
    expect(effectiveCaptionMode('delayed', true)).toBe('delayed')
  })
})
