// Conversation state for the Talk page as a pure reducer: each question the
// learner asks becomes a turn whose answer arrives as emotion-tagged segments,
// and each segment moves queued → speaking → spoken as the avatar says it.

import type { DoneEvent, Emotion, SegmentEvent } from '../../api/types'

export type SegmentState = 'queued' | 'speaking' | 'spoken' | 'skipped'

export type TurnSegment = { index: number; emotion: Emotion; text: string; state: SegmentState }

export type TurnStatus = 'streaming' | 'done' | 'stopped' | 'error'

export type Turn = {
  /** Local id (the server's turn id only arrives with the first segment). */
  id: string
  question: string
  turnId: string | null
  segments: TurnSegment[]
  status: TurnStatus
  /** The whole answer, from the `done` event. */
  text: string
  error: string | null
  /** Captions-off mode: the learner asked to see the words. */
  revealed: boolean
}

export type TurnsAction =
  | { type: 'asked'; id: string; question: string }
  | { type: 'segment'; id: string; segment: SegmentEvent }
  | { type: 'done'; id: string; done: DoneEvent }
  | { type: 'failed'; id: string; message: string }
  | { type: 'stopped'; id: string }
  | { type: 'segmentState'; id: string; index: number; state: 'speaking' | 'spoken' }
  | { type: 'allSpoken'; id: string }
  | { type: 'revealed'; id: string }
  | { type: 'cleared' }

const RANK: Record<SegmentState, number> = { queued: 0, speaking: 1, spoken: 2, skipped: 2 }

function updateTurn(turns: Turn[], id: string, update: (turn: Turn) => Turn): Turn[] {
  let changed = false
  const next = turns.map((turn) => {
    if (turn.id !== id) return turn
    const updated = update(turn)
    if (updated !== turn) changed = true
    return updated
  })
  return changed ? next : turns
}

export function turnsReducer(turns: Turn[], action: TurnsAction): Turn[] {
  switch (action.type) {
    case 'asked':
      return [
        ...turns,
        {
          id: action.id,
          question: action.question,
          turnId: null,
          segments: [],
          status: 'streaming',
          text: '',
          error: null,
          revealed: false,
        },
      ]
    case 'segment':
      return updateTurn(turns, action.id, (turn) => {
        const { segment } = action
        if (turn.segments.some((s) => s.index === segment.index)) return turn
        const segments = [
          ...turn.segments,
          { index: segment.index, emotion: segment.emotion, text: segment.text, state: 'queued' as const },
        ].sort((a, b) => a.index - b.index)
        return { ...turn, turnId: turn.turnId ?? segment.turn_id, segments }
      })
    case 'done':
      return updateTurn(turns, action.id, (turn) => ({
        ...turn,
        status: turn.status === 'stopped' ? 'stopped' : 'done',
        turnId: turn.turnId ?? (action.done.turn_id || null),
        text: action.done.text || turn.segments.map((s) => s.text).join(' '),
      }))
    case 'failed':
      return updateTurn(turns, action.id, (turn) => ({ ...turn, status: 'error', error: action.message }))
    case 'stopped':
      return updateTurn(turns, action.id, (turn) => ({
        ...turn,
        status: turn.status === 'streaming' ? 'stopped' : turn.status,
        // What was being said counts as heard; anything still waiting is dropped.
        segments: turn.segments.map((s) =>
          s.state === 'speaking' ? { ...s, state: 'spoken' } : s.state === 'queued' ? { ...s, state: 'skipped' } : s,
        ),
      }))
    case 'segmentState':
      return updateTurn(turns, action.id, (turn) => {
        const target = turn.segments.find((s) => s.index === action.index)
        if (!target || RANK[target.state] >= RANK[action.state]) return turn
        return {
          ...turn,
          segments: turn.segments.map((s) => (s.index === action.index ? { ...s, state: action.state } : s)),
        }
      })
    case 'allSpoken':
      return updateTurn(turns, action.id, (turn) =>
        hasPendingSpeech(turn)
          ? {
              ...turn,
              segments: turn.segments.map((s) =>
                s.state === 'queued' || s.state === 'speaking' ? { ...s, state: 'spoken' } : s,
              ),
            }
          : turn,
      )
    case 'revealed':
      return updateTurn(turns, action.id, (turn) => (turn.revealed ? turn : { ...turn, revealed: true }))
    case 'cleared':
      return turns.length ? [] : turns
  }
}

/** Segments are still waiting to be (or being) spoken. */
export function hasPendingSpeech(turn: Pick<Turn, 'segments'>): boolean {
  return turn.segments.some((s) => s.state === 'queued' || s.state === 'speaking')
}

/** The answer is still arriving or still being spoken. */
export function isTurnActive(turn: Turn): boolean {
  return (turn.status === 'streaming' && turn.error === null) || hasPendingSpeech(turn)
}

/** A comprehension check makes sense once the whole answer has been heard. */
export function canCheck(turn: Turn): boolean {
  return turn.status === 'done' && turn.turnId !== null && !hasPendingSpeech(turn) && turn.segments.some((s) => s.state === 'spoken')
}

export function speakingSegment(turn: Turn | null | undefined): TurnSegment | null {
  return turn?.segments.find((s) => s.state === 'speaking') ?? null
}

/** The most recent segment the learner has finished hearing. */
export function lastSpokenSegment(turn: Turn | null | undefined): TurnSegment | null {
  if (!turn) return null
  for (let i = turn.segments.length - 1; i >= 0; i -= 1) {
    if (turn.segments[i].state === 'spoken') return turn.segments[i]
  }
  return null
}
