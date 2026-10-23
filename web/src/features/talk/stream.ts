// Turns the raw server-sent events of POST /api/converse/sessions/{id}/messages
// into typed events, tolerating malformed or unexpected data.

import type { SseEvent } from '../../api/sse'
import { EMOTIONS, type DoneEvent, type Emotion, type SegmentEvent } from '../../api/types'

export type TalkStreamEvent =
  | { type: 'segment'; segment: SegmentEvent }
  | { type: 'done'; done: DoneEvent }
  | { type: 'error'; message: string }
  | { type: 'ignored' }

export const STREAM_ERROR_MESSAGE = 'The avatar couldn’t finish that answer. Please try asking again.'

const EMOTION_SET: ReadonlySet<string> = new Set(EMOTIONS)

function parseJson(data: string): Record<string, unknown> | null {
  try {
    const value: unknown = JSON.parse(data)
    return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
  } catch {
    return null
  }
}

const asString = (value: unknown): string | null => (typeof value === 'string' ? value : null)

export function toEmotion(value: unknown): Emotion {
  return typeof value === 'string' && EMOTION_SET.has(value) ? (value as Emotion) : 'neutral'
}

export function parseTalkEvent(event: SseEvent): TalkStreamEvent {
  const data = parseJson(event.data)
  switch (event.event) {
    case 'segment': {
      const text = asString(data?.text)
      const index = data?.index
      if (!data || text === null || typeof index !== 'number' || !Number.isFinite(index)) return { type: 'ignored' }
      return {
        type: 'segment',
        segment: { turn_id: asString(data.turn_id) ?? '', index, emotion: toEmotion(data.emotion), text },
      }
    }
    case 'done':
      return {
        type: 'done',
        done: {
          turn_id: asString(data?.turn_id) ?? '',
          text: asString(data?.text) ?? '',
          echo_text: asString(data?.echo_text) ?? '',
        },
      }
    case 'error': {
      const detail = asString(data?.detail)?.trim()
      return { type: 'error', message: detail || STREAM_ERROR_MESSAGE }
    }
    default:
      return { type: 'ignored' }
  }
}
