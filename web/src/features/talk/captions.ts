// Which words of an answer are visible, per the session's caption mode
// (a progression parameter of the listening track):
//   full    - each segment's text shows while it is spoken;
//   delayed - each segment's text shows once it has been spoken;
//   off     - no text; a "Show the words" button appears after the turn ends.
// Text is never shown for something the learner has not heard yet (except in
// full mode, for the sentence being spoken).

import type { CaptionMode } from '../../api/types'
import { hasPendingSpeech, type Turn } from './turns'

export type CaptionLine = { index: number; text: string; current: boolean }

export type CaptionView = {
  lines: CaptionLine[]
  /** Delayed mode: the current sentence's words appear after it is spoken. */
  waitingForSpeech: boolean
  /** Off mode: the answer is still going, words are hidden. */
  hiddenWhileSpeaking: boolean
  /** Off mode: the turn is over and the words can be revealed. */
  canReveal: boolean
}

/** Without a voice, captions are the only way to get the answer. */
export function effectiveCaptionMode(mode: CaptionMode, voiceAvailable: boolean): CaptionMode {
  return voiceAvailable ? mode : 'full'
}

export function captionView(mode: CaptionMode, turn: Pick<Turn, 'segments' | 'status' | 'revealed'>): CaptionView {
  const spoken = turn.segments.filter((s) => s.state === 'spoken')
  const finished = turn.status !== 'streaming' && !hasPendingSpeech(turn)
  const toLine = (s: Turn['segments'][number]): CaptionLine => ({ index: s.index, text: s.text, current: s.state === 'speaking' })
  const view: CaptionView = { lines: [], waitingForSpeech: false, hiddenWhileSpeaking: false, canReveal: false }

  switch (mode) {
    case 'full':
      view.lines = turn.segments.filter((s) => s.state === 'speaking' || s.state === 'spoken').map(toLine)
      break
    case 'delayed':
      view.lines = spoken.map(toLine)
      view.waitingForSpeech = !finished && turn.segments.some((s) => s.state === 'speaking')
      break
    case 'off':
      view.lines = turn.revealed ? spoken.map(toLine) : []
      view.hiddenWhileSpeaking = !finished
      view.canReveal = finished && !turn.revealed && spoken.length > 0
      break
  }
  return view
}

export const CAPTION_MODE_LABELS: Record<CaptionMode, string> = {
  full: 'Words show while the avatar speaks',
  delayed: 'Words show after each sentence',
  off: 'Listen first, then show the words',
}
