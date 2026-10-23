import { useEffect, useRef, type ReactNode } from 'react'
import type { CaptionMode } from '../../api/types'
import { Icon } from '../../components/Icon'
import { captionView } from './captions'
import type { Turn } from './turns'

type Props = {
  turn: Turn
  captionMode: CaptionMode
  /**
   * Announce new caption lines to screen readers. Only when the words are the
   * main way to get the answer (no voice); otherwise a screen reader would
   * talk over the avatar.
   */
  liveCaptions: boolean
  onReveal: () => void
  children?: ReactNode
}

/** One question and the avatar's answer, with words shown per the caption mode. */
export function TurnCard({ turn, captionMode, liveCaptions, onReveal, children }: Props) {
  const view = captionView(captionMode, turn)
  const wordsRef = useRef<HTMLDivElement>(null)
  const revealed = useRef(turn.revealed)

  // After "Show the words", move focus to them so keyboard and screen-reader users land there.
  useEffect(() => {
    if (turn.revealed && !revealed.current) wordsRef.current?.focus()
    revealed.current = turn.revealed
  }, [turn.revealed])

  const thinking = turn.status === 'streaming' && turn.segments.length === 0
  const stopped = turn.status === 'stopped' || turn.segments.some((s) => s.state === 'skipped')

  return (
    <article className="turn" aria-label={`Your question: ${turn.question}`}>
      <p className="turn__question">
        <span className="turn__who">You asked</span>
        {turn.question}
      </p>
      <div className="turn__answer">
        <span className="turn__who">The avatar</span>
        <div
          ref={wordsRef}
          className="captions"
          role="log"
          aria-live={liveCaptions ? 'polite' : 'off'}
          aria-relevant="additions"
          aria-label="The avatar’s answer"
          tabIndex={-1}
        >
          {view.lines.map((line) => (
            <p key={line.index} className="captions__line" data-current={line.current}>
              {line.text}
            </p>
          ))}
        </div>
        {thinking && (
          <p className="muted">
            <span className="spinner" aria-hidden="true" /> Thinking…
          </p>
        )}
        {view.waitingForSpeech && <p className="muted small">Listen first: the words appear after each sentence.</p>}
        {view.hiddenWhileSpeaking && turn.segments.length > 0 && (
          <p className="muted small">
            <Icon name="speaker" size={16} /> Listen carefully. The words are hidden until the answer ends.
          </p>
        )}
        {view.canReveal && (
          <div>
            <button type="button" className="btn btn--secondary" onClick={onReveal}>
              Show the words
            </button>
          </div>
        )}
        {stopped && <p className="muted small">You stopped this answer.</p>}
        {turn.error && (
          <p className="notice notice--error small" role="alert">
            {turn.error}
          </p>
        )}
      </div>
      {children}
    </article>
  )
}
