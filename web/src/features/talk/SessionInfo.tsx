import type { CaptionMode, ConverseSession } from '../../api/types'
import { CAPTION_MODE_LABELS } from './captions'

function expressionWord(intensity: number): string {
  if (intensity >= 0.8) return 'big and clear'
  if (intensity >= 0.55) return 'medium'
  return 'subtle'
}

/**
 * Today's practice settings, which change with the learner's listening and
 * face-reading levels. Speaking speed and expression strength only apply to
 * the built-in face; the video avatar speaks and moves at its own pace.
 * Captions apply to both.
 */
export function SessionInfo({ session, captionMode }: { session: ConverseSession; captionMode: CaptionMode }) {
  const level = (key: string) => session.tracks.find((t) => t.track === key)?.level
  const listening = level('converse.listening')
  const faces = level('converse.expressions')
  const builtIn = session.provider === 'local'
  return (
    <>
      <ul className="session-info" aria-label="Today’s practice settings">
        {builtIn && <li className="badge">Speed: {Math.round(session.words_per_minute)} words a minute</li>}
        {builtIn && (
          <li className="badge">
            Expressions: {expressionWord(session.expression_intensity)} ({Math.round(session.expression_intensity * 100)}%)
          </li>
        )}
        <li className="badge">{CAPTION_MODE_LABELS[captionMode]}</li>
        {listening !== undefined && <li className="badge badge--accent">Listening level {listening}</li>}
        {faces !== undefined && <li className="badge badge--accent">Faces level {faces}</li>}
      </ul>
      {!builtIn && (
        <p className="muted small" style={{ margin: 0 }}>
          The video avatar speaks and moves at its own pace. Speaking speed and expression strength follow your levels on
          the built-in face.
        </p>
      )}
    </>
  )
}
