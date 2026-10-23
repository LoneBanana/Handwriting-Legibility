import { useSyncExternalStore } from 'react'
import type { Emotion } from '../../api/types'
import { Icon } from '../../components/Icon'
import { AuBars } from './AuBars'
import { EMOTION_FAMILY, EMOTION_WORDS, FEELINGS, type FeelingFamily } from './expressions'
import { strengthWord, type FeelingEstimate } from './faceAnalysis'
import { FaceDiagram } from './FaceDiagram'
import type { FaceStore } from './faceStore'

/** The feeling the avatar was told to show for a sentence the learner has heard. */
export type IntendedFeeling = {
  emotion: Emotion
  /** 1-based sentence number within the answer. */
  sentence: number
  /** What the face reader made of the face as that sentence ended. */
  seenAs: FeelingFamily | null
}

type Props = {
  store: FaceStore
  source: 'avatar' | 'tracker'
  /** Why face reading isn't running yet (video avatar). */
  trackerNote: string | null
  intended: IntendedFeeling | null
  speakingNow: boolean
}

function shownAs(estimate: FeelingEstimate): string {
  if (estimate.family !== 'neutral') return `shown ${strengthWord(estimate.strength)}`
  return estimate.alternative ? `maybe a hint of ${FEELINGS[estimate.alternative].name.toLowerCase()}` : 'relaxed'
}

function Estimate({ estimate }: { estimate: FeelingEstimate }) {
  const feeling = FEELINGS[estimate.family]
  return (
    <>
      <p className="feeling__name">
        {feeling.name} <span className="badge">{shownAs(estimate)}</span>
      </p>
      {estimate.certainty === 'unsure' && estimate.alternative && (
        <p className="small">
          It could also be <strong>{FEELINGS[estimate.alternative].name.toLowerCase()}</strong>: these faces share some
          movements.
        </p>
      )}
      <p className="small">
        <strong>What to look for:</strong> {feeling.lookFor}
      </p>
    </>
  )
}

function Intended({ intended, speakingNow }: { intended: IntendedFeeling | null; speakingNow: boolean }) {
  if (!intended) {
    return (
      <p className="small">
        {speakingNow
          ? 'Watch the face as the avatar talks. What feeling do you see? The answer appears here when the sentence ends.'
          : 'After each sentence, the feeling the avatar meant to show appears here, so you can check your guess.'}
      </p>
    )
  }
  const word = EMOTION_WORDS[intended.emotion]
  const family = EMOTION_FAMILY[intended.emotion]
  const familyName = FEELINGS[family].name
  const kindOf = family !== 'neutral' && familyName.toLowerCase() !== word.word.toLowerCase()
  return (
    <>
      <p className="feeling__name">
        {word.word} <span className="badge">sentence {intended.sentence}</span>
      </p>
      <p className="small">
        That means {word.meaning}
        {kindOf ? ` (a kind of ${familyName.toLowerCase()})` : ''}.
      </p>
      {intended.seenAs &&
        (intended.seenAs === family ? (
          <p className="small feeling__match">
            <Icon name="check" size={16} /> The face reader saw {familyName.toLowerCase()} too.
          </p>
        ) : (
          <p className="small">
            The face reader saw {FEELINGS[intended.seenAs].name.toLowerCase()}. Subtle faces can be read more than one way.
          </p>
        ))}
      {speakingNow && <p className="small muted">Now watch the next sentence…</p>}
    </>
  )
}

/**
 * The live expression panel: which parts of the face are moving (diagram and
 * action-unit bars), what feeling that suggests, and, once each sentence has
 * been heard, the feeling the avatar intended.
 */
export function ExpressionPanel({ store, source, trackerNote, intended, speakingNow }: Props) {
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)
  const waiting = !snapshot.hasData
  const noFace = source === 'tracker' && snapshot.hasData && !snapshot.faceFound
  const dimmed = waiting || noFace

  return (
    <section className="card talk-panel stack" aria-labelledby="face-panel-title">
      <div>
        <h2 id="face-panel-title">Reading the face</h2>
        <p className="muted small" style={{ margin: 0 }}>
          {source === 'avatar'
            ? 'These are the exact movements the built-in avatar is making.'
            : 'Face tracking estimates these movements from the video, so treat them as a good guess.'}
        </p>
      </div>
      {trackerNote && <p className="notice small">{trackerNote}</p>}

      <FaceDiagram regions={snapshot.regions} dimmed={dimmed} />

      <div className="feeling">
        <h3 className="feeling__heading">The face looks</h3>
        {waiting ? (
          <p className="muted small">Waiting for the face…</p>
        ) : noFace ? (
          <p className="muted small">No face in view right now.</p>
        ) : (
          <Estimate estimate={snapshot.estimate} />
        )}
      </div>

      <div className="feeling feeling--intended">
        <h3 className="feeling__heading">The avatar meant to look</h3>
        <Intended intended={intended} speakingNow={speakingNow} />
      </div>

      <div>
        <h3>Face movements</h3>
        <p className="muted small">
          Each movement is an “action unit” (AU) from the Facial Action Coding System, which describes faces muscle by
          muscle.
        </p>
        <AuBars reading={snapshot.reading} dimmed={dimmed} />
      </div>
      <p className="small muted" style={{ margin: 0 }}>
        Tip: while talking, the jaw and lips move with the words. The eyebrows, eyes and cheeks say more about feelings.
      </p>
    </section>
  )
}
