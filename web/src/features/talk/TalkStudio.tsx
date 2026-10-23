import { useEffect, useState } from 'react'
import type { ConverseSession } from '../../api/types'
import { Icon } from '../../components/Icon'
import { LoadingBlock } from '../../components/LoadingBlock'
import { speechSupported } from '../../lib/speech'
import { AskBox } from './AskBox'
import { AvatarController } from './avatarController'
import { effectiveCaptionMode } from './captions'
import { ComprehensionCheck } from './ComprehensionCheck'
import { ExpressionPanel } from './ExpressionPanel'
import { createFaceStore } from './faceStore'
import { LocalAvatar } from './LocalAvatar'
import { SessionInfo } from './SessionInfo'
import { canCheck, speakingSegment, type Turn } from './turns'
import { TurnCard } from './TurnCard'
import { useTalkConversation } from './useTalkConversation'
import { VideoAvatar } from './VideoAvatar'

type Props = {
  session: ConverseSession | null
  loading: boolean
  reducedMotion: boolean
  voiceName: string | null
  onUseBuiltIn: () => void
}

function stageStatus(loading: boolean, ready: boolean, active: Turn | null, video: boolean): string {
  if (loading) return 'Getting the avatar ready…'
  if (!ready) return 'The avatar isn’t available right now.'
  if (!active) return 'Ready for your question.'
  const speaking = speakingSegment(active)
  if (speaking) return `Speaking: sentence ${speaking.index + 1}`
  if (active.status === 'streaming' && active.segments.length === 0) return 'Thinking about your question…'
  if (video && active.status !== 'streaming') return 'The video avatar is about to speak…'
  return 'Answering…'
}

/**
 * One conversation: the avatar, the live expression panel, the question box
 * and the answers. Mounted with a `key` per session, so switching avatars or
 * starting over gives a clean slate.
 */
export function TalkStudio({ session, loading, reducedMotion, voiceName, onUseBuiltIn }: Props) {
  const [store] = useState(() => createFaceStore({ notifyEveryMs: reducedMotion ? 400 : 125 }))
  const [avatar] = useState(() => new AvatarController({ reducedMotion }))
  const [trackerNote, setTrackerNote] = useState<string | null>(null)
  const talk = useTalkConversation({ session, avatar, store, voiceName })

  useEffect(() => {
    avatar.setReducedMotion(reducedMotion)
    // Fewer, calmer panel updates when motion is reduced.
    store.setNotifyInterval(reducedMotion ? 400 : 125)
  }, [avatar, store, reducedMotion])

  const video = session?.provider === 'tavus' && session.tavus ? session.tavus : null
  const canSpeak = speechSupported()
  const voiceAvailable = video !== null || (canSpeak && !talk.voiceBlocked)
  const captionMode = effectiveCaptionMode(session?.captions ?? 'full', voiceAvailable)
  const latest = talk.turns.length > 0 ? talk.turns[talk.turns.length - 1] : null
  const earlier = talk.turns.slice(0, -1).reverse()
  const speakingNow = speakingSegment(talk.activeTurn) !== null

  return (
    <>
      {session && !video && (!canSpeak || talk.voiceBlocked) && (
        <p className="notice notice--warning small talk-voice-notice">
          {canSpeak
            ? 'Your browser didn’t let the avatar speak aloud, so its words are shown as text instead. Check that your sound is on, then reload the page to try again.'
            : 'This browser can’t speak aloud, so the avatar’s words are shown as text while its face moves.'}
        </p>
      )}

      <div className="talk-layout">
        <section className="card talk-stage" aria-labelledby="talk-stage-title">
          <h2 id="talk-stage-title" className="visually-hidden">
            The avatar
          </h2>
          <div className="talk-stage__screen">
            {loading ? (
              <LoadingBlock label="Getting the avatar ready…" />
            ) : video ? (
              <VideoAvatar
                key={video.conversation_id}
                tavus={video}
                store={store}
                onCall={talk.setCall}
                onReplicaSpeaking={talk.onReplicaSpeaking}
                onTrackerNote={setTrackerNote}
                onUseBuiltIn={onUseBuiltIn}
              />
            ) : (
              <LocalAvatar controller={avatar} store={store} />
            )}
          </div>
          <div className="talk-stage__bar">
            <p className="talk-status" data-speaking={speakingNow}>
              {stageStatus(loading, session !== null, talk.activeTurn, video !== null)}
            </p>
            {talk.busy && (
              <button type="button" className="btn btn--secondary" onClick={talk.stop}>
                <Icon name="stop" /> Stop
              </button>
            )}
          </div>
          <p className="visually-hidden" role="status">
            {talk.announcement}
          </p>
          {!video && reducedMotion && (
            <p className="muted small" style={{ margin: 0 }}>
              Motion is reduced, so the face shows each feeling without blinking or moving its lips.
            </p>
          )}
          {session && <SessionInfo session={session} captionMode={captionMode} />}
        </section>

        <ExpressionPanel
          store={store}
          source={video ? 'tracker' : 'avatar'}
          trackerNote={video ? trackerNote : null}
          intended={talk.intended}
          speakingNow={speakingNow}
        />

        <section className="talk-chat stack" aria-labelledby="talk-chat-title">
          <h2 id="talk-chat-title" className="visually-hidden">
            Conversation
          </h2>
          <div className="card">
            <AskBox disabled={!session} busy={talk.busy} showStarters={talk.turns.length === 0} onAsk={talk.ask} />
          </div>
          {latest && (
            <div className="card">
              <TurnCard
                turn={latest}
                captionMode={captionMode}
                liveCaptions={!voiceAvailable}
                onReveal={() => talk.reveal(latest.id)}
              >
                {session && canCheck(latest) && latest.turnId && (
                  <ComprehensionCheck key={latest.id} sessionId={session.session_id} turnId={latest.turnId} />
                )}
              </TurnCard>
            </div>
          )}
          {earlier.length > 0 && (
            <details className="card talk-history">
              <summary>
                Earlier questions <span className="badge">{earlier.length}</span>
              </summary>
              <div className="talk-history__list">
                {earlier.map((turn) => (
                  <TurnCard
                    key={turn.id}
                    turn={turn}
                    captionMode={captionMode}
                    liveCaptions={false}
                    onReveal={() => talk.reveal(turn.id)}
                  />
                ))}
              </div>
            </details>
          )}
        </section>
      </div>
    </>
  )
}
