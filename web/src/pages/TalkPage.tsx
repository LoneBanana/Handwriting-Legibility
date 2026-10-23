import { useState } from 'react'
import type { AvatarProvider } from '../api/types'
import { useAuth } from '../auth/context'
import { ErrorNotice } from '../components/ErrorNotice'
import { Icon } from '../components/Icon'
import { AvatarChooser } from '../features/talk/AvatarChooser'
import { storedProvider, storeProvider, useConverseSession, type SessionRequest } from '../features/talk/session'
import '../features/talk/talk.css'
import { TalkStudio } from '../features/talk/TalkStudio'
import { prefersReducedMotion } from '../lib/settings'
import { usePageTitle } from '../lib/usePageTitle'

export default function TalkPage() {
  usePageTitle('Talk')
  const { settings } = useAuth()
  const [chosen, setChosen] = useState<AvatarProvider>(storedProvider)
  const [request, setRequest] = useState<SessionRequest>(() => ({ provider: chosen, attempt: 0 }))
  const { session, error, loading } = useConverseSession(request)

  // The server falls back to the built-in face when the video avatar isn't configured.
  const fellBack = request.provider === 'tavus' && chosen === 'tavus' && session?.provider === 'local'

  const choose = (next: AvatarProvider) => {
    storeProvider(next)
    setChosen(next)
    if (session && session.provider === next) return
    setRequest((r) => ({ provider: next, attempt: r.attempt + 1 }))
  }
  const retry = () => setRequest((r) => ({ provider: chosen, attempt: r.attempt + 1 }))

  return (
    <>
      <div className="page-header">
        <h1>Talk &amp; faces</h1>
        <p>
          Ask the avatar anything, then listen to its answer. Watch its face as it talks: the panel shows which parts of the
          face move and what feeling that signals.
        </p>
      </div>

      <div className="stack talk-notices">
        <AvatarChooser value={chosen} onChange={choose} disabled={loading} />
        <ErrorNotice error={error} onRetry={retry} />
        {fellBack && (
          <div className="notice" role="status">
            <div className="notice__title">
              <Icon name="info" /> The video avatar isn’t set up yet
            </div>
            <p>
              The realistic video avatar isn’t switched on for this server, so you’re talking with the built-in face. It
              works the same way for practice.
            </p>
            <button type="button" className="btn btn--secondary" onClick={retry}>
              Try the video avatar again
            </button>
          </div>
        )}
        {session?.assistant === 'demo' && (
          <p className="notice small">
            <strong>Demo mode:</strong> answers come from an offline demo, so they are simple practice answers.
          </p>
        )}
      </div>

      <TalkStudio
        key={session?.session_id ?? (loading ? 'loading' : 'none')}
        session={session}
        loading={loading}
        reducedMotion={prefersReducedMotion(settings)}
        voiceName={settings.tts_voice}
        onUseBuiltIn={() => choose('local')}
      />
    </>
  )
}
