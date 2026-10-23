import { Link } from 'react-router'
import { getProgress } from '../api/endpoints'
import type { Track } from '../api/types'
import { useAuth } from '../auth/context'
import { ErrorNotice } from '../components/ErrorNotice'
import { ProgressMeter } from '../components/ProgressMeter'
import { usePageTitle } from '../lib/usePageTitle'
import { useResource } from '../lib/useResource'

const MODULES = [
  {
    to: '/handwriting',
    title: 'Handwriting',
    blurb: 'Snap a photo of handwriting and get clear, readable text — with a read-aloud button.',
    tracks: ['handwriting.legibility'],
    action: 'Transcribe writing',
  },
  {
    to: '/reading',
    title: 'Reading & spelling',
    blurb: 'Short flash-card sessions that build sound-letter patterns, one step at a time.',
    tracks: ['literacy.reading', 'literacy.spelling'],
    action: 'Start cards',
  },
  {
    to: '/math',
    title: 'Math facts',
    blurb: 'Quick rounds of number sense and facts, with pictures when you want them.',
    tracks: ['numeracy.fluency'],
    action: 'Start a round',
  },
  {
    to: '/talk',
    title: 'Talk & faces',
    blurb: 'Ask anything. Listen to a friendly avatar and learn to read its facial expressions.',
    tracks: ['converse.listening', 'converse.expressions'],
    action: 'Start talking',
  },
] as const

export function HomePage() {
  usePageTitle('Home')
  const { user } = useAuth()
  const progress = useResource(getProgress)
  const byKey = new Map<string, Track>((progress.data?.tracks ?? []).map((track) => [track.key, track]))

  return (
    <>
      <div className="page-header">
        <h1>
          Hi {user?.display_name ?? 'there'} <span aria-hidden="true">👋</span>
        </h1>
        <p>Pick something to work on. Every perfect session nudges things forward a tiny bit — small steps add up.</p>
      </div>
      <ErrorNotice error={progress.error} onRetry={progress.reload} />
      <div className="grid">
        {MODULES.map((module) => (
          <section key={module.to} className="card stack" aria-labelledby={`module-${module.to.slice(1)}`}>
            <div>
              <h2 id={`module-${module.to.slice(1)}`}>{module.title}</h2>
              <p className="muted">{module.blurb}</p>
            </div>
            {module.tracks.map((key) => {
              const track = byKey.get(key)
              return (
                <ProgressMeter
                  key={key}
                  compact
                  label={track?.label ?? key.split('.')[1].replace(/^\w/, (c) => c.toUpperCase())}
                  value={track?.progress ?? 0}
                  detail={track ? `Level ${track.level} · streak ${track.streak}` : progress.loading ? 'Loading…' : 'Not started yet'}
                />
              )
            })}
            <div>
              <Link className="btn" to={module.to}>
                {module.action}
              </Link>
            </div>
          </section>
        ))}
      </div>
    </>
  )
}
