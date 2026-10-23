import { Link } from 'react-router'
import { getProgress } from '../api/endpoints'
import { ErrorNotice } from '../components/ErrorNotice'
import { Icon } from '../components/Icon'
import { LoadingBlock } from '../components/LoadingBlock'
import { groupByModule, isBrandNew, MODULE_INFO, MODULE_ORDER } from '../features/progress/progress'
import '../features/progress/progress.css'
import { TrackCard } from '../features/progress/TrackCard'
import { WeekOverview } from '../features/progress/WeekOverview'
import { usePageTitle } from '../lib/usePageTitle'
import { useResource } from '../lib/useResource'

const HOW_LEVELS_WORK =
  'Levels only go up after a perfect session, at most one step a day, and if practice starts to feel hard they ease back a step so it stays comfortable.'

function GettingStarted() {
  return (
    <section className="card stack" aria-labelledby="start-title">
      <h2 id="start-title">Your progress will show up here</h2>
      <p>
        Finish your first practice session and this page will fill in with your levels, streaks and weekly practice. Pick
        anything to start — a few minutes is plenty.
      </p>
      <div className="row">
        {MODULE_ORDER.map((module) => (
          <Link key={module} className="btn btn--secondary" to={MODULE_INFO[module].to}>
            {MODULE_INFO[module].title}
          </Link>
        ))}
      </div>
    </section>
  )
}

export default function ProgressPage() {
  usePageTitle('Progress')
  const progress = useResource(getProgress)
  const data = progress.data
  const groups = data ? groupByModule(data.tracks) : []

  return (
    <>
      <div className="page-header">
        <h1>Your progress</h1>
        <p>{HOW_LEVELS_WORK}</p>
      </div>

      <ErrorNotice error={progress.error} onRetry={progress.reload} />
      {progress.loading && !data && <LoadingBlock label="Gathering your progress…" />}

      {data && isBrandNew(data.tracks) && <GettingStarted />}

      {data && !isBrandNew(data.tracks) && (
        <div className="stack">
          <WeekOverview tracks={data.tracks} weekStart={data.week_start} />
          {groups.map((group) => {
            const info = MODULE_INFO[group.module]
            return (
              <section key={group.module} className="progress-module" aria-labelledby={`module-${group.module}`}>
                <div className="progress-module__head">
                  <h2 id={`module-${group.module}`}>{info.title}</h2>
                  <Link className="btn btn--ghost" to={info.to}>
                    {info.action} <Icon name="arrowRight" />
                  </Link>
                </div>
                <div className="progress-module__tracks">
                  {group.tracks.map((track) => (
                    <TrackCard key={track.key} track={track} weekStart={data.week_start} />
                  ))}
                </div>
              </section>
            )
          })}
        </div>
      )}
    </>
  )
}
