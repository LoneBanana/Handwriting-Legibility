import { useId } from 'react'
import { Link } from 'react-router'
import type { Track } from '../../api/types'
import { LineChart } from '../../components/LineChart'
import { ProgressMeter } from '../../components/ProgressMeter'
import { formatDate, formatParameter, formatValue } from '../../lib/format'
import { historyPoints, MODULE_INFO, plural, streakText, thisWeek } from './progress'
import { WeeklyChart } from './WeeklyChart'

/** Everything about one skill track: level, meters, streaks, settings and history. */
export function TrackCard({ track, weekStart }: { track: Track; weekStart: string }) {
  const titleId = useId()
  const week = thisWeek(track, weekStart)
  const parameters = Object.entries(track.parameters)
  const module = MODULE_INFO[track.module]

  if (track.sessions === 0) {
    return (
      <article className="card stack track-card" aria-labelledby={titleId}>
        <div className="card__title">
          <h3 id={titleId}>{track.label}</h3>
          <span className="badge">Not started</span>
        </div>
        <p className="muted">{track.description}</p>
        <div>
          <Link className="btn btn--secondary" to={module.to}>
            {module.action}
          </Link>
        </div>
      </article>
    )
  }

  return (
    <article className="card stack track-card" aria-labelledby={titleId}>
      <div className="card__title">
        <h3 id={titleId}>{track.label}</h3>
        <span className="badge badge--accent">
          Level {track.level} of {track.max_level}
        </span>
      </div>
      <p className="muted">{track.description}</p>

      <div className="track-card__body">
        <div className="stack">
          <ProgressMeter label="Toward this year’s goal" value={track.progress} />
          <ProgressMeter
            label="This week"
            value={track.weekly_goal > 0 ? week.sessions / track.weekly_goal : 0}
            detail={`${week.sessions} of ${plural(track.weekly_goal, 'session')} · ${week.perfect_sessions} perfect`}
            compact
          />
          <dl className="facts">
            <div>
              <dt>Streak</dt>
              <dd>{streakText(track)}</dd>
            </div>
            <div>
              <dt>All sessions</dt>
              <dd>
                {track.sessions} ({track.perfect_sessions} perfect)
              </dd>
            </div>
            <div>
              <dt>Last level up</dt>
              <dd>{track.last_promoted_on ? formatDate(track.last_promoted_on) : 'Not yet'}</dd>
            </div>
          </dl>
          {parameters.length > 0 && (
            <div>
              <h4>Where things are now</h4>
              <dl className="facts">
                {parameters.map(([key, parameter]) => (
                  <div key={key}>
                    <dt>{parameter.label}</dt>
                    <dd>
                      <strong>{formatParameter(parameter)}</strong>{' '}
                      <span className="muted small">
                        (started at {formatValue(parameter.start, parameter.unit)} → aiming for{' '}
                        {formatValue(parameter.target, parameter.unit)})
                      </span>
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </div>

        <div className="stack">
          <div>
            <h4>Level over time</h4>
            <LineChart points={historyPoints(track)} title={`${track.label}: level over time`} />
          </div>
          <WeeklyChart weeks={track.weekly} goal={track.weekly_goal} title="Sessions per week (last 12 weeks)" />
        </div>
      </div>
    </article>
  )
}
