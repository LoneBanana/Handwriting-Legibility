import type { Track } from '../../api/types'
import { ProgressMeter } from '../../components/ProgressMeter'
import { levelsNote, longDate, plural, signed, weekOverview } from './progress'

/** The "this week" summary at the top of the Progress page. */
export function WeekOverview({ tracks, weekStart }: { tracks: Track[]; weekStart: string }) {
  const week = weekOverview(tracks, weekStart)
  return (
    <section className="card stack" aria-labelledby="week-title">
      <div>
        <h2 id="week-title">This week</h2>
        <p className="muted small" style={{ margin: 0 }}>
          Since {longDate(weekStart)}
        </p>
      </div>
      <ul className="stat-tiles">
        <li className="stat-tile">
          <span className="stat-tile__label">Practice sessions</span>
          <span className="stat-tile__value">{week.sessions}</span>
          {week.goal > 0 && <span className="stat-tile__note">aiming for {week.goal}</span>}
        </li>
        <li className="stat-tile">
          <span className="stat-tile__label">Perfect sessions</span>
          <span className="stat-tile__value">{week.perfectSessions}</span>
          <span className="stat-tile__note">these move levels up</span>
        </li>
        <li className="stat-tile">
          <span className="stat-tile__label">Levels</span>
          <span className="stat-tile__value">{signed(week.levelsGained)}</span>
          <span className="stat-tile__note">{levelsNote(week.levelsGained)}</span>
        </li>
        <li className="stat-tile">
          <span className="stat-tile__label">Skills practised</span>
          <span className="stat-tile__value">{week.tracksPractised}</span>
          <span className="stat-tile__note">of {week.tracksTotal}</span>
        </li>
      </ul>
      {week.goal > 0 && (
        <ProgressMeter
          label="Weekly practice goal"
          value={week.goalProgress}
          detail={week.goalProgress >= 1 ? 'Goal reached — brilliant work!' : 'Every session counts, even a short one.'}
        />
      )}
      {week.longestStreak && (
        <p className="small" style={{ margin: 0 }}>
          Longest streak right now: <strong>{plural(week.longestStreak.streak, 'perfect session')}</strong> in a row in{' '}
          {week.longestStreak.label}.
        </p>
      )}
    </section>
  )
}
