import { useId, useState, type KeyboardEvent } from 'react'
import type { WeekSummary } from '../../api/types'
import { labelledWeeks, plural, shortDate, signed, weekDescription, weeklyGeometry, weeklySummary } from './progress'

const BOX = { width: 560, height: 180, padding: { top: 14, right: 44, bottom: 30, left: 30 } }

/**
 * Practice sessions per week for the last 12 weeks: one column per week in
 * the accent colour, the weekly goal as a reference line. Hover a week, or
 * focus the chart and use the arrow keys, to read it; the same numbers are
 * available as a table.
 */
export function WeeklyChart({ weeks, goal, title }: { weeks: WeekSummary[]; goal: number; title: string }) {
  const titleId = useId()
  const [active, setActive] = useState<number | null>(null)
  const geometry = weeklyGeometry(weeks, goal, BOX)
  const labelled = labelledWeeks(weeks.length)
  const activeBar = active === null ? null : (geometry.bars[active] ?? null)

  if (weeks.length === 0) return <p className="muted small">No weeks to show yet.</p>

  const onKeyDown = (event: KeyboardEvent) => {
    const last = weeks.length - 1
    const current = active ?? last
    let next: number | null = current
    if (event.key === 'ArrowLeft') next = Math.max(0, current - 1)
    else if (event.key === 'ArrowRight') next = Math.min(last, current + 1)
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = last
    else if (event.key === 'Escape') next = null
    else return
    event.preventDefault()
    setActive(next)
  }

  const tooltipLeft = activeBar ? Math.min(85, Math.max(15, (activeBar.cx / BOX.width) * 100)) : 0

  return (
    <figure className="weekly-chart">
      <figcaption id={titleId} className="weekly-chart__title">
        {title}
      </figcaption>
      <div
        className="weekly-chart__plot"
        tabIndex={0}
        role="group"
        aria-labelledby={titleId}
        aria-describedby={`${titleId}-summary`}
        onKeyDown={onKeyDown}
        onFocus={() => setActive((a) => a ?? weeks.length - 1)}
        onBlur={() => setActive(null)}
        onPointerLeave={() => setActive(null)}
      >
        <svg viewBox={`0 0 ${BOX.width} ${BOX.height}`} aria-hidden="true" focusable="false">
          {geometry.ticks.map((tick) => (
            <g key={tick.value}>
              <line
                x1={geometry.plot.left}
                x2={geometry.plot.right}
                y1={tick.y}
                y2={tick.y}
                className={tick.value === 0 ? 'weekly-chart__baseline' : 'weekly-chart__grid'}
              />
              <text x={geometry.plot.left - 8} y={tick.y + 4} textAnchor="end" className="weekly-chart__tick">
                {tick.value}
              </text>
            </g>
          ))}
          {geometry.bars.map((bar) => (
            <g key={bar.week.week_start}>
              {bar.path && <path d={bar.path} className="weekly-chart__bar" data-active={active === bar.index} />}
              {labelled.has(bar.index) && (
                <text x={bar.cx} y={BOX.height - 8} textAnchor="middle" className="weekly-chart__tick">
                  {bar.index === weeks.length - 1 ? 'This week' : shortDate(bar.week.week_start)}
                </text>
              )}
              {/* The whole week's band is the hover target, not just the painted bar. */}
              <rect
                x={bar.bandLeft}
                y={geometry.plot.top}
                width={bar.bandWidth}
                height={geometry.plot.bottom - geometry.plot.top}
                fill="transparent"
                onPointerEnter={() => setActive(bar.index)}
              />
            </g>
          ))}
          {geometry.goalY !== null && (
            <g className="weekly-chart__goal">
              <line x1={geometry.plot.left} x2={geometry.plot.right} y1={geometry.goalY} y2={geometry.goalY} />
              <text x={geometry.plot.right + 6} y={geometry.goalY + 4}>
                goal
              </text>
            </g>
          )}
        </svg>
        {activeBar && (
          <div className="weekly-chart__tooltip" style={{ left: `${tooltipLeft}%` }} aria-hidden="true">
            <strong>{plural(activeBar.week.sessions, 'session')}</strong>
            <span>{activeBar.week.perfect_sessions} perfect</span>
            <span className="muted">Week of {shortDate(activeBar.week.week_start)}</span>
          </div>
        )}
      </div>
      <p id={`${titleId}-summary`} className="visually-hidden">
        {weeklySummary(weeks, goal)} Use the left and right arrow keys to read each week.
      </p>
      <p className="visually-hidden" aria-live="polite">
        {activeBar ? weekDescription(activeBar.week) : ''}
      </p>
      <details className="weekly-chart__table">
        <summary>Show the numbers</summary>
        <table className="table">
          <caption className="visually-hidden">{title}</caption>
          <thead>
            <tr>
              <th scope="col">Week of</th>
              <th scope="col">Sessions</th>
              <th scope="col">Perfect</th>
              <th scope="col">Levels</th>
            </tr>
          </thead>
          <tbody>
            {weeks.map((week) => (
              <tr key={week.week_start}>
                <th scope="row">{shortDate(week.week_start)}</th>
                <td>{week.sessions}</td>
                <td>{week.perfect_sessions}</td>
                <td>{signed(week.levels_gained)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  )
}
