import { useId } from 'react'
import { chartGeometry, type ChartPoint } from '../lib/chart'

type Props = {
  points: ChartPoint[]
  title: string
  yLabel?: string
  height?: number
}

/** Small dependency-free line chart. Shows a text summary to assistive tech. */
export function LineChart({ points, title, yLabel = 'Level', height = 160 }: Props) {
  const titleId = useId()
  const descId = useId()
  const width = 560
  const geometry = chartGeometry(points, { width, height, padding: { top: 12, right: 12, bottom: 28, left: 40 } })

  if (points.length === 0) {
    return <p className="muted small">No sessions yet — your chart will appear here.</p>
  }

  const first = points[0]
  const last = points[points.length - 1]
  return (
    <svg
      className="line-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-labelledby={`${titleId} ${descId}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ width: '100%', height: 'auto' }}
    >
      <title id={titleId}>{title}</title>
      <desc id={descId}>
        {`${yLabel} went from ${first.y} on ${first.label} to ${last.y} on ${last.label} over ${points.length} sessions.`}
      </desc>
      {geometry.yTicks.map((tick) => (
        <g key={tick.value}>
          <line
            x1={geometry.plot.left}
            x2={geometry.plot.right}
            y1={tick.y}
            y2={tick.y}
            stroke="var(--border)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
          <text x={geometry.plot.left - 8} y={tick.y + 4} textAnchor="end" fontSize="11" fill="var(--text-muted)">
            {tick.value}
          </text>
        </g>
      ))}
      <path d={geometry.areaPath} fill="var(--accent-soft)" opacity="0.8" />
      <path d={geometry.linePath} fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinejoin="round" />
      {geometry.points.map((point, index) => (
        <circle key={index} cx={point.x} cy={point.y} r={points.length > 60 ? 1.5 : 3} fill="var(--accent-strong)" />
      ))}
      <text x={geometry.plot.left} y={height - 8} fontSize="11" fill="var(--text-muted)">
        {first.label}
      </text>
      <text x={geometry.plot.right} y={height - 8} fontSize="11" textAnchor="end" fill="var(--text-muted)">
        {last.label}
      </text>
    </svg>
  )
}
