import type { Visual } from '../../api/types'
import { describeVisual, diceGroups, dicePips, lineDots, niceTicks, randomDots, tenFrameCells } from './layouts'

const DOT = 'var(--accent-strong)'

function Dots({ visual, seed }: { visual: Extract<Visual, { type: 'dots' }>; seed: string }) {
  if (visual.layout === 'dice') {
    const groups = diceGroups(visual.count)
    const size = 120
    const gap = 20
    const width = groups.length * size + (groups.length - 1) * gap
    return (
      <svg viewBox={`0 0 ${width} ${size}`} width={Math.min(width, 520)} aria-hidden="true">
        {groups.map((count, g) => (
          <g key={g} transform={`translate(${g * (size + gap)},0)`}>
            <rect x={2} y={2} width={size - 4} height={size - 4} rx={18} fill="var(--surface)" stroke="var(--border)" strokeWidth={3} />
            {dicePips(count).map((p, i) => (
              <circle key={i} cx={p.x * size} cy={p.y * size} r={11} fill={DOT} />
            ))}
          </g>
        ))}
      </svg>
    )
  }
  const points = visual.layout === 'line' ? lineDots(visual.count) : randomDots(visual.count, seed)
  const width = visual.layout === 'line' ? Math.max(200, visual.count * 44) : 240
  const height = visual.layout === 'line' ? 60 : 240
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={Math.min(width, 520)} aria-hidden="true">
      <rect x={1} y={1} width={width - 2} height={height - 2} rx={16} fill="var(--surface)" stroke="var(--border)" strokeWidth={2} />
      {points.map((p, i) => (
        <circle key={i} cx={p.x * width} cy={p.y * height} r={14} fill={DOT} />
      ))}
    </svg>
  )
}

function TenFrame({ visual }: { visual: Extract<Visual, { type: 'ten_frame' }> }) {
  const cells = tenFrameCells(visual.filled, visual.frames)
  const frames = Math.max(...cells.map((c) => c.frame)) + 1
  const cell = 44
  const frameWidth = cell * 5
  const frameGap = 16
  const width = frames * frameWidth + (frames - 1) * frameGap + 4
  const height = cell * 2 + 4
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={Math.min(width, 560)} aria-hidden="true">
      {cells.map((c, i) => {
        const x = 2 + c.frame * (frameWidth + frameGap) + c.col * cell
        const y = 2 + c.row * cell
        return (
          <g key={i}>
            <rect x={x} y={y} width={cell} height={cell} fill="var(--surface)" stroke="var(--text-muted)" strokeWidth={2} />
            {c.filled && <circle cx={x + cell / 2} cy={y + cell / 2} r={cell * 0.32} fill={DOT} />}
          </g>
        )
      })}
    </svg>
  )
}

function ArrayVisual({ visual }: { visual: Extract<Visual, { type: 'array' }> }) {
  const rows = Math.min(visual.rows, 15)
  const cols = Math.min(visual.cols, 15)
  const step = 30
  const width = cols * step + 10
  const height = rows * step + 10
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={Math.min(width * 1.4, 520)} aria-hidden="true">
      {Array.from({ length: rows }, (_, r) =>
        Array.from({ length: cols }, (_, c) => (
          <circle key={`${r}-${c}`} cx={5 + step / 2 + c * step} cy={5 + step / 2 + r * step} r={10} fill={DOT} />
        )),
      )}
    </svg>
  )
}

function NumberLine({ visual }: { visual: Extract<Visual, { type: 'number_line' }> }) {
  const width = 560
  const height = 110
  const left = 30
  const right = width - 30
  const y = 60
  const ticks = niceTicks(visual.min, visual.max, 10)
  const xFor = (v: number) => left + ((v - visual.min) / (visual.max - visual.min || 1)) * (right - left)
  const labelEvery = ticks.length > 12 ? 2 : 1
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ maxWidth: width }} aria-hidden="true">
      <line x1={left} x2={right} y1={y} y2={y} stroke="var(--text)" strokeWidth={3} strokeLinecap="round" />
      {ticks.map((t, i) => {
        const isEnd = i === 0 || i === ticks.length - 1
        return (
          <g key={t}>
            <line x1={xFor(t)} x2={xFor(t)} y1={y - (isEnd ? 14 : 9)} y2={y + (isEnd ? 14 : 9)} stroke="var(--text)" strokeWidth={2} />
            {(isEnd || i % labelEvery === 0) && (
              <text x={xFor(t)} y={y + 34} textAnchor="middle" fontSize={16} fill="var(--text)">
                {t}
              </text>
            )}
          </g>
        )
      })}
      {visual.target !== null && (
        <g>
          <path
            d={`M${xFor(visual.target)},${y - 8} l-10,-22 h20 z`}
            fill="var(--focus)"
            stroke="var(--focus)"
            strokeLinejoin="round"
          />
        </g>
      )}
    </svg>
  )
}

export function MathVisual({ visual, seed }: { visual: Visual; seed: string }) {
  return (
    <figure className="math-visual" style={{ margin: 0 }}>
      {visual.type === 'dots' && <Dots visual={visual} seed={seed} />}
      {visual.type === 'ten_frame' && <TenFrame visual={visual} />}
      {visual.type === 'array' && <ArrayVisual visual={visual} />}
      {visual.type === 'number_line' && <NumberLine visual={visual} />}
      <figcaption className="visually-hidden">{describeVisual(visual)}</figcaption>
    </figure>
  )
}
