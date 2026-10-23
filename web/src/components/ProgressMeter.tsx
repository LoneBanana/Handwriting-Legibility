import { useId } from 'react'
import { percent } from '../lib/format'

type Props = {
  label: string
  value: number // 0..1
  detail?: string
  compact?: boolean
}

export function ProgressMeter({ label, value, detail, compact = false }: Props) {
  const id = useId()
  const clamped = Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0))
  return (
    <div className={compact ? 'meter meter--compact' : 'meter'}>
      <div className="meter__head">
        <span id={id}>{label}</span>
        <span>{percent(clamped)}</span>
      </div>
      <div
        className="meter__track"
        role="progressbar"
        aria-labelledby={id}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(clamped * 100)}
        aria-valuetext={detail ? `${percent(clamped)}, ${detail}` : percent(clamped)}
      >
        <div className="meter__fill" style={{ width: `${clamped * 100}%` }} />
      </div>
      {detail && <span className="muted small">{detail}</span>}
    </div>
  )
}
