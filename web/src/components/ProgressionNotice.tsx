import { useId } from 'react'
import type { ProgressionUpdate } from '../api/types'
import { formatParameter } from '../lib/format'
import { Icon } from './Icon'
import { ProgressMeter } from './ProgressMeter'

const REASON_TEXT: Record<ProgressionUpdate['reason'], string> = {
  perfect: 'Perfect session — you levelled up.',
  held: 'Level kept the same. A perfect session moves you up.',
  daily_cap: 'Perfect! You already levelled up today, so the next step waits until tomorrow.',
  struggling: 'We eased things back one step so practice feels comfortable again.',
  max_level: 'You are at the top level for this skill.',
}

export function ProgressionNotice({ update, title = 'Your progress' }: { update: ProgressionUpdate; title?: string }) {
  const titleId = useId()
  const tone = update.promoted ? 'notice--success' : update.demoted ? 'notice--warning' : ''
  const parameters = Object.entries(update.parameters)
  return (
    <section className={`notice ${tone}`} aria-live="polite" aria-labelledby={titleId}>
      <div className="notice__title" id={titleId}>
        {update.promoted && <Icon name="star" />}
        {title}
      </div>
      <p>{update.message}</p>
      <p className="muted small">{REASON_TEXT[update.reason]}</p>
      <div className="row" style={{ marginBottom: 'var(--space-3)' }}>
        <span className="badge badge--accent">
          Level {update.previous_level}
          {update.level !== update.previous_level && <> → {update.level}</>}
        </span>
      </div>
      <ProgressMeter label="Toward this year’s goal" value={update.progress} compact />
      {parameters.length > 0 && (
        <ul className="small" style={{ margin: 'var(--space-3) 0 0', paddingLeft: '1.2rem' }}>
          {parameters.map(([key, parameter]) => (
            <li key={key}>
              {parameter.label}: <strong>{formatParameter(parameter)}</strong>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
