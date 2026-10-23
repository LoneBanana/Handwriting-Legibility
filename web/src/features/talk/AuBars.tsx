import { ACTION_UNITS, type AuReading } from './faceAnalysis'

const ACTIVE = 0.25

/**
 * One row per action unit: its number, what it means, and how active it is.
 * The text carries the value; the bar is a visual aid.
 */
export function AuBars({ reading, dimmed = false }: { reading: AuReading; dimmed?: boolean }) {
  return (
    <ul className="au-list" data-dimmed={dimmed}>
      {ACTION_UNITS.map((unit) => {
        const value = Math.round(Math.min(1, Math.max(0, reading.au[unit.id])) * 100)
        const oneSided = reading.oneSided[unit.id]
        return (
          <li key={unit.id} className="au-list__row" data-active={value >= ACTIVE * 100}>
            <span className="au-list__id">{unit.id}</span>
            <span className="au-list__meaning">
              {unit.meaning}
              {oneSided && <span className="au-list__side"> · mostly one side</span>}
            </span>
            <span className="au-list__value">{value}%</span>
            <span className="au-list__track" aria-hidden="true">
              <span className="au-list__fill" style={{ width: `${value}%` }} />
            </span>
          </li>
        )
      })}
    </ul>
  )
}
