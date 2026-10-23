import type { TrackParameter } from '../api/types'

export function percent(value: number, digits = 0): string {
  if (!Number.isFinite(value)) return '0%'
  return `${(Math.min(1, Math.max(0, value)) * 100).toFixed(digits)}%`
}

/** Formats a number with up to `digits` decimals and no trailing zeros. */
export function trimNumber(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '–'
  return Number(value.toFixed(digits)).toLocaleString(undefined, { maximumFractionDigits: digits })
}

export function formatValue(value: number, unit: string): string {
  switch (unit) {
    case 'ms':
      return value >= 1000 ? `${trimNumber(value / 1000, 1)} s` : `${Math.round(value)} ms`
    case 's':
      return `${trimNumber(value, 1)} s`
    case '%':
      return `${trimNumber(value, 0)}%`
    case 'x':
    case '×':
      return `×${trimNumber(value, 2)}`
    case '':
      return trimNumber(value, 2)
    default:
      return `${trimNumber(value, 1)} ${unit}`
  }
}

export function formatParameter(parameter: TrackParameter): string {
  return formatValue(parameter.value, parameter.unit)
}

/** Fraction of the way from start to target (handles decreasing parameters such as time limits). */
export function parameterProgress(parameter: TrackParameter): number {
  const span = parameter.target - parameter.start
  if (span === 0) return 1
  return Math.min(1, Math.max(0, (parameter.value - parameter.start) / span))
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '–'
  const date = new Date(iso.length === 10 ? `${iso}T00:00:00` : iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}
