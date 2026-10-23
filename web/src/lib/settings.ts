import type { UserSettings } from '../api/types'

export const DEFAULT_SETTINGS: UserSettings = {
  font: 'default',
  font_scale: 1,
  line_spacing: 1.6,
  theme: 'light',
  reduce_motion: false,
  tts_voice: null,
  store_handwriting_samples: false,
}

export const FONT_SCALE_RANGE = { min: 0.8, max: 2, step: 0.1 } as const
export const LINE_SPACING_RANGE = { min: 1.2, max: 2.5, step: 0.1 } as const

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

/** Fills gaps and clamps out-of-range values so the UI never renders unusably. */
export function normalizeSettings(settings: Partial<UserSettings> | null | undefined): UserSettings {
  const merged = { ...DEFAULT_SETTINGS, ...(settings ?? {}) }
  return {
    ...merged,
    font_scale: clamp(Number(merged.font_scale) || 1, FONT_SCALE_RANGE.min, FONT_SCALE_RANGE.max),
    line_spacing: clamp(Number(merged.line_spacing) || 1.6, LINE_SPACING_RANGE.min, LINE_SPACING_RANGE.max),
  }
}

export type RootPresentation = {
  attributes: Record<string, string>
  properties: Record<string, string>
}

/** Maps settings to `<html>` data attributes and CSS custom properties. */
export function settingsToRoot(settings: UserSettings): RootPresentation {
  const s = normalizeSettings(settings)
  return {
    attributes: {
      'data-theme': s.theme,
      'data-font': s.font,
      'data-reduce-motion': String(s.reduce_motion),
    },
    properties: {
      '--font-scale': s.font_scale.toFixed(2),
      '--line-spacing': s.line_spacing.toFixed(2),
    },
  }
}

export function applySettings(settings: UserSettings, root: HTMLElement = document.documentElement): void {
  const { attributes, properties } = settingsToRoot(settings)
  for (const [name, value] of Object.entries(attributes)) root.setAttribute(name, value)
  for (const [name, value] of Object.entries(properties)) root.style.setProperty(name, value)
}

/** True when either the OS or the user's Legible setting asks for less motion. */
export function prefersReducedMotion(settings?: UserSettings | null): boolean {
  if (settings?.reduce_motion) return true
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  } catch {
    return false
  }
}
