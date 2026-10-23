// The Settings page keeps a draft of the reading-comfort settings, previews it
// live, and saves only what changed.

import type { FontChoice, ThemeChoice, UserSettings } from '../../api/types'
import { FONT_SCALE_RANGE, LINE_SPACING_RANGE, normalizeSettings } from '../../lib/settings'

export const DRAFT_KEYS = ['font', 'font_scale', 'line_spacing', 'theme', 'reduce_motion', 'tts_voice'] as const

export type DraftKey = (typeof DRAFT_KEYS)[number]
export type SettingsDraft = Pick<UserSettings, DraftKey>

export function draftFrom(settings: UserSettings): SettingsDraft {
  const s = normalizeSettings(settings)
  return {
    font: s.font,
    font_scale: s.font_scale,
    line_spacing: s.line_spacing,
    theme: s.theme,
    reduce_motion: s.reduce_motion,
    tts_voice: s.tts_voice ?? null,
  }
}

const round2 = (value: number) => Math.round(value * 100) / 100

function same<K extends DraftKey>(key: K, a: SettingsDraft[K], b: SettingsDraft[K]): boolean {
  if (key === 'font_scale' || key === 'line_spacing') return Math.abs(Number(a) - Number(b)) < 0.005
  if (key === 'tts_voice') return (a ?? null) === (b ?? null)
  return a === b
}

/** Only the settings that differ from what is saved. */
export function draftPatch(saved: UserSettings, draft: SettingsDraft): Partial<UserSettings> {
  const base = draftFrom(saved)
  const patch: Partial<UserSettings> = {}
  for (const key of DRAFT_KEYS) {
    if (same(key, base[key], draft[key])) continue
    const value = draft[key]
    ;(patch as Record<string, unknown>)[key] = typeof value === 'number' ? round2(value) : value
  }
  return patch
}

export function isDirty(saved: UserSettings, draft: SettingsDraft): boolean {
  return Object.keys(draftPatch(saved, draft)).length > 0
}

/** Saved settings with the draft applied, for the live preview. */
export function mergeDraft(saved: UserSettings, draft: SettingsDraft): UserSettings {
  return normalizeSettings({ ...saved, ...draft })
}

export const FONT_OPTIONS: { value: FontChoice; label: string; family: string; hint: string }[] = [
  {
    value: 'default',
    label: 'Your device’s font',
    family: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
    hint: 'The everyday font of your phone or computer.',
  },
  {
    value: 'atkinson',
    label: 'Atkinson Hyperlegible',
    family: "'Atkinson Hyperlegible', system-ui, sans-serif",
    hint: 'Letters that are easy to tell apart, made for low vision.',
  },
  {
    value: 'lexend',
    label: 'Lexend',
    family: "'Lexend', system-ui, sans-serif",
    hint: 'Wide, evenly spaced letters that some readers find faster.',
  },
  {
    value: 'opendyslexic',
    label: 'OpenDyslexic',
    family: "'OpenDyslexic', system-ui, sans-serif",
    hint: 'Heavier letter bottoms help stop letters flipping or swapping.',
  },
]

export const THEME_OPTIONS: { value: ThemeChoice; label: string; swatch: { bg: string; text: string; accent: string } }[] = [
  { value: 'light', label: 'Light', swatch: { bg: '#f6f5f1', text: '#1d2a2e', accent: '#2f6f6a' } },
  { value: 'cream', label: 'Cream (softer glare)', swatch: { bg: '#f7efdf', text: '#2a241f', accent: '#2f6f6a' } },
  { value: 'dark', label: 'Dark', swatch: { bg: '#121819', text: '#e8eef0', accent: '#7cc4bd' } },
  { value: 'high-contrast', label: 'High contrast', swatch: { bg: '#000000', text: '#ffffff', accent: '#ffff00' } },
]

export const SCALE_RANGE = FONT_SCALE_RANGE
export const SPACING_RANGE = LINE_SPACING_RANGE

export const formatScale = (value: number) => `${Math.round(value * 100)}%`
export const formatSpacing = (value: number) => `${value.toFixed(1)}×`

export const DELETE_WORD = 'DELETE'

/** The learner typed the confirmation word (case and surrounding spaces don't matter). */
export function deleteConfirmed(typed: string): boolean {
  return typed.trim().toUpperCase() === DELETE_WORD
}
