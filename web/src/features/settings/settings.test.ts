import { describe, expect, it } from 'vitest'
import type { UserSettings } from '../../api/types'
import { DEFAULT_SETTINGS } from '../../lib/settings'
import { deleteConfirmed, draftFrom, draftPatch, formatScale, formatSpacing, isDirty, mergeDraft } from './settingsForm'
import { groupVoices, splitLang, voiceLabel, type VoiceInfo } from './voices'

const saved: UserSettings = { ...DEFAULT_SETTINGS, font: 'lexend', font_scale: 1.2, store_handwriting_samples: true }

describe('settings draft', () => {
  it('starts equal to what is saved', () => {
    const draft = draftFrom(saved)
    expect(isDirty(saved, draft)).toBe(false)
    expect(draftPatch(saved, draft)).toEqual({})
    expect(draft).not.toHaveProperty('store_handwriting_samples')
  })

  it('saves only what changed, rounding slider values', () => {
    const draft = { ...draftFrom(saved), theme: 'cream' as const, font_scale: 1.2000000000000002, line_spacing: 1.7999999999 }
    expect(draftPatch(saved, draft)).toEqual({ theme: 'cream', line_spacing: 1.8 })
    expect(isDirty(saved, draft)).toBe(true)
  })

  it('treats a missing voice as the device default', () => {
    expect(draftPatch({ ...saved, tts_voice: null }, { ...draftFrom(saved), tts_voice: null })).toEqual({})
    expect(draftPatch(saved, { ...draftFrom(saved), tts_voice: 'Samantha' })).toEqual({ tts_voice: 'Samantha' })
  })

  it('previews the draft on top of the saved settings, within the allowed ranges', () => {
    const merged = mergeDraft(saved, { ...draftFrom(saved), font_scale: 9 })
    expect(merged.font_scale).toBe(2)
    expect(merged.store_handwriting_samples).toBe(true)
  })

  it('formats slider values', () => {
    expect(formatScale(1.2)).toBe('120%')
    expect(formatSpacing(1.6)).toBe('1.6×')
  })

  it('needs DELETE typed to delete the account, in any case', () => {
    expect(deleteConfirmed('DELETE')).toBe(true)
    expect(deleteConfirmed('  delete ')).toBe(true)
    expect(deleteConfirmed('delet')).toBe(false)
    expect(deleteConfirmed('')).toBe(false)
  })
})

describe('voices', () => {
  const voice = (name: string, lang: string, extra: Partial<VoiceInfo> = {}): VoiceInfo => ({
    name,
    lang,
    default: false,
    localService: true,
    ...extra,
  })

  it('splits language tags', () => {
    expect(splitLang('en-US')).toEqual({ language: 'en', region: 'US' })
    expect(splitLang('en_gb')).toEqual({ language: 'en', region: 'GB' })
    expect(splitLang('fr')).toEqual({ language: 'fr', region: null })
  })

  it('puts the learner’s language first, then English, then the rest', () => {
    const groups = groupVoices(
      [
        voice('Thomas', 'fr-FR'),
        voice('Zoe', 'en-US', { localService: false }),
        voice('Alex', 'en-US'),
        voice('Anna', 'de-DE'),
        voice('Samantha', 'en-US', { default: true }),
      ],
      'de-DE',
    )
    expect(groups.map((g) => g.language)).toEqual(['de', 'en', 'fr'])
    expect(groups[1].voices.map((v) => v.name)).toEqual(['Samantha', 'Alex', 'Zoe'])
  })

  it('labels voices with region, default and online needs', () => {
    const label = voiceLabel(voice('Zoe', 'en-US', { default: true, localService: false }))
    expect(label).toMatch(/^Zoe \(/)
    expect(label).toMatch(/device default/)
    expect(label).toMatch(/needs internet/)
    expect(voiceLabel(voice('Plain', ''))).toBe('Plain')
  })
})
