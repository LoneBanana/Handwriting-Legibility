// Grouping and labelling SpeechSynthesis voices for the voice picker. Works on
// plain objects so it can be tested without a browser.

export type VoiceInfo = { name: string; lang: string; default: boolean; localService: boolean }
export type VoiceGroup = { language: string; label: string; voices: VoiceInfo[] }

export function toVoiceInfo(voice: Pick<SpeechSynthesisVoice, 'name' | 'lang' | 'default' | 'localService'>): VoiceInfo {
  return { name: voice.name, lang: voice.lang, default: voice.default, localService: voice.localService }
}

/** "en-US" / "en_US" → { language: "en", region: "US" }. */
export function splitLang(lang: string): { language: string; region: string | null } {
  const [language = '', region] = lang.replace('_', '-').split('-')
  return { language: language.toLowerCase(), region: region ? region.toUpperCase() : null }
}

function displayName(code: string, type: 'language' | 'region'): string | null {
  try {
    return new Intl.DisplayNames(['en'], { type }).of(code) ?? null
  } catch {
    return null
  }
}

export function languageLabel(language: string): string {
  return (language && displayName(language, 'language')) || language || 'Other'
}

/** "Samantha (United States) · device default". */
export function voiceLabel(voice: VoiceInfo): string {
  const { region } = splitLang(voice.lang)
  const where = region ? displayName(region, 'region') ?? region : null
  let label = where ? `${voice.name} (${where})` : voice.name
  if (voice.default) label += ' · device default'
  if (!voice.localService) label += ' · needs internet'
  return label
}

/**
 * Voices grouped by language: the learner's own language first, then English,
 * then the rest alphabetically. Within a group the device default comes first,
 * then voices that work offline, then by name.
 */
export function groupVoices(voices: readonly VoiceInfo[], preferredLang = 'en-US'): VoiceGroup[] {
  const preferred = splitLang(preferredLang).language
  const groups = new Map<string, VoiceInfo[]>()
  for (const voice of voices) {
    const { language } = splitLang(voice.lang)
    const list = groups.get(language) ?? []
    list.push(voice)
    groups.set(language, list)
  }
  const rank = (language: string) => (language === preferred ? 0 : language === 'en' ? 1 : 2)
  return Array.from(groups, ([language, list]) => ({
    language,
    label: languageLabel(language),
    voices: list.slice().sort(
      (a, b) =>
        Number(b.default) - Number(a.default) ||
        Number(b.localService) - Number(a.localService) ||
        a.name.localeCompare(b.name),
    ),
  })).sort((a, b) => rank(a.language) - rank(b.language) || a.label.localeCompare(b.label))
}
