// A simple letter → viseme (mouth shape) approximation for lip-sync. English
// spelling is a rough guide to sound, but mouth shapes only need to look
// plausible: closed lips for m/b/p, a rounded mouth for oo/w, an open jaw for
// "ah", and so on. Timing is estimated from the target speaking rate and
// corrected live by SpeechSynthesis word-boundary events when the voice sends them.

import type { BlendshapeName, BlendshapePatch } from './blendshapes'

export type Viseme =
  | 'rest'
  | 'MBP'
  | 'FV'
  | 'TH'
  | 'DNT'
  | 'SZ'
  | 'CHJ'
  | 'KG'
  | 'R'
  | 'L'
  | 'WQ'
  | 'AA'
  | 'EE'
  | 'IH'
  | 'OH'
  | 'UH'

const VOWELS: ReadonlySet<Viseme> = new Set<Viseme>(['AA', 'EE', 'IH', 'OH', 'UH'])

/** The blendshapes lip-sync drives; everything else belongs to the expression. */
export const VISEME_KEYS: readonly BlendshapeName[] = [
  'jawOpen',
  'mouthClose',
  'mouthFunnel',
  'mouthPucker',
  'mouthRollLower',
  'mouthStretchLeft',
  'mouthStretchRight',
]

const spread = (value: number): BlendshapePatch => ({ mouthStretchLeft: value, mouthStretchRight: value })

export const VISEME_SHAPES: Record<Viseme, BlendshapePatch> = {
  rest: { jawOpen: 0.02 },
  MBP: { jawOpen: 0, mouthClose: 0.6 },
  FV: { jawOpen: 0.1, mouthRollLower: 0.55 },
  TH: { jawOpen: 0.16 },
  DNT: { jawOpen: 0.16 },
  SZ: { jawOpen: 0.08, ...spread(0.12) },
  CHJ: { jawOpen: 0.12, mouthFunnel: 0.5 },
  KG: { jawOpen: 0.26 },
  R: { jawOpen: 0.14, mouthPucker: 0.35, mouthFunnel: 0.15 },
  L: { jawOpen: 0.22 },
  WQ: { jawOpen: 0.08, mouthPucker: 0.8, mouthFunnel: 0.25 },
  AA: { jawOpen: 0.62 },
  EE: { jawOpen: 0.18, ...spread(0.22) },
  IH: { jawOpen: 0.28, ...spread(0.1) },
  OH: { jawOpen: 0.42, mouthFunnel: 0.6 },
  UH: { jawOpen: 0.3, mouthFunnel: 0.3, mouthPucker: 0.2 },
}

const DIGIT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

/** Multi-letter spellings, checked before single letters (longest first). */
const SPELLINGS: readonly (readonly [string, readonly Viseme[]])[] = [
  ['tch', ['CHJ']],
  ['igh', ['AA', 'EE']],
  ['th', ['TH']],
  ['sh', ['CHJ']],
  ['ch', ['CHJ']],
  ['ph', ['FV']],
  ['wh', ['WQ']],
  ['ng', ['KG']],
  ['ck', ['KG']],
  ['qu', ['KG', 'WQ']],
  ['gh', []],
  ['oo', ['WQ']],
  ['ee', ['EE']],
  ['ea', ['EE']],
  ['ai', ['EE']],
  ['ay', ['EE']],
  ['oa', ['OH']],
  ['ou', ['AA', 'WQ']],
  ['ow', ['OH']],
  ['oi', ['OH', 'EE']],
  ['oy', ['OH', 'EE']],
  ['au', ['OH']],
  ['aw', ['OH']],
  ['ew', ['WQ']],
]

const LETTERS: Record<string, readonly Viseme[]> = {
  a: ['AA'],
  b: ['MBP'],
  c: ['KG'],
  d: ['DNT'],
  e: ['EE'],
  f: ['FV'],
  g: ['KG'],
  h: [],
  i: ['IH'],
  j: ['CHJ'],
  k: ['KG'],
  l: ['L'],
  m: ['MBP'],
  n: ['DNT'],
  o: ['OH'],
  p: ['MBP'],
  q: ['KG'],
  r: ['R'],
  s: ['SZ'],
  t: ['DNT'],
  u: ['UH'],
  v: ['FV'],
  w: ['WQ'],
  x: ['KG', 'SZ'],
  y: ['EE'],
  z: ['SZ'],
}

/** Mouth shapes for one written word, with repeated shapes merged ("ll" → one L). */
export function wordVisemes(word: string): Viseme[] {
  let w = word.toLowerCase().replace(/[^a-z0-9]/g, '')
  w = w.replace(/[0-9]/g, (digit) => DIGIT_WORDS[Number(digit)])
  // A silent final "e" after a consonant (make, hope) adds no mouth shape.
  if (w.length > 3 && /[^aeiouy]e$/.test(w)) w = w.slice(0, -1)

  const out: Viseme[] = []
  const push = (visemes: readonly Viseme[]) => {
    for (const viseme of visemes) if (out[out.length - 1] !== viseme) out.push(viseme)
  }
  let i = 0
  while (i < w.length) {
    if (i === 0 && w.startsWith('gh')) {
      push(['KG'])
      i += 2
      continue
    }
    const spelling = SPELLINGS.find(([letters]) => w.startsWith(letters, i))
    if (spelling) {
      push(spelling[1])
      i += spelling[0].length
      continue
    }
    const letter = w[i]
    push(letter === 'c' && /[eiy]/.test(w[i + 1] ?? '') ? ['SZ'] : (LETTERS[letter] ?? []))
    i += 1
  }
  return out
}

export type VisemeKey = { at: number; viseme: Viseme }
export type WordTiming = { charStart: number; charEnd: number; start: number; end: number }
export type VisemeTimeline = { keys: VisemeKey[]; words: WordTiming[]; duration: number }

export const SENTENCE_PAUSE_MS = 320
export const CLAUSE_PAUSE_MS = 160

const WORD_PATTERN = /[A-Za-z0-9'’]+/g

const unitWeight = (viseme: Viseme) => (VOWELS.has(viseme) ? 1.4 : 1)

/**
 * Lays out mouth shapes over time for `text` spoken at `wordsPerMinute`:
 * longer words get more time, vowels a little more than consonants, and
 * punctuation adds a pause with the mouth at rest. The whole utterance lasts
 * about `words / wordsPerMinute` minutes.
 */
export function buildVisemeTimeline(text: string, wordsPerMinute: number): VisemeTimeline {
  const wpm = Number.isFinite(wordsPerMinute) && wordsPerMinute > 0 ? Math.min(320, Math.max(60, wordsPerMinute)) : 160
  const tokens = Array.from(text.matchAll(WORD_PATTERN), (match) => {
    const start = match.index ?? 0
    const visemes = wordVisemes(match[0])
    return { start, end: start + match[0].length, visemes: visemes.length ? visemes : (['IH'] as Viseme[]) }
  })
  if (tokens.length === 0) return { keys: [{ at: 0, viseme: 'rest' }], words: [], duration: 0 }

  const pauses: number[] = tokens.map((token, i) => {
    if (i === tokens.length - 1) return 0
    const gap = text.slice(token.end, tokens[i + 1].start)
    if (/[.!?]/.test(gap)) return SENTENCE_PAUSE_MS
    if (/[,;:–—]|\s-\s/.test(gap)) return CLAUSE_PAUSE_MS
    return 0
  })
  const weights = tokens.map((token) => token.visemes.reduce((sum, viseme) => sum + unitWeight(viseme), 0))
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0)
  const target = (tokens.length * 60_000) / wpm
  const pauseTotal = pauses.reduce((sum, pause) => sum + pause, 0)
  const speechTime = Math.max(target - pauseTotal, target * 0.6)

  const keys: VisemeKey[] = []
  const words: WordTiming[] = []
  let t = 0
  tokens.forEach((token, i) => {
    const wordMs = (speechTime * weights[i]) / totalWeight
    const start = t
    let offset = start
    for (const viseme of token.visemes) {
      keys.push({ at: offset, viseme })
      offset += (wordMs * unitWeight(viseme)) / weights[i]
    }
    t = start + wordMs
    words.push({ charStart: token.start, charEnd: token.end, start, end: t })
    if (pauses[i] > 0) {
      keys.push({ at: t, viseme: 'rest' })
      t += pauses[i]
    }
  })
  keys.push({ at: t, viseme: 'rest' })
  return { keys, words, duration: t }
}

/** The mouth shape at `ms` into the timeline. */
export function visemeAt(timeline: VisemeTimeline, ms: number): Viseme {
  const { keys } = timeline
  if (ms < 0 || ms >= timeline.duration || keys.length === 0) return 'rest'
  let lo = 0
  let hi = keys.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (keys[mid].at <= ms) lo = mid
    else hi = mid - 1
  }
  return keys[lo].viseme
}

/**
 * When the word at (or just after) `charIndex` starts. SpeechSynthesis
 * boundary events report character offsets, so this re-syncs the lips.
 */
export function timeForChar(timeline: VisemeTimeline, charIndex: number): number {
  const word = timeline.words.find((w) => charIndex < w.charEnd)
  return word ? word.start : timeline.duration
}

/** The word being said at `ms` (during a pause: the word before it). */
export function wordAt(timeline: VisemeTimeline, ms: number): WordTiming | null {
  let found: WordTiming | null = null
  for (const word of timeline.words) {
    if (word.start > ms) break
    found = word
  }
  return found
}
