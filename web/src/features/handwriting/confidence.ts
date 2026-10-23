import type { CharConfidence } from '../../api/types'

/** Characters below this confidence are marked as "the model wasn't sure". */
export const LOW_CONFIDENCE = 0.6

export type ConfidenceRun = { text: string; low: boolean; confidence: number }

/**
 * Merges consecutive characters that share the same low/high status into runs,
 * so a line renders as a handful of spans instead of one per character.
 * Whitespace never counts as uncertain.
 */
export function confidenceRuns(chars: CharConfidence[], threshold = LOW_CONFIDENCE): ConfidenceRun[] {
  const runs: ConfidenceRun[] = []
  for (const { char, confidence } of chars) {
    const low = char.trim() !== '' && confidence < threshold
    const last = runs[runs.length - 1]
    if (last && last.low === low) {
      last.text += char
      last.confidence = Math.min(last.confidence, confidence)
    } else {
      runs.push({ text: char, low, confidence })
    }
  }
  return runs
}

export function countUncertain(chars: CharConfidence[], threshold = LOW_CONFIDENCE): number {
  return chars.filter(({ char, confidence }) => char.trim() !== '' && confidence < threshold).length
}

export function legibilityLabel(score: number): string {
  if (score >= 0.9) return 'Very clear'
  if (score >= 0.75) return 'Mostly clear'
  if (score >= 0.5) return 'Some parts were hard to read'
  return 'Hard to read — a clearer photo may help'
}
