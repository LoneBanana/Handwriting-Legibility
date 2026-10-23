export type GraphemeSegment = {
  text: string
  start: number
  end: number
  inFocus: boolean
  /** Cycles 0..3 so neighbouring graphemes always differ in colour. */
  colorIndex: number
}

/**
 * Splits `word` into the server-provided graphemes (e.g. "phone" →
 * ph·o·n·e) and flags those overlapping the skill's focus span. If the
 * graphemes don't spell the word, the whole word becomes one segment so we
 * never display something different from what the learner should read.
 */
export function graphemeSegments(
  word: string,
  graphemes: string[],
  focus: [number, number] | null,
): GraphemeSegment[] {
  const usable = graphemes.length > 0 && graphemes.join('').toLowerCase() === word.toLowerCase()
  const lengths = usable ? graphemes.map((g) => g.length) : [word.length]
  const segments: GraphemeSegment[] = []
  let position = 0
  lengths.forEach((length, index) => {
    const start = position
    const end = position + length
    position = end
    segments.push({
      text: word.slice(start, end),
      start,
      end,
      inFocus: focus !== null && start < focus[1] && end > focus[0],
      colorIndex: index % 4,
    })
  })
  return segments
}

/** Shuffles a copy of `items` with a seeded PRNG (stable across re-renders). */
export function seededShuffle<T>(items: readonly T[], seed: string): T[] {
  let state = 2166136261
  for (let i = 0; i < seed.length; i += 1) {
    state ^= seed.charCodeAt(i)
    state = Math.imul(state, 16777619)
  }
  const random = () => {
    state = Math.imul(state ^ (state >>> 15), 2246822507)
    state = Math.imul(state ^ (state >>> 13), 3266489909)
    state ^= state >>> 16
    return (state >>> 0) / 4294967296
  }
  const copy = items.slice()
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1))
    ;[copy[i], copy[j]] = [copy[j], copy[i]]
  }
  return copy
}
