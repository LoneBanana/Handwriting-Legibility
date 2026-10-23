// Pure geometry for the maths visuals. All coordinates are in a unit square
// (0..1) unless noted, so the SVG components can scale them freely.

import type { Visual } from '../../api/types'

export type Point = { x: number; y: number }

const DICE: Record<number, Point[]> = {
  1: [{ x: 0.5, y: 0.5 }],
  2: [
    { x: 0.25, y: 0.25 },
    { x: 0.75, y: 0.75 },
  ],
  3: [
    { x: 0.25, y: 0.25 },
    { x: 0.5, y: 0.5 },
    { x: 0.75, y: 0.75 },
  ],
  4: [
    { x: 0.25, y: 0.25 },
    { x: 0.75, y: 0.25 },
    { x: 0.25, y: 0.75 },
    { x: 0.75, y: 0.75 },
  ],
  5: [
    { x: 0.25, y: 0.25 },
    { x: 0.75, y: 0.25 },
    { x: 0.5, y: 0.5 },
    { x: 0.25, y: 0.75 },
    { x: 0.75, y: 0.75 },
  ],
  6: [
    { x: 0.25, y: 0.22 },
    { x: 0.75, y: 0.22 },
    { x: 0.25, y: 0.5 },
    { x: 0.75, y: 0.5 },
    { x: 0.25, y: 0.78 },
    { x: 0.75, y: 0.78 },
  ],
}

/** Standard die-face pip positions for 1–6 (0 → no pips). */
export function dicePips(count: number): Point[] {
  if (count <= 0) return []
  return DICE[Math.min(6, Math.round(count))] ?? []
}

/**
 * Splits a count into die faces a learner can subitize: one die up to 6, two
 * near-equal dice up to 12 (8 → 4 + 4, 7 → 4 + 3), then fives.
 */
export function diceGroups(count: number): number[] {
  const n = Math.max(0, Math.round(count))
  if (n <= 6) return n === 0 ? [] : [n]
  if (n <= 12) return [Math.ceil(n / 2), Math.floor(n / 2)]
  const groups: number[] = []
  let remaining = n
  while (remaining > 0) {
    const take = Math.min(5, remaining)
    groups.push(take)
    remaining -= take
  }
  return groups
}

/** Deterministic PRNG (mulberry32) seeded from a string. */
export function seededRandom(seed: string): () => number {
  let h = 1779033703 ^ seed.length
  for (let i = 0; i < seed.length; i += 1) {
    h = Math.imul(h ^ seed.charCodeAt(i), 3432918353)
    h = (h << 13) | (h >>> 19)
  }
  let state = h >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Scatters `count` dots without overlap (minimum spacing shrinks if the square
 * gets crowded). The same seed always gives the same picture.
 */
export function randomDots(count: number, seed: string, margin = 0.1): Point[] {
  const random = seededRandom(seed)
  const n = Math.max(0, Math.round(count))
  const points: Point[] = []
  let minDistance = Math.min(0.22, 0.9 / Math.sqrt(Math.max(1, n)))
  let attempts = 0
  while (points.length < n) {
    const candidate = { x: margin + random() * (1 - 2 * margin), y: margin + random() * (1 - 2 * margin) }
    const clear = points.every((p) => Math.hypot(p.x - candidate.x, p.y - candidate.y) >= minDistance)
    if (clear) points.push(candidate)
    attempts += 1
    if (attempts % 400 === 0) minDistance *= 0.85
    if (attempts > 20000) break
  }
  return points
}

/** Evenly spaced dots on a horizontal line. */
export function lineDots(count: number): Point[] {
  const n = Math.max(0, Math.round(count))
  return Array.from({ length: n }, (_, i) => ({ x: (i + 0.5) / n, y: 0.5 }))
}

export type TenFrameCell = { frame: number; row: number; col: number; filled: boolean }

/** Cells of `frames` ten-frames (2 rows × 5), filled in reading order. */
export function tenFrameCells(filled: number, frames: number): TenFrameCell[] {
  const totalFrames = Math.max(1, Math.round(frames), Math.ceil(Math.max(0, filled) / 10))
  const cells: TenFrameCell[] = []
  for (let frame = 0; frame < totalFrames; frame += 1) {
    for (let row = 0; row < 2; row += 1) {
      for (let col = 0; col < 5; col += 1) {
        const ordinal = frame * 10 + row * 5 + col
        cells.push({ frame, row, col, filled: ordinal < filled })
      }
    }
  }
  return cells
}

/** "Nice" tick values covering [min, max] with roughly `approx` intervals. */
export function niceTicks(min: number, max: number, approx = 10): number[] {
  if (!(max > min)) return [min]
  const raw = (max - min) / approx
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const normalized = raw / magnitude
  const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10) * magnitude
  const ticks: number[] = []
  const first = Math.ceil(min / step) * step
  for (let v = first; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toFixed(10)))
  if (ticks[0] !== min) ticks.unshift(min)
  if (ticks[ticks.length - 1] !== max) ticks.push(max)
  return ticks
}

/**
 * Text alternative for a visual. Counting tasks deliberately don't state the
 * number, since that is the answer being asked for.
 */
export function describeVisual(visual: Visual): string {
  switch (visual.type) {
    case 'dots':
      return `A picture of dots${visual.layout === 'dice' ? ' arranged like dice' : ''}. Count them or recognise the pattern.`
    case 'ten_frame':
      return `${visual.frames === 1 ? 'A ten-frame' : `${visual.frames} ten-frames`} (rows of five boxes) with some boxes filled.`
    case 'array':
      return `An array of dots with ${visual.rows} rows and ${visual.cols} columns.`
    case 'number_line':
      return visual.target === null
        ? `A number line from ${visual.min} to ${visual.max}.`
        : `A number line from ${visual.min} to ${visual.max} with an arrow pointing at a number.`
  }
}
