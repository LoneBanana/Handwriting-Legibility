// The 52 ARKit-style blendshapes shared by the built-in avatar (which animates
// them directly) and MediaPipe Face Landmarker (which estimates them from video).
// "Left" and "Right" always mean the face's own sides, not the viewer's.

export const BLENDSHAPE_NAMES = [
  'browDownLeft',
  'browDownRight',
  'browInnerUp',
  'browOuterUpLeft',
  'browOuterUpRight',
  'cheekPuff',
  'cheekSquintLeft',
  'cheekSquintRight',
  'eyeBlinkLeft',
  'eyeBlinkRight',
  'eyeLookDownLeft',
  'eyeLookDownRight',
  'eyeLookInLeft',
  'eyeLookInRight',
  'eyeLookOutLeft',
  'eyeLookOutRight',
  'eyeLookUpLeft',
  'eyeLookUpRight',
  'eyeSquintLeft',
  'eyeSquintRight',
  'eyeWideLeft',
  'eyeWideRight',
  'jawForward',
  'jawLeft',
  'jawOpen',
  'jawRight',
  'mouthClose',
  'mouthDimpleLeft',
  'mouthDimpleRight',
  'mouthFrownLeft',
  'mouthFrownRight',
  'mouthFunnel',
  'mouthLeft',
  'mouthLowerDownLeft',
  'mouthLowerDownRight',
  'mouthPressLeft',
  'mouthPressRight',
  'mouthPucker',
  'mouthRight',
  'mouthRollLower',
  'mouthRollUpper',
  'mouthShrugLower',
  'mouthShrugUpper',
  'mouthSmileLeft',
  'mouthSmileRight',
  'mouthStretchLeft',
  'mouthStretchRight',
  'mouthUpperUpLeft',
  'mouthUpperUpRight',
  'noseSneerLeft',
  'noseSneerRight',
  'tongueOut',
] as const

export type BlendshapeName = (typeof BLENDSHAPE_NAMES)[number]
export type Blendshapes = Record<BlendshapeName, number>
export type BlendshapePatch = Partial<Blendshapes>

const NAMES: ReadonlySet<string> = new Set(BLENDSHAPE_NAMES)

export function isBlendshapeName(name: string): name is BlendshapeName {
  return NAMES.has(name)
}

export function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0
  return value < 0 ? 0 : value > 1 ? 1 : value
}

/** A relaxed face: every blendshape at rest. */
export function neutralBlendshapes(): Blendshapes {
  const shapes = {} as Blendshapes
  for (const name of BLENDSHAPE_NAMES) shapes[name] = 0
  return shapes
}

/** A complete set of blendshapes: neutral plus `patch` (values clamped to 0..1). */
export function blendshapesFrom(patch: BlendshapePatch): Blendshapes {
  const shapes = neutralBlendshapes()
  for (const [name, value] of Object.entries(patch)) {
    if (isBlendshapeName(name)) shapes[name] = clamp01(value)
  }
  return shapes
}

/**
 * Converts MediaPipe Face Landmarker categories (`categoryName`, `score`) into a
 * blendshape record. Unknown names (e.g. MediaPipe's `_neutral`) are ignored.
 */
export function fromCategories(categories: readonly { categoryName: string; score: number }[]): Blendshapes {
  const shapes = neutralBlendshapes()
  for (const { categoryName, score } of categories) {
    if (isBlendshapeName(categoryName)) shapes[categoryName] = clamp01(score)
  }
  return shapes
}

/**
 * Fraction of the remaining distance covered in `dtMs` by an exponential ease
 * with time constant `tauMs` (after one `tauMs`, about 63% of the way).
 * Frame-rate independent; `tauMs <= 0` jumps straight to the target.
 */
export function easeFactor(dtMs: number, tauMs: number): number {
  if (tauMs <= 0) return 1
  if (!(dtMs > 0)) return 0
  return 1 - Math.exp(-dtMs / tauMs)
}

/** Moves every value of `current` toward `target` (returns a new object). */
export function easeToward(current: Blendshapes, target: Blendshapes, dtMs: number, tauMs: number): Blendshapes {
  const k = easeFactor(dtMs, tauMs)
  const next = {} as Blendshapes
  for (const name of BLENDSHAPE_NAMES) {
    const value = current[name] + (target[name] - current[name]) * k
    // Snap tiny residues so idle frames can be detected as "no change".
    next[name] = Math.abs(value - target[name]) < 1e-4 ? target[name] : value
  }
  return next
}

/** Largest absolute difference between two sets of blendshapes. */
export function maxDifference(a: Blendshapes, b: Blendshapes): number {
  let max = 0
  for (const name of BLENDSHAPE_NAMES) {
    const diff = Math.abs(a[name] - b[name])
    if (diff > max) max = diff
  }
  return max
}
