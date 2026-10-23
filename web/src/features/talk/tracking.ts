// Calibration for blendshapes estimated by MediaPipe Face Landmarker from the
// video avatar, so they read on the same 0..1 scale as the built-in avatar.

import { BLENDSHAPE_NAMES, clamp01, neutralBlendshapes, type BlendshapeName, type Blendshapes } from './blendshapes'

/**
 * Face Landmarker rarely pushes some blendshapes near 1, even for strong
 * expressions (dimples, nose wrinkles, wide eyes...). These gains stretch them
 * toward the built-in avatar's range. They are hand-tuned heuristics, not
 * measurements, which is why the panel calls tracked values estimates.
 */
export const TRACKER_GAIN: Partial<Record<BlendshapeName, number>> = {
  browDownLeft: 1.6,
  browDownRight: 1.6,
  browOuterUpLeft: 1.3,
  browOuterUpRight: 1.3,
  cheekSquintLeft: 1.6,
  cheekSquintRight: 1.6,
  eyeWideLeft: 2.2,
  eyeWideRight: 2.2,
  eyeSquintLeft: 1.2,
  eyeSquintRight: 1.2,
  noseSneerLeft: 2.4,
  noseSneerRight: 2.4,
  mouthUpperUpLeft: 1.6,
  mouthUpperUpRight: 1.6,
  mouthDimpleLeft: 2.5,
  mouthDimpleRight: 2.5,
  mouthFrownLeft: 2,
  mouthFrownRight: 2,
  mouthStretchLeft: 2,
  mouthStretchRight: 2,
  mouthPressLeft: 1.6,
  mouthPressRight: 1.6,
}

export type BaselineOptions = {
  /** How fast the resting level may creep upward, per second. */
  risePerSecond?: number
  /** The resting level never exceeds this, so strong expressions always show. */
  cap?: number
}

/**
 * Learns each blendshape's resting level for the face being watched. It drops
 * straight to any new low and creeps up slowly, so a face's natural resting
 * look (say, brows that sit a little high) stops counting as an expression,
 * while an expression that is held for a few seconds is not absorbed.
 */
export class BaselineTracker {
  private readonly risePerMs: number
  private readonly cap: number
  private values: Blendshapes = neutralBlendshapes()

  constructor({ risePerSecond = 0.008, cap = 0.25 }: BaselineOptions = {}) {
    this.risePerMs = risePerSecond / 1000
    this.cap = cap
  }

  update(raw: Blendshapes, dtMs: number): void {
    const rise = Math.max(0, dtMs) * this.risePerMs
    for (const name of BLENDSHAPE_NAMES) {
      const value = clamp01(raw[name])
      this.values[name] = Math.min(value, this.values[name] + rise, this.cap)
    }
  }

  get baseline(): Blendshapes {
    return { ...this.values }
  }

  reset(): void {
    this.values = neutralBlendshapes()
  }
}

/** Removes the resting level, rescales what is left to 0..1 and applies the gains. */
export function calibrate(raw: Blendshapes, baseline: Blendshapes | null = null): Blendshapes {
  const out = neutralBlendshapes()
  for (const name of BLENDSHAPE_NAMES) {
    const rest = baseline ? Math.min(baseline[name], 0.95) : 0
    const lifted = (clamp01(raw[name]) - rest) / (1 - rest)
    out[name] = clamp01(lifted * (TRACKER_GAIN[name] ?? 1))
  }
  return out
}
