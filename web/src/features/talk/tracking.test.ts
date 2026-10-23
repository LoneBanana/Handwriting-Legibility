import { describe, expect, it } from 'vitest'
import { blendshapesFrom, neutralBlendshapes } from './blendshapes'
import { BaselineTracker, calibrate, TRACKER_GAIN } from './tracking'

describe('calibrate', () => {
  it('stretches movements that face tracking under-reports', () => {
    const out = calibrate(blendshapesFrom({ mouthDimpleLeft: 0.2, jawOpen: 0.5, noseSneerRight: 0.9 }))
    expect(out.mouthDimpleLeft).toBeCloseTo(0.2 * TRACKER_GAIN.mouthDimpleLeft!)
    expect(out.jawOpen).toBeCloseTo(0.5) // no gain
    expect(out.noseSneerRight).toBe(1) // clamped
  })

  it('removes the face’s resting level', () => {
    const baseline = blendshapesFrom({ browInnerUp: 0.2 })
    expect(calibrate(blendshapesFrom({ browInnerUp: 0.2 }), baseline).browInnerUp).toBe(0)
    expect(calibrate(blendshapesFrom({ browInnerUp: 0.6 }), baseline).browInnerUp).toBeCloseTo(0.5)
  })
})

describe('BaselineTracker', () => {
  it('drops to new lows at once, creeps up slowly and never passes the cap', () => {
    const tracker = new BaselineTracker({ risePerSecond: 0.01, cap: 0.25 })
    const resting = blendshapesFrom({ browInnerUp: 0.3, mouthSmileLeft: 0.05 })
    for (let i = 0; i < 10; i += 1) tracker.update(resting, 1000) // 10 s
    expect(tracker.baseline.browInnerUp).toBeCloseTo(0.1)
    expect(tracker.baseline.mouthSmileLeft).toBeCloseTo(0.05)
    for (let i = 0; i < 100; i += 1) tracker.update(resting, 1000) // 100 s more
    expect(tracker.baseline.browInnerUp).toBe(0.25)
    tracker.update(neutralBlendshapes(), 16)
    expect(tracker.baseline.browInnerUp).toBe(0)
    tracker.update(resting, 1000)
    tracker.reset()
    expect(tracker.baseline.browInnerUp).toBe(0)
  })
})
