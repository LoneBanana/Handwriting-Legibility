import { describe, expect, it } from 'vitest'
import { AvatarController, BLINK_MS, blinkAmount, EXPRESSION_EASE_MS, MAX_BOUNDARY_HOLD_MS } from './avatarController'
import { emotionTargets } from './expressions'

/** Steps the controller every 16 ms from `from` to `to`, returning the frames. */
function run(controller: AvatarController, from: number, to: number) {
  const frames = []
  for (let t = from; t <= to; t += 16) frames.push({ t, shapes: controller.step(t) })
  return frames
}

describe('blinkAmount', () => {
  it('closes quickly, holds briefly and opens', () => {
    expect(blinkAmount(0)).toBe(0)
    expect(blinkAmount(35)).toBeGreaterThan(0.3)
    expect(blinkAmount(85)).toBe(1)
    expect(blinkAmount(150)).toBeGreaterThan(0)
    expect(blinkAmount(150)).toBeLessThan(1)
    expect(blinkAmount(BLINK_MS)).toBe(0)
    expect(blinkAmount(-1)).toBe(0)
  })
})

describe('AvatarController', () => {
  it('eases into a new expression', () => {
    const avatar = new AvatarController({ random: () => 0.5 })
    avatar.step(0)
    avatar.setExpression('content', 1)
    const target = emotionTargets('content', 1).mouthSmileLeft
    // After one time constant the face is about 63% of the way there.
    const early = run(avatar, 16, EXPRESSION_EASE_MS).at(-1)!.shapes.mouthSmileLeft
    expect(early).toBeGreaterThan(0.5 * target)
    expect(early).toBeLessThan(0.75 * target)
    const settled = run(avatar, EXPRESSION_EASE_MS + 16, 2000).at(-1)!.shapes
    expect(settled.mouthSmileLeft).toBeCloseTo(target, 3)
  })

  it('blinks every few seconds, briefly', () => {
    const avatar = new AvatarController({ random: () => 0.5 })
    const frames = run(avatar, 0, 12_000)
    const closed = frames.filter((f) => f.shapes.eyeBlinkLeft > 0.9)
    expect(closed.length).toBeGreaterThan(0)
    // Mostly open: well under 10% of frames are mid-blink.
    expect(frames.filter((f) => f.shapes.eyeBlinkLeft > 0.05).length / frames.length).toBeLessThan(0.1)
  })

  it('moves the jaw while speaking and settles after', () => {
    const avatar = new AvatarController({ random: () => 0.5 })
    avatar.step(0)
    avatar.speak('Hello there, how are you today?', 150)
    const speaking = run(avatar, 16, 1500)
    expect(Math.max(...speaking.map((f) => f.shapes.jawOpen))).toBeGreaterThan(0.3)
    avatar.silence()
    const after = run(avatar, 1516, 2000).at(-1)!.shapes
    expect(after.jawOpen).toBeLessThan(0.05)
  })

  it('waits at a word end for the next word boundary, but not forever', () => {
    // At 120 words a minute the two words share 1 s: "ma" ends at about 630 ms.
    const text = 'ma ah'
    const withBoundary = new AvatarController({ random: () => 0.5 })
    withBoundary.step(0)
    withBoundary.speak(text, 120)
    withBoundary.boundary(0)
    const held = run(withBoundary, 16, 900).at(-1)!.shapes
    expect(held.jawOpen).toBeLessThan(0.1) // resting, waiting for the voice to reach "ah"
    withBoundary.boundary(3)
    const moved = run(withBoundary, 916, 1000).at(-1)!.shapes
    expect(moved.jawOpen).toBeGreaterThan(0.4)

    const free = new AvatarController({ random: () => 0.5 })
    free.step(0)
    free.speak(text, 120)
    expect(run(free, 16, 700).at(-1)!.shapes.jawOpen).toBeGreaterThan(0.4) // no boundaries: runs on the estimate

    const stalled = new AvatarController({ random: () => 0.5 })
    stalled.step(0)
    stalled.speak(text, 120)
    stalled.boundary(0)
    const resumed = run(stalled, 16, 630 + MAX_BOUNDARY_HOLD_MS + 200).at(-1)!.shapes
    expect(resumed.jawOpen).toBeGreaterThan(0.4) // boundaries stopped coming: carry on
  })

  it('holds still with reduced motion: instant expressions, no blinks, no lip movement', () => {
    const avatar = new AvatarController({ random: () => 0.5, reducedMotion: true })
    avatar.setExpression('surprised', 1)
    expect(avatar.step(0).browInnerUp).toBeCloseTo(emotionTargets('surprised', 1).browInnerUp)
    avatar.speak('Hello there, how are you today?', 150)
    const frames = run(avatar, 16, 8000)
    expect(Math.max(...frames.map((f) => f.shapes.eyeBlinkLeft))).toBe(0)
    const jaws = new Set(frames.map((f) => f.shapes.jawOpen.toFixed(4)))
    expect(jaws.size).toBe(1)
  })
})
