import { describe, expect, it } from 'vitest'
import { EMOTIONS } from '../../api/types'
import { seededRandom } from '../math/layouts'
import { BLENDSHAPE_NAMES, blendshapesFrom, neutralBlendshapes, type Blendshapes } from './blendshapes'
import { faceGeometry } from './avatarGeometry'
import { emotionTargets } from './expressions'

const allPaths = (shapes: Blendshapes) => JSON.stringify(faceGeometry(shapes))

describe('faceGeometry', () => {
  it('opens the mouth with the jaw', () => {
    const closed = faceGeometry(neutralBlendshapes()).mouth
    const open = faceGeometry(blendshapesFrom({ jawOpen: 0.8 })).mouth
    expect(closed.opening).toBeLessThan(1)
    expect(open.opening).toBeGreaterThan(15)
    expect(open.teeth).not.toBeNull()
    expect(faceGeometry(blendshapesFrom({ jawOpen: 0.8, mouthClose: 1 })).mouth.opening).toBeLessThan(open.opening / 3)
  })

  it('lifts the mouth corners for a smile and drops them for a frown, side by side', () => {
    const rest = faceGeometry(neutralBlendshapes()).mouth.corners
    const smile = faceGeometry(blendshapesFrom({ mouthSmileLeft: 1 })).mouth.corners
    const frown = faceGeometry(blendshapesFrom({ mouthFrownLeft: 1, mouthFrownRight: 1 })).mouth.corners
    const left = (corners: typeof rest) => corners.find((c) => c.side === 'Left')!
    const right = (corners: typeof rest) => corners.find((c) => c.side === 'Right')!
    expect(left(smile).y).toBeLessThan(left(rest).y) // the face's left corner rises…
    expect(right(smile).y).toBeCloseTo(right(rest).y) // …the other stays put
    expect(left(smile).x).toBeGreaterThan(100) // and its left is the viewer's right
    expect(left(frown).y).toBeGreaterThan(left(rest).y)
  })

  it('shows lines for cheeks, frowns and nose wrinkles only when those units are active', () => {
    const rest = faceGeometry(neutralBlendshapes())
    expect(rest.furrow).toBe(0)
    expect(rest.nose.wrinkles).toBe(0)
    expect(rest.eyes.every((eye) => eye.crinkle === 0)).toBe(true)
    const face = faceGeometry(emotionTargets('disgusted', 1))
    expect(face.nose.wrinkles).toBeGreaterThan(0.5)
    expect(face.furrow).toBeGreaterThan(0)
  })

  it('never produces broken paths', () => {
    const random = seededRandom('faces')
    const sets: Blendshapes[] = EMOTIONS.map((emotion) => emotionTargets(emotion, 1))
    for (let i = 0; i < 200; i += 1) {
      const shapes = neutralBlendshapes()
      for (const name of BLENDSHAPE_NAMES) shapes[name] = random()
      sets.push(shapes)
    }
    for (const shapes of sets) expect(allPaths(shapes)).not.toMatch(/NaN|Infinity|undefined/)
  })
})
