import { describe, expect, it } from 'vitest'
import { EMOTIONS } from '../../api/types'
import { BLENDSHAPE_NAMES, easeFactor, easeToward, fromCategories, maxDifference, neutralBlendshapes } from './blendshapes'
import { EMOTION_SHAPES, EMOTION_WORDS, emotionTargets, FEELINGS } from './expressions'

describe('blendshapes', () => {
  it('has the 52 ARKit names', () => {
    expect(BLENDSHAPE_NAMES).toHaveLength(52)
    expect(new Set(BLENDSHAPE_NAMES).size).toBe(52)
  })

  it('reads MediaPipe categories, ignoring unknown names and clamping scores', () => {
    const shapes = fromCategories([
      { categoryName: '_neutral', score: 0.9 },
      { categoryName: 'jawOpen', score: 0.4 },
      { categoryName: 'mouthSmileLeft', score: 1.3 },
    ])
    expect(shapes.jawOpen).toBe(0.4)
    expect(shapes.mouthSmileLeft).toBe(1)
    expect(Object.keys(shapes)).toHaveLength(52)
  })

  it('eases frame-rate independently', () => {
    expect(easeFactor(100, 0)).toBe(1)
    expect(easeFactor(0, 100)).toBe(0)
    expect(easeFactor(100, 100)).toBeCloseTo(1 - Math.exp(-1))
    const target = { ...neutralBlendshapes(), jawOpen: 1 }
    let oneStep = neutralBlendshapes()
    oneStep = easeToward(oneStep, target, 100, 100)
    let manySteps = neutralBlendshapes()
    for (let i = 0; i < 10; i += 1) manySteps = easeToward(manySteps, target, 10, 100)
    expect(oneStep.jawOpen).toBeCloseTo(manySteps.jawOpen, 6)
    expect(maxDifference(oneStep, manySteps)).toBeLessThan(1e-6)
  })
})

describe('emotionTargets', () => {
  it('has a shape, words and a family description for every emotion', () => {
    for (const emotion of EMOTIONS) {
      expect(EMOTION_SHAPES[emotion]).toBeDefined()
      expect(EMOTION_WORDS[emotion].word.length).toBeGreaterThan(0)
    }
    expect(Object.keys(FEELINGS)).toHaveLength(8)
  })

  it('scales the shape by expression intensity and stays within 0..1', () => {
    const full = emotionTargets('excited', 1)
    const subtle = emotionTargets('excited', 0.3)
    expect(subtle.mouthSmileLeft).toBeCloseTo(full.mouthSmileLeft * 0.3)
    for (const emotion of EMOTIONS) {
      for (const value of Object.values(emotionTargets(emotion, 1.4))) {
        expect(value).toBeGreaterThanOrEqual(0)
        expect(value).toBeLessThanOrEqual(1)
      }
    }
  })

  it('keeps neutral fully relaxed and contempt one-sided', () => {
    expect(Object.values(emotionTargets('neutral', 1)).every((v) => v === 0)).toBe(true)
    const contempt = emotionTargets('contempt', 1)
    expect(contempt.mouthSmileRight).toBeGreaterThan(0.3)
    expect(contempt.mouthSmileLeft).toBe(0)
  })
})
