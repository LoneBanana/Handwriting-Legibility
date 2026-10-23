import { describe, expect, it } from 'vitest'
import { EMOTIONS } from '../../api/types'
import { blendshapesFrom, neutralBlendshapes } from './blendshapes'
import { EMOTION_FAMILY, emotionTargets } from './expressions'
import {
  ACTION_UNITS,
  estimateFeeling,
  FACE_REGIONS,
  NEUTRAL_THRESHOLD,
  readActionUnits,
  regionActivations,
  scoreFeelings,
  strengthWord,
} from './faceAnalysis'

describe('readActionUnits', () => {
  it('maps every AU in the table to its blendshapes', () => {
    expect(ACTION_UNITS.map((unit) => unit.id)).toEqual([
      'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU14', 'AU15', 'AU20', 'AU23/24', 'AU26',
    ])
    const reading = readActionUnits(neutralBlendshapes())
    for (const unit of ACTION_UNITS) expect(reading.au[unit.id]).toBe(0)
  })

  it('uses the stronger side for paired muscles and flags one-sided movement', () => {
    const reading = readActionUnits(blendshapesFrom({ mouthSmileRight: 0.6, mouthSmileLeft: 0.1, cheekSquintLeft: 0.4, cheekSquintRight: 0.38 }))
    expect(reading.au.AU12).toBeCloseTo(0.6)
    expect(reading.oneSided.AU12).toBe(true)
    expect(reading.oneSided.AU6).toBeUndefined()
    expect(reading.symmetricSmile).toBeCloseTo(0.1)
    expect(reading.oneSidedSmile).toBeCloseTo(0.5)
  })

  it('lights up regions from their strongest action unit', () => {
    const regions = regionActivations(readActionUnits(blendshapesFrom({ browDownLeft: 0.7, jawOpen: 0.3 })))
    expect(Object.keys(regions)).toEqual([...FACE_REGIONS])
    expect(regions.innerBrows).toBeCloseTo(0.7) // AU4 acts on both brow regions
    expect(regions.outerBrows).toBeCloseTo(0.7)
    expect(regions.jaw).toBeCloseTo(0.3)
    expect(regions.cheeks).toBe(0)
  })
})

describe('emotion prototypes', () => {
  it('covers the whole emotion vocabulary', () => {
    for (const emotion of EMOTIONS) expect(EMOTION_FAMILY[emotion]).toBeDefined()
  })

  for (const intensity of [1, 0.6, 0.3]) {
    it(`recognises every built-in expression at intensity ${intensity}`, () => {
      for (const emotion of EMOTIONS) {
        const estimate = estimateFeeling(readActionUnits(emotionTargets(emotion, intensity)))
        expect(estimate.family, emotion).toBe(EMOTION_FAMILY[emotion])
        // Fear and surprise share most of their movements (as they do for people).
        if (emotion !== 'scared') expect(estimate.certainty, emotion).not.toBe('unsure')
      }
    })
  }

  it('tells surprise from fear by the extra brow, lid and lip movements', () => {
    const surprised = scoreFeelings(readActionUnits(emotionTargets('surprised', 1)))
    const scared = scoreFeelings(readActionUnits(emotionTargets('scared', 1)))
    expect(surprised[0].family).toBe('surprised')
    expect(scared[0].family).toBe('scared')
  })

  it('reads a one-sided smile with a dimple as contempt, and a two-sided smile as happy', () => {
    const smirk = blendshapesFrom({ mouthSmileLeft: 0.5, mouthDimpleLeft: 0.5 })
    const smile = blendshapesFrom({ mouthSmileLeft: 0.7, mouthSmileRight: 0.7, cheekSquintLeft: 0.5, cheekSquintRight: 0.5 })
    expect(estimateFeeling(readActionUnits(smirk)).family).toBe('contempt')
    expect(estimateFeeling(readActionUnits(smile)).family).toBe('happy')
  })

  it('reads a relaxed face, or a talking jaw on its own, as calm', () => {
    expect(estimateFeeling(readActionUnits(neutralBlendshapes()))).toMatchObject({ family: 'neutral', certainty: 'clear' })
    const talking = estimateFeeling(readActionUnits(blendshapesFrom({ jawOpen: 0.25 })))
    expect(talking.family).toBe('neutral')
  })

  it('stays calm below the threshold and mentions a faint hint', () => {
    const faint = estimateFeeling(readActionUnits(blendshapesFrom({ mouthSmileLeft: 0.06, mouthSmileRight: 0.06, cheekSquintLeft: 0.05, cheekSquintRight: 0.05 })))
    expect(faint.family).toBe('neutral')
    expect(faint.scores[0].score).toBeLessThan(NEUTRAL_THRESHOLD)
    expect(faint.alternative).toBe('happy')
  })

  it('flags mixed faces as unsure with the runner-up', () => {
    // Raised brows and wide eyes plus a smile: surprise and happiness at once.
    const mixed = blendshapesFrom({
      browInnerUp: 0.7, browOuterUpLeft: 0.7, browOuterUpRight: 0.7, eyeWideLeft: 0.6, eyeWideRight: 0.6,
      mouthSmileLeft: 0.7, mouthSmileRight: 0.7, cheekSquintLeft: 0.6, cheekSquintRight: 0.6,
    })
    const estimate = estimateFeeling(readActionUnits(mixed))
    expect(estimate.certainty).toBe('unsure')
    expect([estimate.family, estimate.alternative].sort()).toEqual(['happy', 'surprised'])
  })

  it('describes strength in plain words', () => {
    expect(strengthWord(0.1)).toBe('faintly')
    expect(strengthWord(0.3)).toBe('clearly')
    expect(strengthWord(0.8)).toBe('strongly')
  })
})
