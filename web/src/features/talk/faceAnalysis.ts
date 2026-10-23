// Blendshapes → FACS action units → emotion prototypes (EMFACS), following the
// tables in docs/ARCHITECTURE.md. Pure functions so the panel logic is testable.

import type { BlendshapeName, Blendshapes } from './blendshapes'
import type { FeelingFamily } from './expressions'

export type ActionUnitId =
  | 'AU1'
  | 'AU2'
  | 'AU4'
  | 'AU5'
  | 'AU6'
  | 'AU7'
  | 'AU9'
  | 'AU10'
  | 'AU12'
  | 'AU14'
  | 'AU15'
  | 'AU20'
  | 'AU23/24'
  | 'AU26'

export type FaceRegion =
  | 'innerBrows'
  | 'outerBrows'
  | 'eyes'
  | 'cheeks'
  | 'nose'
  | 'upperLip'
  | 'mouthCorners'
  | 'lips'
  | 'jaw'

export type ActionUnitInfo = {
  id: ActionUnitId
  /** FACS name, e.g. "Cheek raiser". */
  name: string
  /** Learner-facing meaning, e.g. "cheeks raised (eyes crinkle)". */
  meaning: string
  blendshapes: readonly BlendshapeName[]
  regions: readonly FaceRegion[]
}

export const ACTION_UNITS: readonly ActionUnitInfo[] = [
  { id: 'AU1', name: 'Inner brow raiser', meaning: 'inner eyebrows raised', blendshapes: ['browInnerUp'], regions: ['innerBrows'] },
  {
    id: 'AU2',
    name: 'Outer brow raiser',
    meaning: 'outer eyebrows raised',
    blendshapes: ['browOuterUpLeft', 'browOuterUpRight'],
    regions: ['outerBrows'],
  },
  {
    id: 'AU4',
    name: 'Brow lowerer',
    meaning: 'eyebrows pulled down and together',
    blendshapes: ['browDownLeft', 'browDownRight'],
    regions: ['innerBrows', 'outerBrows'],
  },
  { id: 'AU5', name: 'Upper lid raiser', meaning: 'eyes opened wide', blendshapes: ['eyeWideLeft', 'eyeWideRight'], regions: ['eyes'] },
  {
    id: 'AU6',
    name: 'Cheek raiser',
    meaning: 'cheeks raised (eyes crinkle)',
    blendshapes: ['cheekSquintLeft', 'cheekSquintRight'],
    regions: ['cheeks'],
  },
  { id: 'AU7', name: 'Lid tightener', meaning: 'eyelids tightened', blendshapes: ['eyeSquintLeft', 'eyeSquintRight'], regions: ['eyes'] },
  { id: 'AU9', name: 'Nose wrinkler', meaning: 'nose wrinkled', blendshapes: ['noseSneerLeft', 'noseSneerRight'], regions: ['nose'] },
  {
    id: 'AU10',
    name: 'Upper lip raiser',
    meaning: 'upper lip raised',
    blendshapes: ['mouthUpperUpLeft', 'mouthUpperUpRight'],
    regions: ['upperLip'],
  },
  {
    id: 'AU12',
    name: 'Lip corner puller',
    meaning: 'mouth corners pulled up',
    blendshapes: ['mouthSmileLeft', 'mouthSmileRight'],
    regions: ['mouthCorners'],
  },
  {
    id: 'AU14',
    name: 'Dimpler',
    meaning: 'dimples (often one-sided)',
    blendshapes: ['mouthDimpleLeft', 'mouthDimpleRight'],
    regions: ['mouthCorners'],
  },
  {
    id: 'AU15',
    name: 'Lip corner depressor',
    meaning: 'mouth corners pulled down',
    blendshapes: ['mouthFrownLeft', 'mouthFrownRight'],
    regions: ['mouthCorners'],
  },
  {
    id: 'AU20',
    name: 'Lip stretcher',
    meaning: 'lips stretched sideways',
    blendshapes: ['mouthStretchLeft', 'mouthStretchRight'],
    regions: ['lips'],
  },
  {
    id: 'AU23/24',
    name: 'Lip tightener and presser',
    meaning: 'lips pressed or tightened',
    blendshapes: ['mouthPressLeft', 'mouthPressRight'],
    regions: ['lips'],
  },
  { id: 'AU26', name: 'Jaw drop', meaning: 'jaw dropped', blendshapes: ['jawOpen'], regions: ['jaw'] },
]

export const FACE_REGIONS: readonly FaceRegion[] = [
  'innerBrows',
  'outerBrows',
  'eyes',
  'cheeks',
  'nose',
  'upperLip',
  'mouthCorners',
  'lips',
  'jaw',
]

export const REGION_LABELS: Record<FaceRegion, string> = {
  innerBrows: 'Inner brows',
  outerBrows: 'Outer brows',
  eyes: 'Eyes',
  cheeks: 'Cheeks',
  nose: 'Nose',
  upperLip: 'Upper lip',
  mouthCorners: 'Mouth corners',
  lips: 'Lips',
  jaw: 'Jaw',
}

export type AuReading = {
  /** Activation 0..1 of each action unit (the stronger side for paired muscles). */
  au: Record<ActionUnitId, number>
  /** Action units clearly stronger on one side of the face. */
  oneSided: Partial<Record<ActionUnitId, boolean>>
  /** Smile shared by both mouth corners: min(left, right). */
  symmetricSmile: number
  /** Smile on one side only: |left − right|. */
  oneSidedSmile: number
}

export function readActionUnits(shapes: Blendshapes): AuReading {
  const au = {} as Record<ActionUnitId, number>
  const oneSided: Partial<Record<ActionUnitId, boolean>> = {}
  for (const unit of ACTION_UNITS) {
    const values = unit.blendshapes.map((name) => shapes[name] ?? 0)
    const strongest = Math.max(...values)
    au[unit.id] = strongest
    if (values.length === 2 && strongest >= 0.2 && Math.abs(values[0] - values[1]) >= 0.5 * strongest) {
      oneSided[unit.id] = true
    }
  }
  const left = shapes.mouthSmileLeft ?? 0
  const right = shapes.mouthSmileRight ?? 0
  return { au, oneSided, symmetricSmile: Math.min(left, right), oneSidedSmile: Math.abs(left - right) }
}

/** How active each labelled face region is: the strongest action unit acting on it. */
export function regionActivations(reading: AuReading): Record<FaceRegion, number> {
  const regions = {} as Record<FaceRegion, number>
  for (const region of FACE_REGIONS) regions[region] = 0
  for (const unit of ACTION_UNITS) {
    for (const region of unit.regions) regions[region] = Math.max(regions[region], reading.au[unit.id])
  }
  return regions
}

// ------------------------------------------------------------ prototype matching

type Feature = Exclude<ActionUnitId, 'AU12'> | 'symmetricSmile' | 'oneSidedSmile'

export type ScoredFamily = Exclude<FeelingFamily, 'neutral'>

/** EMFACS prototypes from docs/ARCHITECTURE.md (contempt = one-sided AU12 + AU14). */
export const PROTOTYPES: Record<ScoredFamily, readonly Feature[]> = {
  happy: ['AU6', 'symmetricSmile'],
  sad: ['AU1', 'AU4', 'AU15'],
  surprised: ['AU1', 'AU2', 'AU5', 'AU26'],
  scared: ['AU1', 'AU2', 'AU4', 'AU5', 'AU7', 'AU20', 'AU26'],
  angry: ['AU4', 'AU5', 'AU7', 'AU23/24'],
  disgusted: ['AU9', 'AU10', 'AU15'],
  contempt: ['oneSidedSmile', 'AU14'],
}

/**
 * Talking moves the jaw and lips all the time, so those units count for less
 * as evidence of a feeling than brows, eyes and cheeks do.
 */
const RELIABILITY: Partial<Record<Feature, number>> = { AU26: 0.5, AU20: 0.8, 'AU23/24': 0.8 }

const weightOf = (feature: Feature) => RELIABILITY[feature] ?? 1

function weightedFeatures(reading: AuReading): Record<Feature, number> {
  const { au } = reading
  // AU12 enters as its two halves: shared smiles (happy) and one-sided ones (contempt).
  const raw: Record<Feature, number> = {
    AU1: au.AU1,
    AU2: au.AU2,
    AU4: au.AU4,
    AU5: au.AU5,
    AU6: au.AU6,
    AU7: au.AU7,
    AU9: au.AU9,
    AU10: au.AU10,
    AU14: au.AU14,
    AU15: au.AU15,
    AU20: au.AU20,
    'AU23/24': au['AU23/24'],
    AU26: au.AU26,
    symmetricSmile: reading.symmetricSmile,
    oneSidedSmile: reading.oneSidedSmile,
  }
  const weighted = {} as Record<Feature, number>
  for (const key of Object.keys(raw) as Feature[]) weighted[key] = raw[key] * weightOf(key)
  return weighted
}

export type FeelingScore = {
  family: ScoredFamily
  /** strength × the share of the face's movement this prototype explains. */
  score: number
  /** Weighted mean activation of the prototype's units, 0..1. */
  strength: number
}

/**
 * Scores each prototype. A prototype ranks high when its own units are active
 * (strength) and little else is moving (explained share), which separates
 * look-alikes such as surprised (1+2+5+26) and scared (the same plus 4, 7, 20).
 */
export function scoreFeelings(reading: AuReading): FeelingScore[] {
  const features = weightedFeatures(reading)
  const total = Object.values(features).reduce((sum, value) => sum + value, 0)
  const scores = (Object.keys(PROTOTYPES) as ScoredFamily[]).map((family) => {
    const units = PROTOTYPES[family]
    const matched = units.reduce((sum, unit) => sum + features[unit], 0)
    const strength = matched / units.reduce((sum, unit) => sum + weightOf(unit), 0)
    const explained = total > 1e-9 ? matched / total : 0
    return { family, strength, score: strength * explained }
  })
  return scores.sort((a, b) => b.score - a.score)
}

/** Below this best score the face reads as calm. */
export const NEUTRAL_THRESHOLD = 0.06

export type Certainty = 'clear' | 'likely' | 'unsure'

export type FeelingEstimate = {
  family: FeelingFamily
  /** How strongly the feeling is shown, 0..1 (0 when calm). */
  strength: number
  certainty: Certainty
  /** A close runner-up (or, when calm, a faint hint of a feeling). */
  alternative: ScoredFamily | null
  scores: FeelingScore[]
}

export function estimateFeeling(reading: AuReading): FeelingEstimate {
  const scores = scoreFeelings(reading)
  const [best, second] = scores
  if (!best || best.score < NEUTRAL_THRESHOLD) {
    const hint = best && best.score >= NEUTRAL_THRESHOLD * 0.6 ? best.family : null
    return { family: 'neutral', strength: 0, certainty: hint ? 'likely' : 'clear', alternative: hint, scores }
  }
  const close = second !== undefined && second.score >= 0.7 * best.score
  return {
    family: best.family,
    strength: best.strength,
    certainty: close ? 'unsure' : best.score >= 0.25 ? 'clear' : 'likely',
    alternative: close ? second.family : null,
    scores,
  }
}

/** "faintly" / "clearly" / "strongly": how visibly a feeling is shown. */
export function strengthWord(strength: number): string {
  if (strength >= 0.55) return 'strongly'
  if (strength >= 0.2) return 'clearly'
  return 'faintly'
}
