import type { Emotion } from '../../api/types'
import { blendshapesFrom, clamp01, type BlendshapePatch, type Blendshapes } from './blendshapes'

/** The emotion prototypes the face panel can recognise (EMFACS families). */
export type FeelingFamily = 'neutral' | 'happy' | 'sad' | 'surprised' | 'scared' | 'angry' | 'disgusted' | 'contempt'

export const EMOTION_FAMILY: Record<Emotion, FeelingFamily> = {
  neutral: 'neutral',
  content: 'happy',
  excited: 'happy',
  elated: 'happy',
  surprised: 'surprised',
  sad: 'sad',
  dejected: 'sad',
  scared: 'scared',
  angry: 'angry',
  disgusted: 'disgusted',
  contempt: 'contempt',
}

/** Learner-facing words for the eleven emotions the avatar is told to show. */
export const EMOTION_WORDS: Record<Emotion, { word: string; meaning: string }> = {
  neutral: { word: 'Neutral', meaning: 'calm, not showing a strong feeling' },
  content: { word: 'Content', meaning: 'quietly happy and comfortable' },
  excited: { word: 'Excited', meaning: 'happy and full of energy' },
  elated: { word: 'Elated', meaning: 'extremely happy' },
  surprised: { word: 'Surprised', meaning: 'not expecting this' },
  sad: { word: 'Sad', meaning: 'unhappy' },
  dejected: { word: 'Dejected', meaning: 'very sad and low, like giving up' },
  scared: { word: 'Scared', meaning: 'afraid or worried' },
  angry: { word: 'Angry', meaning: 'cross or annoyed' },
  disgusted: { word: 'Disgusted', meaning: 'something seems gross or wrong' },
  contempt: { word: 'Contempt', meaning: 'looking down on someone or something' },
}

/** Names and plain-language "what to look for" tips for each feeling family. */
export const FEELINGS: Record<FeelingFamily, { name: string; lookFor: string }> = {
  neutral: {
    name: 'Calm',
    lookFor: 'The face is relaxed: eyebrows level, eyes open normally, mouth soft.',
  },
  happy: {
    name: 'Happy',
    lookFor:
      'Cheeks lift and the eyes crinkle at the sides, and the mouth corners pull up. When the eyes join in, the smile is usually real.',
  },
  sad: {
    name: 'Sad',
    lookFor: 'The inner ends of the eyebrows rise and pull together, and the mouth corners turn down.',
  },
  surprised: {
    name: 'Surprised',
    lookFor: 'Eyebrows shoot up, the eyes open wide, and the jaw drops open. It usually comes and goes quickly.',
  },
  scared: {
    name: 'Scared',
    lookFor: 'Eyebrows go up and pull together, the eyes open wide but look tense, and the lips stretch sideways.',
  },
  angry: {
    name: 'Angry',
    lookFor: 'Eyebrows pull down and together, the eyes stare hard, and the lips press tight.',
  },
  disgusted: {
    name: 'Disgusted',
    lookFor: 'The nose wrinkles and the upper lip lifts, as if something smells bad.',
  },
  contempt: {
    name: 'Contempt',
    lookFor: 'Only one corner of the mouth tightens and lifts: a one-sided half-smile.',
  },
}

type Sided =
  | 'browDown'
  | 'browOuterUp'
  | 'cheekSquint'
  | 'eyeBlink'
  | 'eyeLookDown'
  | 'eyeSquint'
  | 'eyeWide'
  | 'mouthFrown'
  | 'mouthPress'
  | 'mouthSmile'
  | 'mouthStretch'
  | 'mouthUpperUp'
  | 'noseSneer'

/** The same value on both sides of the face. */
function both(base: Sided, value: number): BlendshapePatch {
  return { [`${base}Left`]: value, [`${base}Right`]: value } as BlendshapePatch
}

/**
 * Each emotion at full strength, built from the FACS prototypes in
 * docs/ARCHITECTURE.md (happy = AU6 + AU12, sad = AU1 + AU4 + AU15, ...).
 * Within a family, the stronger members use more of the same movements
 * (content → excited → elated) plus a few natural companions such as a
 * slightly open jaw or a lowered gaze.
 */
export const EMOTION_SHAPES: Record<Emotion, BlendshapePatch> = {
  neutral: {},
  content: { ...both('cheekSquint', 0.35), ...both('mouthSmile', 0.55) },
  excited: {
    ...both('cheekSquint', 0.6),
    ...both('mouthSmile', 0.85),
    browInnerUp: 0.25,
    ...both('browOuterUp', 0.3),
    ...both('eyeWide', 0.15),
    jawOpen: 0.15,
  },
  elated: {
    ...both('cheekSquint', 0.85),
    ...both('mouthSmile', 1),
    browInnerUp: 0.2,
    ...both('browOuterUp', 0.35),
    ...both('eyeSquint', 0.2),
    jawOpen: 0.3,
  },
  surprised: {
    browInnerUp: 0.8,
    ...both('browOuterUp', 0.85),
    ...both('eyeWide', 0.8),
    jawOpen: 0.55,
    mouthFunnel: 0.15,
  },
  sad: { browInnerUp: 0.7, ...both('browDown', 0.3), ...both('mouthFrown', 0.6), ...both('eyeLookDown', 0.15) },
  dejected: {
    browInnerUp: 0.85,
    ...both('browDown', 0.4),
    ...both('mouthFrown', 0.85),
    ...both('eyeBlink', 0.3),
    ...both('eyeLookDown', 0.45),
  },
  // Fear shares surprise's raised brows and wide eyes; the brows pulling
  // together (AU4), tense lids (AU7) and sideways lips (AU20) tell them apart.
  scared: {
    browInnerUp: 0.85,
    ...both('browOuterUp', 0.6),
    ...both('browDown', 0.45),
    ...both('eyeWide', 0.85),
    ...both('eyeSquint', 0.4),
    ...both('mouthStretch', 0.75),
    jawOpen: 0.3,
  },
  angry: { ...both('browDown', 0.85), ...both('eyeWide', 0.35), ...both('eyeSquint', 0.55), ...both('mouthPress', 0.7) },
  disgusted: {
    ...both('noseSneer', 0.8),
    ...both('mouthUpperUp', 0.65),
    ...both('mouthFrown', 0.45),
    ...both('browDown', 0.2),
  },
  // One-sided AU12 + AU14 on the face's right side.
  contempt: { mouthSmileRight: 0.55, mouthDimpleRight: 0.6 },
}

/**
 * Target blendshapes for `emotion`, scaled by the session's
 * `expression_intensity` (1 = exaggerated, ~0.3 = subtle).
 */
export function emotionTargets(emotion: Emotion, intensity: number): Blendshapes {
  const scale = clamp01(intensity)
  const shape = EMOTION_SHAPES[emotion] ?? {}
  const scaled: BlendshapePatch = {}
  for (const [name, value] of Object.entries(shape)) {
    scaled[name as keyof BlendshapePatch] = value * scale
  }
  return blendshapesFrom(scaled)
}
