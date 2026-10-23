// Pure geometry for the built-in avatar: blendshapes in, SVG shapes out.
// Coordinates are in a 200 × 240 viewBox. The avatar faces the viewer, so the
// face's own LEFT side is drawn on the viewer's RIGHT (larger x).

import type { Blendshapes } from './blendshapes'

export const FACE_VIEWBOX = '0 0 200 240'

type Side = 'Left' | 'Right'
const SIDES: readonly Side[] = ['Right', 'Left']

/** The blendshape for one side of the face, e.g. sided('eyeBlink', 'Left'). */
const sided = <Base extends string>(base: Base, side: Side) => `${base}${side}` as `${Base}${Side}`

/** +1 toward the face's own left (viewer's right), −1 toward its right. */
const outwardOf = (side: Side) => (side === 'Left' ? 1 : -1)

const f = (n: number) => Number(n.toFixed(2))

export type EyeShape = {
  side: Side
  /** Visible eye opening (white of the eye), also used as a clip path. */
  opening: string
  upperLid: string
  lowerLid: string
  iris: { x: number; y: number }
  /** Laughter lines beside the eye (AU6), 0..1. */
  crinkle: number
  crinklePath: string
}

export type BrowShape = { side: Side; path: string }

export type MouthShape = {
  /** Outline of the lips; filled dark when open. */
  outline: string
  /** Line where the lips meet, for a closed mouth. */
  seam: string
  /** How far apart the lips are, in viewBox units. */
  opening: number
  lipWidth: number
  teeth: { x: number; y: number; width: number; height: number } | null
  tongue: { cx: number; cy: number; rx: number; ry: number } | null
  corners: { side: Side; x: number; y: number }[]
  dimples: { side: Side; path: string; opacity: number }[]
}

export type FaceShape = {
  head: string
  brows: BrowShape[]
  /** Vertical frown lines between the brows (AU4), 0..1. */
  furrow: number
  /** Horizontal forehead lines (AU1/AU2), 0..1. */
  foreheadLines: number
  eyes: EyeShape[]
  cheeks: { side: Side; cx: number; cy: number; opacity: number }[]
  nose: { bottom: string; nostrils: { cx: number; cy: number; rx: number }[]; wrinkles: number }
  mouth: MouthShape
}

const EYE_Y = 108
const EYE_HALF_WIDTH = 15
const eyeX = (side: Side) => (side === 'Left' ? 128 : 72)

function eye(side: Side, s: Blendshapes): EyeShape {
  const cx = eyeX(side)
  const out = outwardOf(side)
  const blink = s[sided('eyeBlink', side)]
  const wide = s[sided('eyeWide', side)]
  const squint = s[sided('eyeSquint', side)]
  const cheek = s[sided('cheekSquint', side)]
  const open = 1 - blink
  const up = Math.max(0, 9.5 * open * (1 + 0.55 * wide - 0.3 * squint - 0.12 * cheek))
  const down = Math.max(0.3, 6 * open * (1 + 0.25 * wide - 0.5 * squint - 0.45 * cheek))
  const y = EYE_Y + 1.5 * blink
  const left = cx - EYE_HALF_WIDTH
  const right = cx + EYE_HALF_WIDTH
  const gazeX = (s[sided('eyeLookOut', side)] - s[sided('eyeLookIn', side)]) * 5 * out
  const gazeY = (s[sided('eyeLookDown', side)] - s[sided('eyeLookUp', side)]) * 4
  const cornerX = cx + out * (EYE_HALF_WIDTH + 3)
  return {
    side,
    opening: `M${f(left)},${f(y)} Q${f(cx)},${f(y - 2 * up)} ${f(right)},${f(y)} Q${f(cx)},${f(y + 2 * down)} ${f(left)},${f(y)} Z`,
    upperLid: `M${f(left - 1)},${f(y + 0.5)} Q${f(cx)},${f(y - 2 * up - 1)} ${f(right + 1)},${f(y + 0.5)}`,
    lowerLid: `M${f(left + 2)},${f(y + 0.5)} Q${f(cx)},${f(y + 2 * down)} ${f(right - 2)},${f(y + 0.5)}`,
    iris: { x: f(cx + gazeX), y: f(EYE_Y + gazeY) },
    crinkle: cheek,
    crinklePath:
      `M${f(cornerX)},${f(EYE_Y - 3)} l${f(out * 5)},-3 ` +
      `M${f(cornerX + out)},${f(EYE_Y + 0.5)} l${f(out * 6)},0 ` +
      `M${f(cornerX)},${f(EYE_Y + 4)} l${f(out * 5)},3`,
  }
}

function brow(side: Side, s: Blendshapes): BrowShape {
  const out = outwardOf(side)
  const cx = eyeX(side)
  const innerUp = s.browInnerUp
  const outerUp = s[sided('browOuterUp', side)]
  const down = s[sided('browDown', side)]
  // Lowered brows (AU4) drop most at the inner end and pull toward the nose.
  const inner = { x: cx - out * 14 - out * 3 * down, y: 86 - 10 * innerUp + 9 * down }
  const outer = { x: cx + out * 16, y: 87 - 9 * outerUp + 2 * down }
  const mid = { x: (inner.x + outer.x) / 2, y: Math.min(inner.y, outer.y) - 4 + 2 * down - 1.5 * outerUp }
  const control = { x: mid.x, y: 2 * mid.y - (inner.y + outer.y) / 2 }
  return { side, path: `M${f(inner.x)},${f(inner.y)} Q${f(control.x)},${f(control.y)} ${f(outer.x)},${f(outer.y)}` }
}

function mouth(s: Blendshapes): MouthShape {
  const avg = (a: number, b: number) => (a + b) / 2
  const cx = 100 + (s.mouthLeft - s.mouthRight) * 6
  const baseY = 164
  const press = avg(s.mouthPressLeft, s.mouthPressRight)
  const upperUp = avg(s.mouthUpperUpLeft, s.mouthUpperUpRight)
  const lowerDown = avg(s.mouthLowerDownLeft, s.mouthLowerDownRight)

  const corners = SIDES.map((side) => {
    const out = outwardOf(side)
    const smile = s[sided('mouthSmile', side)]
    const halfWidth =
      21 + 6 * smile + 7 * s[sided('mouthStretch', side)] + 2 * s[sided('mouthDimple', side)] - 9 * s.mouthPucker - 6 * s.mouthFunnel - 2 * press
    return {
      side,
      x: cx + out * Math.max(7, halfWidth),
      y: baseY - 7 * smile + 7 * s[sided('mouthFrown', side)] + 1.5 * s[sided('mouthStretch', side)] - 1.5 * s[sided('mouthDimple', side)],
    }
  })
  const [right, left] = corners // viewer's left, viewer's right
  const cornerY = (right.y + left.y) / 2
  // A raised upper lip (AU10) lifts away from the lower one and bares the top teeth.
  const upperY = baseY - 1 - 5 * upperUp - 1.5 * s.mouthFunnel + press
  const opening = Math.max(
    0,
    26 * s.jawOpen * (1 - 0.85 * s.mouthClose) + 3.5 * lowerDown + 3 * s.mouthFunnel + 4 * upperUp - 2 * press - 2 * s.mouthRollLower,
  )
  const lowerY = upperY + 1.5 + opening
  const upperControl = 2 * upperY - cornerY
  const lowerControl = 2 * lowerY - cornerY
  const halfSpan = (left.x - right.x) / 2

  const outline =
    `M${f(right.x)},${f(right.y)} Q${f(cx)},${f(upperControl)} ${f(left.x)},${f(left.y)} ` +
    `Q${f(cx)},${f(lowerControl)} ${f(right.x)},${f(right.y)} Z`
  const seamY = (upperY + lowerY) / 2
  const seam = `M${f(right.x)},${f(right.y)} Q${f(cx)},${f(2 * seamY - cornerY)} ${f(left.x)},${f(left.y)}`

  const teeth =
    opening > 2.5
      ? { x: f(cx - halfSpan * 0.62), y: f(upperY - 1), width: f(halfSpan * 1.24), height: f(Math.min(5.5, 1.5 + opening * 0.3)) }
      : null
  const tongue = opening > 9 ? { cx: f(cx), cy: f(lowerY - 2), rx: f(halfSpan * 0.5), ry: f(Math.min(5, opening * 0.22)) } : null

  return {
    outline,
    seam,
    opening: f(opening),
    lipWidth: f(3.6 * (1 - 0.5 * press) * (1 - 0.3 * s.mouthRollLower)),
    teeth,
    tongue,
    corners: corners.map((c) => ({ side: c.side, x: f(c.x), y: f(c.y) })),
    dimples: corners.map((c) => {
      const out = outwardOf(c.side)
      return {
        side: c.side,
        path: `M${f(c.x + out * 4)},${f(c.y - 3)} q${f(out * 2.5)},3 0,6`,
        opacity: f(Math.min(1, s[sided('mouthDimple', c.side)] * 0.9)),
      }
    }),
  }
}

export function faceGeometry(s: Blendshapes): FaceShape {
  const chinY = 198 + 9 * s.jawOpen
  const jawCurve = 186 + 7 * s.jawOpen
  const head =
    `M100,34 C138,34 164,62 164,108 C164,152 140,${f(jawCurve)} 100,${f(chinY)} ` +
    `C60,${f(jawCurve)} 36,152 36,108 C36,62 62,34 100,34 Z`
  const sneer = Math.max(s.noseSneerLeft, s.noseSneerRight)
  const mouthShape = mouth(s)
  const noseLift = 2 * sneer

  return {
    head,
    brows: SIDES.map((side) => brow(side, s)),
    furrow: Math.min(1, Math.max(s.browDownLeft, s.browDownRight) * 0.85),
    foreheadLines: Math.min(1, Math.max(s.browInnerUp, s.browOuterUpLeft, s.browOuterUpRight) * 0.6),
    eyes: SIDES.map((side) => eye(side, s)),
    cheeks: SIDES.map((side) => {
      const cheek = s[sided('cheekSquint', side)]
      return { side, cx: side === 'Left' ? 134 : 66, cy: f(141 - 5 * cheek), opacity: f(0.18 + 0.45 * cheek) }
    }),
    nose: {
      bottom: `M90,${f(139 - noseLift)} Q100,${f(146 - noseLift)} 110,${f(139 - noseLift)}`,
      nostrils: [94, 106].map((cx) => ({ cx, cy: f(139 - noseLift), rx: f(2.6 + sneer) })),
      wrinkles: Math.min(1, sneer),
    },
    mouth: mouthShape,
  }
}
