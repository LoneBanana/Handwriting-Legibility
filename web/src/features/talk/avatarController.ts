// Drives the built-in avatar's 52 blendshapes over time. Three layers combine
// each frame:
//   1. expression - FACS-based targets for the current emotion, eased in;
//   2. lip-sync   - mouth shapes from the viseme timeline while speaking;
//   3. blinking   - natural, slightly irregular blinks.
// Time is passed in (`step(now)`), so the controller is deterministic in tests.
// With reduced motion the face holds each expression without easing, blinking
// or lip movement.

import type { Emotion } from '../../api/types'
import { clamp01, easeToward, neutralBlendshapes, blendshapesFrom, type Blendshapes } from './blendshapes'
import { emotionTargets } from './expressions'
import {
  buildVisemeTimeline,
  timeForChar,
  visemeAt,
  wordAt,
  VISEME_KEYS,
  VISEME_SHAPES,
  type Viseme,
  type VisemeTimeline,
} from './visemes'

/** Time constant for easing between expressions. */
export const EXPRESSION_EASE_MS = 200
/** Time constant for easing between mouth shapes (co-articulation). */
export const MOUTH_EASE_MS = 50
/** How long the lips wait at a word's end for the next boundary event. */
export const MAX_BOUNDARY_HOLD_MS = 650
export const BLINK_MS = 200

const VISEME_TARGETS: Record<Viseme, Blendshapes> = Object.fromEntries(
  Object.entries(VISEME_SHAPES).map(([viseme, patch]) => [viseme, blendshapesFrom(patch)]),
) as Record<Viseme, Blendshapes>

const smoothstep = (x: number) => x * x * (3 - 2 * x)

/** Eyelid closure (0..1) `elapsed` ms into a blink: quick close, brief hold, slower open. */
export function blinkAmount(elapsed: number): number {
  if (elapsed < 0 || elapsed >= BLINK_MS) return 0
  if (elapsed < 70) return smoothstep(elapsed / 70)
  if (elapsed < 100) return 1
  return 1 - smoothstep((elapsed - 100) / (BLINK_MS - 100))
}

type SpeechState = {
  timeline: VisemeTimeline
  playhead: number
  /** While boundary events arrive, the lips wait at the end of the current word. */
  holdUntil: number | null
  heldFor: number
}

export type AvatarControllerOptions = {
  random?: () => number
  reducedMotion?: boolean
}

export class AvatarController {
  private expression = neutralBlendshapes()
  private target = neutralBlendshapes()
  private mouth = neutralBlendshapes()
  private speech: SpeechState | null = null
  private last: number | null = null
  private nextBlinkAt: number | null = null
  private blinkStartedAt: number | null = null
  private reducedMotion: boolean
  private readonly random: () => number
  private output = neutralBlendshapes()

  constructor({ random = Math.random, reducedMotion = false }: AvatarControllerOptions = {}) {
    this.random = random
    this.reducedMotion = reducedMotion
  }

  setReducedMotion(value: boolean): void {
    this.reducedMotion = value
    if (value) this.blinkStartedAt = null
  }

  /** Starts easing toward `emotion` at the session's expression intensity. */
  setExpression(emotion: Emotion, intensity: number): void {
    this.target = emotionTargets(emotion, intensity)
  }

  /** Starts lip-sync for an utterance that has just begun playing. */
  speak(text: string, wordsPerMinute: number): void {
    this.speech = { timeline: buildVisemeTimeline(text, wordsPerMinute), playhead: 0, holdUntil: null, heldFor: 0 }
  }

  /** A SpeechSynthesis word boundary: jump the lips to that word. */
  boundary(charIndex: number): void {
    const speech = this.speech
    if (!speech) return
    speech.playhead = timeForChar(speech.timeline, charIndex)
    speech.holdUntil = wordAt(speech.timeline, speech.playhead)?.end ?? null
    speech.heldFor = 0
  }

  /** The utterance ended: let the mouth settle. */
  silence(): void {
    this.speech = null
  }

  get speaking(): boolean {
    return this.speech !== null
  }

  get current(): Blendshapes {
    return this.output
  }

  step(now: number): Blendshapes {
    const dt = this.last === null ? 16 : Math.min(Math.max(now - this.last, 0), 100)
    this.last = now

    if (this.reducedMotion) {
      this.expression = { ...this.target }
      this.mouth = neutralBlendshapes()
    } else {
      this.expression = easeToward(this.expression, this.target, dt, EXPRESSION_EASE_MS)
      this.mouth = easeToward(this.mouth, VISEME_TARGETS[this.advanceSpeech(dt)], dt, MOUTH_EASE_MS)
    }
    const blink = this.reducedMotion ? 0 : this.blink(now)

    const out = { ...this.expression }
    const talking = this.speech !== null && !this.reducedMotion
    for (const name of VISEME_KEYS) {
      // While talking, words move the jaw; the expression keeps part of its own opening.
      const own = name === 'jawOpen' && talking ? this.expression.jawOpen * 0.6 : this.expression[name]
      out[name] = clamp01(own + this.mouth[name])
    }
    out.eyeBlinkLeft = Math.max(this.expression.eyeBlinkLeft, blink)
    out.eyeBlinkRight = Math.max(this.expression.eyeBlinkRight, blink)
    this.output = out
    return out
  }

  private advanceSpeech(dt: number): Viseme {
    const speech = this.speech
    if (!speech) return 'rest'
    const { timeline } = speech
    speech.playhead += dt
    if (speech.holdUntil !== null && speech.playhead >= speech.holdUntil) {
      speech.playhead = speech.holdUntil
      speech.heldFor += dt
      if (speech.heldFor < MAX_BOUNDARY_HOLD_MS) return 'rest'
      speech.holdUntil = null // boundary events stopped: run on the estimate again
    }
    if (timeline.duration > 0 && speech.playhead >= timeline.duration) {
      // The voice is slower than estimated: keep the lips moving until it ends.
      speech.playhead %= timeline.duration
    }
    return visemeAt(timeline, speech.playhead)
  }

  private blink(now: number): number {
    if (this.nextBlinkAt === null) this.nextBlinkAt = now + this.blinkInterval()
    if (this.blinkStartedAt === null && now >= this.nextBlinkAt) this.blinkStartedAt = now
    if (this.blinkStartedAt === null) return 0
    const elapsed = now - this.blinkStartedAt
    if (elapsed >= BLINK_MS) {
      this.blinkStartedAt = null
      // Now and then people blink twice in quick succession.
      this.nextBlinkAt = now + (this.random() < 0.12 ? 90 : this.blinkInterval())
      return 0
    }
    return blinkAmount(elapsed)
  }

  private blinkInterval(): number {
    return 2200 + this.random() * 3800
  }
}
