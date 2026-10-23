// A tiny external store between whatever produces blendshapes (the built-in
// avatar at 60 fps, or face tracking on the video at ~12 fps) and the
// expression panel. It smooths the values, analyses them, and notifies the
// panel at a gentle rate so the page doesn't re-render every frame and nothing
// on screen flickers.

import { easeToward, neutralBlendshapes, type Blendshapes } from './blendshapes'
import {
  estimateFeeling,
  readActionUnits,
  regionActivations,
  type AuReading,
  type FaceRegion,
  type FeelingEstimate,
} from './faceAnalysis'

export type FaceSnapshot = {
  /** Something has been published since the last reset. */
  hasData: boolean
  /** False when face tracking can't see a face in the video. */
  faceFound: boolean
  reading: AuReading
  regions: Record<FaceRegion, number>
  estimate: FeelingEstimate
}

export type FaceStore = {
  publish: (shapes: Blendshapes | null, now: number) => void
  subscribe: (listener: () => void) => () => void
  getSnapshot: () => FaceSnapshot
  reset: () => void
  setNotifyInterval: (ms: number) => void
}

export type FaceStoreOptions = {
  /** Minimum time between panel updates. */
  notifyEveryMs?: number
  /** Smoothing for the bars and diagram. */
  displayEaseMs?: number
  /** Heavier smoothing for the feeling estimate, so talking doesn't sway it. */
  estimateEaseMs?: number
}

function buildSnapshot(display: Blendshapes, forEstimate: Blendshapes, hasData: boolean, faceFound: boolean): FaceSnapshot {
  const reading = readActionUnits(display)
  return {
    hasData,
    faceFound,
    reading,
    regions: regionActivations(reading),
    estimate: estimateFeeling(readActionUnits(forEstimate)),
  }
}

export function createFaceStore({
  notifyEveryMs = 125,
  displayEaseMs = 90,
  estimateEaseMs = 350,
}: FaceStoreOptions = {}): FaceStore {
  let interval = notifyEveryMs
  let display = neutralBlendshapes()
  let forEstimate = neutralBlendshapes()
  let lastPublish: number | null = null
  let lastNotify: number | null = null
  let hasData = false
  let faceFound = false
  let snapshot = buildSnapshot(display, forEstimate, false, false)
  const listeners = new Set<() => void>()

  const notify = () => {
    snapshot = buildSnapshot(display, forEstimate, hasData, faceFound)
    for (const listener of listeners) listener()
  }

  return {
    publish(shapes, now) {
      const dt = lastPublish === null ? 0 : Math.min(Math.max(now - lastPublish, 0), 250)
      const first = !hasData
      lastPublish = now
      hasData = true
      faceFound = shapes !== null
      // With no face in view the bars relax rather than freezing on stale values.
      const target = shapes ?? neutralBlendshapes()
      if (first && shapes) {
        display = { ...shapes }
        forEstimate = { ...shapes }
      } else {
        display = easeToward(display, target, dt, displayEaseMs)
        forEstimate = easeToward(forEstimate, target, dt, estimateEaseMs)
      }
      if (lastNotify === null || now - lastNotify >= interval) {
        lastNotify = now
        notify()
      }
    },
    subscribe(listener) {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    getSnapshot: () => snapshot,
    reset() {
      display = neutralBlendshapes()
      forEstimate = neutralBlendshapes()
      lastPublish = null
      lastNotify = null
      hasData = false
      faceFound = false
      notify()
    },
    setNotifyInterval(ms) {
      interval = ms
    },
  }
}
