// MediaPipe Face Landmarker on the video avatar's <video>, for the expression
// panel. The library and its WASM runtime load on demand (self-hosted under
// VITE_MEDIAPIPE_WASM_BASE by scripts/copy-mediapipe-wasm.mjs).

import type { FaceLandmarker } from '@mediapipe/tasks-vision'
import { fromCategories, type Blendshapes } from './blendshapes'

const WASM_BASE = import.meta.env.VITE_MEDIAPIPE_WASM_BASE || '/mediapipe/wasm'
const MODEL_URL =
  import.meta.env.VITE_FACE_LANDMARKER_MODEL ||
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

/** Analysis rate: plenty for expressions, light on the CPU/GPU. */
export const TRACKING_FPS = 12

export const TRACKING_UNAVAILABLE_MESSAGE =
  'Face reading isn’t available in this browser, so the panel can’t follow the video avatar’s face. The video still works.'

export function canTrackFaces(): boolean {
  return typeof WebAssembly === 'object' && typeof document !== 'undefined' && typeof requestAnimationFrame === 'function'
}

function hasWebGL2(): boolean {
  try {
    return Boolean(document.createElement('canvas').getContext('webgl2'))
  } catch {
    return false
  }
}

/** Loads Face Landmarker (GPU when WebGL2 works, otherwise CPU). Null if `signal` aborts first. */
export async function loadFaceLandmarker(signal: AbortSignal): Promise<FaceLandmarker | null> {
  const vision = await import('@mediapipe/tasks-vision')
  if (signal.aborted) return null
  const fileset = await vision.FilesetResolver.forVisionTasks(WASM_BASE)
  if (signal.aborted) return null
  const create = (delegate: 'GPU' | 'CPU') =>
    vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
    })
  let landmarker: FaceLandmarker
  if (hasWebGL2()) {
    try {
      landmarker = await create('GPU')
    } catch {
      landmarker = await create('CPU')
    }
  } else {
    landmarker = await create('CPU')
  }
  if (signal.aborted) {
    landmarker.close()
    return null
  }
  return landmarker
}

/**
 * Runs face tracking on `video` about `fps` times a second until the returned
 * function is called. `onFrame` receives raw blendshapes, or null when no face
 * is visible.
 */
export function trackVideo(
  landmarker: FaceLandmarker,
  video: HTMLVideoElement,
  onFrame: (shapes: Blendshapes | null, now: number) => void,
  onError: (error: unknown) => void,
  fps = TRACKING_FPS,
): () => void {
  let frame = 0
  let stopped = false
  let lastRun = -Infinity
  let lastVideoTime = -1
  let lastTimestamp = 0

  const tick = (now: number) => {
    if (stopped) return
    frame = requestAnimationFrame(tick)
    if (now - lastRun < 1000 / fps) return
    if (video.readyState < 2 || video.videoWidth === 0 || video.currentTime === lastVideoTime) return
    lastRun = now
    lastVideoTime = video.currentTime
    // detectForVideo needs strictly increasing timestamps.
    const timestamp = Math.max(Math.round(now), lastTimestamp + 1)
    lastTimestamp = timestamp
    try {
      const result = landmarker.detectForVideo(video, timestamp)
      const categories = result.faceBlendshapes?.[0]?.categories
      onFrame(categories && categories.length > 0 ? fromCategories(categories) : null, now)
    } catch (error) {
      stopped = true
      cancelAnimationFrame(frame)
      onError(error)
    }
  }
  frame = requestAnimationFrame(tick)
  return () => {
    stopped = true
    cancelAnimationFrame(frame)
  }
}
