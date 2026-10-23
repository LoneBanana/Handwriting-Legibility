// Joins the Tavus conversation room with daily-js as a call object (not the
// prebuilt iframe), so the replica's video and audio tracks can be attached to
// our own <video>/<audio> elements and analysed by the face panel. daily-js is
// imported on demand so other pages never download it.

import type { DailyCall, DailyParticipant, DailyTrackState } from '@daily-co/daily-js'
import { replicaSpeakingFrom, type AppMessageSender } from './tavus'

export type ReplicaTracks = { video: MediaStreamTrack | null; audio: MediaStreamTrack | null }

export type VideoCallEvents = {
  onTracks: (tracks: ReplicaTracks) => void
  onReplicaSpeaking: (speaking: boolean) => void
  onError: (message: string) => void
}

export type VideoCall = AppMessageSender & { leave: () => Promise<void> }

export const UNSUPPORTED_BROWSER_MESSAGE =
  'This browser can’t play the video avatar. A recent Chrome, Edge, Firefox or Safari works, or you can use the built-in face.'

function usableTrack(state: DailyTrackState | undefined): MediaStreamTrack | null {
  if (!state || state.state === 'off' || state.state === 'blocked') return null
  // persistentTrack avoids a Safari bug with unmuting remote tracks.
  return state.persistentTrack ?? state.track ?? null
}

function replicaTracks(call: DailyCall): ReplicaTracks {
  const remote: DailyParticipant[] = Object.values(call.participants()).filter((p) => !p.local)
  const withVideo = remote.find((p) => usableTrack(p.tracks.video)) ?? remote[0]
  return { video: usableTrack(withVideo?.tracks.video), audio: usableTrack(withVideo?.tracks.audio) }
}

async function teardown(call: DailyCall): Promise<void> {
  try {
    await call.leave()
  } catch {
    // Already left or never joined.
  }
  try {
    if (!call.isDestroyed()) await call.destroy()
  } catch {
    // Nothing more to clean up.
  }
}

/**
 * Joins `url` without sending the learner's camera or microphone. Resolves to
 * null if `signal` aborts first (the page was left while connecting).
 */
export async function joinVideoCall(
  url: string,
  token: string | null,
  events: VideoCallEvents,
  signal: AbortSignal,
): Promise<VideoCall | null> {
  const { default: Daily } = await import('@daily-co/daily-js')
  if (signal.aborted) return null
  if (!Daily.supportedBrowser().supported) throw new Error(UNSUPPORTED_BROWSER_MESSAGE)

  const call = Daily.createCallObject({
    // React's development double-mount can briefly overlap two instances.
    allowMultipleCallInstances: true,
    videoSource: false,
    audioSource: false,
    subscribeToTracksAutomatically: true,
  })
  const sync = () => {
    if (!call.isDestroyed()) events.onTracks(replicaTracks(call))
  }
  call
    .on('track-started', sync)
    .on('track-stopped', sync)
    .on('participant-joined', sync)
    .on('participant-updated', sync)
    .on('participant-left', sync)
    .on('app-message', (event) => {
      const speaking = replicaSpeakingFrom(event?.data)
      if (speaking !== null) events.onReplicaSpeaking(speaking)
    })
    .on('error', (event) => events.onError(event?.errorMsg || 'The video call stopped unexpectedly.'))

  try {
    await call.join({ url, token: token ?? undefined, startVideoOff: true, startAudioOff: true })
  } catch (error) {
    await teardown(call)
    throw error
  }
  if (signal.aborted) {
    await teardown(call)
    return null
  }
  sync()
  return {
    sendAppMessage: (data, to) => call.sendAppMessage(data, to),
    leave: () => teardown(call),
  }
}
