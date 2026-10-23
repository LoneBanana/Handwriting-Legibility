import { useEffect, useRef, useState } from 'react'
import type { ConverseSession } from '../../api/types'
import { Icon } from '../../components/Icon'
import { errorMessage } from '../../lib/errors'
import { canTrackFaces, loadFaceLandmarker, trackVideo, TRACKING_UNAVAILABLE_MESSAGE } from './faceTracker'
import type { FaceStore } from './faceStore'
import { BaselineTracker, calibrate } from './tracking'
import { joinVideoCall, type ReplicaTracks, type VideoCall } from './videoCall'

type Props = {
  tavus: NonNullable<ConverseSession['tavus']>
  store: FaceStore
  /** The connected call (for sending echo/interrupt messages), or null when gone. */
  onCall: (call: VideoCall | null) => void
  onReplicaSpeaking: (speaking: boolean) => void
  /** A note for the expression panel about face reading, or null when it is running. */
  onTrackerNote: (note: string | null) => void
  onUseBuiltIn: () => void
}

type Status = 'connecting' | 'waiting' | 'live' | 'error'

function statusOf(problem: string | null, tracks: ReplicaTracks, joined: boolean): Status {
  if (problem) return 'error'
  if (tracks.video) return 'live'
  return joined ? 'waiting' : 'connecting'
}

function attach(element: HTMLMediaElement | null, track: MediaStreamTrack | null): HTMLMediaElement | null {
  if (!element) return null
  const current = element.srcObject instanceof MediaStream ? element.srcObject.getTracks()[0] : null
  if (current === track) return element
  element.srcObject = track ? new MediaStream([track]) : null
  return track ? element : null
}

/**
 * The realistic (Tavus) avatar: joins the conversation room, shows the
 * replica's video in our own <video> element, plays its voice, and runs face
 * tracking on the video for the expression panel. Mount it with a `key` per
 * conversation so its state starts fresh.
 */
export function VideoAvatar({ tavus, store, onCall, onReplicaSpeaking, onTrackerNote, onUseBuiltIn }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const [joined, setJoined] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)
  const [tracks, setTracks] = useState<ReplicaTracks>({ video: null, audio: null })
  const [soundBlocked, setSoundBlocked] = useState(false)
  const status = statusOf(problem, tracks, joined)
  const callbacks = useRef({ onCall, onReplicaSpeaking, onTrackerNote })

  useEffect(() => {
    callbacks.current = { onCall, onReplicaSpeaking, onTrackerNote }
  }, [onCall, onReplicaSpeaking, onTrackerNote])

  // Join the room for this conversation; leave it when the session changes or the page closes.
  useEffect(() => {
    const controller = new AbortController()
    let call: VideoCall | null = null
    joinVideoCall(
      tavus.conversation_url,
      tavus.meeting_token,
      {
        onTracks: setTracks,
        onReplicaSpeaking: (speaking) => callbacks.current.onReplicaSpeaking(speaking),
        onError: (message) => setProblem(message),
      },
      controller.signal,
    ).then(
      (joined) => {
        if (!joined) return
        call = joined
        callbacks.current.onCall(joined)
        setJoined(true)
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setProblem(errorMessage(error))
      },
    )
    return () => {
      controller.abort()
      callbacks.current.onCall(null)
      if (call) void call.leave()
    }
  }, [tavus.conversation_url, tavus.meeting_token])

  useEffect(() => {
    const video = attach(videoRef.current, tracks.video)
    void video?.play().catch(() => undefined) // muted video may always autoplay
  }, [tracks.video])

  useEffect(() => {
    const audio = attach(audioRef.current, tracks.audio)
    if (!audio) return
    audio.play().then(
      () => setSoundBlocked(false),
      () => setSoundBlocked(true),
    )
  }, [tracks.audio])

  // Face tracking on the replica's video, feeding the expression panel.
  useEffect(() => {
    const video = videoRef.current
    if (!tracks.video || !video) return
    if (!canTrackFaces()) {
      callbacks.current.onTrackerNote(TRACKING_UNAVAILABLE_MESSAGE)
      return
    }
    const controller = new AbortController()
    let stop: (() => void) | null = null
    let close: (() => void) | null = null
    const baseline = new BaselineTracker()
    let last: number | null = null
    callbacks.current.onTrackerNote('Starting face reading…')
    loadFaceLandmarker(controller.signal).then(
      (landmarker) => {
        if (!landmarker) return
        close = () => landmarker.close()
        callbacks.current.onTrackerNote(null)
        stop = trackVideo(
          landmarker,
          video,
          (raw, now) => {
            if (!raw) {
              store.publish(null, now)
              return
            }
            baseline.update(raw, last === null ? 0 : now - last)
            last = now
            store.publish(calibrate(raw, baseline.baseline), now)
          },
          () => callbacks.current.onTrackerNote(TRACKING_UNAVAILABLE_MESSAGE),
        )
      },
      () => {
        if (!controller.signal.aborted) callbacks.current.onTrackerNote(TRACKING_UNAVAILABLE_MESSAGE)
      },
    )
    return () => {
      controller.abort()
      stop?.()
      close?.()
      callbacks.current.onTrackerNote(null)
    }
  }, [tracks.video, store])

  const unmute = () => {
    audioRef.current?.play().then(
      () => setSoundBlocked(false),
      () => setSoundBlocked(true),
    )
  }

  return (
    <div className="video-avatar" data-status={status}>
      <video ref={videoRef} autoPlay playsInline muted aria-label="Video of the avatar" />
      <audio ref={audioRef} autoPlay />
      {(status === 'connecting' || status === 'waiting') && (
        <div className="video-avatar__overlay" role="status">
          <span className="spinner" aria-hidden="true" />{' '}
          {status === 'connecting' ? 'Connecting to the video avatar…' : 'Waiting for the avatar to join…'}
        </div>
      )}
      {status === 'error' && (
        <div className="video-avatar__overlay">
          <div className="notice notice--error" role="alert">
            <div className="notice__title">The video avatar couldn’t connect</div>
            <p>{problem}</p>
            <button type="button" className="btn btn--secondary" onClick={onUseBuiltIn}>
              Use the built-in face instead
            </button>
          </div>
        </div>
      )}
      {soundBlocked && status === 'live' && (
        <button type="button" className="btn video-avatar__sound" onClick={unmute}>
          <Icon name="speaker" /> Turn on the avatar’s voice
        </button>
      )}
    </div>
  )
}
