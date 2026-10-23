import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { streamConverseMessage } from '../../api/endpoints'
import type { ConverseSession, Emotion } from '../../api/types'
import { errorMessage } from '../../lib/errors'
import { speechSupported, stopSpeaking } from '../../lib/speech'
import type { AvatarController } from './avatarController'
import type { IntendedFeeling } from './ExpressionPanel'
import type { FaceStore } from './faceStore'
import { primeSpeech, SpeechQueue } from './speechQueue'
import { parseTalkEvent } from './stream'
import { sendEcho, sendInterrupt } from './tavus'
import { canCheck, isTurnActive, turnsReducer, type Turn } from './turns'
import type { VideoCall } from './videoCall'

/** If the video avatar never says it started talking, pace captions anyway after this long. */
const REPLICA_START_FALLBACK_MS = 2500
/**
 * The video avatar speaks at its own pace (the session's words_per_minute only
 * drives the built-in voice), so captions are paced by a typical rate and
 * finished off when the replica reports it stopped speaking.
 */
const REPLICA_WORDS_PER_MINUTE = 160
/** "Stopped speaking" this long without starting again means the answer is over. */
const REPLICA_STOP_GRACE_MS = 1200
/** After an answer, the built-in face relaxes back to neutral. */
const RELAX_AFTER_MS = 2500

const ANSWERING = 'The avatar is answering.'
const CUT_OFF = 'The answer stopped part-way. Please try asking again.'

type Segment = { index: number; emotion: Emotion; text: string }

let turnCounter = 0
const newTurnId = () => `turn-${Date.now().toString(36)}-${(turnCounter += 1)}`

const isAbort = (error: unknown) => error instanceof Error && error.name === 'AbortError'

type Options = {
  session: ConverseSession | null
  avatar: AvatarController
  store: FaceStore
  voiceName: string | null
}

export type TalkConversation = {
  turns: Turn[]
  /** The turn being answered (streamed or spoken), if any. */
  activeTurn: Turn | null
  busy: boolean
  intended: IntendedFeeling | null
  /** The browser refused to speak; answers are shown as text instead. */
  voiceBlocked: boolean
  /** Short status for screen readers: only the big moments, so it doesn't talk over the avatar. */
  announcement: string
  ask: (question: string) => void
  stop: () => void
  reveal: (turnId: string) => void
  setCall: (call: VideoCall | null) => void
  onReplicaSpeaking: (speaking: boolean) => void
}

/**
 * Everything that happens when the learner asks a question: stream the answer,
 * speak it segment by segment (or hand it to the video avatar), drive the
 * built-in face, reveal each sentence's intended feeling, and stop cleanly.
 * State belongs to one session: mount the caller with a `key` per session.
 */
export function useTalkConversation({ session, avatar, store, voiceName }: Options): TalkConversation {
  const [turns, dispatch] = useReducer(turnsReducer, [])
  const [intended, setIntended] = useState<IntendedFeeling | null>(null)
  const [voiceBlocked, setVoiceBlocked] = useState(false)
  const [announcement, setAnnouncement] = useState('')
  const [queue] = useState(() => new SpeechQueue({ voice: false, wordsPerMinute: 160, voiceName: null }))

  const sessionRef = useRef(session)
  const streamRef = useRef<AbortController | null>(null)
  const callRef = useRef<VideoCall | null>(null)
  const replicaRef = useRef<{ id: string; segments: Segment[]; started: boolean } | null>(null)
  const timers = useRef<{ relax?: number; replicaStart?: number; replicaStop?: number }>({})
  const voiceFailures = useRef(0)

  const latest = turns.length > 0 ? turns[turns.length - 1] : null
  const activeTurn = latest && isTurnActive(latest) ? latest : null
  const activeRef = useRef(activeTurn)

  useEffect(() => {
    sessionRef.current = session
  }, [session])
  useEffect(() => {
    activeRef.current = activeTurn
  }, [activeTurn])

  const usesVideo = session?.provider === 'tavus' && session.tavus !== null
  const speakAloud = !usesVideo && speechSupported() && !voiceBlocked

  useEffect(() => {
    // The video avatar speaks for itself; the queue then only paces captions.
    queue.configure({
      voice: speakAloud,
      wordsPerMinute: usesVideo ? REPLICA_WORDS_PER_MINUTE : (session?.words_per_minute ?? 160),
      voiceName,
    })
  }, [queue, speakAloud, usesVideo, session, voiceName])

  const clearTimers = useCallback(() => {
    window.clearTimeout(timers.current.relax)
    window.clearTimeout(timers.current.replicaStart)
    window.clearTimeout(timers.current.replicaStop)
  }, [])

  const relaxSoon = useCallback(
    (delay = RELAX_AFTER_MS) => {
      window.clearTimeout(timers.current.relax)
      timers.current.relax = window.setTimeout(() => avatar.setExpression('neutral', 1), delay)
    },
    [avatar],
  )

  const revealIntended = useCallback(
    (segment: Segment) => {
      const snapshot = store.getSnapshot()
      setIntended({
        emotion: segment.emotion,
        sentence: segment.index + 1,
        seenAs: snapshot.hasData && snapshot.faceFound ? snapshot.estimate.family : null,
      })
    },
    [store],
  )

  useEffect(() => {
    queue.setHandlers({
      onStart: (item, voiced) => {
        if (voiced) voiceFailures.current = 0
        window.clearTimeout(timers.current.relax)
        dispatch({ type: 'segmentState', id: item.turnKey, index: item.index, state: 'speaking' })
        const current = sessionRef.current
        if (current?.provider !== 'tavus') {
          avatar.setExpression(item.emotion, current?.expression_intensity ?? 0.8)
          avatar.speak(item.text, current?.words_per_minute ?? 160)
        }
        setAnnouncement(ANSWERING)
      },
      onBoundary: (_item, charIndex) => avatar.boundary(charIndex),
      onEnd: (item) => {
        dispatch({ type: 'segmentState', id: item.turnKey, index: item.index, state: 'spoken' })
        avatar.silence()
        revealIntended(item)
      },
      onVoiceFailed: () => {
        // Two failures in a row means the browser won't speak: switch to words on screen.
        voiceFailures.current += 1
        if (voiceFailures.current >= 2) setVoiceBlocked(true)
      },
      onIdle: () => relaxSoon(),
    })
  }, [queue, avatar, revealIntended, relaxSoon])

  // Announce when an answer ends (errors announce themselves as alerts).
  const wasActive = useRef(false)
  useEffect(() => {
    const isActive = activeTurn !== null
    if (wasActive.current && !isActive && latest && latest.status !== 'error') {
      const stopped = latest.status === 'stopped' || latest.segments.some((s) => s.state === 'skipped')
      setAnnouncement(
        stopped
          ? 'Stopped.'
          : `The avatar finished answering.${canCheck(latest) ? ' You can check your understanding below.' : ''}`,
      )
    }
    wasActive.current = isActive
  }, [activeTurn, latest])

  // Leaving (or switching sessions, which remounts the conversation) stops everything.
  useEffect(() => {
    return () => {
      streamRef.current?.abort()
      streamRef.current = null
      replicaRef.current = null
      queue.clear()
      stopSpeaking()
      avatar.silence()
      clearTimers()
    }
  }, [queue, avatar, clearTimers])

  const beginReplicaPacing = useCallback(() => {
    const pending = replicaRef.current
    if (!pending || pending.started) return
    pending.started = true
    window.clearTimeout(timers.current.replicaStart)
    for (const s of pending.segments) queue.enqueue({ turnKey: pending.id, index: s.index, text: s.text, emotion: s.emotion })
  }, [queue])

  /**
   * Hands a finished answer to the video avatar: the `done` event's echo_text,
   * once per answer. Its emotion tags steer the replica's face.
   */
  const speakWithReplica = useCallback(
    (id: string, segments: Segment[], echoText: string) => {
      const call = callRef.current
      const conversationId = sessionRef.current?.tavus?.conversation_id
      if (!call || !conversationId || segments.length === 0 || !echoText.trim()) {
        dispatch({ type: 'allSpoken', id })
        return
      }
      try {
        sendEcho(call, conversationId, echoText)
      } catch {
        dispatch({ type: 'allSpoken', id })
        return
      }
      replicaRef.current = { id, segments, started: false }
      window.clearTimeout(timers.current.replicaStart)
      timers.current.replicaStart = window.setTimeout(beginReplicaPacing, REPLICA_START_FALLBACK_MS)
    },
    [beginReplicaPacing],
  )

  const onReplicaSpeaking = useCallback(
    (speaking: boolean) => {
      window.clearTimeout(timers.current.replicaStop)
      if (speaking) {
        beginReplicaPacing()
        return
      }
      timers.current.replicaStop = window.setTimeout(() => {
        const pending = replicaRef.current
        if (!pending?.started) return
        replicaRef.current = null
        queue.clear()
        dispatch({ type: 'allSpoken', id: pending.id })
        const last = pending.segments[pending.segments.length - 1]
        if (last) revealIntended(last)
      }, REPLICA_STOP_GRACE_MS)
    },
    [beginReplicaPacing, queue, revealIntended],
  )

  const ask = useCallback(
    async (question: string) => {
      const current = sessionRef.current
      if (!current || streamRef.current || activeRef.current) return
      const video = current.provider === 'tavus' && current.tavus !== null
      if (!video) primeSpeech() // still inside the click / key press
      const id = newTurnId()
      const controller = new AbortController()
      streamRef.current = controller
      dispatch({ type: 'asked', id, question })
      setIntended(null)
      setAnnouncement('Thinking about your question.')

      const segments: Segment[] = []
      let finished = false
      try {
        for await (const raw of streamConverseMessage(current.session_id, question, controller.signal)) {
          const event = parseTalkEvent(raw)
          if (event.type === 'segment') {
            const { segment } = event
            if (segments.some((s) => s.index === segment.index)) continue
            segments.push({ index: segment.index, emotion: segment.emotion, text: segment.text })
            dispatch({ type: 'segment', id, segment })
            if (!video) queue.enqueue({ turnKey: id, index: segment.index, text: segment.text, emotion: segment.emotion })
          } else if (event.type === 'done') {
            finished = true
            dispatch({ type: 'done', id, done: event.done })
            if (video) speakWithReplica(id, segments, event.done.echo_text)
          } else if (event.type === 'error') {
            // May follow some segments with no `done`: that answer can't be checked.
            finished = true
            dispatch({ type: 'failed', id, message: event.message })
            if (video) dispatch({ type: 'allSpoken', id })
            break
          }
        }
        if (!finished) {
          dispatch({ type: 'failed', id, message: segments.length > 0 ? CUT_OFF : 'No answer came back. Please try asking again.' })
          if (video) dispatch({ type: 'allSpoken', id })
        }
      } catch (error) {
        if (isAbort(error) || controller.signal.aborted) return // the learner pressed Stop
        // Includes 429 (hourly limit): the server's own message is shown.
        dispatch({ type: 'failed', id, message: errorMessage(error) })
        if (video) dispatch({ type: 'allSpoken', id })
      } finally {
        if (streamRef.current === controller) streamRef.current = null
      }
    },
    [queue, speakWithReplica],
  )

  const stop = useCallback(() => {
    streamRef.current?.abort()
    streamRef.current = null
    queue.clear()
    stopSpeaking()
    avatar.silence()
    replicaRef.current = null
    window.clearTimeout(timers.current.replicaStart)
    window.clearTimeout(timers.current.replicaStop)
    const current = sessionRef.current
    if (current?.provider === 'tavus' && current.tavus && callRef.current) {
      try {
        sendInterrupt(callRef.current, current.tavus.conversation_id)
      } catch {
        // The call is gone; nothing is playing anyway.
      }
    }
    const active = activeRef.current
    if (active) dispatch({ type: 'stopped', id: active.id })
    relaxSoon(800)
  }, [queue, avatar, relaxSoon])

  const reveal = useCallback((turnId: string) => dispatch({ type: 'revealed', id: turnId }), [])

  const setCall = useCallback((call: VideoCall | null) => {
    callRef.current = call
  }, [])

  return {
    turns,
    activeTurn,
    busy: activeTurn !== null,
    intended,
    voiceBlocked,
    announcement,
    ask: (question: string) => void ask(question),
    stop,
    reveal,
    setCall,
    onReplicaSpeaking,
  }
}
