import { useEffect, useState } from 'react'
import { apiFetch } from '../../api/client'
import { createConverseSession, endConverseSession } from '../../api/endpoints'
import type { AvatarProvider, ConverseSession } from '../../api/types'
import { readStorage, writeStorage } from '../../lib/storage'

const PROVIDER_KEY = 'legible.talk.avatar'

/** The avatar this learner picked last time on this device. */
export function storedProvider(): AvatarProvider {
  return readStorage(PROVIDER_KEY) === 'tavus' ? 'tavus' : 'local'
}

export function storeProvider(provider: AvatarProvider): void {
  writeStorage(PROVIDER_KEY, provider)
}

/**
 * Ends a session while the tab is closing. A normal request may be cancelled
 * when the page unloads, so this sends the same call with `keepalive`
 * (`endConverseSession` has no option for it). Matters most for video avatar
 * sessions, which hold a paid Tavus conversation open.
 */
export function endSessionOnUnload(sessionId: string): void {
  void apiFetch(`/api/converse/sessions/${encodeURIComponent(sessionId)}/end`, { method: 'POST', keepalive: true }).catch(
    () => undefined,
  )
}

/** A new object starts a new session (and ends the previous one). */
export type SessionRequest = { provider: AvatarProvider; attempt: number }

export type SessionState = { session: ConverseSession | null; error: unknown; loading: boolean }

/**
 * Creates a conversation session for `request` and ends it when the request
 * changes, the page is left, or the tab closes. A tab restored from the
 * back/forward cache gets a fresh session.
 */
export function useConverseSession(request: SessionRequest): SessionState {
  const [revival, setRevival] = useState(0)
  const [settled, setSettled] = useState<{
    request: SessionRequest
    revival: number
    session: ConverseSession | null
    error: unknown
  } | null>(null)

  useEffect(() => {
    let cancelled = false
    let current: ConverseSession | null = null
    let ended = false
    const end = (unloading: boolean) => {
      if (!current || ended) return
      ended = true
      if (unloading) endSessionOnUnload(current.session_id)
      else void endConverseSession(current.session_id).catch(() => undefined)
    }

    // Deferred by a tick so React's development double-mount doesn't create
    // (and, for the video avatar, pay for) two sessions.
    const timer = window.setTimeout(() => {
      createConverseSession(request.provider).then(
        (session) => {
          if (cancelled) {
            void endConverseSession(session.session_id).catch(() => undefined)
            return
          }
          current = session
          setSettled({ request, revival, session, error: null })
        },
        (error: unknown) => {
          if (!cancelled) setSettled({ request, revival, session: null, error })
        },
      )
    }, 0)

    const onPageHide = () => end(true)
    const onPageShow = (event: PageTransitionEvent) => {
      if (event.persisted && ended) setRevival((n) => n + 1)
    }
    window.addEventListener('pagehide', onPageHide)
    window.addEventListener('pageshow', onPageShow)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
      window.removeEventListener('pagehide', onPageHide)
      window.removeEventListener('pageshow', onPageShow)
      end(false)
    }
  }, [request, revival])

  // Results that belong to an older request are ignored, so a new request reads as loading.
  const current = settled && settled.request === request && settled.revival === revival ? settled : null
  return { session: current?.session ?? null, error: current?.error ?? null, loading: current === null }
}
