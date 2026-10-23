import { readStorage, writeStorage } from '../lib/storage'

const TOKEN_KEY = 'legible.access_token'
const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')

export const tokenStore = {
  get: (): string | null => readStorage(TOKEN_KEY),
  set: (token: string): void => writeStorage(TOKEN_KEY, token),
  clear: (): void => writeStorage(TOKEN_KEY, null),
}

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

let unauthorizedHandler: (() => void) | null = null

/** The learner's IANA time zone, so "one level per day" follows their own calendar. */
function localTimeZone(): string | null {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || null
  } catch {
    return null
  }
}

/** Called once when any authenticated request comes back 401. */
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler
}

/**
 * Turns a FastAPI error body into one readable sentence. FastAPI sends
 * `{"detail": "..."}` for HTTPException and `{"detail": [{loc, msg, type}]}` for
 * validation errors.
 */
export function describeErrorBody(body: unknown, status: number): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail
    if (typeof detail === 'string' && detail.trim()) return detail
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => {
          if (!item || typeof item !== 'object') return null
          const { msg, loc } = item as { msg?: unknown; loc?: unknown }
          if (typeof msg !== 'string') return null
          const field = Array.isArray(loc) ? loc.filter((p) => p !== 'body').join('.') : ''
          return field ? `${field}: ${msg}` : msg
        })
        .filter((m): m is string => Boolean(m))
      if (messages.length) return messages.join('; ')
    }
  }
  if (status === 0) return 'Could not reach the server. Check your connection and try again.'
  if (status >= 500) return 'The server had a problem. Please try again in a moment.'
  return `Request failed (${status}).`
}

type RequestOptions = Omit<RequestInit, 'body'> & {
  /** JSON-serialisable body; sets Content-Type automatically. */
  json?: unknown
  /** Raw body (e.g. FormData for uploads). */
  body?: BodyInit
  /** Send the bearer token (default true). */
  auth?: boolean
}

export async function apiFetch(path: string, options: RequestOptions = {}): Promise<Response> {
  const { json, body, auth = true, headers, ...rest } = options
  const finalHeaders = new Headers(headers)
  if (!finalHeaders.has('Accept')) finalHeaders.set('Accept', 'application/json')
  let finalBody: BodyInit | undefined = body
  if (json !== undefined) {
    finalHeaders.set('Content-Type', 'application/json')
    finalBody = JSON.stringify(json)
  }
  const token = auth ? tokenStore.get() : null
  if (token) finalHeaders.set('Authorization', `Bearer ${token}`)
  const timeZone = localTimeZone()
  if (timeZone && !finalHeaders.has('X-Timezone')) finalHeaders.set('X-Timezone', timeZone)

  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, { ...rest, headers: finalHeaders, body: finalBody })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError(0, describeErrorBody(null, 0))
  }

  if (!response.ok) {
    if (response.status === 401 && auth) {
      tokenStore.clear()
      unauthorizedHandler?.()
    }
    let parsed: unknown = null
    try {
      parsed = await response.json()
    } catch {
      parsed = null
    }
    throw new ApiError(response.status, describeErrorBody(parsed, response.status))
  }
  return response
}

export async function apiJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await apiFetch(path, options)
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}
