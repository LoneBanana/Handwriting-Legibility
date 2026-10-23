import { apiJson } from './client'
import { postEventStream, type SseEvent } from './sse'
import type {
  AnswerResult,
  AuthResponse,
  AvatarProvider,
  CheckResult,
  ComprehensionCheck,
  ConverseSession,
  DecoderChoice,
  HandwritingAnswerResult,
  HandwritingStatus,
  LiteracyCompletion,
  LiteracySession,
  NumeracyRound,
  ProblemAnswer,
  ProgressSummary,
  RoundResult,
  TranscribeMode,
  TranscribeResponse,
  User,
  UserSettings,
} from './types'

const enc = encodeURIComponent

// ----------------------------------------------------------------------- auth

export const health = () => apiJson<{ status: string; version: string }>('/api/health', { auth: false })

export const register = (body: { email: string; password: string; display_name: string }) =>
  apiJson<AuthResponse>('/api/auth/register', { method: 'POST', json: body, auth: false })

export const login = (body: { email: string; password: string }) =>
  apiJson<AuthResponse>('/api/auth/login', { method: 'POST', json: body, auth: false })

export const getMe = () => apiJson<User>('/api/auth/me')

export const updateSettings = (patch: Partial<UserSettings>) =>
  apiJson<User>('/api/auth/me/settings', { method: 'PATCH', json: patch })

export const deleteAccount = () => apiJson<void>('/api/auth/me', { method: 'DELETE' })

// ---------------------------------------------------------------- handwriting

export type TranscribeOptions = {
  mode: TranscribeMode
  clarify: boolean
  decoder: DecoderChoice
  /** "This is my own handwriting": feeds the handwriting clarity track. */
  trackProgress: boolean
}

export function transcribe(file: Blob, options: TranscribeOptions, signal?: AbortSignal) {
  const form = new FormData()
  form.append('file', file, file instanceof File ? file.name : 'handwriting.png')
  form.append('mode', options.mode)
  form.append('clarify', String(options.clarify))
  form.append('decoder', options.decoder)
  form.append('track_progress', String(options.trackProgress))
  return apiJson<TranscribeResponse>('/api/handwriting/transcribe', { method: 'POST', body: form, signal })
}

export const submitCorrection = (sampleId: string, text: string) =>
  apiJson<{ ok: true }>(`/api/handwriting/samples/${enc(sampleId)}/correction`, {
    method: 'POST',
    json: { text },
  })

export const handwritingStatus = () => apiJson<HandwritingStatus>('/api/handwriting/status')

export const handwritingSampleCount = () =>
  apiJson<{ samples: number; corrected: number }>('/api/handwriting/samples/count')

/** Deletes every stored handwriting photo, correction, and the personal adapter. */
export const deleteHandwritingSamples = () => apiJson<void>('/api/handwriting/samples', { method: 'DELETE' })

// ------------------------------------------------------------------- literacy

export const getLiteracySession = () => apiJson<LiteracySession>('/api/literacy/session')

export const answerCard = (sessionId: string, body: { card_id: string; response: string; latency_ms: number }) =>
  apiJson<AnswerResult>(`/api/literacy/session/${enc(sessionId)}/answer`, { method: 'POST', json: body })

export function answerCardHandwriting(sessionId: string, cardId: string, file: Blob, latencyMs: number) {
  const form = new FormData()
  form.append('file', file, file instanceof File ? file.name : 'word.png')
  form.append('card_id', cardId)
  form.append('latency_ms', String(Math.round(latencyMs)))
  return apiJson<HandwritingAnswerResult>(`/api/literacy/session/${enc(sessionId)}/answer-handwriting`, {
    method: 'POST',
    body: form,
  })
}

export const completeLiteracySession = (sessionId: string) =>
  apiJson<LiteracyCompletion>(`/api/literacy/session/${enc(sessionId)}/complete`, { method: 'POST' })

// ------------------------------------------------------------------- numeracy

export const getNumeracyRound = () => apiJson<NumeracyRound>('/api/numeracy/round')

export const submitNumeracyRound = (roundId: string, answers: ProblemAnswer[]) =>
  apiJson<RoundResult>(`/api/numeracy/round/${enc(roundId)}/submit`, { method: 'POST', json: { answers } })

// ------------------------------------------------------------------- converse

export const createConverseSession = (provider?: AvatarProvider) =>
  apiJson<ConverseSession>('/api/converse/sessions', {
    method: 'POST',
    json: provider ? { provider } : {},
  })

export const streamConverseMessage = (sessionId: string, text: string, signal?: AbortSignal): AsyncGenerator<SseEvent> =>
  postEventStream(`/api/converse/sessions/${enc(sessionId)}/messages`, { text }, signal)

export const createCheck = (sessionId: string, turnId: string) =>
  apiJson<ComprehensionCheck>(`/api/converse/sessions/${enc(sessionId)}/turns/${enc(turnId)}/check`, {
    method: 'POST',
  })

export const submitCheck = (checkId: string, answers: { question_id: string; choice: string }[]) =>
  apiJson<CheckResult>(`/api/converse/checks/${enc(checkId)}/submit`, { method: 'POST', json: { answers } })

export const endConverseSession = (sessionId: string) =>
  apiJson<{ ok: true }>(`/api/converse/sessions/${enc(sessionId)}/end`, { method: 'POST' })

// ------------------------------------------------------------------- progress

export const getProgress = () => apiJson<ProgressSummary>('/api/progress')
