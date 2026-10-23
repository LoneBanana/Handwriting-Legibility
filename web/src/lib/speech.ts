// Thin wrapper over the Web Speech API (SpeechSynthesis). Every function is a
// no-op where the API is missing, so pages degrade to text-only.

/** Typical speaking rate of browser voices at rate 1.0, in words per minute. */
export const BASE_WORDS_PER_MINUTE = 170

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

/** Maps a target speaking speed to a SpeechSynthesisUtterance rate. */
export function wpmToRate(wordsPerMinute: number): number {
  if (!Number.isFinite(wordsPerMinute) || wordsPerMinute <= 0) return 1
  return clamp(wordsPerMinute / BASE_WORDS_PER_MINUTE, 0.5, 2)
}

/** Rough duration of speaking `text` at `wordsPerMinute`, used to pace delayed captions. */
export function estimateSpeechMs(text: string, wordsPerMinute: number): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length
  const wpm = wordsPerMinute > 0 ? wordsPerMinute : BASE_WORDS_PER_MINUTE
  return Math.round((words / wpm) * 60_000)
}

// Chrome can garbage-collect an utterance nothing references and then never fire its
// end event, which would leave callers waiting forever; keep each alive until it ends.
const liveUtterances = new Set<SpeechSynthesisUtterance>()

export function speechSupported(): boolean {
  return typeof window !== 'undefined' && 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window
}

export function findVoice(name: string | null | undefined): SpeechSynthesisVoice | null {
  if (!name || !speechSupported()) return null
  return window.speechSynthesis.getVoices().find((voice) => voice.name === name) ?? null
}

export type SpeakOptions = {
  rate?: number
  voiceName?: string | null
  lang?: string
  onStart?: () => void
  onBoundary?: (charIndex: number) => void
  signal?: AbortSignal
}

/** Speaks `text` and resolves when speech ends (or immediately if unsupported). */
export function speak(text: string, options: SpeakOptions = {}): Promise<void> {
  if (!speechSupported() || !text.trim()) return Promise.resolve()
  return new Promise((resolve) => {
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = options.rate ?? 1
    utterance.lang = options.lang ?? 'en-US'
    const voice = findVoice(options.voiceName)
    if (voice) utterance.voice = voice
    let settled = false
    const finish = () => {
      if (settled) return
      settled = true
      liveUtterances.delete(utterance)
      options.signal?.removeEventListener('abort', onAbort)
      resolve()
    }
    const onAbort = () => {
      window.speechSynthesis.cancel()
      finish()
    }
    utterance.onstart = () => options.onStart?.()
    utterance.onboundary = (event) => options.onBoundary?.(event.charIndex)
    utterance.onend = finish
    utterance.onerror = finish
    if (options.signal?.aborted) {
      finish()
      return
    }
    options.signal?.addEventListener('abort', onAbort)
    liveUtterances.add(utterance)
    window.speechSynthesis.speak(utterance)
  })
}

export function stopSpeaking(): void {
  if (speechSupported()) window.speechSynthesis.cancel()
}
