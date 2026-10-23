// Feature detection and helpers for the Web Speech API's SpeechRecognition
// (the "ask with your voice" mic button). TypeScript's DOM types include the
// result events but not the constructor, so a minimal shape is declared here.

export type RecognitionInstance = {
  lang: string
  continuous: boolean
  interimResults: boolean
  maxAlternatives: number
  onresult: ((event: SpeechRecognitionEvent) => void) | null
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
  abort: () => void
}

type RecognitionConstructor = new () => RecognitionInstance

export function recognitionConstructor(): RecognitionConstructor | null {
  if (typeof window === 'undefined') return null
  const w = window as unknown as { SpeechRecognition?: RecognitionConstructor; webkitSpeechRecognition?: RecognitionConstructor }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null
}

/** The avatar answers in English, so listen for English in the learner's own variety. */
export function recognitionLanguage(preferred: string | undefined): string {
  return preferred && /^en(-|$)/i.test(preferred) ? preferred : 'en-US'
}

type ResultLike = { isFinal: boolean; length: number; [index: number]: { transcript: string } }

/** Joins recognition results into one transcript; `final` once every part is settled. */
export function joinTranscript(results: ArrayLike<ResultLike>): { text: string; final: boolean } {
  let text = ''
  let final = results.length > 0
  for (let i = 0; i < results.length; i += 1) {
    const result = results[i]
    if (result.length > 0) text += result[0].transcript
    if (!result.isFinal) final = false
  }
  return { text: text.replace(/\s+/g, ' ').trim(), final }
}

/** A plain-language message for a SpeechRecognition error code, or null to stay quiet. */
export function recognitionErrorMessage(code: string): string | null {
  switch (code) {
    case 'aborted':
      return null
    case 'no-speech':
      return 'I didn’t hear anything. Try again, a little closer to the microphone.'
    case 'audio-capture':
      return 'No microphone was found. You can type your question instead.'
    case 'not-allowed':
    case 'service-not-allowed':
      return 'The microphone is blocked for this site. You can allow it in your browser, or type instead.'
    case 'network':
      return 'Voice typing needs an internet connection. You can type your question instead.'
    case 'language-not-supported':
      return 'Voice typing isn’t available for this language here. You can type instead.'
    default:
      return 'Voice typing stopped working. You can type your question instead.'
  }
}
