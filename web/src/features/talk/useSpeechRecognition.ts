import { useCallback, useEffect, useRef, useState } from 'react'
import {
  joinTranscript,
  recognitionConstructor,
  recognitionErrorMessage,
  recognitionLanguage,
  type RecognitionInstance,
} from './recognition'

export type SpeechRecognitionControls = {
  supported: boolean
  listening: boolean
  /** A learner-facing problem from the last attempt, if any. */
  problem: string | null
  start: () => void
  stop: () => void
}

/**
 * Voice typing with the Web Speech API where the browser has it. Calls
 * `onTranscript` with the words heard so far (`final` once settled).
 */
export function useSpeechRecognition(onTranscript: (text: string, final: boolean) => void): SpeechRecognitionControls {
  const [supported] = useState(() => recognitionConstructor() !== null)
  const [listening, setListening] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)
  const recognition = useRef<RecognitionInstance | null>(null)
  const callback = useRef(onTranscript)

  useEffect(() => {
    callback.current = onTranscript
  }, [onTranscript])

  const stop = useCallback(() => recognition.current?.stop(), [])

  const start = useCallback(() => {
    const Recognition = recognitionConstructor()
    if (!Recognition || recognition.current) return
    let instance: RecognitionInstance
    try {
      instance = new Recognition()
    } catch {
      setProblem(recognitionErrorMessage('unknown'))
      return
    }
    instance.lang = recognitionLanguage(typeof navigator === 'undefined' ? undefined : navigator.language)
    instance.continuous = false
    instance.interimResults = true
    instance.maxAlternatives = 1
    instance.onresult = (event) => {
      const { text, final } = joinTranscript(event.results)
      if (text) callback.current(text, final)
    }
    instance.onerror = (event) => setProblem(recognitionErrorMessage(event.error))
    instance.onend = () => {
      if (recognition.current === instance) recognition.current = null
      setListening(false)
    }
    recognition.current = instance
    setProblem(null)
    try {
      instance.start()
      setListening(true)
    } catch {
      recognition.current = null
      setProblem(recognitionErrorMessage('unknown'))
    }
  }, [])

  useEffect(
    () => () => {
      recognition.current?.abort()
      recognition.current = null
    },
    [],
  )

  return { supported, listening, problem, start, stop }
}
