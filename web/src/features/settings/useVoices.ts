import { useEffect, useState } from 'react'
import { speechSupported } from '../../lib/speech'

export type VoicesState = {
  supported: boolean
  voices: SpeechSynthesisVoice[]
  /** Still waiting for the browser to list its voices. */
  loading: boolean
}

/**
 * The browser's SpeechSynthesis voices. Chrome lists them asynchronously (the
 * `voiceschanged` event) and some browsers never fire that event, so this
 * also checks a few times while the page opens.
 */
export function useVoices(): VoicesState {
  const [supported] = useState(speechSupported)
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>(() =>
    supported ? window.speechSynthesis.getVoices() : [],
  )
  const [gaveUp, setGaveUp] = useState(false)

  useEffect(() => {
    if (!supported) return
    const synth = window.speechSynthesis
    const update = () => {
      const list = synth.getVoices()
      if (list.length > 0) setVoices(list)
    }
    if (typeof synth.addEventListener === 'function') synth.addEventListener('voiceschanged', update)
    else synth.onvoiceschanged = update
    const poll = window.setInterval(update, 300)
    const stop = window.setTimeout(() => {
      window.clearInterval(poll)
      setGaveUp(true)
    }, 3000)
    return () => {
      if (typeof synth.removeEventListener === 'function') synth.removeEventListener('voiceschanged', update)
      else synth.onvoiceschanged = null
      window.clearInterval(poll)
      window.clearTimeout(stop)
    }
  }, [supported])

  return { supported, voices, loading: supported && voices.length === 0 && !gaveUp }
}
