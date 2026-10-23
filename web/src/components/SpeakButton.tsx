import { useEffect, useRef, useState } from 'react'
import { useAuth } from '../auth/context'
import { speak, speechSupported, stopSpeaking } from '../lib/speech'
import { Icon } from './Icon'

type Props = {
  text: string
  label?: string
  rate?: number
  className?: string
}

/** Toggle button that reads `text` aloud with the learner's chosen voice. */
export function SpeakButton({ text, label = 'Read aloud', rate = 1, className = 'btn btn--secondary' }: Props) {
  const { settings } = useAuth()
  const [speaking, setSpeaking] = useState(false)
  const controller = useRef<AbortController | null>(null)

  useEffect(() => () => controller.current?.abort(), [])

  if (!speechSupported()) return null

  const toggle = async () => {
    if (speaking) {
      controller.current?.abort()
      stopSpeaking()
      setSpeaking(false)
      return
    }
    stopSpeaking()
    controller.current = new AbortController()
    setSpeaking(true)
    await speak(text, { rate, voiceName: settings.tts_voice, signal: controller.current.signal })
    setSpeaking(false)
  }

  return (
    <button type="button" className={className} onClick={toggle} aria-pressed={speaking} disabled={!text.trim()}>
      <Icon name={speaking ? 'stop' : 'speaker'} />
      {speaking ? 'Stop' : label}
    </button>
  )
}
