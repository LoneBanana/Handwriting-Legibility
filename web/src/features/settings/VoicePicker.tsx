import { useEffect, useId, useRef, useState } from 'react'
import { Icon } from '../../components/Icon'
import { speak, stopSpeaking } from '../../lib/speech'
import { useVoices } from './useVoices'
import { groupVoices, toVoiceInfo, voiceLabel } from './voices'

const SAMPLE = 'Hi! This is how I will sound when I read to you.'

/** Read-aloud voice picker with a "Test voice" button. `value` is a voice name, or null for the device default. */
export function VoicePicker({ value, onChange }: { value: string | null; onChange: (name: string | null) => void }) {
  const selectId = useId()
  const hintId = useId()
  const { supported, voices, loading } = useVoices()
  const [testing, setTesting] = useState(false)
  const controller = useRef<AbortController | null>(null)

  useEffect(
    () => () => {
      controller.current?.abort()
    },
    [],
  )

  if (!supported) {
    return (
      <p className="notice small" style={{ margin: 0 }}>
        This browser can’t read text aloud, so there are no voices to choose from. Read-aloud buttons will be hidden, and
        the talking avatar will show its words as text.
      </p>
    )
  }

  const groups = groupVoices(voices.map(toVoiceInfo), typeof navigator === 'undefined' ? 'en-US' : navigator.language)
  const missing = value !== null && voices.length > 0 && !voices.some((voice) => voice.name === value)

  const test = async () => {
    if (testing) {
      controller.current?.abort()
      stopSpeaking()
      setTesting(false)
      return
    }
    stopSpeaking()
    const current = new AbortController()
    controller.current = current
    setTesting(true)
    await speak(SAMPLE, { voiceName: value, signal: current.signal })
    if (controller.current === current) setTesting(false)
  }

  return (
    <div className="field">
      <label htmlFor={selectId}>Voice</label>
      <div className="row">
        <select
          id={selectId}
          className="voice-select"
          value={value ?? ''}
          onChange={(event) => onChange(event.target.value || null)}
          aria-describedby={hintId}
        >
          <option value="">Device default</option>
          {value !== null && (missing || voices.length === 0) && (
            <option value={value}>{missing ? `${value} (not on this device)` : value}</option>
          )}
          {groups.map((group) => (
            <optgroup key={group.language} label={group.label}>
              {group.voices.map((voice) => (
                <option key={`${voice.name}|${voice.lang}`} value={voice.name}>
                  {voiceLabel(voice)}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        <button type="button" className="btn btn--secondary" onClick={test}>
          <Icon name={testing ? 'stop' : 'speaker'} />
          {testing ? 'Stop' : 'Test voice'}
        </button>
      </div>
      <span id={hintId} className="hint">
        {loading
          ? 'Looking for voices on this device…'
          : missing
            ? 'Your saved voice isn’t on this device, so the device default is used here.'
            : 'Used by read-aloud buttons and the talking avatar. Voices differ from device to device.'}
      </span>
    </div>
  )
}
