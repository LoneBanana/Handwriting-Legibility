import { useCallback, useId, useRef, useState, type FormEvent } from 'react'
import { Icon } from '../../components/Icon'
import { useSpeechRecognition } from './useSpeechRecognition'

const STARTER_QUESTIONS = [
  'Why is the sky blue?',
  'How do bees make honey?',
  'Why do cats purr?',
  'What makes a rainbow?',
  'How do volcanoes work?',
] as const

type Props = {
  /** No session yet: nothing can be asked. */
  disabled: boolean
  /** An answer is in progress. */
  busy: boolean
  showStarters: boolean
  onAsk: (question: string) => void
}

/** The question box: type and press Enter, or speak into the microphone. */
export function AskBox({ disabled, busy, showStarters, onAsk }: Props) {
  const inputId = useId()
  const hintId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [text, setText] = useState('')
  const [heard, setHeard] = useState<string | null>(null)

  const onTranscript = useCallback((transcript: string, final: boolean) => {
    setText(transcript)
    if (final) {
      setHeard(transcript)
      inputRef.current?.focus()
    }
  }, [])
  const mic = useSpeechRecognition(onTranscript)
  const blocked = disabled || busy

  const send = (question: string) => {
    const trimmed = question.trim()
    if (!trimmed || blocked) return
    mic.stop()
    onAsk(trimmed)
    setText('')
    setHeard(null)
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    send(text)
  }

  const hint = busy
    ? 'The avatar is answering. Press Stop if you want to ask something else.'
    : `Press Enter to ask.${mic.supported ? ' You can also press Speak and say your question.' : ''}`

  return (
    <form className="ask stack-sm" onSubmit={submit}>
      <label htmlFor={inputId} className="label">
        Ask anything
      </label>
      <div className="ask__row">
        <input
          id={inputId}
          ref={inputRef}
          type="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="For example: why do cats purr?"
          autoComplete="off"
          enterKeyHint="send"
          maxLength={500}
          aria-describedby={hintId}
          disabled={disabled}
        />
        {mic.supported && (
          <button
            type="button"
            className="btn btn--secondary"
            onClick={mic.listening ? mic.stop : mic.start}
            disabled={blocked && !mic.listening}
          >
            <Icon name={mic.listening ? 'stop' : 'mic'} />
            {mic.listening ? 'Stop listening' : 'Speak'}
          </button>
        )}
        <button type="submit" className="btn" disabled={blocked || !text.trim()}>
          <Icon name="send" /> Ask
        </button>
      </div>
      <p id={hintId} className="muted small" style={{ margin: 0 }}>
        {hint}
      </p>
      <p className="small" role="status" style={{ margin: 0 }}>
        {mic.listening
          ? 'Listening… say your question.'
          : (mic.problem ?? (heard ? `I heard: “${heard}”. Press Enter to ask, or change it first.` : ''))}
      </p>
      {showStarters && (
        <div className="stack-sm">
          <span className="small muted" id={`${inputId}-ideas`}>
            Need an idea? Try one of these:
          </span>
          <div className="starters" role="group" aria-labelledby={`${inputId}-ideas`}>
            {STARTER_QUESTIONS.map((question) => (
              <button key={question} type="button" className="starter" onClick={() => send(question)} disabled={blocked}>
                {question}
              </button>
            ))}
          </div>
        </div>
      )}
    </form>
  )
}
