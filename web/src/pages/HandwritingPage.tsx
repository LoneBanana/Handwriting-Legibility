import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { Link } from 'react-router'
import { handwritingStatus, submitCorrection, transcribe } from '../api/endpoints'
import type { DecoderChoice, TranscribeMode, TranscribeResponse, TranscribedLine } from '../api/types'
import { ErrorNotice } from '../components/ErrorNotice'
import { Icon } from '../components/Icon'
import { ImagePicker } from '../components/ImagePicker'
import { ProgressionNotice } from '../components/ProgressionNotice'
import { ProgressMeter } from '../components/ProgressMeter'
import { SpeakButton } from '../components/SpeakButton'
import { confidenceRuns, countUncertain, legibilityLabel, LOW_CONFIDENCE } from '../features/handwriting/confidence'
import '../features/handwriting/handwriting.css'
import { errorMessage } from '../lib/errors'
import { usePageTitle } from '../lib/usePageTitle'
import { useResource } from '../lib/useResource'

const MODES: { value: TranscribeMode; label: string }[] = [
  { value: 'auto', label: 'Detect for me' },
  { value: 'page', label: 'A page or paragraph' },
  { value: 'line', label: 'A single line' },
]

const DECODERS: { value: DecoderChoice; label: string }[] = [
  { value: 'accurate', label: 'Careful (slower)' },
  { value: 'fast', label: 'Quick' },
]

function LineText({ line }: { line: TranscribedLine }) {
  if (line.chars.length === 0) return <>{line.text}</>
  const uncertain = countUncertain(line.chars)
  return (
    <>
      {confidenceRuns(line.chars).map((run, index) =>
        run.low ? (
          <mark key={index} className="char-low" title={`Not sure (${Math.round(run.confidence * 100)}%)`}>
            {run.text}
          </mark>
        ) : (
          <span key={index}>{run.text}</span>
        ),
      )}
      {uncertain > 0 && (
        <span className="visually-hidden">
          {` (${uncertain} uncertain ${uncertain === 1 ? 'letter' : 'letters'})`}
        </span>
      )}
    </>
  )
}

function Preview({ url, result, activeLine }: { url: string; result: TranscribeResponse | null; activeLine: number | null }) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null)
  return (
    <div className="preview">
      <img
        src={url}
        alt="Your handwriting photo"
        onLoad={(event) => setSize({ w: event.currentTarget.naturalWidth, h: event.currentTarget.naturalHeight })}
      />
      {result && size && (
        <svg viewBox={`0 0 ${size.w} ${size.h}`} preserveAspectRatio="none" aria-hidden="true">
          {result.lines.map((line) => {
            const [x0, y0, x1, y1] = line.bbox
            return (
              <rect
                key={line.index}
                x={x0}
                y={y0}
                width={Math.max(0, x1 - x0)}
                height={Math.max(0, y1 - y0)}
                rx={4}
                data-active={activeLine === line.index}
              />
            )
          })}
        </svg>
      )}
    </div>
  )
}

function CorrectionBox({ result }: { result: TranscribeResponse }) {
  const id = useId()
  const [text, setText] = useState(result.text)
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [problem, setProblem] = useState<string | null>(null)

  if (!result.sample_id) {
    return (
      <p className="muted small">
        Want Legible to learn your handwriting? Turn on <Link to="/settings">“Keep my handwriting samples”</Link> in
        Settings, then you can correct mistakes here.
      </p>
    )
  }

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setStatus('saving')
    setProblem(null)
    try {
      await submitCorrection(result.sample_id!, text)
      setStatus('saved')
    } catch (error) {
      setStatus('error')
      setProblem(errorMessage(error))
    }
  }

  return (
    <form className="stack-sm" onSubmit={onSubmit}>
      <label htmlFor={id} className="label">
        Fix any mistakes (this teaches Legible your handwriting)
      </label>
      <span id={`${id}-hint`} className="muted small">
        Keep one line of text for each line of handwriting, so each line can be matched with your writing.
      </span>
      <textarea
        id={id}
        aria-describedby={`${id}-hint`}
        rows={Math.max(3, result.lines.length + 1)}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="row">
        <button className="btn btn--secondary" type="submit" disabled={status === 'saving' || !text.trim()}>
          {status === 'saving' ? 'Saving…' : 'Save correction'}
        </button>
        <span role="status" className="small">
          {status === 'saved' && 'Thanks! Saved — this helps the model learn your writing.'}
          {status === 'error' && problem}
        </span>
      </div>
    </form>
  )
}

export default function HandwritingPage() {
  usePageTitle('Handwriting')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [mode, setMode] = useState<TranscribeMode>('auto')
  const [decoder, setDecoder] = useState<DecoderChoice>('accurate')
  const [clarify, setClarify] = useState(true)
  const [ownWriting, setOwnWriting] = useState(true)
  const [result, setResult] = useState<TranscribeResponse | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)
  const [activeLine, setActiveLine] = useState<number | null>(null)
  const [copied, setCopied] = useState(false)
  const resultRef = useRef<HTMLDivElement>(null)
  const controller = useRef<AbortController | null>(null)
  // Also warms the model up on the server while the learner picks a photo.
  const status = useResource(handwritingStatus)
  const unavailable = status.data !== null && !status.data.model_loaded

  useEffect(() => () => controller.current?.abort(), [])

  useEffect(() => {
    if (!file) return
    // An object URL is an external resource: create it here so the cleanup can revoke it.
    const url = URL.createObjectURL(file)
    // oxlint-disable-next-line react/set-state-in-effect
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const chooseFile = (next: File) => {
    setFile(next)
    setResult(null)
    setError(null)
    setActiveLine(null)
  }

  const run = async () => {
    if (!file) return
    controller.current?.abort()
    controller.current = new AbortController()
    setBusy(true)
    setError(null)
    try {
      const response = await transcribe(
        file,
        { mode, decoder, clarify, trackProgress: ownWriting },
        controller.current.signal,
      )
      setResult(response)
      setCopied(false)
      requestAnimationFrame(() => resultRef.current?.focus())
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) setError(err)
    } finally {
      setBusy(false)
    }
  }

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
    } catch {
      setCopied(false)
    }
  }

  const bestText = result?.clarified?.text ?? result?.text ?? ''

  return (
    <>
      <div className="page-header">
        <h1>Handwriting to text</h1>
        <p>Take or upload a photo of handwriting. Legible reads it and shows where it wasn’t sure.</p>
        {status.data?.personalized && <span className="badge badge--success">Tuned to your handwriting</span>}
      </div>

      {unavailable && (
        <div className="notice notice--warning" role="status" style={{ marginBottom: 'var(--space-5)' }}>
          <div className="notice__title">Reading handwriting isn’t set up yet</div>
          <p>This server doesn’t have a handwriting model installed, so photos can’t be read right now.</p>
        </div>
      )}

      <div className="split">
        <div className="stack">
          <ImagePicker onFile={chooseFile} acceptPaste prompt="Drop a photo of handwriting here, or choose one" />
          {previewUrl && <Preview url={previewUrl} result={result} activeLine={activeLine} />}
        </div>

        <form
          className="card stack"
          onSubmit={(event) => {
            event.preventDefault()
            void run()
          }}
        >
          <h2>Options</h2>
          <fieldset className="field">
            <legend>What is in the photo?</legend>
            <div className="option-list">
              {MODES.map((option) => (
                <label key={option.value} className="option">
                  <input
                    type="radio"
                    name="mode"
                    value={option.value}
                    checked={mode === option.value}
                    onChange={() => setMode(option.value)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset className="field">
            <legend>Reading style</legend>
            <div className="option-list">
              {DECODERS.map((option) => (
                <label key={option.value} className="option">
                  <input
                    type="radio"
                    name="decoder"
                    value={option.value}
                    checked={decoder === option.value}
                    onChange={() => setDecoder(option.value)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </fieldset>
          <label className="checkbox">
            <input type="checkbox" checked={clarify} onChange={(e) => setClarify(e.target.checked)} />
            <span>
              <strong>Also suggest what was meant</strong>
              <br />
              <span className="muted small">Fixes spelling and grammar without adding new ideas.</span>
            </span>
          </label>
          <label className="checkbox">
            <input type="checkbox" checked={ownWriting} onChange={(e) => setOwnWriting(e.target.checked)} />
            <span>
              <strong>This is my own handwriting</strong>
              <br />
              <span className="muted small">Counts toward your handwriting clarity progress.</span>
            </span>
          </label>
          <button className="btn btn--large" type="submit" disabled={!file || busy || unavailable} aria-busy={busy}>
            {busy ? (
              <>
                <span className="spinner" aria-hidden="true" /> Reading your handwriting…
              </>
            ) : (
              <>
                <Icon name="sparkle" /> Transcribe
              </>
            )}
          </button>
          {!file && <p className="muted small">Choose a photo first.</p>}
          <ErrorNotice error={error} />
        </form>
      </div>

      {result && (
        <div
          className="stack"
          style={{ marginTop: 'var(--space-6)' }}
          ref={resultRef}
          tabIndex={-1}
          aria-label="Transcription result"
        >
          <section className="card stack" aria-labelledby="literal-title">
            <div className="card__title">
              <h2 id="literal-title">What it says</h2>
              <span className="badge">{(result.elapsed_ms / 1000).toFixed(1)} s</span>
            </div>
            <ProgressMeter label="Legibility" value={result.legibility} detail={legibilityLabel(result.legibility)} />
            {result.lines.length > 0 ? (
              <ol className="transcript" aria-label="Transcribed lines">
                {result.lines.map((line) => (
                  <li
                    key={line.index}
                    data-active={activeLine === line.index}
                    onMouseEnter={() => setActiveLine(line.index)}
                    onMouseLeave={() => setActiveLine(null)}
                  >
                    <span className="transcript__text">
                      <LineText line={line} />
                    </span>
                    <span className="transcript__conf">{Math.round(line.confidence * 100)}%</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="muted">No writing was found. Try a closer, brighter photo.</p>
            )}
            <p className="muted small">
              <mark className="char-low">Wavy-underlined</mark> letters are ones the model was less than{' '}
              {Math.round(LOW_CONFIDENCE * 100)}% sure about.
            </p>
            <div className="row">
              <SpeakButton text={result.text} />
              <button type="button" className="btn btn--secondary" onClick={() => copy(result.text)}>
                <Icon name="copy" /> Copy text
              </button>
              <span role="status" className="small">
                {copied ? 'Copied!' : ''}
              </span>
            </div>
          </section>

          {result.clarified && (
            <section className="card stack" aria-labelledby="clarified-title">
              <h2 id="clarified-title">What was probably meant</h2>
              <p className="clarified">{result.clarified.text}</p>
              {result.clarified.changes.length > 0 && (
                <>
                  <h3>Changes</h3>
                  <ul className="change-list">
                    {result.clarified.changes.map((change, index) => (
                      <li key={index}>
                        <del>{change.original}</del> → <ins>{change.corrected}</ins>{' '}
                        <span className="muted small">({change.reason})</span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {result.clarified.uncertain.length > 0 && (
                <p className="small">
                  <strong>Still unsure about:</strong> {result.clarified.uncertain.join(', ')}
                </p>
              )}
              <div className="row">
                <SpeakButton text={result.clarified.text} />
                <button type="button" className="btn btn--secondary" onClick={() => copy(bestText)}>
                  <Icon name="copy" /> Copy
                </button>
              </div>
            </section>
          )}

          <section className="card">
            <CorrectionBox key={result.sample_id ?? result.text} result={result} />
          </section>
          {result.progressions.map((update) => (
            <ProgressionNotice key={update.track} update={update} title="Handwriting clarity" />
          ))}
          <p className="muted small">
            Model {result.model_version}
            {result.personalized && ' · tuned to your handwriting'}
          </p>
        </div>
      )}
    </>
  )
}
