import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { answerCard, answerCardHandwriting, completeLiteracySession, getLiteracySession } from '../api/endpoints'
import type { AnswerResult, Card, HandwritingAnswerResult, LiteracyCompletion, LiteracySession } from '../api/types'
import { useAuth } from '../auth/context'
import { ErrorNotice } from '../components/ErrorNotice'
import { Icon } from '../components/Icon'
import { ImagePicker } from '../components/ImagePicker'
import { LoadingBlock } from '../components/LoadingBlock'
import { ProgressionNotice } from '../components/ProgressionNotice'
import { GraphemeWord } from '../features/reading/GraphemeWord'
import { seededShuffle } from '../features/reading/graphemes'
import '../features/reading/reading.css'
import { speak, speechSupported, stopSpeaking } from '../lib/speech'
import { usePageTitle } from '../lib/usePageTitle'

type Feedback = (AnswerResult | HandwritingAnswerResult) & { response: string }

const ERROR_LABELS: Record<AnswerResult['errors'][number]['type'], string> = {
  reversal: 'Letter flipped',
  transposition: 'Letters swapped',
  omission: 'Letter missing',
  insertion: 'Extra letter',
  substitution: 'Different letter',
  vowel: 'Vowel sound',
  doubling: 'Double letter',
}

function ReadCard({ card, onAnswer }: { card: Card; onAnswer: (response: string, latencyMs: number) => void }) {
  const [phase, setPhase] = useState<'ready' | 'flash' | 'choose'>('ready')
  const shownAt = useRef(0)
  const choices = useMemo(() => seededShuffle(card.choices ?? [card.word], card.card_id), [card])

  useEffect(() => {
    if (phase !== 'flash') return
    const timer = window.setTimeout(() => {
      setPhase('choose')
      shownAt.current = performance.now()
    }, card.exposure_ms ?? 1500)
    return () => window.clearTimeout(timer)
  }, [phase, card.exposure_ms])

  useEffect(() => {
    if (phase !== 'choose') return
    const onKey = (event: KeyboardEvent) => {
      const index = Number(event.key) - 1
      if (Number.isInteger(index) && index >= 0 && index < choices.length) {
        onAnswer(choices[index], performance.now() - shownAt.current)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [phase, choices, onAnswer])

  return (
    <div className="stack">
      <p className="muted">{card.prompt}</p>
      <div className="flash-stage" aria-live="polite">
        {phase === 'ready' && (
          <button type="button" className="btn btn--large" autoFocus onClick={() => setPhase('flash')}>
            <Icon name="play" /> Show the word
          </button>
        )}
        {phase === 'flash' && <GraphemeWord word={card.word} graphemes={card.graphemes} focus={card.focus} />}
        {phase === 'choose' && <p style={{ margin: 0 }}>Which word did you see?</p>}
      </div>
      {phase === 'ready' && (
        <p className="muted small">
          The word appears for {((card.exposure_ms ?? 1500) / 1000).toFixed(1)} seconds. Get comfortable first.
        </p>
      )}
      {phase === 'choose' && (
        <div className="choice-grid" role="group" aria-label="Choices">
          {choices.map((choice, index) => (
            <button
              key={choice}
              type="button"
              className="choice"
              autoFocus={index === 0}
              onClick={() => onAnswer(choice, performance.now() - shownAt.current)}
            >
              <span className="choice__key" aria-hidden="true">
                {index + 1}
              </span>
              {choice}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function SpellCard({
  card,
  voiceName,
  onAnswer,
  onHandwriting,
}: {
  card: Card
  voiceName: string | null
  onAnswer: (response: string, latencyMs: number) => void
  onHandwriting: (file: File, latencyMs: number) => void
}) {
  const [value, setValue] = useState(card.hint?.first_letter ?? '')
  const [speaking, setSpeaking] = useState(false)
  const startedAt = useRef(0)
  const hint = card.hint

  useEffect(() => {
    startedAt.current = performance.now() // the card is on screen from here
  }, [])

  const sayIt = useCallback(async () => {
    stopSpeaking()
    setSpeaking(true)
    await speak(card.word, { rate: 0.85, voiceName })
    if (card.sentence) await speak(card.sentence, { rate: 0.9, voiceName })
    setSpeaking(false)
  }, [card.word, card.sentence, voiceName])

  useEffect(() => () => stopSpeaking(), [])

  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (!value.trim()) return
    onAnswer(value.trim(), performance.now() - startedAt.current)
  }

  return (
    <div className="stack">
      <p className="muted">{card.prompt}</p>
      <div className="flash-stage">
        <div className="stack" style={{ alignItems: 'center' }}>
          {speechSupported() ? (
            <button type="button" className="btn btn--large" onClick={sayIt} autoFocus aria-pressed={speaking}>
              <Icon name="speaker" /> {speaking ? 'Listening…' : 'Hear the word'}
            </button>
          ) : (
            <p>Your browser can’t speak words aloud. Ask a helper to read: “{card.word}”.</p>
          )}
          {card.sentence && <p className="muted small" style={{ margin: 0 }}>You’ll also hear it in a sentence.</p>}
        </div>
      </div>
      <form className="stack-sm" onSubmit={submit}>
        <label htmlFor="spell-input" className="label">
          Type the word
        </label>
        <input
          id="spell-input"
          className="spell-input"
          type="text"
          autoComplete="off"
          autoCapitalize="none"
          autoCorrect="off"
          spellCheck={false}
          aria-describedby={hint ? 'spell-hint' : undefined}
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
        {hint && (
          <>
            <div className="letter-boxes" aria-hidden="true">
              {Array.from({ length: hint.length }, (_, i) => (
                <span key={i} data-filled={Boolean(value[i])} data-hint={i === 0 && hint.first_letter !== null}>
                  {value[i] ?? ''}
                </span>
              ))}
            </div>
            <p id="spell-hint" className="muted small" style={{ margin: 0 }}>
              Hint: the word has {hint.length} letters
              {hint.first_letter ? ` and starts with “${hint.first_letter}”` : ''}.
            </p>
          </>
        )}
        <button className="btn" type="submit" disabled={!value.trim()}>
          Check
        </button>
      </form>
      <details>
        <summary>Or write it on paper and upload a photo</summary>
        <div style={{ marginTop: 'var(--space-3)' }}>
          <ImagePicker compact prompt="A photo of just this word works best." onFile={(file) => onHandwriting(file, performance.now() - startedAt.current)} />
        </div>
      </details>
    </div>
  )
}

function FeedbackPanel({ card, feedback, onNext, isLast }: { card: Card; feedback: Feedback; onNext: () => void; isLast: boolean }) {
  const recognized = 'recognized' in feedback ? feedback.recognized : null
  return (
    <div className="stack" aria-live="polite">
      <div className={`notice ${feedback.correct ? 'notice--success' : 'notice--warning'}`}>
        <div className="notice__title">
          <Icon name={feedback.correct ? 'check' : 'info'} />
          {feedback.correct ? 'Correct!' : 'Not quite — here’s the word'}
        </div>
        <p>{feedback.feedback}</p>
        {recognized !== null && (
          <p className="small">
            We read your writing as: <strong>{recognized || '(nothing)'}</strong>
          </p>
        )}
        {!feedback.correct && feedback.phonetically_plausible && (
          <p className="small">Your spelling sounds right — this word just uses a different pattern.</p>
        )}
      </div>
      <div className="flash-stage" style={{ minHeight: 140 }}>
        <div className="stack-sm" style={{ alignItems: 'center' }}>
          <GraphemeWord word={feedback.expected} graphemes={card.graphemes} focus={card.focus} />
          <span className="badge badge--accent">{card.skill_label}</span>
        </div>
      </div>
      {!feedback.correct && feedback.errors.length > 0 && (
        <div>
          <h3>What changed</h3>
          <ul className="error-list">
            {feedback.errors.map((error, index) => (
              <li key={index}>
                <strong>{ERROR_LABELS[error.type]}:</strong> {error.detail}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div>
        <button type="button" className="btn btn--large" onClick={onNext} autoFocus>
          {isLast ? 'Finish session' : 'Next card'} <Icon name="arrowRight" />
        </button>
      </div>
    </div>
  )
}

export default function ReadingPage() {
  usePageTitle('Reading')
  const { settings } = useAuth()
  const [session, setSession] = useState<LiteracySession | null>(null)
  const [index, setIndex] = useState(0)
  const [feedback, setFeedback] = useState<Feedback | null>(null)
  const [completion, setCompletion] = useState<LiteracyCompletion | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)
  const [results, setResults] = useState<boolean[]>([])

  const start = useCallback(async () => {
    setBusy(true)
    setError(null)
    setCompletion(null)
    setFeedback(null)
    setIndex(0)
    setResults([])
    try {
      setSession(await getLiteracySession())
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    // Loading on mount is the effect's job; start() flips the loading flags itself.
    // oxlint-disable-next-line react/set-state-in-effect
    void start()
  }, [start])

  const card = session?.cards[index] ?? null

  const handleAnswer = useCallback(
    async (response: string, latencyMs: number) => {
      if (!session || !card || busy) return
      setBusy(true)
      setError(null)
      try {
        const result = await answerCard(session.session_id, {
          card_id: card.card_id,
          response,
          latency_ms: Math.round(latencyMs),
        })
        setFeedback({ ...result, response })
        setResults((r) => [...r, result.correct])
      } catch (err) {
        setError(err)
      } finally {
        setBusy(false)
      }
    },
    [session, card, busy],
  )

  const handleHandwriting = async (file: File, latencyMs: number) => {
    if (!session || !card) return
    setBusy(true)
    setError(null)
    try {
      const result = await answerCardHandwriting(session.session_id, card.card_id, file, latencyMs)
      setFeedback({ ...result, response: result.recognized })
      setResults((r) => [...r, result.correct])
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  const next = async () => {
    if (!session) return
    setFeedback(null)
    if (index + 1 < session.cards.length) {
      setIndex(index + 1)
      return
    }
    setBusy(true)
    try {
      setCompletion(await completeLiteracySession(session.session_id))
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="page-header">
        <h1>Reading & spelling cards</h1>
        <p>Short sessions. Colours show the sound chunks in each word; the highlighted chunk is today’s pattern.</p>
      </div>

      <ErrorNotice error={error} onRetry={session ? undefined : start} />

      {!session && busy && <LoadingBlock label="Preparing your cards…" />}

      {completion && session && (
        <section className="card stack" aria-labelledby="done-title">
          <h2 id="done-title">Session complete</h2>
          <p className="word word--small">
            {completion.correct} / {completion.total} correct
          </p>
          {completion.progressions.map((update) => (
            <ProgressionNotice
              key={update.track}
              update={update}
              title={update.track === 'literacy.spelling' ? 'Spelling progress' : 'Reading progress'}
            />
          ))}
          <div>
            <button type="button" className="btn btn--large" onClick={start} autoFocus>
              <Icon name="refresh" /> Another session
            </button>
          </div>
        </section>
      )}

      {session && card && !completion && (
        <section className="card stack" aria-labelledby="card-title">
          <div className="card-progress">
            <h2 id="card-title" style={{ margin: 0 }}>
              Card {index + 1} of {session.cards.length}
            </h2>
            <span className="badge badge--accent">{card.mode === 'read' ? 'Read' : 'Spell'}</span>
          </div>
          <div className="dots" aria-hidden="true">
            {session.cards.map((c, i) => (
              <span key={c.card_id} data-state={i < index ? 'done' : i === index ? 'current' : 'todo'} />
            ))}
          </div>
          <p className="small muted" style={{ margin: 0 }}>
            Pattern: <strong>{card.skill_label}</strong>
          </p>
          {feedback ? (
            <FeedbackPanel
              card={card}
              feedback={feedback}
              onNext={next}
              isLast={index + 1 === session.cards.length}
            />
          ) : card.mode === 'read' ? (
            <ReadCard key={card.card_id} card={card} onAnswer={handleAnswer} />
          ) : (
            <SpellCard
              key={card.card_id}
              card={card}
              voiceName={settings.tts_voice}
              onAnswer={handleAnswer}
              onHandwriting={handleHandwriting}
            />
          )}
          {busy && <LoadingBlock label="Checking…" />}
          <p className="visually-hidden" aria-live="polite">
            {results.length > 0 && `${results.filter(Boolean).length} correct so far.`}
          </p>
        </section>
      )}

      {session && session.cards.length === 0 && !completion && (
        <p className="empty">No cards are due right now — come back later!</p>
      )}
    </>
  )
}
