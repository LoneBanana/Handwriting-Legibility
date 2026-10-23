import { useCallback, useEffect, useRef, useState } from 'react'
import { getNumeracyRound, submitNumeracyRound } from '../api/endpoints'
import type { NumeracyRound, Problem, ProblemAnswer, RoundResult } from '../api/types'
import { useAuth } from '../auth/context'
import { ErrorNotice } from '../components/ErrorNotice'
import { Icon } from '../components/Icon'
import { LoadingBlock } from '../components/LoadingBlock'
import { ProgressionNotice } from '../components/ProgressionNotice'
import { NumberPad } from '../features/math/NumberPad'
import { TimerBar } from '../features/math/TimerBar'
import { MathVisual } from '../features/math/Visuals'
import '../features/math/math.css'
import '../features/reading/reading.css'
import { prefersReducedMotion } from '../lib/settings'
import { usePageTitle } from '../lib/usePageTitle'

const MAX_DIGITS = 6

function ProblemView({
  problem,
  limitMs,
  reducedMotion,
  onAnswer,
}: {
  problem: Problem
  limitMs: number
  reducedMotion: boolean
  onAnswer: (answer: string, latencyMs: number) => void
}) {
  const [value, setValue] = useState('')
  const valueRef = useRef('')
  const shownAt = useRef(0)
  const submitted = useRef(false)

  useEffect(() => {
    shownAt.current = performance.now() // the problem is on screen from here
  }, [])

  const update = useCallback((fn: (v: string) => string) => {
    valueRef.current = fn(valueRef.current)
    setValue(valueRef.current)
  }, [])

  const submit = useCallback(
    (answer: string) => {
      if (!answer || answer === '-' || submitted.current) return
      submitted.current = true
      onAnswer(answer, performance.now() - shownAt.current)
    },
    [onAnswer],
  )

  const addDigit = useCallback(
    (digit: string) => update((v) => (v.length >= MAX_DIGITS ? v : v === '0' ? digit : v + digit)),
    [update],
  )

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return
      // A focused button handles its own Enter/Space activation.
      if (event.target instanceof HTMLButtonElement && (event.key === 'Enter' || event.key === ' ')) return
      if (problem.answer_format === 'choice') {
        const index = Number(event.key) - 1
        const choices = problem.choices ?? []
        if (Number.isInteger(index) && index >= 0 && index < choices.length) submit(choices[index])
        return
      }
      if (/^[0-9]$/.test(event.key)) addDigit(event.key)
      else if (event.key === 'Backspace') update((v) => v.slice(0, -1))
      else if (event.key === '-') update((v) => (v.startsWith('-') ? v.slice(1) : `-${v}`))
      else if (event.key === 'Enter') {
        event.preventDefault()
        submit(valueRef.current)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [problem, submit, addDigit, update])

  return (
    <div className="problem">
      <TimerBar limitMs={limitMs} resetKey={problem.problem_id} reducedMotion={reducedMotion} />
      {problem.visual && <MathVisual visual={problem.visual} seed={problem.problem_id} />}
      <p className="problem__prompt" aria-live="polite">
        {problem.prompt}
      </p>
      {problem.answer_format === 'choice' ? (
        <div className="choice-grid" role="group" aria-label="Choose an answer" style={{ width: 'min(100%, 560px)' }}>
          {(problem.choices ?? []).map((choice, index) => (
            <button key={choice} type="button" className="choice" autoFocus={index === 0} onClick={() => submit(choice)}>
              <span className="choice__key" aria-hidden="true">
                {index + 1}
              </span>
              {choice}
            </button>
          ))}
        </div>
      ) : (
        <>
          <div className="answer-display" data-empty={!value} aria-live="polite" aria-label="Your answer">
            {value || '?'}
          </div>
          <NumberPad
            onDigit={addDigit}
            onBackspace={() => update((v) => v.slice(0, -1))}
            onSubmit={() => submit(valueRef.current)}
            canSubmit={Boolean(value) && value !== '-'}
          />
          <p className="muted small">Type with your keyboard or tap the numbers. Press Enter or OK to check.</p>
        </>
      )}
    </div>
  )
}

function Results({ round, answers, result, onAgain }: { round: NumeracyRound; answers: ProblemAnswer[]; result: RoundResult; onAgain: () => void }) {
  const byId = new Map(round.problems.map((p) => [p.problem_id, p]))
  const answerById = new Map(answers.map((a) => [a.problem_id, a]))
  const correct = result.results.filter((r) => r.correct).length
  return (
    <section className="card stack" aria-labelledby="round-done">
      <h2 id="round-done">Round complete</h2>
      <div className="row">
        <span className="badge badge--success">
          <Icon name="check" size={16} /> {correct} / {result.total} correct
        </span>
        <span className="badge badge--accent">
          <Icon name="bolt" size={16} /> {result.fluent_count} quick and correct
        </span>
      </div>
      {result.progressions.map((update) => (
        <ProgressionNotice key={update.track} update={update} />
      ))}
      <table className="table">
        <caption className="visually-hidden">Your answers</caption>
        <thead>
          <tr>
            <th scope="col">Problem</th>
            <th scope="col">You said</th>
            <th scope="col">Answer</th>
            <th scope="col">Result</th>
          </tr>
        </thead>
        <tbody>
          {result.results.map((r) => {
            const problem = byId.get(r.problem_id)
            const answer = answerById.get(r.problem_id)
            return (
              <tr key={r.problem_id}>
                <td>
                  <strong>{problem?.prompt ?? r.problem_id}</strong>
                  {r.strategy_hint && <div className="small muted">Tip: {r.strategy_hint}</div>}
                </td>
                <td>{answer?.answer ?? '–'}</td>
                <td>{r.expected}</td>
                <td>
                  <span className="result-mark" data-ok={r.correct}>
                    <Icon name={r.correct ? 'check' : 'cross'} size={16} />
                    {r.correct ? 'Right' : 'Not yet'}
                  </span>
                  {r.correct && r.fluent && (
                    <span className="result-mark" style={{ marginLeft: 'var(--space-2)' }}>
                      <Icon name="bolt" size={16} /> quick
                    </span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div>
        <button type="button" className="btn btn--large" onClick={onAgain} autoFocus>
          <Icon name="refresh" /> Another round
        </button>
      </div>
    </section>
  )
}

export default function MathPage() {
  usePageTitle('Math')
  const { settings } = useAuth()
  const reducedMotion = prefersReducedMotion(settings)
  const [round, setRound] = useState<NumeracyRound | null>(null)
  const [phase, setPhase] = useState<'intro' | 'playing' | 'submitting' | 'done'>('intro')
  const [index, setIndex] = useState(0)
  const [answers, setAnswers] = useState<ProblemAnswer[]>([])
  const [result, setResult] = useState<RoundResult | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    setAnswers([])
    setIndex(0)
    setPhase('intro')
    try {
      setRound(await getNumeracyRound())
    } catch (err) {
      setError(err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    // Loading on mount is the effect's job; load() flips the loading flags itself.
    // oxlint-disable-next-line react/set-state-in-effect
    void load()
  }, [load])

  const submitRound = useCallback(
    async (all: ProblemAnswer[]) => {
      if (!round) return
      setPhase('submitting')
      try {
        setResult(await submitNumeracyRound(round.round_id, all))
        setPhase('done')
      } catch (err) {
        setError(err)
        setPhase('playing')
      }
    },
    [round],
  )

  const onAnswer = useCallback(
    (answer: string, latencyMs: number) => {
      if (!round) return
      const problem = round.problems[index]
      const next = [...answers, { problem_id: problem.problem_id, answer, latency_ms: Math.round(latencyMs) }]
      setAnswers(next)
      if (index + 1 < round.problems.length) setIndex(index + 1)
      else void submitRound(next)
    },
    [round, index, answers, submitRound],
  )

  return (
    <>
      <div className="page-header">
        <h1>Math facts</h1>
        <p>Short rounds. Accuracy first — speed comes with practice, and the timer never cuts you off.</p>
      </div>
      <ErrorNotice error={error} onRetry={round ? undefined : load} />
      {loading && <LoadingBlock label="Setting up your round…" />}

      {round && phase === 'intro' && (
        <section className="card stack" aria-labelledby="round-title">
          <h2 id="round-title">{round.skill_label}</h2>
          <p>
            {round.problems.length} problems. Try to answer each within{' '}
            <strong>{(round.time_limit_ms / 1000).toFixed(1)} seconds</strong> — but there’s no rush to be perfect.
          </p>
          <div className="row">
            <span className="badge badge--accent">Level {round.level}</span>
          </div>
          <div>
            <button type="button" className="btn btn--large" onClick={() => setPhase('playing')} autoFocus>
              <Icon name="play" /> Start
            </button>
          </div>
        </section>
      )}

      {round && (phase === 'playing' || phase === 'submitting') && round.problems[index] && (
        <section className="card stack" aria-labelledby="problem-count">
          <div className="card-progress">
            <h2 id="problem-count" style={{ margin: 0 }}>
              Problem {index + 1} of {round.problems.length}
            </h2>
            <span className="badge">{round.skill_label}</span>
          </div>
          {phase === 'submitting' ? (
            <LoadingBlock label="Checking your answers…" />
          ) : (
            <ProblemView
              key={round.problems[index].problem_id}
              problem={round.problems[index]}
              limitMs={round.time_limit_ms}
              reducedMotion={reducedMotion}
              onAnswer={onAnswer}
            />
          )}
        </section>
      )}

      {round && phase === 'done' && result && <Results round={round} answers={answers} result={result} onAgain={load} />}
    </>
  )
}
