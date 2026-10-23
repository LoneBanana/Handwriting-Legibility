import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { createCheck, submitCheck } from '../../api/endpoints'
import type { CheckQuestion, CheckResult, ComprehensionCheck as Check } from '../../api/types'
import { ErrorNotice } from '../../components/ErrorNotice'
import { Icon } from '../../components/Icon'
import { LoadingBlock } from '../../components/LoadingBlock'
import { ProgressionNotice } from '../../components/ProgressionNotice'

const KIND_LABELS: Record<CheckQuestion['kind'], string> = {
  main_idea: 'Big idea',
  detail: 'Detail',
  emotion: 'Feelings',
  inference: 'Reading between the lines',
}

const TRACK_TITLES: Record<string, string> = {
  'converse.listening': 'Listening progress',
  'converse.expressions': 'Reading faces progress',
}

type Phase = 'idle' | 'loading' | 'answering' | 'submitting' | 'done'

/** "Check my understanding": a few multiple-choice questions about one answer. */
export function ComprehensionCheck({ sessionId, turnId }: { sessionId: string; turnId: string }) {
  const formId = useId()
  const [phase, setPhase] = useState<Phase>('idle')
  const [check, setCheck] = useState<Check | null>(null)
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [result, setResult] = useState<CheckResult | null>(null)
  const [error, setError] = useState<unknown>(null)
  const headingRef = useRef<HTMLHeadingElement>(null)

  useEffect(() => {
    if (phase === 'answering' || phase === 'done') headingRef.current?.focus()
  }, [phase])

  const start = async () => {
    setPhase('loading')
    setError(null)
    try {
      const next = await createCheck(sessionId, turnId)
      setCheck(next)
      setAnswers({})
      setPhase('answering')
    } catch (err) {
      setError(err)
      setPhase('idle')
    }
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!check) return
    setPhase('submitting')
    setError(null)
    try {
      const outcome = await submitCheck(
        check.check_id,
        check.questions.map((q) => ({ question_id: q.question_id, choice: answers[q.question_id] ?? '' })),
      )
      setResult(outcome)
      setPhase('done')
    } catch (err) {
      setError(err)
      setPhase('answering')
    }
  }

  if (phase === 'idle') {
    return (
      <div className="check stack-sm">
        <div className="row">
          <button type="button" className="btn" onClick={start}>
            <Icon name="check" /> Check my understanding
          </button>
          <span className="muted small">A few quick questions about this answer.</span>
        </div>
        <ErrorNotice error={error} />
      </div>
    )
  }

  if (phase === 'loading') return <LoadingBlock label="Writing a few questions for you…" />

  if (phase === 'done' && check && result) {
    const byId = new Map(check.questions.map((q) => [q.question_id, q]))
    const right = result.results.filter((r) => r.correct).length
    return (
      <section className="check stack" aria-labelledby={`${formId}-results`}>
        <h3 id={`${formId}-results`} ref={headingRef} tabIndex={-1}>
          {result.perfect ? 'Perfect — well done!' : 'How you did'}
        </h3>
        <p role="status" style={{ margin: 0 }}>
          <span className="badge badge--success">
            <Icon name="check" size={16} /> {right} of {result.results.length} right
          </span>
        </p>
        <ol className="check__results">
          {result.results.map((r) => {
            const question = byId.get(r.question_id)
            const chosen = answers[r.question_id]
            return (
              <li key={r.question_id} className="check__result" data-correct={r.correct}>
                <p className="check__prompt">{question?.prompt ?? 'Question'}</p>
                <p className="result-line">
                  <Icon name={r.correct ? 'check' : 'cross'} size={18} />
                  <span>
                    {r.correct ? (
                      <>
                        Right! You chose <strong>{chosen}</strong>.
                      </>
                    ) : (
                      <>
                        You chose <strong>{chosen || 'nothing'}</strong>. The answer was <strong>{r.expected}</strong>.
                      </>
                    )}
                  </span>
                </p>
                {r.explanation && <p className="small muted">{r.explanation}</p>}
              </li>
            )
          })}
        </ol>
        {result.progressions.map((update) => (
          <ProgressionNotice key={update.track} update={update} title={TRACK_TITLES[update.track] ?? 'Your progress'} />
        ))}
      </section>
    )
  }

  if (!check) return null
  const answered = check.questions.filter((q) => answers[q.question_id]).length
  const complete = answered === check.questions.length
  return (
    <form className="check stack" onSubmit={submit} aria-labelledby={`${formId}-title`}>
      <h3 id={`${formId}-title`} ref={headingRef} tabIndex={-1}>
        Check your understanding
      </h3>
      {check.questions.map((question, index) => (
        <fieldset key={question.question_id} className="field check__question">
          <legend>
            <span className="badge">{KIND_LABELS[question.kind] ?? 'Question'}</span>{' '}
            <span>
              {index + 1}. {question.prompt}
            </span>
          </legend>
          <div className="option-list">
            {question.choices.map((choice) => (
              <label key={choice} className="option">
                <input
                  type="radio"
                  name={`${formId}-${question.question_id}`}
                  value={choice}
                  checked={answers[question.question_id] === choice}
                  onChange={() => setAnswers((a) => ({ ...a, [question.question_id]: choice }))}
                  disabled={phase === 'submitting'}
                />
                {choice}
              </label>
            ))}
          </div>
        </fieldset>
      ))}
      <ErrorNotice error={error} />
      <div className="row">
        <button type="submit" className="btn" disabled={!complete || phase === 'submitting'} aria-busy={phase === 'submitting'}>
          {phase === 'submitting' ? 'Checking…' : 'Check my answers'}
        </button>
        <span className="muted small">
          {complete ? 'All answered.' : `${answered} of ${check.questions.length} answered`}
        </span>
      </div>
    </form>
  )
}
