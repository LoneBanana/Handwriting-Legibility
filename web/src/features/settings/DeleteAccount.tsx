import { useId, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router'
import { deleteAccount } from '../../api/endpoints'
import { useAuth } from '../../auth/context'
import { ErrorNotice } from '../../components/ErrorNotice'
import { deleteConfirmed, DELETE_WORD } from './settingsForm'

/** "Delete my account", with a typed confirmation step. */
export function DeleteAccount() {
  const { logout } = useAuth()
  const navigate = useNavigate()
  const inputId = useId()
  const hintId = useId()
  const [confirming, setConfirming] = useState(false)
  const [typed, setTyped] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const ready = deleteConfirmed(typed)

  const cancel = () => {
    setConfirming(false)
    setTyped('')
    setError(null)
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!ready || busy) return
    setBusy(true)
    setError(null)
    try {
      await deleteAccount()
      logout()
      navigate('/login', { replace: true })
    } catch (err) {
      setError(err)
      setBusy(false)
    }
  }

  return (
    <section className="card stack danger-zone" aria-labelledby="delete-title">
      <h2 id="delete-title">Delete my account</h2>
      <p style={{ margin: 0 }}>
        This permanently deletes your account, all your progress, any saved handwriting samples and your personal
        handwriting model. It can’t be undone.
      </p>
      {!confirming ? (
        <div>
          <button type="button" className="btn btn--danger-outline" onClick={() => setConfirming(true)}>
            Delete my account…
          </button>
        </div>
      ) : (
        <form className="stack-sm" onSubmit={submit}>
          <div className="field">
            <label htmlFor={inputId}>To confirm, type {DELETE_WORD}</label>
            <input
              id={inputId}
              type="text"
              autoComplete="off"
              autoCapitalize="characters"
              spellCheck={false}
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              aria-describedby={hintId}
              autoFocus
            />
            <span id={hintId} className="hint">
              {ready ? 'Ready. “Delete my account forever” will now delete everything.' : 'Capital or small letters both work.'}
            </span>
          </div>
          <ErrorNotice error={error} />
          <div className="row">
            <button type="submit" className="btn btn--danger" disabled={!ready || busy} aria-busy={busy}>
              {busy ? 'Deleting…' : 'Delete my account forever'}
            </button>
            <button type="button" className="btn btn--secondary" onClick={cancel} disabled={busy}>
              Keep my account
            </button>
          </div>
        </form>
      )}
    </section>
  )
}
