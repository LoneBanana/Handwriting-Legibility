import { useState } from 'react'
import { Link } from 'react-router'
import { deleteHandwritingSamples, handwritingSampleCount } from '../../api/endpoints'
import { useAuth } from '../../auth/context'
import { ErrorNotice } from '../../components/ErrorNotice'
import { useResource } from '../../lib/useResource'

const count = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`

/**
 * Consent to keep handwriting photos for personalisation (saved straight
 * away, since it is a privacy choice), plus how many samples are stored and a
 * way to delete them.
 */
export function HandwritingSamples() {
  const { settings, saveSettings } = useAuth()
  const [pending, setPending] = useState<boolean | null>(null)
  const [consentMessage, setConsentMessage] = useState('')
  const [consentError, setConsentError] = useState<unknown>(null)
  const samples = useResource(handwritingSampleCount)
  const [confirming, setConfirming] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<unknown>(null)
  const [deletedMessage, setDeletedMessage] = useState('')

  const keep = pending ?? settings.store_handwriting_samples

  const toggle = async (next: boolean) => {
    setPending(next)
    setConsentError(null)
    setConsentMessage('')
    try {
      await saveSettings({ store_handwriting_samples: next })
      setConsentMessage(
        next
          ? 'Saved. From now on, the handwriting photos you transcribe will be kept to help Legible learn your writing.'
          : 'Saved. New handwriting photos won’t be kept.',
      )
    } catch (err) {
      setConsentError(err)
    } finally {
      setPending(null)
    }
  }

  const remove = async () => {
    setDeleting(true)
    setDeleteError(null)
    try {
      await deleteHandwritingSamples()
      setConfirming(false)
      setDeletedMessage(
        'Deleted. Your saved handwriting samples, corrections and personal handwriting model are gone.' +
          (settings.store_handwriting_samples ? ' New photos will be kept again while “Keep my handwriting samples” is on.' : ''),
      )
      samples.reload()
    } catch (err) {
      setDeleteError(err)
    } finally {
      setDeleting(false)
    }
  }

  const stored = samples.data?.samples ?? 0
  const corrected = samples.data?.corrected ?? 0

  return (
    <section className="card stack" aria-labelledby="samples-title">
      <h2 id="samples-title">Handwriting samples</h2>
      <div className="stack-sm">
        <p style={{ margin: 0 }}>
          When you turn this on, Legible keeps the photos you transcribe on the <Link to="/handwriting">Handwriting</Link>{' '}
          page, along with any corrections you type. It uses them to train a personal model that learns your handwriting,
          so its reading of your writing gets more accurate over time.
        </p>
        <p className="muted small" style={{ margin: 0 }}>
          When it’s off, your photos are read and then not kept. You can delete saved samples at any time below, and
          deleting your account removes them too.
        </p>
      </div>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={keep}
          onChange={(event) => void toggle(event.target.checked)}
          disabled={pending !== null}
        />
        <span>
          <strong>Keep my handwriting samples</strong>
          <br />
          <span className="muted small">Saved as soon as you change it.</span>
        </span>
      </label>
      <p className="small" role="status" style={{ margin: 0 }}>
        {pending !== null ? 'Saving…' : consentMessage}
      </p>
      <ErrorNotice error={consentError} />

      <div className="stack-sm samples-store">
        <h3 style={{ margin: 0 }}>Saved samples</h3>
        {samples.loading && !samples.data ? (
          <p className="muted small" style={{ margin: 0 }}>
            Checking for saved samples…
          </p>
        ) : samples.error ? (
          <ErrorNotice error={samples.error} onRetry={samples.reload} />
        ) : stored === 0 ? (
          <p className="small" style={{ margin: 0 }}>
            No handwriting samples are saved right now.
          </p>
        ) : (
          <p className="small" style={{ margin: 0 }}>
            You have <strong>{count(stored, 'saved handwriting photo', 'saved handwriting photos')}</strong>
            {corrected > 0 ? `, ${corrected} with your corrections` : ''}.
          </p>
        )}
        {stored > 0 && !confirming && (
          <div>
            <button type="button" className="btn btn--danger-outline" onClick={() => setConfirming(true)}>
              Delete my saved handwriting samples…
            </button>
          </div>
        )}
        {confirming && (
          <div className="notice notice--warning stack-sm" role="group" aria-labelledby="samples-confirm-title">
            <div className="notice__title" id="samples-confirm-title">
              Delete all {count(stored, 'sample', 'samples')}?
            </div>
            <p style={{ margin: 0 }}>
              This deletes the photos, your corrections and your personal handwriting model. Transcriptions go back to the
              standard model. It can’t be undone.
            </p>
            <div className="row">
              <button type="button" className="btn btn--danger" onClick={() => void remove()} disabled={deleting} aria-busy={deleting}>
                {deleting ? 'Deleting…' : 'Yes, delete them'}
              </button>
              <button type="button" className="btn btn--secondary" onClick={() => setConfirming(false)} disabled={deleting} autoFocus>
                Keep them
              </button>
            </div>
            <ErrorNotice error={deleteError} />
          </div>
        )}
        <p className="small" role="status" style={{ margin: 0 }}>
          {deletedMessage}
        </p>
      </div>
    </section>
  )
}
