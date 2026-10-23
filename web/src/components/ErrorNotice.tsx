import { errorMessage } from '../lib/errors'

export function ErrorNotice({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  if (!error) return null
  return (
    <div className="notice notice--error" role="alert">
      <div className="notice__title">That didn’t work</div>
      <p>{errorMessage(error)}</p>
      {onRetry && (
        <button type="button" className="btn btn--secondary" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}
