import { Link } from 'react-router'
import { usePageTitle } from '../lib/usePageTitle'

export function NotFoundPage() {
  usePageTitle('Page not found')
  return (
    <div className="empty">
      <h1>We couldn’t find that page</h1>
      <p style={{ margin: '0 auto var(--space-4)' }}>The link may be old, or there may be a typo in the address.</p>
      <Link className="btn" to="/">
        Go home
      </Link>
    </div>
  )
}
