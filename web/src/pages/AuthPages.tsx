import { useState, type FormEvent, type ReactNode } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router'
import { useAuth } from '../auth/context'
import { ErrorNotice } from '../components/ErrorNotice'
import { BrandMark } from '../components/Layout'
import { usePageTitle } from '../lib/usePageTitle'

function AuthShell({ title, children }: { title: string; children: ReactNode }) {
  return (
    <main id="main" className="auth-page">
      <div className="row" style={{ marginBottom: 'var(--space-5)' }}>
        <span className="brand" style={{ fontSize: '1.5rem' }}>
          <BrandMark />
          Legible
        </span>
      </div>
      <div className="card">
        <h1>{title}</h1>
        {children}
      </div>
    </main>
  )
}

function useRedirectTarget(): string {
  const location = useLocation()
  const from = (location.state as { from?: string } | null)?.from
  return from && from !== '/login' && from !== '/register' ? from : '/'
}

export function LoginPage() {
  usePageTitle('Log in')
  const { status, login } = useAuth()
  const navigate = useNavigate()
  const target = useRedirectTarget()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)

  if (status === 'authenticated') return <Navigate to={target} replace />

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await login(email.trim(), password)
      navigate(target, { replace: true })
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <AuthShell title="Welcome back">
      <form className="stack" onSubmit={onSubmit} noValidate>
        <div className="field">
          <label htmlFor="login-email">Email</label>
          <input
            id="login-email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="login-password">Password</label>
          <input
            id="login-password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        <ErrorNotice error={error} />
        <button className="btn btn--large btn--block" type="submit" disabled={busy || !email || !password}>
          {busy ? 'Logging in…' : 'Log in'}
        </button>
        <p className="muted">
          New here? <Link to="/register">Create an account</Link>
        </p>
      </form>
    </AuthShell>
  )
}

export function RegisterPage() {
  usePageTitle('Create account')
  const { status, register } = useAuth()
  const navigate = useNavigate()
  const [displayName, setDisplayName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)

  if (status === 'authenticated') return <Navigate to="/" replace />

  const tooShort = password.length > 0 && password.length < 8

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (password.length < 8) return
    setBusy(true)
    setError(null)
    try {
      await register(email.trim(), password, displayName.trim())
      navigate('/', { replace: true })
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <AuthShell title="Create your account">
      <form className="stack" onSubmit={onSubmit} noValidate>
        <div className="field">
          <label htmlFor="reg-name">What should we call you?</label>
          <input
            id="reg-name"
            type="text"
            autoComplete="nickname"
            required
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="reg-email">Email</label>
          <input
            id="reg-email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="reg-password">Password</label>
          <input
            id="reg-password"
            type="password"
            autoComplete="new-password"
            required
            minLength={8}
            aria-describedby="reg-password-hint"
            aria-invalid={tooShort}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <span id="reg-password-hint" className="hint">
            {tooShort ? `At least 8 characters — ${8 - password.length} more to go.` : 'At least 8 characters.'}
          </span>
        </div>
        <ErrorNotice error={error} />
        <button
          className="btn btn--large btn--block"
          type="submit"
          disabled={busy || !email || !displayName || password.length < 8}
        >
          {busy ? 'Creating account…' : 'Create account'}
        </button>
        <p className="muted">
          Already have an account? <Link to="/login">Log in</Link>
        </p>
      </form>
    </AuthShell>
  )
}
