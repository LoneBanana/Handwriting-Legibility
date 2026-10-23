import type { ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router'
import { LoadingBlock } from '../components/LoadingBlock'
import { useAuth } from './context'

export function RequireAuth({ children }: { children: ReactNode }) {
  const { status } = useAuth()
  const location = useLocation()
  if (status === 'loading') return <LoadingBlock label="Signing you in…" />
  if (status === 'anonymous') return <Navigate to="/login" replace state={{ from: location.pathname }} />
  return children
}
