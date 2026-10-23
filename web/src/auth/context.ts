import { createContext, useContext } from 'react'
import type { User, UserSettings } from '../api/types'

// The context and its hook live apart from <AuthProvider>, so editing the provider
// during development (fast refresh) never recreates the context object.

export type AuthStatus = 'loading' | 'authenticated' | 'anonymous'

export type AuthValue = {
  status: AuthStatus
  user: User | null
  settings: UserSettings
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, displayName: string) => Promise<void>
  logout: () => void
  /** Applies settings to the page immediately without saving them. */
  previewSettings: (settings: UserSettings) => void
  saveSettings: (patch: Partial<UserSettings>) => Promise<User>
}

export const AuthContext = createContext<AuthValue | null>(null)

export function useAuth(): AuthValue {
  const value = useContext(AuthContext)
  if (!value) throw new Error('useAuth must be used inside <AuthProvider>')
  return value
}
