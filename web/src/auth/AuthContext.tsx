import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { setUnauthorizedHandler, tokenStore } from '../api/client'
import * as api from '../api/endpoints'
import type { User, UserSettings } from '../api/types'
import { applySettings, DEFAULT_SETTINGS, normalizeSettings } from '../lib/settings'
import { AuthContext, type AuthStatus, type AuthValue } from './context'

function systemTheme(): UserSettings['theme'] {
  try {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>(() => (tokenStore.get() ? 'loading' : 'anonymous'))
  const [user, setUser] = useState<User | null>(null)

  useEffect(() => {
    if (!tokenStore.get()) return
    let cancelled = false
    api
      .getMe()
      .then((me) => {
        if (cancelled) return
        setUser(me)
        setStatus('authenticated')
      })
      .catch(() => {
        if (cancelled) return
        tokenStore.clear()
        setUser(null)
        setStatus('anonymous')
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    setUnauthorizedHandler(() => {
      setUser(null)
      setStatus('anonymous')
    })
    return () => setUnauthorizedHandler(null)
  }, [])

  const settings = useMemo(
    () => (user ? normalizeSettings(user.settings) : { ...DEFAULT_SETTINGS, theme: systemTheme() }),
    [user],
  )

  useEffect(() => {
    applySettings(settings)
  }, [settings])

  const login = useCallback(async (email: string, password: string) => {
    const response = await api.login({ email, password })
    tokenStore.set(response.access_token)
    setUser(response.user)
    setStatus('authenticated')
  }, [])

  const register = useCallback(async (email: string, password: string, displayName: string) => {
    const response = await api.register({ email, password, display_name: displayName })
    tokenStore.set(response.access_token)
    let user = response.user
    // Start new accounts in the theme the learner's system already uses.
    const theme = systemTheme()
    if (theme !== user.settings.theme) {
      user = await api.updateSettings({ theme }).catch(() => user)
    }
    setUser(user)
    setStatus('authenticated')
  }, [])

  const logout = useCallback(() => {
    tokenStore.clear()
    setUser(null)
    setStatus('anonymous')
  }, [])

  const previewSettings = useCallback((next: UserSettings) => applySettings(next), [])

  const saveSettings = useCallback(async (patch: Partial<UserSettings>) => {
    const updated = await api.updateSettings(patch)
    setUser(updated)
    return updated
  }, [])

  const value = useMemo<AuthValue>(
    () => ({ status, user, settings, login, register, logout, previewSettings, saveSettings }),
    [status, user, settings, login, register, logout, previewSettings, saveSettings],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
