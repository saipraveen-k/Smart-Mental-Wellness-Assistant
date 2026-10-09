import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { api, getToken, setToken } from '../services/api'
import type { Consent, User } from '../types'

interface AuthState {
  user: User | null
  ready: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, name: string) => Promise<void>
  logout: () => void
  refreshUser: () => Promise<void>
  updateConsent: (patch: Partial<Consent>) => Promise<Consent>
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [ready, setReady] = useState(false)

  const fetchMe = useCallback(async () => {
    // The backend has no /me endpoint; recover the user via privacy/consent.
    try {
      const data = await api<{ consent: Consent; data_we_store: string[] }>('/api/privacy/consent')
      setUser((prev) => (prev ? { ...prev, consent: data.consent } : prev))
    } catch {
      setToken(null)
      setUser(null)
    }
  }, [])

  useEffect(() => {
    const stored = localStorage.getItem('smwa_user')
    const token = getToken()
    if (stored && token) {
      setUser(JSON.parse(stored))
      fetchMe().finally(() => setReady(true))
    } else {
      setReady(true)
    }
    const onUnauthorized = () => {
      setUser(null)
      localStorage.removeItem('smwa_user')
    }
    window.addEventListener('smwa:unauthorized', onUnauthorized)
    return () => window.removeEventListener('smwa:unauthorized', onUnauthorized)
  }, [fetchMe])

  const persist = (u: User | null) => {
    setUser(u)
    if (u) localStorage.setItem('smwa_user', JSON.stringify(u))
    else localStorage.removeItem('smwa_user')
  }

  const login = useCallback(async (email: string, password: string) => {
    const data = await api<{ token: string; user: User }>('/api/auth/login', {
      method: 'POST',
      body: { email, password },
      auth: false,
    })
    setToken(data.token)
    persist(data.user)
  }, [])

  const register = useCallback(async (email: string, password: string, name: string) => {
    const data = await api<{ token: string; user: User }>('/api/auth/register', {
      method: 'POST',
      body: { email, password, name },
      auth: false,
    })
    setToken(data.token)
    persist(data.user)
  }, [])

  const logout = useCallback(() => {
    setToken(null)
    persist(null)
  }, [])

  const updateConsent = useCallback(
    async (patch: Partial<Consent>) => {
      const data = await api<{ consent: Consent }>('/api/privacy/consent', {
        method: 'PUT',
        body: patch,
      })
      setUser((prev) => (prev ? { ...prev, consent: data.consent } : prev))
      return data.consent
    },
    [],
  )

  const value = useMemo(
    () => ({ user, ready, login, register, logout, refreshUser: fetchMe, updateConsent }),
    [user, ready, login, register, logout, fetchMe, updateConsent],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
