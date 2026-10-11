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

const GUEST_USER: User = {
  id: 'guest_user',
  email: 'guest@smwa.local',
  name: 'Guest User',
  created_at: new Date().toISOString(),
  consent: {
    save_mood_history: true,
    camera_access: true,
    store_images: false,
    consent_updated_at: new Date().toISOString(),
  },
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User>(GUEST_USER)
  const [ready, setReady] = useState(false)

  const initGuest = useCallback(async () => {
    try {
      const data = await api<{ token: string; user: User }>('/api/auth/guest', {
        method: 'POST',
        auth: false,
      })
      setToken(data.token)
      setUser(data.user)
      localStorage.setItem('smwa_user', JSON.stringify(data.user))
    } catch {
      setUser(GUEST_USER)
      localStorage.setItem('smwa_user', JSON.stringify(GUEST_USER))
    }
  }, [])

  const fetchMe = useCallback(async () => {
    try {
      const data = await api<{ consent: Consent; data_we_store: string[] }>('/api/privacy/consent')
      setUser((prev) => ({ ...prev, consent: data.consent }))
    } catch {
      await initGuest()
    }
  }, [initGuest])

  useEffect(() => {
    const stored = localStorage.getItem('smwa_user')
    const token = getToken()
    if (stored && token) {
      try {
        setUser(JSON.parse(stored))
      } catch {
        setUser(GUEST_USER)
      }
      fetchMe().finally(() => setReady(true))
    } else {
      initGuest().finally(() => setReady(true))
    }
    const onUnauthorized = () => {
      initGuest()
    }
    window.addEventListener('smwa:unauthorized', onUnauthorized)
    return () => window.removeEventListener('smwa:unauthorized', onUnauthorized)
  }, [fetchMe, initGuest])

  const persist = (u: User) => {
    setUser(u)
    localStorage.setItem('smwa_user', JSON.stringify(u))
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
    initGuest()
  }, [initGuest])

  const updateConsent = useCallback(
    async (patch: Partial<Consent>) => {
      const data = await api<{ consent: Consent }>('/api/privacy/consent', {
        method: 'PUT',
        body: patch,
      })
      setUser((prev) => ({ ...prev, consent: data.consent }))
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
