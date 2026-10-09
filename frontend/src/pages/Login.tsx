import { useState } from 'react'
import { ApiError } from '../services/api'
import { useAuth } from '../hooks/useAuth'

export default function Login() {
  const { login, register } = useAuth()
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setBusy(true)
    try {
      if (mode === 'login') await login(email, password)
      else await register(email, password, name)
    } catch (err) {
      const e2 = err as ApiError
      setError(`${e2.message}${e2.suggestion ? ` ${e2.suggestion}` : ''}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-page">
      <div className="card login-card">
        <div className="row" style={{ gap: 10, marginBottom: 14 }}>
          <span style={{ fontSize: 30 }}>🤍</span>
          <div>
            <h2 style={{ margin: 0, fontSize: 20 }}>Wellness Assistant</h2>
            <p className="muted small" style={{ margin: 0 }}>AI-assisted wellness support</p>
          </div>
        </div>

        <div className="row" style={{ marginBottom: 6 }}>
          <button className={mode === 'login' ? '' : 'secondary'} onClick={() => setMode('login')} type="button">
            Sign in
          </button>
          <button className={mode === 'register' ? '' : 'secondary'} onClick={() => setMode('register')} type="button">
            Create account
          </button>
        </div>

        {error && <div className="alert error" role="alert">{error}</div>}

        <form onSubmit={submit}>
          {mode === 'register' && (
            <>
              <label htmlFor="name">Name</label>
              <input id="name" value={name} onChange={(e) => setName(e.target.value)} required maxLength={80} />
            </>
          )}
          <label htmlFor="email">Email</label>
          <input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          <label htmlFor="password">Password</label>
          <input
            id="password" type="password" value={password}
            onChange={(e) => setPassword(e.target.value)}
            required minLength={mode === 'register' ? 8 : 1}
          />
          <div style={{ marginTop: 18 }}>
            <button disabled={busy} style={{ width: '100%' }}>
              {busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}
            </button>
          </div>
        </form>

        <p className="disclaimer">
          This application provides AI-assisted wellness support and is not a medical diagnostic
          system, therapist, or emergency service.
        </p>
      </div>
    </div>
  )
}
