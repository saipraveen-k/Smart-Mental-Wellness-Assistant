import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'

const links = [
  { to: '/', label: 'Dashboard', icon: '🏠' },
  { to: '/assistant', label: 'AI Assistant', icon: '💬' },
  { to: '/camera', label: 'Camera', icon: '📷' },
  { to: '/multimodal', label: 'Multimodal', icon: '🧠' },
  { to: '/mood', label: 'Mood Dashboard', icon: '📈' },
  { to: '/wellness', label: 'Wellness Center', icon: '🌿' },
  { to: '/privacy', label: 'Privacy', icon: '🔒' },
]

export default function Layout({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth()

  const toggleTheme = () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'
    document.documentElement.setAttribute('data-theme', next)
    localStorage.setItem('smwa_theme', next)
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">🤍</span>
          <span>Wellness Assistant</span>
        </div>
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.to === '/'}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            <span aria-hidden>{l.icon}</span> {l.label}
          </NavLink>
        ))}
        <div className="sidebar-footer">
          <div className="row between">
            <span>{user?.name}</span>
            <button className="small secondary" onClick={toggleTheme} aria-label="Toggle dark mode">
              🌓
            </button>
          </div>
          <div style={{ marginTop: 8 }}>
            <button className="small secondary" onClick={logout}>Sign out</button>
          </div>
          <p style={{ marginTop: 12 }}>
            AI-assisted wellness support — not a medical diagnostic system, therapist, or emergency service.
          </p>
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  )
}
