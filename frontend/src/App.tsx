import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import { useAuth } from './hooks/useAuth'
import Assistant from './pages/Assistant'
import Camera from './pages/Camera'
import Dashboard from './pages/Dashboard'
import Login from './pages/Login'
import MoodDashboard from './pages/MoodDashboard'
import Multimodal from './pages/Multimodal'
import Privacy from './pages/Privacy'
import Wellness from './pages/Wellness'

export default function App() {
  const { ready } = useAuth()

  if (!ready) {
    return (
      <div className="login-page">
        <div className="card login-card">
          <div className="skeleton" style={{ height: 22, width: '60%' }} />
          <div className="skeleton" style={{ marginTop: 12 }} />
          <div className="skeleton" style={{ marginTop: 8 }} />
        </div>
      </div>
    )
  }

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/assistant" element={<Assistant />} />
        <Route path="/camera" element={<Camera />} />
        <Route path="/multimodal" element={<Multimodal />} />
        <Route path="/mood" element={<MoodDashboard />} />
        <Route path="/wellness" element={<Wellness />} />
        <Route path="/privacy" element={<Privacy />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}

