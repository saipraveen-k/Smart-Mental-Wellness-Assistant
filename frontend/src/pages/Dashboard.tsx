import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import { useAuth } from '../hooks/useAuth'
import type { ForecastResult, MoodRecord, TrendsResult } from '../types'
import { ConfidenceBar, EmotionBadge, ErrorState, Skeleton, WellnessBadge } from '../components/shared'

function greeting() {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}

export default function Dashboard() {
  const { user } = useAuth()
  const [trends, setTrends] = useState<TrendsResult | null>(null)
  const [history, setHistory] = useState<MoodRecord[] | null>(null)
  const [forecast, setForecast] = useState<ForecastResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [t, h, f] = await Promise.all([
        api<TrendsResult>('/api/mood/trends'),
        api<{ records: MoodRecord[] }>('/api/mood/history?limit=10'),
        api<ForecastResult>('/api/mood/forecast'),
      ])
      setTrends(t)
      setHistory(h.records)
      setForecast(f)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const latest = history && history.length > 0 ? history[0] : null

  return (
    <div>
      <h1 className="page-title">{greeting()}, {user?.name?.split(' ')[0] ?? 'there'} 👋</h1>
      <p className="page-sub">Here is your current wellness picture — built only from your own check-ins.</p>

      {error && <ErrorState error={error} onRetry={load} />}

      {loading ? (
        <div className="card"><Skeleton lines={4} /></div>
      ) : !latest ? (
        <div className="card" style={{ textAlign: 'center', padding: 44 }}>
          <h3>No mood history yet</h3>
          <p className="muted">Start your first check-in to see your wellness state, trends and personal recommendations.</p>
          <Link to="/assistant"><button>Start check-in</button></Link>
        </div>
      ) : (
        <>
          <div className="grid cols-3">
            <div className="card">
              <div className="kpi-label">Latest check-in</div>
              <div style={{ margin: '10px 0' }}><EmotionBadge emotion={latest.fused_emotion} /></div>
              <ConfidenceBar value={latest.fused_confidence} label="Confidence" />
              <p className="muted small" style={{ marginTop: 8 }}>
                {new Date(latest.timestamp).toLocaleString()}
              </p>
            </div>
            <div className="card">
              <div className="kpi-label">Wellness indicator</div>
              <div style={{ margin: '10px 0' }}>
                <WellnessBadge state={latest.wellness_state || 'STABLE'} />
              </div>
              <p className="muted small">
                Non-clinical indicator from your latest check-in signals.
              </p>
            </div>
            <div className="card">
              <div className="kpi-label">Latest stressor</div>
              <div className="kpi" style={{ fontSize: 20, marginTop: 8 }}>
                {latest.stressor ?? 'None recorded'}
              </div>
              <p className="muted small">{latest.severity ? `Intensity: ${latest.severity}` : '—'}</p>
            </div>
          </div>

          <div className="card">
            <div className="row between">
              <h3 style={{ margin: 0 }}>Today's check-in</h3>
              <Link to="/assistant"><button>Start check-in</button></Link>
            </div>
            <p className="muted small" style={{ marginBottom: 0 }}>
              Write what's on your mind — optionally add a facial-expression estimate from your camera.
            </p>
          </div>

          <div className="grid cols-2">
            <div className="card">
              <h3>7-day trend</h3>
              {trends?.available ? (
                <div className="stack">
                  <div className="row between">
                    <span>Direction: <strong>{trends.trend?.direction}</strong></span>
                    <span>Recent mean valence: <strong>{trends.weekly?.recent_7d_mean_valence ?? '—'}</strong></span>
                  </div>
                  <p className="muted small" style={{ margin: 0 }}>
                    {trends.trend?.declining
                      ? 'Your recent check-ins trend slightly downward — a gentle exercise below may help.'
                      : 'No declining trend detected in your recent check-ins.'}
                  </p>
                </div>
              ) : (
                <p className="muted small">{trends?.message ?? 'Not enough history yet.'}</p>
              )}
            </div>
            <div className="card">
              <h3>Forecast</h3>
              {forecast?.available ? (
                <p className="muted small">
                  A {forecast.horizon_days || (forecast as any).horizon || 7}-day projection ({forecast.method}) from {forecast.observations} daily
                  observations is available on the Mood Dashboard.
                </p>
              ) : (
                <p className="muted small">
                  {forecast?.message ?? 'Not enough mood history for forecasting. Continue checking in to unlock personalized trend analysis.'}
                </p>
              )}
            </div>
          </div>
        </>
      )}

      <p className="footer-note">
        AI-assisted wellness support — not a medical diagnostic system, therapist, or emergency service.
      </p>
    </div>
  )
}
