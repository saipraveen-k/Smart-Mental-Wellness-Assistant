import { useCallback, useEffect, useState } from 'react'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line,
  LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api } from '../services/api'
import type { ForecastResult, TrendsResult } from '../types'
import { EmptyState, ErrorState, Skeleton } from '../components/shared'

const EMOTION_COLORS: Record<string, string> = {
  Joy: '#16a34a', Sadness: '#0ea5e9', Anger: '#dc2626', Anxiety: '#d97706', Neutral: '#94a3b8',
}

export default function MoodDashboard() {
  const [trends, setTrends] = useState<TrendsResult | null>(null)
  const [forecast, setForecast] = useState<ForecastResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [t, f] = await Promise.all([
        api<TrendsResult>('/api/mood/trends'),
        api<ForecastResult>('/api/mood/forecast'),
      ])
      setTrends(t)
      setForecast(f)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  if (loading) return <div className="card"><Skeleton lines={6} /></div>
  if (error) return <ErrorState error={error} onRetry={load} />

  if (!trends?.available) {
    return (
      <div>
        <h1 className="page-title">Mood dashboard</h1>
        <EmptyState message={trends?.message ?? 'No mood history yet. Start your first check-in.'} />
      </div>
    )
  }

  const emotionData = Object.entries(trends.emotion_distribution || {}).map(
    ([emotion, count]) => ({ emotion, count }),
  )
  const stressorData = Object.entries(trends.stressor_distribution || {}).map(
    ([stressor, count]) => ({ stressor, count }),
  )
  const forecastData = forecast?.available && forecast.forecast
    ? forecast.forecast.map((f) => ({
        date: f.date.slice(5),
        valence: f.forecast_valence,
        stress: f.potential_stress_period,
      }))
    : []

  const seriesData = [
    ...trends.daily_series!.map((d) => ({ date: d.date.slice(5), valence: d.valence, type: 'history' as const })),
    ...forecastData.map((f) => ({ date: f.date, valence: f.valence, type: 'forecast' as const })),
  ]

  return (
    <div>
      <h1 className="page-title">Mood dashboard</h1>
      <p className="page-sub">
        All charts are computed from your own {trends.records} stored check-in{trends.records === 1 ? '' : 's'}.
      </p>

      <div className="grid cols-2">
        <div className="card">
          <h3>Daily mood valence (history + forecast)</h3>
          <div className="chart-box">
            <ResponsiveContainer>
              <LineChart data={seriesData} margin={{ top: 8, right: 16, left: -14, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={24} />
                <YAxis domain={[-1, 1]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <ReferenceLine y={0} strokeDasharray="3 3" />
                <Line data={trends.daily_series} dataKey="valence" name="History" stroke="var(--primary)" dot strokeWidth={2} />
                {forecast?.available && (
                  <Line data={forecastData} dataKey="valence" name="Forecast" stroke="var(--accent)" strokeDasharray="6 4" dot={false} strokeWidth={2} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="muted small">
            Valence mapping: Joy +1.0×confidence, Sadness −0.6, Anxiety −0.7, Anger −0.5, Neutral 0.
          </p>
        </div>

        <div className="card">
          <h3>Emotion distribution</h3>
          <div className="chart-box">
            <ResponsiveContainer>
              <BarChart data={emotionData} margin={{ top: 8, right: 16, left: -14, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="emotion" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                  {emotionData.map((d) => <Cell key={d.emotion} fill={EMOTION_COLORS[d.emotion] || '#94a3b8'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h3>Stressor frequency</h3>
          {stressorData.length === 0 ? (
            <p className="muted small">No stressors recorded yet.</p>
          ) : (
            <div className="chart-box">
              <ResponsiveContainer>
                <BarChart data={stressorData} layout="vertical" margin={{ top: 8, right: 16, left: 24, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="stressor" width={90} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="var(--accent)" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="card">
          <h3>Weekly comparison</h3>
          <table className="data">
            <tbody>
              <tr><td>Recent 7-day mean valence</td><td style={{ textAlign: 'right' }}>{trends.weekly?.recent_7d_mean_valence ?? '—'}</td></tr>
              <tr><td>Previous 7-day mean valence</td><td style={{ textAlign: 'right' }}>{trends.weekly?.previous_7d_mean_valence ?? '—'}</td></tr>
              <tr><td>Check-ins (recent / previous)</td><td style={{ textAlign: 'right' }}>{trends.weekly?.recent_checkins} / {trends.weekly?.previous_checkins}</td></tr>
              <tr><td>Trend direction</td><td style={{ textAlign: 'right' }}>{trends.trend?.direction ?? '—'}</td></tr>
            </tbody>
          </table>
          {trends.trend?.recurring_patterns && (
            <p className="muted small" style={{ marginTop: 10 }}>
              Recurring pattern: weaker mood on {trends.trend.recurring_patterns.weakest_weekdays.join(', ')}.
            </p>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Forecast</h3>
        {forecast?.available ? (
          <>
            <p className="muted small">
              Method: <strong>{forecast.method}</strong> over {forecast.observations} daily observations,
              {forecast.horizon_days}-day horizon. Forecasts are statistical projections of your own
              history — not clinical predictions.
            </p>
            <table className="data">
              <thead>
                <tr><th>Date</th><th>Forecast valence</th><th>Potential stress period</th></tr>
              </thead>
              <tbody>
                {forecast.forecast!.map((f) => (
                  <tr key={f.date}>
                    <td>{f.date}</td>
                    <td>{f.forecast_valence.toFixed(3)}</td>
                    <td>{f.potential_stress_period ? '⚠ yes' : 'no'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <p className="muted">
            {forecast?.message ?? 'Not enough historical data for forecasting.'}
          </p>
        )}
      </div>

      <div className="card">
        <h3>Intervention effectiveness (your feedback)</h3>
        {trends.intervention_effectiveness && trends.intervention_effectiveness.length > 0 ? (
          <table className="data">
            <thead>
              <tr><th>Intervention</th><th>Ratings</th><th>Mean score</th></tr>
            </thead>
            <tbody>
              {trends.intervention_effectiveness.map((eff) => (
                <tr key={eff.intervention_id}>
                  <td>{eff.intervention_id}</td>
                  <td>{Object.entries(eff.ratings).map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`).join(' · ')}</td>
                  <td>{eff.mean_score > 0 ? '+' : ''}{eff.mean_score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted small">No feedback recorded yet — rate an exercise in the Wellness Center.</p>
        )}
      </div>

      <p className="footer-note">All statistics are computed live from your stored records — nothing is fabricated.</p>
    </div>
  )
}
