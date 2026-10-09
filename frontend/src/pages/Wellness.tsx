import { useCallback, useEffect, useState } from 'react'
import { api } from '../services/api'
import { useAuth } from '../hooks/useAuth'
import type { Intervention, ModelsStatus } from '../types'
import { ErrorState, InterventionCard, Skeleton } from '../components/shared'

interface CatalogResult {
  interventions: Intervention[]
}

export default function Wellness() {
  const { user } = useAuth()
  const [catalog, setCatalog] = useState<Intervention[] | null>(null)
  const [models, setModels] = useState<ModelsStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [feedbackMsg, setFeedbackMsg] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [c, m] = await Promise.all([
        api<CatalogResult>('/api/interventions/catalog'),
        api<ModelsStatus>('/api/models/status'),
      ])
      setCatalog(c.interventions)
      setModels(m)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const sendFeedback = useCallback(async (id: string, rating: string) => {
    try {
      const res = await api<{ intervention_stats: { total: number; mean_score: number } | null }>(
        '/api/interventions/feedback',
        { method: 'POST', body: { intervention_id: id, rating } },
      )
      const stats = res.intervention_stats
      setFeedbackMsg(
        `Thanks! Your feedback for ${id.replace(/_/g, ' ')} is stored${
          stats ? ` (mean score ${stats.mean_score > 0 ? '+' : ''}${stats.mean_score} over ${stats.total} rating${stats.total === 1 ? '' : 's'})` : ''
        }. Future recommendations adapt to it.`,
      )
    } catch (err) {
      setFeedbackMsg((err as Error).message)
    }
  }, [])

  return (
    <div>
      <h1 className="page-title">Wellness center</h1>
      <p className="page-sub">
        Personalized, CBT-inspired and mindfulness-oriented exercises. These are wellness
        activities — not medical treatment. Signed in as {user?.name}.
      </p>

      {error && <ErrorState error={error} onRetry={load} />}
      {feedbackMsg && <div className="alert success">{feedbackMsg}</div>}

      {loading ? (
        <div className="card"><Skeleton lines={5} /></div>
      ) : (
        <div className="grid cols-2">
          {catalog?.map((iv) => (
            <InterventionCard key={iv.id} intervention={iv} onFeedback={sendFeedback} />
          ))}
        </div>
      )}

      {models && (
        <div className="card" style={{ marginTop: 18 }}>
          <h3>Model status</h3>
          <p className="muted small">
            Device: <strong>{models.device.selected}</strong>
            {models.device.error ? ` (${models.device.error})` : models.device.note ? ` — ${models.device.note}` : ''}
            {models.device.torch_version ? ` · torch ${models.device.torch_version}` : ''}
          </p>
          <table className="data">
            <thead>
              <tr><th>Model</th><th>Available</th><th>Test accuracy</th><th>Macro F1</th></tr>
            </thead>
            <tbody>
              {Object.entries(models.models).map(([name, info]) => {
                const metrics = models.saved_metrics?.[name] as Record<string, unknown> | undefined
                return (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{info.available ? '✅ loaded' : (info.type ? '⚠ not trained yet' : '—')}</td>
                    <td>{metrics?.accuracy != null ? (metrics.accuracy as number).toFixed(4) : '—'}</td>
                    <td>{metrics?.macro_f1 != null ? (metrics.macro_f1 as number).toFixed(4) : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <p className="muted small" style={{ marginTop: 8 }}>
            Metrics shown are the saved results of the actual training runs (see ml/artifacts/*/metrics.json).
          </p>
        </div>
      )}
    </div>
  )
}
