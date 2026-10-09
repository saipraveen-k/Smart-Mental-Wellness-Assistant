import type { CrisisResources, CrisisResult, Intervention, InterventionResult } from '../types'

export function ConfidenceBar({ value, label }: { value: number; label?: string }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div>
      {label && <div className="row between small muted"><span>{label}</span><span>{pct}%</span></div>}
      {!label && <div className="small muted">{pct}%</div>}
      <div className="meter" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function EmotionBadge({ emotion }: { emotion: string }) {
  return <span className={`badge ${emotion.toLowerCase()}`}>{emotion}</span>
}

export function WellnessBadge({ state }: { state: string }) {
  const cls = state === 'STABLE' ? 'stable' : state === 'NEEDS_ATTENTION' ? 'attention' : 'high_concern'
  const label = state.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase())
  return <span className={`badge ${cls}`}>{label}</span>
}

export function Skeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="stack" aria-busy>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="skeleton" style={{ width: `${90 - i * 12}%` }} />
      ))}
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="alert error" role="alert">
      <strong>Something went wrong.</strong> {error}
      {onRetry && (
        <div style={{ marginTop: 8 }}>
          <button className="small secondary" onClick={onRetry}>Try again</button>
        </div>
      )}
    </div>
  )
}

export function EmptyState({ message, action }: { message: string; action?: React.ReactNode }) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: 40 }}>
      <p className="muted" style={{ marginBottom: action ? 14 : 0 }}>{message}</p>
      {action}
    </div>
  )
}

export function CrisisPanel({ crisis, resources }: { crisis: CrisisResult; resources?: CrisisResources | null }) {
  if (crisis.level === 'LOW') return null
  return (
    <div className="crisis-panel" role="alert">
      <h2>{crisis.level === 'HIGH' ? 'Immediate support matters right now' : 'This may be a serious moment'}</h2>
      <p>{resources?.message ?? "We're concerned that you may be going through a serious situation. You don't have to handle this alone."}</p>
      <p className="small">Screening signal: <strong>{crisis.level}</strong> — this is an AI screening estimate, not a diagnosis or a guarantee.</p>
      {resources?.resources?.map((r) => (
        <div className="crisis-resource" key={r.name}>
          <strong>{r.name}</strong>
          <div className="small muted">{r.description}</div>
          <div>{r.type === 'directory' ? <a href={r.contact} target="_blank" rel="noreferrer">{r.contact}</a> : <strong>{r.contact}</strong>}</div>
          <div className="small muted">Source: <a href={r.source} target="_blank" rel="noreferrer">{r.source}</a></div>
        </div>
      ))}
      {resources?.international?.map((r) => (
        <div className="crisis-resource" key={r.name}>
          <strong>{r.name}</strong>
          <div className="small muted">{r.description}</div>
          <a href={r.contact} target="_blank" rel="noreferrer">{r.contact}</a>
        </div>
      ))}
      {resources?.emergency && <p className="small" style={{ marginTop: 12 }}>{resources.emergency.description}</p>}
      {resources?.note && <p className="small muted">{resources.note}</p>}
    </div>
  )
}

export function InterventionList({ result }: { result: InterventionResult }) {
  return (
    <div className="stack">
      <h3>{result.headline}</h3>
      {result.interventions.map((iv: Intervention) => (
        <InterventionCard key={iv.id} intervention={iv} />
      ))}
      <p className="muted small">{result.learning_note}</p>
      <p className="disclaimer">{result.disclaimer}</p>
    </div>
  )
}

export function InterventionCard({
  intervention,
  onFeedback,
}: {
  intervention: Intervention
  onFeedback?: (id: string, rating: string) => Promise<void>
}) {
  return (
    <div className="card" style={{ marginBottom: 0 }}>
      <div className="row between">
        <h3 style={{ margin: 0 }}>{intervention.name}</h3>
        <span className="badge">{intervention.duration_minutes > 0 ? `${intervention.duration_minutes} min` : 'support'}</span>
      </div>
      <p className="muted">{intervention.description}</p>
      <ol style={{ margin: '8px 0 8px 18px', padding: 0, fontSize: 14 }}>
        {intervention.steps.map((s, i) => <li key={i} style={{ marginBottom: 4 }}>{s}</li>)}
      </ol>
      <div className="small muted">Why recommended: {intervention.rationale.join(' ')}</div>
      <div className="small muted">Priority {Math.round(intervention.priority * 100) / 100}</div>
      {onFeedback && (
        <div className="row" style={{ marginTop: 10 }}>
          <span className="small muted">Did this help?</span>
          {['much_better', 'a_little_better', 'no_change', 'worse'].map((r) => (
            <button key={r} className="small secondary" onClick={() => onFeedback(intervention.id, r)}>
              {r.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      )}
      <div className="disclaimer">{intervention.disclaimer}</div>
    </div>
  )
}

export function LatencyTable({ latencies }: { latencies: Record<string, number> }) {
  return (
    <table className="data">
      <tbody>
        {Object.entries(latencies).map(([k, v]) => (
          <tr key={k}>
            <td style={{ textTransform: 'capitalize' }}>{k.replace(/_/g, ' ').replace(/ms$/, '')}</td>
            <td style={{ textAlign: 'right' }}>{v.toFixed(1)} ms</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
