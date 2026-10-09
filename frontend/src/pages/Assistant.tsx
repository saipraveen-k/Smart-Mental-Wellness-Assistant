import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../services/api'
import type { ChatResult, CrisisResources } from '../types'
import {
  ConfidenceBar,
  CrisisPanel,
  EmotionBadge,
  ErrorState,
  InterventionList,
  LatencyTable,
  Skeleton,
  WellnessBadge,
} from '../components/shared'

interface ChatEntry {
  role: 'user' | 'assistant'
  text: string
  result?: ChatResult
}

export default function Assistant() {
  const [entries, setEntries] = useState<ChatEntry[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saveCheckin, setSaveCheckin] = useState(true)
  const [resources, setResources] = useState<CrisisResources | null>(null)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api<CrisisResources>('/api/safety/resources', { auth: false }).then(setResources).catch(() => setResources(null))
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' })
  }, [entries])

  const send = useCallback(async () => {
    const text = input.trim()
    if (!text || busy) return
    setBusy(true)
    setError(null)
    setInput('')
    setEntries((prev) => [...prev, { role: 'user', text }])
    try {
      const result = await api<ChatResult>('/api/chat/message', {
        method: 'POST',
        body: { text, save_checkin: saveCheckin },
      })
      setEntries((prev) => [...prev, { role: 'assistant', text: result.interventions.headline, result }])
    } catch (err) {
      const e = err as Error & { suggestion?: string }
      setError(`${e.message}${e.suggestion ? ` ${e.suggestion}` : ''}`)
    } finally {
      setBusy(false)
    }
  }, [input, busy, saveCheckin])

  const sendFeedback = useCallback(async (id: string, rating: string) => {
    await api('/api/interventions/feedback', { method: 'POST', body: { intervention_id: id, rating } })
    setEntries((prev) => [...prev, {
      role: 'assistant',
      text: `Thanks — your feedback for this exercise (${rating.replace(/_/g, ' ')}) is stored and will adjust future recommendations.`,
    }])
  }, [])

  return (
    <div>
      <h1 className="page-title">AI Assistant</h1>
      <p className="page-sub">
        Check in with a message. The assistant screens for crisis risk first, then analyzes emotion,
        stressors, and suggests support.
      </p>

      {error && <ErrorState error={error} />}

      <div className="chat-log" ref={logRef}>
        {entries.length === 0 && (
          <div className="bubble assistant">
            Hi — what's on your mind? You can write freely. Your message is analyzed for emotion,
            stressors and wellbeing signals; it is never stored as raw text.
          </div>
        )}
        {entries.map((entry, i) => (
          <div key={i} className={`bubble ${entry.role}`}>
            <div style={{ whiteSpace: 'pre-wrap' }}>{entry.text}</div>
            {entry.result && <ResultBlock result={entry.result} resources={resources} onFeedback={sendFeedback} />}
          </div>
        ))}
        {busy && (
          <div className="bubble assistant"><Skeleton lines={3} /></div>
        )}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <label htmlFor="chat-input">What's on your mind?</label>
        <textarea
          id="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
          placeholder="Write anything — how your day went, what's stressing you…"
          maxLength={4000}
        />
        <div className="row between" style={{ marginTop: 10 }}>
          <label style={{ margin: 0, display: 'flex', gap: 8, alignItems: 'center', fontWeight: 400 }}>
            <input
              type="checkbox" checked={saveCheckin} onChange={(e) => setSaveCheckin(e.target.checked)}
              style={{ width: 'auto' }}
            />
            Save this check-in to my mood history
          </label>
          <button onClick={send} disabled={busy || !input.trim()}>
            {busy ? 'Analyzing…' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}

function ResultBlock({
  result,
  resources,
  onFeedback,
}: {
  result: ChatResult
  resources: CrisisResources | null
  onFeedback: (id: string, rating: string) => Promise<void>
}) {
  return (
    <div style={{ marginTop: 10 }}>
      {result.crisis && result.crisis.level !== 'LOW' && (
        <CrisisPanel crisis={result.crisis} resources={resources} />
      )}

      <div style={{ marginBottom: 8 }} className="row">
        <span className="small muted">Estimated emotion:</span>
        <EmotionBadge emotion={result.emotion} />
        <span className="small muted">{Math.round(result.confidence * 100)}%</span>
        <span className="small muted">·</span>
        <WellnessBadge state={result.wellness.state} />
      </div>

      <ConfidenceBar value={result.confidence} label={`AI-estimated ${result.emotion}`} />

      {result.text_signal?.explanation?.contributing_tokens && result.text_signal.explanation.contributing_tokens.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="small muted">Signals contributing to this prediction:</div>
          <div>
            {result.text_signal.explanation.contributing_tokens.map((t, i) => (
              <span key={i} className="token-chip" title={`attribution ${t.score}`}>
                {t.token}
              </span>
            ))}
          </div>
        </div>
      )}

      {result.stressor && (
        <p className="small" style={{ marginTop: 8 }}>
          Stressor: <strong>{result.stressor.category ?? 'none detected'}</strong>
          {result.stressor.category && (
            <> · intensity {result.stressor.intensity} · cues: {result.stressor.matched_cues.map((c) => c.cue).join(', ')}</>
          )}
        </p>
      )}

      <div style={{ marginTop: 12 }}>
        <InterventionList result={result.interventions} />
        <div className="row" style={{ marginTop: 6 }}>
          {result.interventions.interventions.slice(0, 1).map((iv) => (
            <button key={iv.id} className="small secondary" onClick={() => onFeedback(iv.id, 'much_better')}>
              👍 This helped
            </button>
          ))}
        </div>
      </div>

      <details style={{ marginTop: 8 }}>
        <summary className="small muted">Performance & technical details</summary>
        <LatencyTable latencies={result.latency_ms} />
        <p className="small muted">
          Fusion: {result.fusion.mode} (text {Math.round(result.fusion.text_weight * 100)}% ·
          vision {Math.round(result.fusion.vision_weight * 100)}%)
        </p>
        {result.saved && (
          <p className="small muted">
            {result.saved.saved
              ? 'Check-in saved to your mood history.'
              : `Check-in not saved: ${result.saved.message ?? result.saved.reason}`}
          </p>
        )}
      </details>
      <p className="small muted" style={{ marginTop: 6 }}>{result.disclaimer}</p>
    </div>
  )
}
