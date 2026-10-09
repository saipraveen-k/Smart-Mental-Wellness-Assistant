import { useCallback, useEffect, useState } from 'react'
import { api } from '../services/api'
import type { CrisisResources, CrisisResult, Distribution, FusionInfo, StressorResult, TextSignal, VisionSignal } from '../types'
import {
  ConfidenceBar,
  CrisisPanel,
  EmotionBadge,
  ErrorState,
  LatencyTable,
  Skeleton,
} from '../components/shared'

interface MultimodalResult {
  emotion: string
  confidence: number
  distribution: Distribution
  text_signal: TextSignal | null
  vision_signal: VisionSignal | null
  fusion: FusionInfo
  latency_ms: Record<string, number>
  disclaimer: string
}

export default function Multimodal() {
  const [text, setText] = useState('')
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null)
  const [result, setResult] = useState<MultimodalResult | null>(null)
  const [crisis, setCrisis] = useState<CrisisResult | null>(null)
  const [resources, setResources] = useState<CrisisResources | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api<CrisisResources>('/api/safety/resources', { auth: false }).then(setResources).catch(() => null)
  }, [])

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => setImageDataUrl(String(reader.result))
    reader.readAsDataURL(file)
  }

  const analyze = useCallback(async () => {
    if (!text.trim() && !imageDataUrl) return
    setBusy(true)
    setError(null)
    setResult(null)
    setCrisis(null)
    try {
      const body: Record<string, unknown> = { explain: true }
      if (text.trim()) body.text = text.trim()
      if (imageDataUrl) body.image_base64 = imageDataUrl
      const data = await api<MultimodalResult>('/api/multimodal/analyze', { method: 'POST', body })
      setResult(data)
      if (text.trim()) {
        try {
          const c = await api<CrisisResult>('/api/crisis/analyze', { method: 'POST', body: { text: text.trim() } })
          setCrisis(c)
        } catch { /* crisis endpoint requires auth token; non-fatal for this page */ }
      }
    } catch (err) {
      const e = err as Error & { suggestion?: string }
      setError(`${e.message}${e.suggestion ? ` ${e.suggestion}` : ''}`)
    } finally {
      setBusy(false)
    }
  }, [text, imageDataUrl])

  return (
    <div>
      <h1 className="page-title">Multimodal analysis</h1>
      <p className="page-sub">
        Combine what you write with a facial-expression estimate. Missing modalities are never
        fabricated — single-modality analysis is clearly labelled.
      </p>

      <div className="grid cols-2">
        <div className="card">
          <h3>Text</h3>
          <label htmlFor="mm-text">What's on your mind?</label>
          <textarea id="mm-text" value={text} onChange={(e) => setText(e.target.value)} maxLength={4000} />
        </div>
        <div className="card">
          <h3>Face (optional)</h3>
          <label htmlFor="mm-image">Upload a photo of a single face</label>
          <input id="mm-image" type="file" accept="image/jpeg,image/png,image/webp" onChange={onFile} onChangeCapture={() => undefined} />
          {imageDataUrl && (
            <div className="camera-stage" style={{ marginTop: 12, aspectRatio: '16/10' }}>
              <img src={imageDataUrl} alt="Selected for facial-expression analysis" />
            </div>
          )}
          <div className="row" style={{ marginTop: 12 }}>
            <button onClick={analyze} disabled={busy || (!text.trim() && !imageDataUrl)}>
              {busy ? 'Analyzing…' : 'Analyze'}
            </button>
            {imageDataUrl && <button className="secondary" onClick={() => setImageDataUrl(null)}>Remove image</button>}
          </div>
        </div>
      </div>

      {error && <ErrorState error={error} />}
      {crisis && crisis.level !== 'LOW' && <CrisisPanel crisis={crisis} resources={resources} />}
      {busy && <div className="card"><Skeleton lines={5} /></div>}

      {result && (
        <>
          <div className="grid cols-3">
            <div className="card">
              <div className="kpi-label">Text</div>
              {result.text_signal && 'distribution' in result.text_signal ? (
                <>
                  <div style={{ margin: '10px 0' }}><EmotionBadge emotion={result.text_signal.emotion} /></div>
                  <ConfidenceBar value={result.text_signal.confidence} label={`${result.text_signal.emotion} ${Math.round(result.text_signal.confidence * 100)}%`} />
                </>
              ) : (
                <p className="muted small">{(result.text_signal as any)?.error ? 'Text modality failed.' : 'Not provided.'}</p>
              )}
            </div>
            <div className="card">
              <div className="kpi-label">Face</div>
              {result.vision_signal?.distribution ? (
                <>
                  <div style={{ margin: '10px 0' }}><EmotionBadge emotion={result.vision_signal.emotion!} /></div>
                  <ConfidenceBar value={result.vision_signal.confidence!} label={`${result.vision_signal.expression?.display_label} → ${result.vision_signal.emotion}`} />
                </>
              ) : (
                <p className="muted small">{result.vision_signal?.error ? `Vision failed: ${result.vision_signal.error}` : 'Not provided.'}</p>
              )}
            </div>
            <div className="card" style={{ borderColor: 'var(--primary)' }}>
              <div className="kpi-label">Fused result</div>
              <div style={{ margin: '10px 0' }}><EmotionBadge emotion={result.emotion} /></div>
              <ConfidenceBar value={result.confidence} label={`${result.emotion} ${Math.round(result.confidence * 100)}%`} />
            </div>
          </div>

          <div className="card">
            <h3>Fusion details</h3>
            <p className="muted small">
              Mode: <strong>{result.fusion.mode.replace('_', ' ')}</strong> · text weight{' '}
              {Math.round(result.fusion.text_weight * 100)}% · vision weight{' '}
              {Math.round(result.fusion.vision_weight * 100)}%
            </p>
            <p className="muted small">Formula: {result.fusion.formula}</p>
            <h4>Full distribution</h4>
            {Object.entries((result as any).distribution || {})
              .sort((a: any, b: any) => Number(b[1]) - Number(a[1]))
              .map(([k, v]) => <ConfidenceBar key={k} value={Number(v)} label={k} />)}
            <h4>Performance</h4>
            <LatencyTable latencies={result.latency_ms} />
            <p className="disclaimer">{result.disclaimer}</p>
          </div>
        </>
      )}
    </div>
  )
}
