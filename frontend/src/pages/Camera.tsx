import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../services/api'
import type { CrisisResources, CrisisResult, VisionSignal } from '../types'
import {
  ConfidenceBar,
  CrisisPanel,
  ErrorState,
  LatencyTable,
  Skeleton,
} from '../components/shared'

interface VisionResult {
  face: { detected: boolean; faces_found: number; selected_face: Record<string, number>; selection_policy: string }
  expression: { label: string; display_label: string; confidence: number; distribution: Record<string, number> }
  wellness_mapping: { emotion: string; confidence: number; distribution: Record<string, number> }
  explanation?: { method: string | null; note?: string; error?: string; peak_intensity?: number } | null
  latency_ms: Record<string, number>
  image_retention: string
  framing: string
}

export default function Camera() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [permission, setPermission] = useState<'idle' | 'granted' | 'denied'>('idle')
  const [result, setResult] = useState<VisionResult | null>(null)
  const [error, setError] = useState<{ message: string; code?: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [allowPrimaryFace, setAllowPrimaryFace] = useState(false)
  const [crisis, setCrisis] = useState<CrisisResult | null>(null)
  const [resources, setResources] = useState<CrisisResources | null>(null)

  useEffect(() => {
    api<CrisisResources>('/api/safety/resources', { auth: false }).then(setResources).catch(() => null)
  }, [])

  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream])

  const startCamera = useCallback(async () => {
    setError(null)
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      setStream(s)
      setPermission('granted')
      if (videoRef.current) {
        videoRef.current.srcObject = s
        await videoRef.current.play()
      }
    } catch {
      setPermission('denied')
      setError({
        message: 'Camera permission was denied or no camera is available. You can still upload an image below.',
        code: 'CAMERA_UNAVAILABLE',
      })
    }
  }, [])

  const stopCamera = useCallback(() => {
    stream?.getTracks().forEach((t) => t.stop())
    setStream(null)
  }, [stream])

  const analyzeBlob = useCallback(async (blob: Blob, faceIndex?: number) => {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', blob, 'capture.jpg')
      fd.append('explain', 'true')
      if (faceIndex !== undefined) fd.append('face_index', String(faceIndex))
      else if (allowPrimaryFace) fd.append('face_index', '0')
      const data = await api<VisionResult>('/api/vision/analyze', { method: 'POST', formData: fd })
      setResult(data)
    } catch (err) {
      const e = err as Error & { code?: string; suggestion?: string }
      setError({ message: `${e.message}${e.suggestion ? ` ${e.suggestion}` : ''}`, code: e.code })
    } finally {
      setBusy(false)
    }
  }, [allowPrimaryFace])

  const capture = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !video.videoWidth) return
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(video, 0, 0)
    canvas.toBlob((blob) => blob && analyzeBlob(blob, allowPrimaryFace ? 0 : undefined), 'image/jpeg', 0.92)
  }, [analyzeBlob, allowPrimaryFace])

  const onFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) analyzeBlob(file, allowPrimaryFace ? 0 : undefined)
  }, [analyzeBlob, allowPrimaryFace])

  return (
    <div>
      <h1 className="page-title">Facial-expression analysis</h1>
      <p className="page-sub">
        This module estimates the <strong>visible facial expression</strong> in a photo. It cannot
        determine your true internal emotional state. Images are processed transiently and never stored.
      </p>

      <div className="grid cols-2">
        <div className="card">
          <h3>Camera</h3>
          {permission === 'denied' && (
            <div className="alert warning">
              Camera unavailable. Check browser permissions, or use image upload instead.
            </div>
          )}
          <div className="camera-stage">
            <video ref={videoRef} playsInline muted />
            {!stream && <span style={{ color: '#888', position: 'absolute' }}>Camera off</span>}
          </div>
          <canvas ref={canvasRef} style={{ display: 'none' }} />
          <div className="row" style={{ marginTop: 12 }}>
            {!stream
              ? <button onClick={startCamera}>Enable camera</button>
              : <>
                  <button onClick={capture} disabled={busy}>Capture & analyze</button>
                  <button className="secondary" onClick={stopCamera}>Stop camera</button>
                </>}
          </div>
          <p className="muted small" style={{ marginTop: 8 }}>
            Camera permission: {permission === 'granted' ? 'granted ✓' : permission === 'denied' ? 'denied' : 'not requested'}
          </p>
        </div>

        <div className="card">
          <h3>Upload an image</h3>
          <label htmlFor="file">JPEG or PNG, single person (max 4 MB)</label>
          <input id="file" type="file" accept="image/jpeg,image/png,image/webp" onChange={onFile} />
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontWeight: 400, marginTop: 12 }}>
            <input
              type="checkbox" style={{ width: 'auto' }} checked={allowPrimaryFace}
              onChange={(e) => setAllowPrimaryFace(e.target.checked)}
            />
            If multiple faces are detected, use the primary (largest) face
          </label>
          <p className="muted small">
            Default behavior: multi-person images are rejected instead of silently picking someone.
          </p>
        </div>
      </div>

      {error && <ErrorState error={error.message} />}

      {busy && <div className="card"><Skeleton lines={4} /></div>}

      {result && (
        <div className="card">
          <div className="row between">
            <h3 style={{ margin: 0 }}>Result</h3>
            <span className="badge">{result.face.faces_found} face{result.face.faces_found === 1 ? '' : 's'} detected</span>
          </div>
          <p className="muted small">{result.framing}</p>

          <h4>Facial-expression estimate</h4>
          <div className="row" style={{ marginBottom: 8 }}>
            <strong>{result.expression.display_label}</strong>
            <span className="muted small">Confidence: {Math.round(result.expression.confidence * 100)}%</span>
          </div>
          <ConfidenceBar value={result.expression.confidence} label={result.expression.display_label} />

          <h4>Expression distribution</h4>
          {Object.entries(result.expression.distribution)
            .sort((a, b) => b[1] - a[1])
            .map(([label, p]) => (
              <ConfidenceBar key={label} value={p} label={label.charAt(0).toUpperCase() + label.slice(1)} />
            ))}

          {result.explanation?.method === 'grad_cam' && (
            <>
              <h4>Grad-CAM explanation</h4>
              <p className="muted small">
                Heatmap computed (peak intensity {result.explanation.peak_intensity?.toFixed(3)},
                mean {(result.explanation as any).mean_intensity?.toFixed(3) || 'N/A'}). Brighter regions contributed more
                to the predicted expression class.
              </p>
            </>
          )}
          {result.explanation?.error && (
            <p className="muted small">Visual explanation unavailable for this frame.</p>
          )}

          <h4>Wellness taxonomy mapping</h4>
          <ConfidenceBar value={result.wellness_mapping.confidence} label={result.wellness_mapping.emotion} />

          <h4>Performance</h4>
          <LatencyTable latencies={result.latency_ms} />
          <p className="disclaimer">Image retention: {result.image_retention}.</p>
        </div>
      )}

      {crisis && <CrisisPanel crisis={crisis} resources={resources} />}
    </div>
  )
}
