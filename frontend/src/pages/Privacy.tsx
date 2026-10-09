import { useCallback, useEffect, useState } from 'react'
import { api, getToken } from '../services/api'
import { useAuth } from '../hooks/useAuth'
import type { Consent } from '../types'
import { ErrorState } from '../components/shared'

interface ConsentInfo {
  consent: Consent
  data_we_store: string[]
  data_we_never_store: string[]
}

export default function Privacy() {
  const { user, updateConsent, logout } = useAuth()
  const [info, setInfo] = useState<ConsentInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const load = useCallback(async () => {
    try {
      setInfo(await api<ConsentInfo>('/api/privacy/consent'))
    } catch (err) {
      setError((err as Error).message)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const toggle = useCallback(async (key: keyof Consent, value: boolean) => {
    setBusy(true)
    setNote(null)
    try {
      await updateConsent({ [key]: value })
      setNote('Consent updated.')
      await load()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }, [updateConsent, load])

  const exportData = useCallback(async () => {
    setNote(null)
    try {
      const res = await fetch('/api/privacy/export', {
        headers: { Authorization: `Bearer ${getToken()}` },
      })
      if (!res.ok) throw new Error('Export failed.')
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'my_wellness_data.json'
      a.click()
      URL.revokeObjectURL(url)
      setNote('Your data export has been downloaded.')
    } catch (err) {
      setError((err as Error).message)
    }
  }, [])

  const deleteData = useCallback(async () => {
    setBusy(true)
    try {
      await api('/api/privacy/data', { method: 'DELETE' })
      setNote('All your data has been permanently deleted. You have been signed out.')
      setTimeout(() => logout(), 1200)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }, [logout])

  const consent = info?.consent ?? user?.consent

  return (
    <div>
      <h1 className="page-title">Privacy & consent</h1>
      <p className="page-sub">You are in control of what this app stores about you.</p>

      {error && <ErrorState error={error} />}
      {note && <div className="alert success">{note}</div>}

      <div className="card">
        <h3>Consent controls</h3>
        <div className="toggle-row">
          <div>
            <strong>Save mood history</strong>
            <div className="muted small">Store check-in results so trends and forecasts can be computed.</div>
          </div>
          <label className="switch">
            <input
              type="checkbox" checked={!!consent?.save_mood_history} disabled={busy}
              onChange={(e) => toggle('save_mood_history', e.target.checked)}
            />
            <span className="slider" />
          </label>
        </div>
        <div className="toggle-row">
          <div>
            <strong>Camera permission</strong>
            <div className="muted small">Allow the camera page to request your webcam for expression analysis.</div>
          </div>
          <label className="switch">
            <input
              type="checkbox" checked={!!consent?.camera_access} disabled={busy}
              onChange={(e) => toggle('camera_access', e.target.checked)}
            />
            <span className="slider" />
          </label>
        </div>
        <div className="toggle-row">
          <div>
            <strong>Store captured images</strong>
            <div className="muted small">
              Off by default and unused: images are processed transiently and discarded — the backend
              never writes camera frames to disk or the database.
            </div>
          </div>
          <label className="switch">
            <input
              type="checkbox" checked={!!consent?.store_images} disabled
              onChange={() => undefined}
            />
            <span className="slider" />
          </label>
        </div>
        <p className="muted small" style={{ marginTop: 10 }}>
          Consent updated: {consent?.consent_updated_at ? new Date(consent.consent_updated_at).toLocaleString() : '—'}
        </p>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h3>What we store</h3>
          <ul className="muted small" style={{ paddingLeft: 18 }}>
            {info?.data_we_store.map((s) => <li key={s}>{s}</li>)}
          </ul>
          <h4>What we never store</h4>
          <ul className="muted small" style={{ paddingLeft: 18 }}>
            {info?.data_we_never_store.map((s) => <li key={s}>{s}</li>)}
          </ul>
        </div>
        <div className="card">
          <h3>Your data</h3>
          <div className="stack">
            <button className="secondary" onClick={exportData}>Export my data (JSON)</button>
            {!confirmDelete ? (
              <button className="danger" onClick={() => setConfirmDelete(true)}>Delete my data…</button>
            ) : (
              <>
                <div className="alert error">
                  This permanently deletes your account and every record. It cannot be undone.
                </div>
                <div className="row">
                  <button className="danger" disabled={busy} onClick={deleteData}>Yes, delete everything</button>
                  <button className="secondary" onClick={() => setConfirmDelete(false)}>Cancel</button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      <p className="footer-note">
        Data flows are documented in docs/PRIVACY.md. Raw conversation text is never written to logs.
      </p>
    </div>
  )
}
