/**
 * App.jsx — Funding Studio (Engine 7 control), Cycle 54b.
 * Two views: Control (start + list sessions) and Monitor (always-on regime
 * panel + per-session panel). The frontend only calls the backend API; it
 * holds no trade logic.
 */
import { useState, useEffect, useCallback } from 'react'
import { getJSON, postJSON } from './api.js'
import { C } from './shared/styles.js'
import Control from './views/Control.jsx'
import Monitor from './views/Monitor.jsx'

export default function App() {
  const [view, setView] = useState('control')
  const [selectedId, setSelectedId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [signals, setSignals] = useState([])
  const [alerts, setAlerts] = useState([])
  const [health, setHealth] = useState(null)
  const [err, setErr] = useState(null)

  const refreshSessions = useCallback(async () => {
    try { setSessions(await getJSON('/api/sessions')) }
    catch (e) { setErr(String(e.message || e)) }
  }, [])

  const refreshRegime = useCallback(async () => {
    try {
      const [sig, al] = await Promise.all([
        getJSON('/api/signals?limit=60'),
        getJSON('/api/alerts?limit=30'),
      ])
      setSignals(sig)
      setAlerts(al)
      try { setHealth(await getJSON('/api/health')) } catch { /* non-fatal */ }
    } catch (e) { setErr(String(e.message || e)) }
  }, [])

  useEffect(() => {
    refreshSessions(); refreshRegime()
    const a = setInterval(refreshSessions, 5000)
    const b = setInterval(refreshRegime, 30000)
    return () => { clearInterval(a); clearInterval(b) }
  }, [refreshSessions, refreshRegime])

  const startSession = useCallback(async (body) => {
    setErr(null)
    try {
      const { session_id } = await postJSON('/api/sessions', body)
      setSelectedId(session_id)
      setView('monitor')
      refreshSessions()
    } catch (e) { setErr(String(e.message || e)) }
  }, [refreshSessions])

  const stopSession = useCallback(async (id) => {
    try { await postJSON(`/api/sessions/${id}/stop`, {}); refreshSessions() }
    catch (e) { setErr(String(e.message || e)) }
  }, [refreshSessions])

  const openSession = useCallback((id) => { setSelectedId(id); setView('monitor') }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 16, padding: '10px 16px',
        background: C.panel, borderBottom: `1px solid ${C.border}`, flexShrink: 0,
      }}>
        <span style={{ fontSize: 16, fontWeight: 700, color: C.accent }}>
          ◈ Funding Studio
          <span style={{ color: C.muted, fontSize: 12, fontWeight: 400 }}> · Engine 7 control</span>
        </span>
        <div style={{ display: 'flex', gap: 4, marginLeft: 12 }}>
          {[['control', '▶ Control'], ['monitor', '◉ Monitor']].map(([id, label]) => (
            <button key={id} onClick={() => setView(id)} style={navBtn(view === id)}>{label}</button>
          ))}
        </div>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: C.muted }}>
          paper · backend :8002{health?.checked_at_utc ? ` · health ${String(health.checked_at_utc).slice(11, 19)}Z` : ''}
        </span>
      </div>

      {err && (
        <div style={{ padding: '4px 16px', background: 'rgba(239,83,80,0.12)', color: C.red, fontSize: 11, flexShrink: 0 }}>
          ⚠ {err}
        </div>
      )}

      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        {view === 'control'
          ? <Control sessions={sessions} onStart={startSession} onStop={stopSession} onOpen={openSession} />
          : <Monitor signals={signals} alerts={alerts} sessionId={selectedId} onStop={stopSession} />}
      </div>
    </div>
  )
}

function navBtn(active) {
  return {
    padding: '6px 14px', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 12,
    fontWeight: active ? 700 : 400,
    background: active ? C.panel2 : 'transparent',
    color: active ? C.accent : C.muted,
  }
}
