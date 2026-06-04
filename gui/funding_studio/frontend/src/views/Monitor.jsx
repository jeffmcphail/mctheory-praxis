/**
 * Monitor.jsx — always-on regime panel (SignalFeed + AlertsPanel) + a
 * per-session panel that fills in when a session is selected. Useful with NO
 * session running, which is the common case in the current sit-out regime.
 */
import { C } from '../shared/styles.js'
import SignalFeed from '../components/SignalFeed.jsx'
import AlertsPanel from '../components/AlertsPanel.jsx'
import SessionPanel from '../components/SessionPanel.jsx'

export default function Monitor({ signals, alerts, sessionId, onStop }) {
  return (
    <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div style={{ flex: '2 1 520px', minWidth: 320 }}><SignalFeed signals={signals} /></div>
        <div style={{ flex: '1 1 240px', minWidth: 220 }}><AlertsPanel alerts={alerts} /></div>
      </div>
      {sessionId
        ? <SessionPanel sessionId={sessionId} onStop={onStop} />
        : <NoSession />}
    </div>
  )
}

function NoSession() {
  return (
    <div style={{
      border: `1px dashed ${C.border}`, borderRadius: 6, padding: '28px 16px',
      textAlign: 'center', color: C.muted,
    }}>
      <div style={{ fontSize: 28, opacity: 0.3, marginBottom: 8 }}>◉</div>
      <div style={{ fontSize: 13 }}>No session selected.</div>
      <div style={{ fontSize: 12, marginTop: 4 }}>
        The regime above updates live. Start a session (Control) or pick one to see its bookings.
      </div>
    </div>
  )
}
