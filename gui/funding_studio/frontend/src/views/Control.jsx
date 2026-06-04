/**
 * Control.jsx — start a session + the session list.
 */
import SessionKickoff from '../components/SessionKickoff.jsx'
import SessionList from '../components/SessionList.jsx'

export default function Control({ sessions, onStart, onStop, onOpen }) {
  return (
    <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>
      <SessionKickoff onStart={onStart} />
      <SessionList sessions={sessions} onStop={onStop} onOpen={onOpen} />
    </div>
  )
}
