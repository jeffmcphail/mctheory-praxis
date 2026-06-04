/**
 * SessionHeader.jsx — session identity + config summary + live freshness.
 * Renders the kill-switch / config-gate STATE from the session's config_json
 * (visible, not editable). For live sessions, shows ws connection + last run.
 */
import { C, S, STATUS_COLOR } from '../shared/styles.js'
import { shortId, fmtTime } from '../shared/format.js'

export default function SessionHeader({ session, lastState, connected, onStop }) {
  if (!session) {
    return <div style={{ ...S.panel, padding: 14, color: C.muted }}>Loading session…</div>
  }
  let cfg = {}
  try { cfg = JSON.parse(session.config_json || '{}') } catch { /* ignore */ }
  const phase = lastState?.phase || lastState?.type
  const lastRun = lastState?.last_run_at
  const running = session.status === 'running'

  return (
    <div style={{ ...S.panel, padding: '12px 14px', display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 14, fontWeight: 700 }}>
            {session.mode === 'paper_replay' ? '⟲ Replay' : '● Live'} session
          </span>
          <span style={{ fontSize: 11, color: C.muted, fontFamily: 'monospace' }}>{shortId(session.session_id)}</span>
          <span style={S.badge(STATUS_COLOR[session.status] || C.sub)}>{session.status}</span>
          {phase && <span style={{ fontSize: 11, color: C.accent }}>{phase}</span>}
        </div>
        <div style={{ fontSize: 11, color: C.muted }}>
          notional ${cfg.max_notional_per_asset_usd ?? '—'}/asset
          {' · '}config gate {cfg.enforce_config_gate === false ? 'OFF' : 'ON'}
          {' · '}kill-switch {cfg.kill_switch_on ? '⛔ ON' : 'off'}
          {session.replay_start ? ` · window ${session.replay_start}→${session.replay_end}` : ''}
        </div>
      </div>

      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
        {session.mode === 'paper_live' && (
          <span style={{ fontSize: 11, color: connected ? C.green : C.muted }}>
            {connected ? '● live' : '○ ws'}{lastRun ? ` · ran ${fmtTime(lastRun).slice(11)}` : ''}
          </span>
        )}
        {running && (
          <button style={{ ...S.btn(C.red, false), padding: '5px 14px' }}
                  onClick={() => onStop(session.session_id)}>■ Stop</button>
        )}
      </div>
    </div>
  )
}
