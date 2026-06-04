/**
 * SessionList.jsx — sortable-ish session ledger (newest first from the API).
 * Net P&L is read from the persisted pnl_rollup_json (replay) or shown — for
 * a live session that hasn't booked. View → Monitor; Stop for running ones.
 */
import { C, S, STATUS_COLOR } from '../shared/styles.js'
import { fmtTime, fmtUsd, shortId, pnlColor } from '../shared/format.js'

function rollupNet(s) {
  if (!s.pnl_rollup_json) return null
  try { return JSON.parse(s.pnl_rollup_json).net_pnl_usd } catch { return null }
}

export default function SessionList({ sessions, onStop, onOpen }) {
  return (
    <div style={{ ...S.panel, overflow: 'hidden' }}>
      <div style={{ ...S.sectionTitle, padding: '10px 14px', borderBottom: `1px solid ${C.border}` }}>
        Sessions <span style={{ color: C.muted, fontWeight: 400 }}>({sessions.length})</span>
      </div>
      <div style={{ maxHeight: 360, overflowY: 'auto' }}>
        {sessions.length === 0 ? (
          <div style={{ padding: 16, color: C.muted, fontSize: 12, textAlign: 'center' }}>
            No sessions yet — start one above.
          </div>
        ) : (
          <table style={S.table}>
            <thead>
              <tr>
                {['Created', 'Mode', 'Trigger', 'Status', 'Net P&L', 'Window', '', ''].map((h, i) => (
                  <th key={i} style={S.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sessions.map((s, i) => {
                const net = rollupNet(s)
                return (
                  <tr key={s.session_id} style={{ background: i % 2 ? C.panel2 : 'transparent' }}>
                    <td style={S.td}>{fmtTime(s.created_at)}</td>
                    <td style={S.td}>{s.mode === 'paper_replay' ? '⟲ replay' : '● live'}</td>
                    <td style={{ ...S.td, color: C.muted }}>{s.trigger_source}</td>
                    <td style={S.td}><span style={S.badge(STATUS_COLOR[s.status] || C.sub)}>{s.status}</span></td>
                    <td style={{ ...S.td, color: pnlColor(net), fontWeight: 700 }}>{net == null ? '—' : fmtUsd(net)}</td>
                    <td style={{ ...S.td, color: C.muted, fontSize: 11 }}>
                      {s.replay_start ? `${s.replay_start}→${s.replay_end}` : '—'}
                    </td>
                    <td style={S.td}>
                      <button style={{ ...S.btn(C.border, false), padding: '3px 10px', fontSize: 11 }}
                              onClick={() => onOpen(s.session_id)}>View</button>
                    </td>
                    <td style={S.td}>
                      {s.status === 'running' && (
                        <button style={{ ...S.btn(C.red, false), padding: '3px 10px', fontSize: 11 }}
                                onClick={() => onStop(s.session_id)}>Stop</button>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
