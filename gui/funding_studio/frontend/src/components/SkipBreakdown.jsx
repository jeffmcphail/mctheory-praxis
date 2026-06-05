/**
 * SkipBreakdown.jsx — deep skip-reason breakdown for a session. Aggregates the
 * skip decisions by normalized reason. Pure render of paper_trades rows.
 *
 * Note: GUI replay sessions gate alerts upstream (monitor_gate), so the
 * executor rarely skips -> this commonly shows the honest empty state; it
 * populates for live sessions / pre-fix harnesses where skips occur.
 */
import { C } from '../shared/styles.js'

export default function SkipBreakdown({ trades }) {
  const skips = (trades || []).filter((t) => t.decision === 'skip')
  if (skips.length === 0) {
    return (
      <div style={{ padding: 14, color: C.muted, fontSize: 12 }}>
        No skips — every gated alert in this session was entered.
      </div>
    )
  }
  const counts = {}
  for (const t of skips) {
    const reason = (t.skip_reason || 'unknown').split(/[(;]/)[0].trim()
    counts[reason] = (counts[reason] || 0) + 1
  }
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1])
  const max = Math.max(...rows.map((r) => r[1]))
  return (
    <div style={{ padding: '8px 0' }}>
      {rows.map(([reason, n]) => (
        <div key={reason} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '4px 14px' }}>
          <div style={{ width: 200, fontSize: 12, color: C.sub, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{reason}</div>
          <div style={{ flex: 1, height: 14, background: C.panel2, borderRadius: 3, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${(n / max) * 100}%`, background: C.amber, borderRadius: 3 }} />
          </div>
          <div style={{ width: 30, textAlign: 'right', fontSize: 12, fontWeight: 600 }}>{n}</div>
        </div>
      ))}
    </div>
  )
}
