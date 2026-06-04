/**
 * SignalFeed.jsx — the GLOBAL regime view (not session-scoped): latest
 * funding_signals row per asset, glanceable "how close to the gate + is
 * funding favorable." NULL min_pct_positive renders as em-dash (never crash).
 */
import { C, S } from '../shared/styles.js'
import { fmtNum, fmtAnnPct, fmtTime } from '../shared/format.js'

export default function SignalFeed({ signals }) {
  const latest = {}
  let window = null
  for (const r of signals || []) {
    if (!(r.asset in latest)) latest[r.asset] = r
    if (!window || r.datetime > window) window = r.datetime
  }
  const rows = Object.values(latest).sort((a, b) => a.asset.localeCompare(b.asset))

  return (
    <div style={{ ...S.panel, overflow: 'hidden' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', borderBottom: `1px solid ${C.border}` }}>
        <span style={S.sectionTitle}>Signal feed — regime</span>
        <span style={{ fontSize: 11, color: C.muted }}>{window ? `window ${fmtTime(window)} UTC` : 'no signals'}</span>
      </div>

      {rows.length === 0 ? (
        <div style={{ padding: 16, color: C.muted, fontSize: 12, textAlign: 'center' }}>No signals yet.</div>
      ) : (
        <table style={S.table}>
          <thead>
            <tr>{['Asset', 'P(profit)', 'vs gate', 'Funding ann', '% positive', 'State'].map((h) => <th key={h} style={S.th}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const gate = r.gate_threshold ?? 0.70
              const frac = Math.max(0, Math.min(1, (r.p_profitable || 0) / gate))
              const fired = r.above_gate === 1
              const pctPos = r.pct_positive
              const minPct = r.min_pct_positive
              const pctOk = pctPos != null && minPct != null && pctPos >= minPct
              return (
                <tr key={r.asset} style={{ background: i % 2 ? C.panel2 : 'transparent' }}>
                  <td style={{ ...S.td, fontWeight: 700 }}>{r.asset}</td>
                  <td style={{ ...S.td, color: fired ? C.green : C.text, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                    {fmtNum(r.p_profitable, 3)}
                  </td>
                  <td style={S.td}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ width: 80, height: 5, background: C.border, borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${frac * 100}%`, background: fired ? C.green : C.amber, borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 10, color: C.muted }}>{(frac * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td style={{ ...S.td, color: (r.ann_rate ?? 0) >= 0 ? C.green : C.red, fontVariantNumeric: 'tabular-nums' }}>
                    {fmtAnnPct(r.ann_rate)}
                  </td>
                  <td style={{ ...S.td, color: pctOk ? C.green : C.sub, fontVariantNumeric: 'tabular-nums' }}>
                    {fmtNum(pctPos, 3)} <span style={{ color: C.muted, fontSize: 10 }}>/ {minPct == null ? '—' : fmtNum(minPct, 2)}</span>
                  </td>
                  <td style={S.td}>
                    {fired ? <span style={S.badge(C.green)}>● FIRED</span> : <span style={{ color: C.muted }}>○ sit-out</span>}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}

      <div style={{ padding: '6px 14px', borderTop: `1px solid ${C.border}`, fontSize: 10, color: C.muted }}>
        gate fires at P &gt; 0.70 · funding favorable when ann &gt; 0 · entry also needs % positive ≥ its config threshold
      </div>
    </div>
  )
}
