/**
 * ComparisonTable.jsx — side-by-side session metrics (one column per session).
 * Pure render of each session's rollup (fetched from GET /{id}); no recompute.
 *
 * items: [{ session, rollup, color }]
 */
import { C, S } from '../shared/styles.js'
import { fmtUsd, pnlColor, shortId } from '../shared/format.js'

export default function ComparisonTable({ items }) {
  const Row = ({ label, get, fmt, color }) => (
    <tr>
      <td style={{ ...S.td, color: C.muted }}>{label}</td>
      {items.map((it, i) => {
        const v = get(it.rollup)
        return (
          <td key={i} style={{ ...S.td, textAlign: 'right', fontWeight: 600, color: color ? color(v) : C.text }}>
            {fmt ? fmt(v) : (v ?? '—')}
          </td>
        )
      })}
    </tr>
  )
  return (
    <table style={S.table}>
      <thead>
        <tr>
          <th style={S.th}>Metric</th>
          {items.map((it, i) => (
            <th key={i} style={{ ...S.th, textAlign: 'right', color: it.color }}>
              {it.session.mode === 'paper_replay' ? '⟲' : '●'} {shortId(it.session.session_id)}
            </th>
          ))}
        </tr>
        <tr>
          <th style={{ ...S.th, fontWeight: 400, textTransform: 'none' }}>config</th>
          {items.map((it, i) => {
            let cfg = {}
            try { cfg = JSON.parse(it.session.config_json || '{}') } catch { /* ignore */ }
            const win = it.session.replay_start ? `${it.session.replay_start}→${it.session.replay_end}` : 'live'
            return (
              <th key={i} style={{ ...S.th, textAlign: 'right', fontWeight: 400, textTransform: 'none', color: C.muted }}>
                {win}{cfg.gate != null ? ` · g${cfg.gate}` : ''}
              </th>
            )
          })}
        </tr>
      </thead>
      <tbody>
        <Row label="Net P&L" get={(r) => r?.net_pnl_usd} fmt={fmtUsd} color={pnlColor} />
        <Row label="Funding" get={(r) => r?.funding_payments_usd} fmt={fmtUsd} color={pnlColor} />
        <Row label="TC" get={(r) => r?.transaction_costs_usd} fmt={(v) => `−$${Math.abs(v || 0).toFixed(2)}`} />
        <Row label="Entries" get={(r) => r?.entries ?? 0} />
        <Row label="Exits" get={(r) => r?.exits ?? 0} />
        <Row label="Skips" get={(r) => r?.skips ?? 0} />
      </tbody>
    </table>
  )
}
