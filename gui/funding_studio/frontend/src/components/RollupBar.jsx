/**
 * RollupBar.jsx — the P&L headline (numbers, not the time-series curve; the
 * charted equity curve is 54c). Reads the rollup the backend returns.
 */
import { C, S } from '../shared/styles.js'
import { fmtUsd, pnlColor } from '../shared/format.js'

export default function RollupBar({ rollup }) {
  if (!rollup) return null
  const metrics = [
    ['Net P&L', fmtUsd(rollup.net_pnl_usd), pnlColor(rollup.net_pnl_usd)],
    ['Funding', fmtUsd(rollup.funding_payments_usd), pnlColor(rollup.funding_payments_usd)],
    ['TC', `−$${Math.abs(rollup.transaction_costs_usd || 0).toFixed(2)}`, C.amber],
    ['Entries', rollup.entries ?? 0, C.text],
    ['Exits', rollup.exits ?? 0, C.text],
    ['Skips', rollup.skips ?? 0, C.muted],
  ]
  return (
    <div style={{ ...S.panel, padding: '10px 16px', display: 'flex', gap: 28, flexWrap: 'wrap' }}>
      {metrics.map(([label, value, color]) => (
        <div key={label} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span style={{ fontSize: 9, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>
            {label}
          </span>
          <span style={{ fontSize: 15, fontWeight: 700, color: color || C.text, fontVariantNumeric: 'tabular-nums' }}>
            {value}
          </span>
        </div>
      ))}
    </div>
  )
}
