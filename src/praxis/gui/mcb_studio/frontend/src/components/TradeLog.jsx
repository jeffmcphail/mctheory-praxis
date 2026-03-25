/**
 * TradeLog.jsx
 * ============
 * Scrollable table of completed trades with P&L coloring.
 * Updates live as the backtest streams.
 */

export default function TradeLog({ trades = [] }) {
  const S = {
    wrap: {
      background: '#1a2035',
      borderTop: '1px solid #2a3248',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
    },
    header: {
      padding: '6px 12px',
      borderBottom: '1px solid #2a3248',
      fontSize: 10,
      fontWeight: 700,
      color: '#5d6b8a',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      display: 'flex',
      justifyContent: 'space-between',
    },
    tableWrap: {
      flex: 1,
      overflowY: 'auto',
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse',
      fontSize: 11,
    },
    th: {
      padding: '4px 8px',
      background: '#1e2430',
      color: '#5d6b8a',
      fontWeight: 600,
      textAlign: 'left',
      fontSize: 10,
      whiteSpace: 'nowrap',
      position: 'sticky',
      top: 0,
      borderBottom: '1px solid #2a3248',
    },
    td: {
      padding: '4px 8px',
      borderBottom: '1px solid #1e2430',
      whiteSpace: 'nowrap',
      color: '#d1d4dc',
    },
    empty: {
      padding: 16,
      color: '#5d6b8a',
      fontSize: 12,
      textAlign: 'center',
    },
  }

  const pnlColor = v => {
    if (v == null) return '#d1d4dc'
    return v > 0 ? '#26a69a' : v < 0 ? '#ef5350' : '#d1d4dc'
  }

  const fmtTime = t => {
    if (!t) return '—'
    try {
      const d = new Date(t)
      return d.toISOString().slice(0, 16).replace('T', ' ')
    } catch { return t }
  }

  const fmtPx = v => v != null ? Number(v).toLocaleString('en-US', { maximumFractionDigits: 2 }) : '—'
  const fmtPct = v => v != null ? `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}%` : '—'

  return (
    <div style={S.wrap}>
      <div style={S.header}>
        <span>Trade Log</span>
        <span>{trades.length} trades</span>
      </div>
      <div style={S.tableWrap}>
        {trades.length === 0 ? (
          <div style={S.empty}>No completed trades yet</div>
        ) : (
          <table style={S.table}>
            <thead>
              <tr>
                {['#', 'Entry Time', 'Entry Px', 'Exit Time', 'Exit Px', 'P&L %', 'Exit Reason'].map(h => (
                  <th key={h} style={S.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...trades].reverse().map((t, i) => {
                const pnl = t.pnl_pct
                return (
                  <tr key={i} style={{ background: i % 2 === 0 ? 'transparent' : '#181f30' }}>
                    <td style={S.td}>{trades.length - i}</td>
                    <td style={S.td}>{fmtTime(t.entry_time)}</td>
                    <td style={S.td}>{fmtPx(t.entry_price)}</td>
                    <td style={S.td}>{fmtTime(t.exit_time)}</td>
                    <td style={S.td}>{fmtPx(t.exit_price)}</td>
                    <td style={{ ...S.td, color: pnlColor(pnl), fontWeight: 700 }}>
                      {fmtPct(pnl)}
                    </td>
                    <td style={{ ...S.td, color: '#5d6b8a', fontStyle: 'italic' }}>
                      {t.exit_reason || '—'}
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
