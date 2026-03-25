/**
 * StatsBar.jsx
 * ============
 * Horizontal bar showing live-updating key performance metrics during replay.
 */

export default function StatsBar({ stats, progress, status, running, error }) {
  const fmt = (v, suffix = '') => v !== null && v !== undefined ? `${v}${suffix}` : '—'
  const pct = v => {
    if (v === null || v === undefined) return { text: '—', color: '#5d6b8a' }
    const n = Number(v)
    return {
      text: `${n > 0 ? '+' : ''}${n.toFixed(2)}%`,
      color: n > 0 ? '#26a69a' : n < 0 ? '#ef5350' : '#d1d4dc',
    }
  }

  const ret   = pct(stats?.total_return_pct)
  const opnl  = pct(stats?.open_pnl_pct)
  const inPos = stats?.in_position

  const S = {
    bar: {
      background: '#1a2035',
      borderBottom: '1px solid #2a3248',
      padding: '6px 16px',
      display: 'flex',
      alignItems: 'center',
      gap: 24,
      minHeight: 36,
    },
    metric: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'flex-start',
      gap: 1,
    },
    metricLabel: {
      fontSize: 9,
      color: '#5d6b8a',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      fontWeight: 600,
    },
    metricValue: (color) => ({
      fontSize: 13,
      fontWeight: 700,
      color: color || '#d1d4dc',
      fontVariantNumeric: 'tabular-nums',
    }),
    divider: {
      width: 1,
      height: 24,
      background: '#2a3248',
    },
    progressWrap: {
      flex: 1,
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      justifyContent: 'flex-end',
    },
    progressBar: {
      width: 120,
      height: 4,
      background: '#2a3248',
      borderRadius: 2,
      overflow: 'hidden',
    },
    progressFill: {
      height: '100%',
      background: '#26a69a',
      borderRadius: 2,
      transition: 'width 0.1s ease',
      width: `${progress ?? 0}%`,
    },
    statusText: {
      fontSize: 11,
      color: '#5d6b8a',
      maxWidth: 200,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    },
    posTag: {
      background: inPos ? 'rgba(38,166,154,0.15)' : '#1e2430',
      border: `1px solid ${inPos ? '#26a69a' : '#2a3248'}`,
      color: inPos ? '#26a69a' : '#5d6b8a',
      borderRadius: 3,
      padding: '2px 8px',
      fontSize: 11,
      fontWeight: 600,
    },
    errorText: {
      color: '#ef5350',
      fontSize: 12,
    },
  }

  if (error) {
    return (
      <div style={S.bar}>
        <span style={{ fontSize: 14 }}>⚠</span>
        <span style={S.errorText}>{error}</span>
      </div>
    )
  }

  return (
    <div style={S.bar}>
      <Metric label="Return" value={ret.text} color={ret.color} />
      <div style={S.divider} />
      <Metric label="Equity" value={stats ? `${stats.equity?.toFixed(2)}` : '100.00'} />
      <div style={S.divider} />
      <Metric label="Trades" value={fmt(stats?.total_trades)} />
      <div style={S.divider} />
      <Metric label="Win Rate" value={stats?.win_rate != null ? `${stats.win_rate}%` : '—'} />
      <div style={S.divider} />
      <Metric label="Sharpe" value={fmt(stats?.sharpe)} color={
        stats?.sharpe > 1 ? '#26a69a' : stats?.sharpe < 0 ? '#ef5350' : '#d1d4dc'
      } />
      <div style={S.divider} />
      <Metric label="Max DD" value={stats?.max_drawdown_pct != null ? `-${stats.max_drawdown_pct}%` : '—'} color="#ef5350" />
      <div style={S.divider} />

      <span style={S.posTag}>{inPos ? '● LONG' : '○ FLAT'}</span>

      {inPos && (
        <>
          <div style={S.divider} />
          <Metric label="Open P&L" value={opnl.text} color={opnl.color} />
        </>
      )}

      <div style={S.progressWrap}>
        {running && (
          <>
            <span style={S.statusText}>{status}</span>
            <div style={S.progressBar}><div style={S.progressFill} /></div>
            <span style={{ ...S.statusText, color: '#40c4ff', minWidth: 32 }}>{progress ?? 0}%</span>
          </>
        )}
        {!running && stats && (
          <span style={{ ...S.statusText, color: '#26a69a' }}>✓ Complete</span>
        )}
      </div>
    </div>
  )
}

function Metric({ label, value, color }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 1,
    }}>
      <span style={{ fontSize: 9, color: '#5d6b8a', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>
        {label}
      </span>
      <span style={{ fontSize: 13, fontWeight: 700, color: color || '#d1d4dc', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </span>
    </div>
  )
}
