/**
 * BatchRunner.jsx
 * ===============
 * Cross-year / cross-asset comparison panel.
 * Runs multiple (symbol, year, strategy) combos and shows a sortable table.
 */

import { useState, useRef, useCallback } from 'react'

const WS_BASE = `ws://${window.location.hostname}:8001`

const PRESET_YEARS = [
  { label: '2021', start: '2021-01-01', end: '2022-01-01' },
  { label: '2022', start: '2022-01-01', end: '2023-01-01' },
  { label: '2023', start: '2023-01-01', end: '2024-01-01' },
  { label: '2024', start: '2024-01-01', end: '2025-01-01' },
]

const PRESET_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']

const STRATEGY_LABELS = {
  anchor_trigger:       'Anchor & Trigger',
  zero_line_rejection:  'Zero Line Rejection',
  bullish_divergence:   'Bullish Divergence',
  mfi_momentum:         'MFI Momentum',
}

const S = {
  wrap: {
    display: 'flex', flexDirection: 'column', height: '100%',
    background: '#0f1520', overflow: 'hidden',
  },
  toolbar: {
    padding: '10px 16px', borderBottom: '1px solid #2a3248',
    display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap',
    background: '#1a2035',
  },
  group: { display: 'flex', flexDirection: 'column', gap: 4 },
  label: { fontSize: 10, fontWeight: 600, color: '#5d6b8a',
           textTransform: 'uppercase', letterSpacing: '0.5px' },
  checkRow: { display: 'flex', gap: 8, flexWrap: 'wrap' },
  chip: (active) => ({
    padding: '3px 10px', borderRadius: 3, fontSize: 11, cursor: 'pointer',
    border: `1px solid ${active ? '#40c4ff' : '#2a3248'}`,
    background: active ? 'rgba(64,196,255,0.12)' : '#131722',
    color: active ? '#40c4ff' : '#9ba8bf',
    userSelect: 'none',
  }),
  select: {
    background: '#131722', border: '1px solid #2a3248',
    borderRadius: 4, color: '#d1d4dc', padding: '4px 8px',
    fontSize: 12, outline: 'none', cursor: 'pointer',
  },
  input: {
    background: '#131722', border: '1px solid #2a3248',
    borderRadius: 4, color: '#d1d4dc', padding: '4px 8px',
    fontSize: 12, width: 60, outline: 'none',
  },
  btn: (color, disabled) => ({
    padding: '6px 16px', borderRadius: 4, border: 'none', fontSize: 12,
    fontWeight: 600, cursor: disabled ? 'not-allowed' : 'pointer',
    background: disabled ? '#2a3248' : color, color: disabled ? '#5d6b8a' : '#fff',
  }),
  tableWrap: { flex: 1, overflowY: 'auto', padding: '0 0 8px 0' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 12 },
  th: {
    padding: '8px 12px', background: '#1a2035', color: '#5d6b8a',
    fontWeight: 600, textAlign: 'left', fontSize: 10,
    textTransform: 'uppercase', letterSpacing: '0.5px',
    position: 'sticky', top: 0, borderBottom: '1px solid #2a3248',
    whiteSpace: 'nowrap', cursor: 'pointer',
  },
  td: {
    padding: '7px 12px', borderBottom: '1px solid #1a2035',
    whiteSpace: 'nowrap', color: '#d1d4dc',
  },
  statusBar: {
    padding: '6px 16px', borderTop: '1px solid #2a3248',
    fontSize: 11, color: '#5d6b8a', display: 'flex', gap: 16,
    alignItems: 'center', background: '#131722',
  },
  progressBarWrap: { flex: 1, height: 3, background: '#2a3248', borderRadius: 2 },
  progressBarFill: (pct) => ({
    height: '100%', background: '#26a69a', borderRadius: 2,
    width: `${pct}%`, transition: 'width 0.3s ease',
  }),
  empty: { padding: 32, textAlign: 'center', color: '#5d6b8a' },
}

function pctColor(v) {
  if (v == null) return '#9ba8bf'
  return v > 0 ? '#26a69a' : v < 0 ? '#ef5350' : '#9ba8bf'
}
function sharpeColor(v) {
  if (v == null) return '#9ba8bf'
  return v > 1.5 ? '#26a69a' : v > 0.5 ? '#ffb74d' : v < 0 ? '#ef5350' : '#9ba8bf'
}
function fmt(v, decimals = 2, suffix = '') {
  if (v == null || v === undefined) return '—'
  const n = Number(v)
  if (isNaN(n)) return '—'
  return `${n > 0 && suffix === '%' ? '+' : ''}${n.toFixed(decimals)}${suffix}`
}

export default function BatchRunner({ strategies = [], symbols = [], intervals = [] }) {
  const [selYears,    setSelYears]    = useState(['2022', '2023', '2024'])
  const [selSymbols,  setSelSymbols]  = useState(['BTC/USDT'])
  const [selStrategy, setSelStrategy] = useState('anchor_trigger')
  const [selInterval, setSelInterval] = useState('4h')
  const [tcPct,       setTcPct]       = useState(0.1)

  const [running,     setRunning]     = useState(false)
  const [results,     setResults]     = useState([])
  const [currentRun,  setCurrentRun]  = useState(null)   // {label, pct}
  const [sortCol,     setSortCol]     = useState('label')
  const [sortAsc,     setSortAsc]     = useState(true)

  const wsRef = useRef(null)

  const toggle = (val, sel, setSel) => {
    setSel(prev => prev.includes(val) ? prev.filter(x => x !== val) : [...prev, val])
  }

  const totalRuns = selYears.length * selSymbols.length

  const handleRun = useCallback(async () => {
    if (running) return
    setRunning(true)
    setResults([])
    setCurrentRun(null)

    // Build run list
    const runs = []
    for (const sym of selSymbols) {
      for (const yr of selYears) {
        const preset = PRESET_YEARS.find(y => y.label === yr)
        if (!preset) continue
        runs.push({
          label: `${sym.split('/')[0]} ${yr}`,
          symbol: sym, interval: selInterval,
          start: preset.start, end: preset.end,
          strategy: selStrategy,
          params: {},
          tc_pct: tcPct,
        })
      }
    }

    // Create session
    let sessionId
    try {
      const res = await fetch('/api/batch/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ runs }),
      })
      const data = await res.json()
      sessionId = data.session_id
    } catch (e) {
      setRunning(false)
      return
    }

    const ws = new WebSocket(`${WS_BASE}/ws/batch/${sessionId}`)
    wsRef.current = ws

    ws.onmessage = (ev) => {
      const frame = JSON.parse(ev.data)
      if (frame.type === 'run_start') {
        setCurrentRun({ label: frame.label, pct: 0 })
      } else if (frame.type === 'run_progress') {
        setCurrentRun(prev => prev ? { ...prev, pct: frame.pct } : null)
      } else if (frame.type === 'run_result') {
        setResults(prev => [...prev, frame])
        setCurrentRun(null)
      } else if (frame.type === 'batch_done') {
        setRunning(false)
        setCurrentRun(null)
      }
    }
    ws.onerror = () => setRunning(false)
    ws.onclose = () => setRunning(false)
  }, [running, selYears, selSymbols, selStrategy, selInterval, tcPct])

  const handleStop = () => {
    wsRef.current?.close()
    setRunning(false)
  }

  // Sort results
  const colKey = (r) => {
    const s = r.stats || {}
    switch (sortCol) {
      case 'return':    return s.total_return_pct ?? -999
      case 'sharpe':    return s.sharpe ?? -999
      case 'winrate':   return s.win_rate ?? -999
      case 'maxdd':     return -(s.max_drawdown_pct ?? 999)
      case 'trades':    return r.n_trades ?? 0
      default:          return r.label || ''
    }
  }

  const sorted = [...results].sort((a, b) => {
    const av = colKey(a), bv = colKey(b)
    if (typeof av === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av)
    return sortAsc ? av - bv : bv - av
  })

  const onSort = (col) => {
    if (sortCol === col) setSortAsc(a => !a)
    else { setSortCol(col); setSortAsc(false) }
  }

  const thProps = (col) => ({
    style: { ...S.th, color: sortCol === col ? '#40c4ff' : '#5d6b8a' },
    onClick: () => onSort(col),
  })

  return (
    <div style={S.wrap}>
      {/* Toolbar */}
      <div style={S.toolbar}>
        {/* Years */}
        <div style={S.group}>
          <span style={S.label}>Years</span>
          <div style={S.checkRow}>
            {PRESET_YEARS.map(y => (
              <span key={y.label}
                style={S.chip(selYears.includes(y.label))}
                onClick={() => toggle(y.label, selYears, setSelYears)}>
                {y.label}
              </span>
            ))}
          </div>
        </div>

        {/* Symbols */}
        <div style={S.group}>
          <span style={S.label}>Assets</span>
          <div style={S.checkRow}>
            {PRESET_SYMBOLS.map(s => (
              <span key={s}
                style={S.chip(selSymbols.includes(s))}
                onClick={() => toggle(s, selSymbols, setSelSymbols)}>
                {s.split('/')[0]}
              </span>
            ))}
          </div>
        </div>

        {/* Strategy */}
        <div style={S.group}>
          <span style={S.label}>Strategy</span>
          <select style={S.select} value={selStrategy}
            onChange={e => setSelStrategy(e.target.value)} disabled={running}>
            {Object.entries(STRATEGY_LABELS).map(([id, name]) => (
              <option key={id} value={id}>{name}</option>
            ))}
          </select>
        </div>

        {/* Interval */}
        <div style={S.group}>
          <span style={S.label}>Interval</span>
          <select style={S.select} value={selInterval}
            onChange={e => setSelInterval(e.target.value)} disabled={running}>
            {(intervals.length ? intervals : ['15m','1h','4h','1d']).map(i => (
              <option key={i} value={i}>{i}</option>
            ))}
          </select>
        </div>

        {/* TC */}
        <div style={S.group}>
          <span style={S.label}>TC %/leg</span>
          <input type="number" style={S.input} value={tcPct}
            step={0.05} min={0} max={2} disabled={running}
            onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) setTcPct(v) }} />
        </div>

        {/* Buttons */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          {!running ? (
            <button style={S.btn('#26a69a', totalRuns === 0)}
              onClick={handleRun} disabled={totalRuns === 0}>
              ▶ Run {totalRuns} backtests
            </button>
          ) : (
            <button style={S.btn('#ef5350', false)} onClick={handleStop}>
              ■ Stop
            </button>
          )}
          {results.length > 0 && !running && (
            <button style={S.btn('#2a3248', false)} onClick={() => setResults([])}>
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Results table */}
      <div style={S.tableWrap}>
        {results.length === 0 && !running ? (
          <div style={S.empty}>
            Select years, assets, strategy and click Run to compare performance across market conditions.
          </div>
        ) : (
          <table style={S.table}>
            <thead>
              <tr>
                <th {...thProps('label')}>Run {sortCol==='label' ? (sortAsc?'↑':'↓') : ''}</th>
                <th {...thProps('return')}>Return {sortCol==='return' ? (sortAsc?'↑':'↓') : ''}</th>
                <th {...thProps('sharpe')}>Sharpe {sortCol==='sharpe' ? (sortAsc?'↑':'↓') : ''}</th>
                <th {...thProps('winrate')}>Win Rate {sortCol==='winrate' ? (sortAsc?'↑':'↓') : ''}</th>
                <th {...thProps('trades')}>Trades {sortCol==='trades' ? (sortAsc?'↑':'↓') : ''}</th>
                <th {...thProps('maxdd')}>Max DD {sortCol==='maxdd' ? (sortAsc?'↑':'↓') : ''}</th>
                <th style={S.th}>Avg Win</th>
                <th style={S.th}>Avg Loss</th>
                <th style={S.th}>Status</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, i) => {
                const s = r.stats || {}
                const isError = !!r.error
                return (
                  <tr key={i} style={{ background: i % 2 === 0 ? 'transparent' : '#131722' }}>
                    <td style={{ ...S.td, fontWeight: 600, color: '#d1d4dc' }}>{r.label}</td>
                    <td style={{ ...S.td, color: pctColor(s.total_return_pct), fontWeight: 700 }}>
                      {fmt(s.total_return_pct, 2, '%')}
                    </td>
                    <td style={{ ...S.td, color: sharpeColor(s.sharpe), fontWeight: 700 }}>
                      {fmt(s.sharpe, 2)}
                    </td>
                    <td style={S.td}>{fmt(s.win_rate, 1, '%')}</td>
                    <td style={S.td}>{r.n_trades ?? '—'}</td>
                    <td style={{ ...S.td, color: '#ef5350' }}>
                      {s.max_drawdown_pct != null ? `-${s.max_drawdown_pct.toFixed(2)}%` : '—'}
                    </td>
                    <td style={{ ...S.td, color: '#26a69a' }}>{fmt(s.avg_win_pct, 2, '%')}</td>
                    <td style={{ ...S.td, color: '#ef5350' }}>{fmt(s.avg_loss_pct, 2, '%')}</td>
                    <td style={{ ...S.td, fontSize: 11, color: isError ? '#ef5350' : '#26a69a' }}>
                      {isError ? `Error: ${r.error}` : '✓'}
                    </td>
                  </tr>
                )
              })}
              {/* Pending run row */}
              {currentRun && (
                <tr style={{ background: 'rgba(64,196,255,0.06)' }}>
                  <td style={{ ...S.td, color: '#40c4ff', fontWeight: 600 }}>
                    {currentRun.label}
                  </td>
                  <td colSpan={8} style={{ ...S.td }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{ ...S.progressBarWrap }}>
                        <div style={S.progressBarFill(currentRun.pct)} />
                      </div>
                      <span style={{ color: '#40c4ff', fontSize: 11, minWidth: 32 }}>
                        {currentRun.pct}%
                      </span>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>

      {/* Status bar */}
      {(running || results.length > 0) && (
        <div style={S.statusBar}>
          <span>{results.length}/{totalRuns} complete</span>
          {results.length > 0 && (
            <>
              <span style={{ color: '#26a69a' }}>
                Positive: {results.filter(r => (r.stats?.total_return_pct ?? 0) > 0).length}
              </span>
              <span style={{ color: '#ef5350' }}>
                Negative: {results.filter(r => (r.stats?.total_return_pct ?? 0) < 0).length}
              </span>
              <span>
                Avg Sharpe: {(results.reduce((s, r) => s + (r.stats?.sharpe ?? 0), 0) / results.length).toFixed(2)}
              </span>
            </>
          )}
        </div>
      )}
    </div>
  )
}
