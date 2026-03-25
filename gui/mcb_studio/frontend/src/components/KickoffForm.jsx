/**
 * KickoffForm.jsx
 * ===============
 * The control panel at the top of the app.
 * Lets users pick symbol, interval, dates, strategy, and strategy params.
 * Collapses while a backtest is running to give charts maximum space.
 */

import { useState, useEffect } from 'react'

const S = {
  // Layout
  panel: {
    background: '#1e2430',
    borderBottom: '1px solid #2a3248',
    padding: '12px 16px',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginBottom: 10,
  },
  logo: {
    fontSize: 18,
    fontWeight: 700,
    color: '#40c4ff',
    letterSpacing: '-0.5px',
  },
  logoSub: {
    color: '#5d6b8a',
    fontSize: 12,
    fontWeight: 400,
  },
  row: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: 12,
    flexWrap: 'wrap',
  },
  fieldGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  label: {
    fontSize: 10,
    fontWeight: 600,
    color: '#5d6b8a',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  select: {
    background: '#131722',
    border: '1px solid #2a3248',
    borderRadius: 4,
    color: '#d1d4dc',
    padding: '5px 8px',
    fontSize: 13,
    cursor: 'pointer',
    minWidth: 140,
    outline: 'none',
  },
  input: {
    background: '#131722',
    border: '1px solid #2a3248',
    borderRadius: 4,
    color: '#d1d4dc',
    padding: '5px 8px',
    fontSize: 13,
    width: 110,
    outline: 'none',
  },
  btn: (color, disabled) => ({
    background: disabled ? '#2a3248' : color,
    border: 'none',
    borderRadius: 4,
    color: disabled ? '#5d6b8a' : '#fff',
    padding: '7px 20px',
    fontSize: 13,
    fontWeight: 600,
    cursor: disabled ? 'not-allowed' : 'pointer',
    transition: 'background 0.2s',
    whiteSpace: 'nowrap',
  }),
  paramsSection: {
    marginTop: 10,
    borderTop: '1px solid #2a3248',
    paddingTop: 10,
  },
  paramRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
    fontSize: 12,
  },
  paramLabel: {
    color: '#9ba8bf',
    width: 200,
  },
  paramInput: {
    background: '#131722',
    border: '1px solid #2a3248',
    borderRadius: 4,
    color: '#d1d4dc',
    padding: '3px 6px',
    fontSize: 12,
    width: 80,
    outline: 'none',
  },
  paramHelp: {
    color: '#5d6b8a',
    fontSize: 11,
    fontStyle: 'italic',
  },
  speedRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    fontSize: 12,
  },
  speedLabel: { color: '#9ba8bf', width: 80 },
  strategyDesc: {
    fontSize: 11,
    color: '#5d6b8a',
    marginTop: 4,
    lineHeight: 1.4,
    maxWidth: 500,
  },
}

function ParamControl({ spec, value, onChange }) {
  if (spec.type === 'bool') {
    return (
      <input
        type="checkbox"
        checked={!!value}
        onChange={e => onChange(e.target.checked)}
        style={{ cursor: 'pointer' }}
      />
    )
  }
  if (spec.type === 'select') {
    return (
      <select style={S.paramInput} value={value} onChange={e => onChange(e.target.value)}>
        {(spec.options || []).map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    )
  }
  // float or int
  return (
    <input
      type="number"
      style={S.paramInput}
      value={value}
      step={spec.step ?? (spec.type === 'int' ? 1 : 0.1)}
      min={spec.min ?? undefined}
      max={spec.max ?? undefined}
      onChange={e => {
        const v = spec.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value)
        if (!isNaN(v)) onChange(v)
      }}
    />
  )
}

export default function KickoffForm({
  strategies,
  symbols,
  intervals,
  running,
  onStart,
  onStop,
  collapsed,
  onToggleCollapse,
}) {
  const [symbol,   setSymbol]   = useState('BTC/USDT')
  const [interval, setInterval] = useState('4h')
  const [start,    setStart]    = useState('2024-01-01')
  const [end,      setEnd]      = useState('2025-01-01')
  const [stratId,  setStratId]  = useState('anchor_trigger')
  const [params,   setParams]   = useState({})
  const [speed,    setSpeed]    = useState(30)
  const [tcPct,    setTcPct]    = useState(0.1)
  const [showParams, setShowParams] = useState(true)

  const currentStrat = strategies.find(s => s.id === stratId)

  // Reset params when strategy changes
  useEffect(() => {
    if (currentStrat) {
      const defaults = {}
      currentStrat.params.forEach(p => { defaults[p.name] = p.default })
      setParams(defaults)
    }
  }, [stratId, strategies])

  const handleStart = () => {
    onStart({
      symbol, interval, start, end,
      strategy: stratId,
      params,
      replay_speed_ms: speed,
      tc_pct: tcPct,
    })
  }

  if (collapsed) {
    return (
      <div style={{ ...S.panel, padding: '8px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <span style={S.logo}>⬡ MCb Studio</span>
        <span style={{ color: '#5d6b8a', fontSize: 12 }}>
          {symbol} · {interval} · {currentStrat?.name}
        </span>
        <button style={S.btn('#2a3248', false)} onClick={onToggleCollapse}>▾ Expand</button>
        <button style={S.btn('#ef5350', false)} onClick={onStop}>■ Stop</button>
      </div>
    )
  }

  return (
    <div style={S.panel}>
      <div style={S.header}>
        <span style={S.logo}>⬡ MCb<span style={S.logoSub}> Backtest Studio</span></span>
        {running && (
          <button style={S.btn('#2a3248', false)} onClick={onToggleCollapse}>▲ Collapse</button>
        )}
      </div>

      <div style={S.row}>
        {/* Symbol */}
        <div style={S.fieldGroup}>
          <span style={S.label}>Symbol</span>
          <select style={S.select} value={symbol} onChange={e => setSymbol(e.target.value)} disabled={running}>
            {symbols.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>

        {/* Interval */}
        <div style={S.fieldGroup}>
          <span style={S.label}>Interval</span>
          <select style={S.select} value={interval} onChange={e => setInterval(e.target.value)} disabled={running} style={{...S.select, minWidth: 90}}>
            {intervals.map(i => <option key={i} value={i}>{i}</option>)}
          </select>
        </div>

        {/* Date range */}
        <div style={S.fieldGroup}>
          <span style={S.label}>From</span>
          <input type="date" style={S.input} value={start} onChange={e => setStart(e.target.value)} disabled={running} />
        </div>
        <div style={S.fieldGroup}>
          <span style={S.label}>To</span>
          <input type="date" style={S.input} value={end} onChange={e => setEnd(e.target.value)} disabled={running} />
        </div>

        {/* Strategy */}
        <div style={S.fieldGroup}>
          <span style={S.label}>Strategy</span>
          <select style={{...S.select, minWidth: 200}} value={stratId}
            onChange={e => setStratId(e.target.value)} disabled={running}>
            {strategies.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </div>

        {/* Speed */}
        <div style={S.fieldGroup}>
          <span style={S.label}>Replay Speed</span>
          <div style={S.speedRow}>
            <span style={{ color: '#5d6b8a', fontSize: 11 }}>Fast</span>
            <input type="range" min={0} max={500} step={10} value={speed}
              onChange={e => setSpeed(Number(e.target.value))}
              style={{ width: 80, cursor: 'pointer' }} />
            <span style={{ color: '#5d6b8a', fontSize: 11 }}>Slow</span>
          </div>
        </div>

        {/* TC */}
        <div style={S.fieldGroup}>
          <span style={S.label}>TC % (one-way)</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input
              type="number"
              style={{ ...S.input, width: 70 }}
              value={tcPct}
              step={0.05}
              min={0}
              max={2}
              disabled={running}
              onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) setTcPct(v) }}
            />
            <span style={{ color: '#5d6b8a', fontSize: 11 }}>
              {tcPct === 0 ? 'no TC' : `${(tcPct * 2).toFixed(2)}% RT`}
            </span>
          </div>
        </div>

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          {!running ? (
            <button style={S.btn('#26a69a', false)} onClick={handleStart}>▶ Run Backtest</button>
          ) : (
            <button style={S.btn('#ef5350', false)} onClick={onStop}>■ Stop</button>
          )}
          <button
            style={{ ...S.btn('#2a3248', false), fontSize: 11, padding: '7px 10px' }}
            onClick={() => setShowParams(p => !p)}>
            {showParams ? '▲ Params' : '▼ Params'}
          </button>
        </div>
      </div>

      {/* Strategy description */}
      {currentStrat && (
        <div style={S.strategyDesc}>{currentStrat.description}</div>
      )}

      {/* Strategy params */}
      {showParams && currentStrat && currentStrat.params.length > 0 && (
        <div style={S.paramsSection}>
          <div style={{ fontSize: 11, color: '#5d6b8a', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Strategy Parameters
          </div>
          {currentStrat.params.map(spec => (
            <div key={spec.name} style={S.paramRow}>
              <span style={S.paramLabel}>{spec.label}</span>
              <ParamControl
                spec={spec}
                value={params[spec.name] ?? spec.default}
                onChange={v => setParams(p => ({ ...p, [spec.name]: v }))}
              />
              {spec.help && <span style={S.paramHelp}>{spec.help}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
