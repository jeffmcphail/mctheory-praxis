/**
 * App.jsx
 * =======
 * MCb Backtest Studio — main layout.
 *
 * Layout (top → bottom):
 *   ┌─────────────────────────────────────────┐
 *   │  KickoffForm (collapsible)              │
 *   ├─────────────────────────────────────────┤
 *   │  StatsBar                               │
 *   ├──────────────────────┬──────────────────┤
 *   │                      │  EquityChart     │
 *   │  PriceChart          ├──────────────────┤
 *   │                      │  TradeLog        │
 *   ├──────────────────────┴──────────────────┤
 *   │  MCBPane (oscillator)                   │
 *   └─────────────────────────────────────────┘
 */

import { useState, useEffect, useRef, useCallback } from 'react'

import KickoffForm  from './components/KickoffForm.jsx'
import StatsBar     from './components/StatsBar.jsx'
import PriceChart   from './components/PriceChart.jsx'
import MCBPane      from './components/MCBPane.jsx'
import TradeLog     from './components/TradeLog.jsx'
import EquityChart  from './components/EquityChart.jsx'
import { useBacktestSocket } from './hooks/useBacktestSocket.js'

const API = '/api'

export default function App() {
  // ---- Metadata from API ----
  const [strategies, setStrategies] = useState([])
  const [symbols,    setSymbols]    = useState([])
  const [intervals,  setIntervals]  = useState([])
  const [formCollapsed, setFormCollapsed] = useState(false)

  // ---- Chart refs ----
  const priceChartRef  = useRef(null)
  const mcbPaneRef     = useRef(null)
  const equityChartRef = useRef(null)

  // ---- Fetch metadata ----
  useEffect(() => {
    fetch(`${API}/strategies`).then(r => r.json()).then(setStrategies).catch(console.error)
    fetch(`${API}/symbols`).then(r => r.json()).then(setSymbols).catch(console.error)
    fetch(`${API}/intervals`).then(r => r.json()).then(setIntervals).catch(console.error)
  }, [])

  // ---- Bar callback ----
  const onBar = useCallback((frame) => {
    priceChartRef.current?.addBar(frame)
    mcbPaneRef.current?.addBar(frame)
    if (frame.stats?.equity != null) {
      equityChartRef.current?.addEquity(frame.stats.equity)
    }
  }, [])

  // ---- Socket ----
  const {
    running, status, stats, trades, error,
    progress, startBacktest, stopBacktest,
  } = useBacktestSocket({ onBar })

  // ---- Start / stop ----
  const handleStart = useCallback((config) => {
    priceChartRef.current?.clear()
    mcbPaneRef.current?.clear()
    equityChartRef.current?.clear()
    setFormCollapsed(true)
    startBacktest(config)
  }, [startBacktest])

  const handleStop = useCallback(() => {
    stopBacktest()
    setFormCollapsed(false)
  }, [stopBacktest])

  // Layout heights (px)
  const PRICE_H  = 360
  const MCB_H    = 180
  const EQUITY_H = 130
  const SIDEBAR_W = 320

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>

      {/* Kickoff form */}
      <KickoffForm
        strategies={strategies}
        symbols={symbols}
        intervals={intervals}
        running={running}
        onStart={handleStart}
        onStop={handleStop}
        collapsed={formCollapsed}
        onToggleCollapse={() => setFormCollapsed(c => !c)}
      />

      {/* Stats bar */}
      <StatsBar
        stats={stats}
        status={status}
        progress={progress}
        running={running}
        error={error}
      />

      {/* Main content area */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

        {/* Left: charts column */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

          {/* Price chart */}
          <div style={{
            flex: 'none',
            height: PRICE_H,
            borderBottom: '2px solid #2a3248',
            position: 'relative',
          }}>
            <PriceChart ref={priceChartRef} height={PRICE_H} />
            {/* Empty state overlay */}
            {!running && !stats && (
              <EmptyOverlay message="Configure a backtest above and click ▶ Run" />
            )}
          </div>

          {/* MCb oscillator */}
          <div style={{
            flex: 'none',
            height: MCB_H,
            borderBottom: '1px solid #2a3248',
            background: '#131722',
          }}>
            <MCBPane ref={mcbPaneRef} height={MCB_H} />
          </div>

        </div>

        {/* Right: sidebar */}
        <div style={{
          width: SIDEBAR_W,
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          borderLeft: '1px solid #2a3248',
          background: '#0f1520',
          overflow: 'hidden',
        }}>

          {/* Equity curve */}
          <div style={{
            height: EQUITY_H,
            flexShrink: 0,
            borderBottom: '1px solid #2a3248',
            background: '#131722',
          }}>
            <EquityChart ref={equityChartRef} height={EQUITY_H} />
          </div>

          {/* Detailed stats card */}
          {stats && <StatsCard stats={stats} />}

          {/* Trade log — fills remaining space */}
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <TradeLog trades={trades} />
          </div>

        </div>
      </div>
    </div>
  )
}

// ---- Detailed stats card ----
function StatsCard({ stats }) {
  const rows = [
    ['Avg Win',      stats.avg_win_pct  != null ? `+${stats.avg_win_pct}%`  : '—', '#26a69a'],
    ['Avg Loss',     stats.avg_loss_pct != null ? `${stats.avg_loss_pct}%`  : '—', '#ef5350'],
    ['Open trades',  stats.open_trades ?? 0,  null],
  ]

  return (
    <div style={{
      padding: '8px 12px',
      borderBottom: '1px solid #2a3248',
      fontSize: 11,
      flexShrink: 0,
    }}>
      {rows.map(([label, value, color]) => (
        <div key={label} style={{
          display: 'flex', justifyContent: 'space-between',
          marginBottom: 4,
        }}>
          <span style={{ color: '#5d6b8a' }}>{label}</span>
          <span style={{ color: color || '#d1d4dc', fontWeight: 600 }}>{value}</span>
        </div>
      ))}
    </div>
  )
}

// ---- Empty state overlay ----
function EmptyOverlay({ message }) {
  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(19,23,34,0.85)',
      zIndex: 10,
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 32, marginBottom: 12, opacity: 0.3 }}>⬡</div>
        <div style={{ color: '#5d6b8a', fontSize: 13 }}>{message}</div>
      </div>
    </div>
  )
}
