/**
 * App.jsx — MCb Backtest Studio
 * Tabs: Backtest | Compare
 * Resizable panels, oscillator synced to price chart viewport.
 */

import { useState, useEffect, useRef, useCallback } from 'react'

import KickoffForm  from './components/KickoffForm.jsx'
import StatsBar     from './components/StatsBar.jsx'
import PriceChart   from './components/PriceChart.jsx'
import MCBPane      from './components/MCBPane.jsx'
import TradeLog     from './components/TradeLog.jsx'
import EquityChart  from './components/EquityChart.jsx'
import BatchRunner  from './components/BatchRunner.jsx'
import { useBacktestSocket } from './hooks/useBacktestSocket.js'

const API = '/api'

function useDragResize(initial, min, max, direction = 'vertical', invert = false) {
  const [size, setSize] = useState(initial)
  const dragging = useRef(false)
  const startPos = useRef(0)
  const startSize = useRef(initial)

  const onMouseDown = useCallback((e) => {
    dragging.current = true
    startPos.current = direction === 'vertical' ? e.clientY : e.clientX
    startSize.current = size
    e.preventDefault()
  }, [size, direction])

  useEffect(() => {
    const onMove = (e) => {
      if (!dragging.current) return
      const raw = direction === 'vertical'
        ? e.clientY - startPos.current
        : e.clientX - startPos.current
      const delta = invert ? -raw : raw
      setSize(Math.max(min, Math.min(max, startSize.current + delta)))
    }
    const onUp = () => { dragging.current = false }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [min, max, direction, invert])

  return [size, onMouseDown]
}

const VDIVIDER = {
  height: 5, cursor: 'row-resize', background: '#1a2035',
  borderTop: '1px solid #2a3248', borderBottom: '1px solid #2a3248',
  flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
}
const HDIVIDER = {
  width: 5, cursor: 'col-resize', background: '#1a2035',
  borderLeft: '1px solid #2a3248', borderRight: '1px solid #2a3248', flexShrink: 0,
}
const GRIP = { width: 30, height: 2, background: '#3a4468', borderRadius: 1 }

export default function App() {
  const [strategies, setStrategies] = useState([])
  const [symbols,    setSymbols]    = useState([])
  const [intervals,  setIntervals]  = useState([])
  const [formCollapsed, setFormCollapsed] = useState(false)
  const [visibleRange,  setVisibleRange]  = useState(null)
  const [tcPct,         setTcPct]         = useState(0.1)
  const [activeTab,     setActiveTab]     = useState('backtest')

  const [priceH,   onPriceDrag]   = useDragResize(360, 150, 700, 'vertical')
  const [mcbH,     onMcbDrag]     = useDragResize(180, 80,  400, 'vertical')
  const [sidebarW, onSidebarDrag] = useDragResize(320, 200, 600, 'horizontal', true)
  const [equityH,  onEquityDrag]  = useDragResize(130, 80,  300, 'vertical')

  const priceChartRef  = useRef(null)
  const mcbPaneRef     = useRef(null)
  const equityChartRef = useRef(null)

  useEffect(() => {
    const safeFetch = (url, setter) =>
      fetch(url).then(r => r.ok ? r.json() : [])
        .then(d => { if (Array.isArray(d)) setter(d) }).catch(console.error)
    safeFetch(`${API}/strategies`, setStrategies)
    safeFetch(`${API}/symbols`,    setSymbols)
    safeFetch(`${API}/intervals`,  setIntervals)
  }, [])

  const onBar = useCallback((frame) => {
    priceChartRef.current?.addBar(frame)
    mcbPaneRef.current?.addBar(frame)
    if (frame.stats?.equity != null)
      equityChartRef.current?.addEquity(frame.stats.equity)
  }, [])

  const { running, status, stats, trades, error, progress, startBacktest, stopBacktest, sendSpeed } =
    useBacktestSocket({ onBar })

  const handleStart = useCallback((config) => {
    priceChartRef.current?.clear()
    mcbPaneRef.current?.clear()
    equityChartRef.current?.clear()
    setVisibleRange(null)
    setTcPct(config.tc_pct ?? 0.1)
    setFormCollapsed(true)
    startBacktest(config)
  }, [startBacktest])

  const handleStop = useCallback(() => {
    stopBacktest()
    setFormCollapsed(false)
  }, [stopBacktest])

  const onVisibleRangeChange = useCallback((range) => setVisibleRange(range), [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>

      {/* Tab bar */}
      <div style={{
        display: 'flex', background: '#131722',
        borderBottom: '1px solid #2a3248', flexShrink: 0,
      }}>
        {[['backtest', '⬡ Backtest'], ['compare', '⊞ Compare']].map(([id, label]) => (
          <button key={id} onClick={() => setActiveTab(id)} style={{
            padding: '8px 20px', border: 'none', cursor: 'pointer', fontSize: 12,
            fontWeight: activeTab === id ? 700 : 400,
            background: activeTab === id ? '#1e2430' : 'transparent',
            color: activeTab === id ? '#40c4ff' : '#5d6b8a',
            borderBottom: activeTab === id ? '2px solid #40c4ff' : '2px solid transparent',
          }}>{label}</button>
        ))}
      </div>

      {/* Compare tab */}
      {activeTab === 'compare' && (
        <div style={{ flex: 1, overflow: 'hidden', minHeight: 0 }}>
          <BatchRunner strategies={strategies} symbols={symbols} intervals={intervals} />
        </div>
      )}

      {/* Backtest tab */}
      {activeTab === 'backtest' && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 0 }}>

          <KickoffForm
            strategies={strategies} symbols={symbols} intervals={intervals}
            running={running} onStart={handleStart} onStop={handleStop}
            collapsed={formCollapsed} onToggleCollapse={() => setFormCollapsed(c => !c)}
            onSpeedChange={sendSpeed}
          />
          <StatsBar stats={stats} status={status} progress={progress} running={running} error={error} tcPct={tcPct} />

          <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

            {/* Charts column */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
              <div style={{ height: priceH, flexShrink: 0, position: 'relative' }}>
                <PriceChart ref={priceChartRef} height={priceH} onVisibleRangeChange={onVisibleRangeChange} />
                {!running && !stats && <EmptyOverlay message="Configure a backtest above and click ▶ Run" />}
              </div>
              <div style={VDIVIDER} onMouseDown={onPriceDrag}><div style={GRIP} /></div>
              <div style={{ height: mcbH, flexShrink: 0, background: '#131722' }}>
                <MCBPane ref={mcbPaneRef} height={mcbH} visibleRange={visibleRange} />
              </div>
              <div style={VDIVIDER} onMouseDown={onMcbDrag}><div style={GRIP} /></div>
            </div>

            {/* Sidebar drag handle */}
            <div style={HDIVIDER} onMouseDown={onSidebarDrag} />

            {/* Sidebar */}
            <div style={{
              width: sidebarW, flexShrink: 0, display: 'flex', flexDirection: 'column',
              background: '#0f1520', overflow: 'hidden',
            }}>
              <div style={{ height: equityH, flexShrink: 0, borderBottom: '1px solid #2a3248', background: '#131722' }}>
                <EquityChart ref={equityChartRef} height={equityH} />
              </div>
              <div style={VDIVIDER} onMouseDown={onEquityDrag}><div style={GRIP} /></div>
              {stats && <StatsCard stats={stats} />}
              <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                <TradeLog trades={trades} />
              </div>
            </div>

          </div>
        </div>
      )}

    </div>
  )
}

function StatsCard({ stats }) {
  const rows = [
    ['Avg Win',     stats.avg_win_pct  != null ? `+${stats.avg_win_pct}%`  : '—', '#26a69a'],
    ['Avg Loss',    stats.avg_loss_pct != null ? `${stats.avg_loss_pct}%`  : '—', '#ef5350'],
    ['Open trades', stats.open_trades ?? 0, null],
  ]
  return (
    <div style={{ padding: '8px 12px', borderBottom: '1px solid #2a3248', fontSize: 11, flexShrink: 0 }}>
      {rows.map(([label, value, color]) => (
        <div key={label} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ color: '#5d6b8a' }}>{label}</span>
          <span style={{ color: color || '#d1d4dc', fontWeight: 600 }}>{value}</span>
        </div>
      ))}
    </div>
  )
}

function EmptyOverlay({ message }) {
  return (
    <div style={{
      position: 'absolute', inset: 0, zIndex: 10,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(19,23,34,0.85)',
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 32, marginBottom: 12, opacity: 0.3 }}>⬡</div>
        <div style={{ color: '#5d6b8a', fontSize: 13 }}>{message}</div>
      </div>
    </div>
  )
}
