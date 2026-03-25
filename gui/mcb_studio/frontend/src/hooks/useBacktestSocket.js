/**
 * useBacktestSocket.js
 * ====================
 * Custom hook that manages the WebSocket lifecycle for a backtest session.
 *
 * Returns:
 *   { running, status, bars, trades, stats, error, startBacktest, stopBacktest }
 */

import { useState, useRef, useCallback } from 'react'

const API_BASE = '/api'
const WS_BASE  = `ws://${window.location.hostname}:8001`

export function useBacktestSocket({ onBar, onTradeEvent, onDone }) {
  const [running,   setRunning]   = useState(false)
  const [status,    setStatus]    = useState('')
  const [stats,     setStats]     = useState(null)
  const [trades,    setTrades]    = useState([])
  const [error,     setError]     = useState(null)
  const [totalBars, setTotalBars] = useState(0)
  const [doneBar,   setDoneBar]   = useState(0)

  const wsRef = useRef(null)

  const stopBacktest = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setRunning(false)
  }, [])

  const startBacktest = useCallback(async (config) => {
    // Clean up any existing session
    stopBacktest()
    setError(null)
    setStats(null)
    setTrades([])
    setDoneBar(0)
    setTotalBars(0)
    setStatus('Starting…')
    setRunning(true)

    // Create session
    let sessionId
    try {
      const res = await fetch(`${API_BASE}/backtest/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      sessionId = data.session_id
    } catch (err) {
      setError(`Failed to start session: ${err.message}`)
      setRunning(false)
      return
    }

    // Open WebSocket — use the proxy if same origin, else direct
    const wsUrl = `${WS_BASE}/ws/${sessionId}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onmessage = (event) => {
      let frame
      try { frame = JSON.parse(event.data) } catch { return }

      switch (frame.type) {
        case 'status':
          setStatus(frame.message)
          if (frame.total_bars) setTotalBars(frame.total_bars)
          break

        case 'bar':
          setDoneBar(frame.i + 1)
          if (frame.stats) setStats(frame.stats)
          if (onBar) onBar(frame)
          if (frame.trade_event) {
            if (onTradeEvent) onTradeEvent(frame.trade_event, frame)
            if (frame.trade_event.type === 'EXIT' && frame.trade_event.trade) {
              setTrades(prev => [...prev, frame.trade_event.trade])
            }
          }
          break

        case 'final_stats':
          setStats(frame.stats)
          setTrades(frame.trades || [])
          if (onDone) onDone(frame)
          break

        case 'done':
          setStatus('Complete')
          setRunning(false)
          break

        case 'error':
          setError(frame.message)
          setRunning(false)
          break

        default:
          break
      }
    }

    ws.onerror = () => {
      setError('WebSocket connection error')
      setRunning(false)
    }

    ws.onclose = () => {
      setRunning(false)
    }
  }, [onBar, onTradeEvent, onDone, stopBacktest])

  const progress = totalBars > 0 ? Math.round(doneBar / totalBars * 100) : 0

  return {
    running,
    status,
    stats,
    trades,
    error,
    progress,
    totalBars,
    doneBar,
    startBacktest,
    stopBacktest,
  }
}
