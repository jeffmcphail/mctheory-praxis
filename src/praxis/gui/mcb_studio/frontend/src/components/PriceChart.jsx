/**
 * PriceChart.jsx
 * ==============
 * TradingView Lightweight Charts candlestick chart.
 * Renders OHLCV bars as they stream in and overlays entry/exit markers.
 *
 * Exposes imperative `addBar(frame)` and `clear()` methods via ref.
 */

import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react'
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts'

const TV_THEME = {
  background:      { type: ColorType.Solid, color: '#131722' },
  textColor:       '#9ba8bf',
  grid: {
    vertLines:   { color: '#1e2430', style: 1 },
    horzLines:   { color: '#1e2430', style: 1 },
  },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: { color: '#3a4468', width: 1, style: 1 },
    horzLine: { color: '#3a4468', width: 1, style: 1 },
  },
  rightPriceScale: {
    borderColor: '#2a3248',
    scaleMargins: { top: 0.1, bottom: 0.15 },
  },
  timeScale: {
    borderColor:      '#2a3248',
    timeVisible:      true,
    secondsVisible:   false,
    fixLeftEdge:      false,
    fixRightEdge:     false,
  },
}

const PriceChart = forwardRef(function PriceChart({ height = 380, onCrosshairMove }, ref) {
  const containerRef = useRef(null)
  const chartRef     = useRef(null)
  const candleRef    = useRef(null)
  const markersRef   = useRef([])   // accumulated markers
  const volumeRef    = useRef(null)

  // Init chart
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      ...TV_THEME,
      width:  containerRef.current.clientWidth,
      height,
      layout: TV_THEME,
    })
    chartRef.current = chart

    // Candle series
    const candle = chart.addCandlestickSeries({
      upColor:   '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor:   '#26a69a',
      wickDownColor: '#ef5350',
    })
    candleRef.current = candle

    // Volume series (tiny histogram at bottom)
    const vol = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'vol',
    })
    chart.priceScale('vol').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
      visible: false,
    })
    volumeRef.current = vol

    // Crosshair sync
    if (onCrosshairMove) {
      chart.subscribeCrosshairMove(onCrosshairMove)
    }

    // Resize observer
    const observer = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.resize(containerRef.current.clientWidth, height)
      }
    })
    observer.observe(containerRef.current)

    return () => {
      observer.disconnect()
      chart.remove()
      chartRef.current  = null
      candleRef.current = null
      volumeRef.current = null
      markersRef.current = []
    }
  }, [height])

  // Imperative API
  useImperativeHandle(ref, () => ({
    addBar(frame) {
      if (!candleRef.current) return

      const bar = {
        time:  frame.time,
        open:  frame.open,
        high:  frame.high,
        low:   frame.low,
        close: frame.close,
      }
      candleRef.current.update(bar)

      if (volumeRef.current) {
        volumeRef.current.update({
          time:  frame.time,
          value: frame.volume,
          color: frame.close >= frame.open ? 'rgba(38,166,154,0.4)' : 'rgba(239,83,80,0.4)',
        })
      }

      // Entry/exit markers
      if (frame.trade_event) {
        const ev = frame.trade_event
        if (ev.type === 'ENTRY') {
          markersRef.current.push({
            time:      frame.time,
            position:  'belowBar',
            color:     '#26a69a',
            shape:     'arrowUp',
            text:      `BUY @ ${frame.close.toFixed(2)}`,
            size:      1,
          })
        } else if (ev.type === 'EXIT') {
          const sign = (ev.pnl_pct || 0) >= 0 ? '+' : ''
          markersRef.current.push({
            time:      frame.time,
            position:  'aboveBar',
            color:     (ev.pnl_pct || 0) >= 0 ? '#26a69a' : '#ef5350',
            shape:     'arrowDown',
            text:      `SELL @ ${frame.close.toFixed(2)} (${sign}${(ev.pnl_pct || 0).toFixed(2)}%)`,
            size:      1,
          })
        }
        // Sort markers by time (required by LWC)
        markersRef.current.sort((a, b) => a.time - b.time)
        candleRef.current.setMarkers(markersRef.current)
      }
    },

    clear() {
      if (!candleRef.current) return
      candleRef.current.setData([])
      volumeRef.current?.setData([])
      markersRef.current = []
      candleRef.current.setMarkers([])
    },

    scrollToLive() {
      chartRef.current?.timeScale().scrollToRealTime()
    },

    getTimeScale() {
      return chartRef.current?.timeScale()
    },
  }))

  return (
    <div ref={containerRef} style={{ width: '100%', height, position: 'relative' }}>
      {/* Overlay label */}
      <div style={{
        position: 'absolute', top: 8, left: 10, zIndex: 2,
        fontSize: 11, color: '#5d6b8a', pointerEvents: 'none',
      }}>
        Price (USDT)
      </div>
    </div>
  )
})

export default PriceChart
