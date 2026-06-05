/**
 * EquityCurve.jsx — realized-P&L equity curve(s) via lightweight-charts.
 * Stepped line (the curve is honestly a step function — P&L jumps at each exit,
 * flat between). Single series or multi-series overlay (same time axis).
 * Pure render of the backend's equity series; no computation here.
 *
 * seriesList: [{ label, color, points: [{ time (UTC seconds), value }] }]
 */
import { useEffect, useRef } from 'react'
import { createChart, ColorType, LineType, CrosshairMode } from 'lightweight-charts'
import { C } from '../shared/styles.js'

const PALETTE = [C.accent, C.amber, C.green, '#b388ff', '#80cbc4', C.red]

export default function EquityCurve({ seriesList, height = 280 }) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const chart = createChart(el, {
      width: el.clientWidth,
      height,
      layout: { background: { type: ColorType.Solid, color: C.bg }, textColor: C.sub, fontSize: 11 },
      grid: { vertLines: { color: C.panel2 }, horzLines: { color: C.panel2 } },
      rightPriceScale: { borderColor: C.border },
      timeScale: { borderColor: C.border, timeVisible: true, secondsVisible: false },
      crosshair: { mode: CrosshairMode.Normal },
    })
    let firstSeries = null
    ;(seriesList || []).forEach((s, i) => {
      const ser = chart.addLineSeries({
        color: s.color || PALETTE[i % PALETTE.length],
        lineWidth: 2,
        lineType: LineType.WithSteps,
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
        title: s.label || '',
        priceLineVisible: false,
      })
      ser.setData((s.points || []).slice().sort((a, b) => a.time - b.time))
      if (!firstSeries) firstSeries = ser
    })
    if (firstSeries) {
      firstSeries.createPriceLine({ price: 0, color: C.muted, lineWidth: 1, lineStyle: 2, axisLabelVisible: false })
    }
    chart.timeScale().fitContent()
    const obs = new ResizeObserver(() => { if (ref.current) chart.resize(ref.current.clientWidth, height) })
    obs.observe(el)
    return () => { obs.disconnect(); chart.remove() }
  }, [seriesList, height])

  return <div ref={ref} style={{ width: '100%', height }} />
}
