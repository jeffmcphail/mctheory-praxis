/**
 * EquityChart.jsx
 * ===============
 * Lightweight canvas equity curve. Updates bar-by-bar from equity snapshots.
 */

import { useRef, useImperativeHandle, forwardRef, useEffect, useCallback } from 'react'

const EquityChart = forwardRef(function EquityChart({ height = 120 }, ref) {
  const canvasRef  = useRef(null)
  const equityRef  = useRef([100])
  const dirtyRef   = useRef(false)
  const animRef    = useRef(null)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height
    const data = equityRef.current
    if (data.length < 2) return

    ctx.clearRect(0, 0, w, h)
    ctx.fillStyle = '#131722'
    ctx.fillRect(0, 0, w, h)

    const minV = Math.min(...data)
    const maxV = Math.max(...data)
    const range = maxV - minV || 1
    const pad = 4

    const toY = v => pad + (1 - (v - minV) / range) * (h - pad * 2)
    const toX = i => (i / (data.length - 1)) * w

    // Zero line (equity = 100)
    const y100 = toY(100)
    ctx.beginPath()
    ctx.strokeStyle = '#2a3248'
    ctx.lineWidth = 0.5
    ctx.setLineDash([4, 4])
    ctx.moveTo(0, y100); ctx.lineTo(w, y100)
    ctx.stroke()
    ctx.setLineDash([])

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, h)
    const last = data[data.length - 1]
    const color = last >= 100 ? '38,166,154' : '239,83,80'
    grad.addColorStop(0, `rgba(${color},0.3)`)
    grad.addColorStop(1, `rgba(${color},0)`)

    ctx.beginPath()
    ctx.moveTo(toX(0), toY(data[0]))
    data.forEach((v, i) => ctx.lineTo(toX(i), toY(v)))
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath()
    ctx.fillStyle = grad
    ctx.fill()

    // Line
    ctx.beginPath()
    ctx.strokeStyle = last >= 100 ? '#26a69a' : '#ef5350'
    ctx.lineWidth = 1.5
    ctx.moveTo(toX(0), toY(data[0]))
    data.forEach((v, i) => ctx.lineTo(toX(i), toY(v)))
    ctx.stroke()

    // Current value label
    ctx.font = '10px monospace'
    ctx.fillStyle = last >= 100 ? '#26a69a' : '#ef5350'
    ctx.textAlign = 'right'
    ctx.fillText(`${last.toFixed(2)}`, w - 4, toY(last) - 4)

    // Title
    ctx.font = '9px -apple-system, sans-serif'
    ctx.fillStyle = '#5d6b8a'
    ctx.textAlign = 'left'
    ctx.fillText('Equity (indexed to 100)', 4, 12)

    dirtyRef.current = false
  }, [])

  useEffect(() => {
    const loop = () => {
      if (dirtyRef.current) draw()
      animRef.current = requestAnimationFrame(loop)
    }
    animRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(animRef.current)
  }, [draw])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const obs = new ResizeObserver(() => {
      canvas.width  = canvas.offsetWidth
      canvas.height = height
      dirtyRef.current = true
    })
    obs.observe(canvas)
    canvas.width  = canvas.offsetWidth || 400
    canvas.height = height
    return () => obs.disconnect()
  }, [height])

  useImperativeHandle(ref, () => ({
    addEquity(equity) {
      equityRef.current.push(equity)
      dirtyRef.current = true
    },
    clear() {
      equityRef.current = [100]
      dirtyRef.current = true
    },
  }))

  return (
    <canvas
      ref={canvasRef}
      style={{ display: 'block', width: '100%', height }}
    />
  )
})

export default EquityChart
