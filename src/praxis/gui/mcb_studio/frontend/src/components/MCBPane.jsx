/**
 * MCBPane.jsx
 * ===========
 * Market Cipher B oscillator panel — rendered on an HTML Canvas.
 * Shows: WT1 wave, WT2 wave, MFI fill, RSI line, buy/sell/gold dots.
 *
 * Draws into a fixed-height canvas that is kept in sync with the price
 * chart's time scale via the shared `timeRange` prop.
 *
 * Exposes imperative addBar(frame) and clear() via ref.
 */

import { useRef, useImperativeHandle, forwardRef, useEffect, useCallback } from 'react'

// Visual constants — mirrors MCb default color scheme
const C = {
  bg:         '#131722',
  zero:       '#2a3248',
  ob:         'rgba(239,83,80,0.25)',
  os:         'rgba(38,166,154,0.25)',
  obLine:     '#ef5350',
  osLine:     '#26a69a',
  wt1:        '#40c4ff',
  wt2:        '#1565c0',
  mfiBull:    'rgba(38,166,154,0.35)',
  mfiBear:    'rgba(239,83,80,0.30)',
  rsi:        '#e040fb',
  dotBuy:     '#26a69a',
  dotSell:    '#ef5350',
  dotGold:    '#ffd54f',
  dotEntry:   '#00e676',   // strategy entry
  dotExit:    '#ff7043',   // strategy exit
  text:       '#5d6b8a',
  gridLine:   '#1e2430',
}

const OB = 60
const OS = -60
const Y_MIN = -100
const Y_MAX = 100
const Y_RANGE = Y_MAX - Y_MIN

function yPx(val, h) {
  return h - ((val - Y_MIN) / Y_RANGE) * h
}

const MCBPane = forwardRef(function MCBPane({ height = 180 }, ref) {
  const canvasRef = useRef(null)
  const barsRef   = useRef([])     // accumulated bar data
  const animRef   = useRef(null)
  const dirtyRef  = useRef(false)

  // ---- Draw ----
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height
    const bars = barsRef.current

    ctx.clearRect(0, 0, w, h)

    // Background
    ctx.fillStyle = C.bg
    ctx.fillRect(0, 0, w, h)

    // Grid levels
    for (const level of [OB, 0, OS]) {
      const y = yPx(level, h)
      ctx.beginPath()
      ctx.strokeStyle = level === 0 ? C.zero : C.gridLine
      ctx.lineWidth = level === 0 ? 1.5 : 0.5
      ctx.setLineDash(level === 0 ? [] : [4, 4])
      ctx.moveTo(0, y); ctx.lineTo(w, y)
      ctx.stroke()
      ctx.setLineDash([])

      // Level labels
      ctx.fillStyle = C.text
      ctx.font = '9px monospace'
      ctx.textAlign = 'right'
      ctx.fillText(level.toString(), w - 4, y - 3)
    }

    // OB/OS zone fill
    const yOB = yPx(OB, h)
    const yOS = yPx(OS, h)
    ctx.fillStyle = C.ob
    ctx.fillRect(0, 0, w, yOB)
    ctx.fillStyle = C.os
    ctx.fillRect(0, yOS, w, h - yOS)

    if (bars.length < 2) return

    const barW = Math.max(1, w / bars.length)

    // ---- MFI fill ----
    for (let i = 0; i < bars.length; i++) {
      const b = bars[i]
      if (b.rsi_mfi == null) continue
      const x = i * barW
      const yZero = yPx(0, h)
      const yMfi  = yPx(b.rsi_mfi, h)
      ctx.fillStyle = b.rsi_mfi > 0 ? C.mfiBull : C.mfiBear
      ctx.fillRect(x, Math.min(yZero, yMfi), barW, Math.abs(yZero - yMfi))
    }

    // ---- RSI line ----
    const drawLine = (key, color, lw = 1) => {
      ctx.beginPath()
      ctx.strokeStyle = color
      ctx.lineWidth = lw
      let started = false
      for (let i = 0; i < bars.length; i++) {
        const v = bars[i][key]
        if (v == null) continue
        const x = i * barW + barW / 2
        const y = yPx(Math.max(Y_MIN, Math.min(Y_MAX, v)), h)
        if (!started) { ctx.moveTo(x, y); started = true }
        else            ctx.lineTo(x, y)
      }
      ctx.stroke()
    }

    drawLine('rsi', C.rsi, 0.8)

    // ---- WT2 (thicker, behind WT1) ----
    drawLine('wt2', C.wt2, 2.5)

    // ---- WT1 (thinner, on top) ----
    drawLine('wt1', C.wt1, 1.2)

    // ---- Dots ----
    const dotR = Math.max(2.5, barW * 0.4)
    for (let i = 0; i < bars.length; i++) {
      const b = bars[i]
      const x = i * barW + barW / 2

      if (b.gold_dot && b.wt2 != null) {
        drawDot(ctx, x, yPx(b.wt2, h), dotR * 1.3, C.dotGold)
      } else if (b.buy_dot && b.wt2 != null) {
        drawDot(ctx, x, yPx(b.wt2, h), dotR, C.dotBuy)
      }
      if (b.sell_dot && b.wt2 != null) {
        drawDot(ctx, x, yPx(b.wt2, h), dotR, C.dotSell)
      }

      // Divergence triangles
      if (b.bull_div && b.wt2 != null) {
        drawTriangle(ctx, x, yPx(b.wt2, h) + dotR + 4, dotR, '#9c27b0', 'up')
      }
      if (b.bear_div && b.wt2 != null) {
        drawTriangle(ctx, x, yPx(b.wt2, h) - dotR - 4, dotR, '#9c27b0', 'down')
      }

      // Strategy entry/exit (on top of MCb dots)
      if (b.entry && b.wt2 != null) {
        drawDot(ctx, x, yPx(b.wt2, h), dotR * 1.4, C.dotEntry, true)
      }
      if (b.exit_signal && !b.entry && b.wt2 != null) {
        drawDot(ctx, x, yPx(b.wt2, h), dotR * 1.2, C.dotExit, true)
      }
    }

    // Legend
    const legendItems = [
      { color: C.wt1, label: 'WT1' },
      { color: C.wt2, label: 'WT2' },
      { color: C.rsi, label: 'RSI' },
      { color: C.mfiBull, label: 'MFI+' },
      { color: C.dotBuy, label: '● Buy dot' },
      { color: C.dotGold, label: '● Gold' },
      { color: C.dotEntry, label: '▲ Entry' },
      { color: C.dotExit, label: '▼ Exit' },
    ]
    ctx.font = '9px -apple-system, sans-serif'
    ctx.textAlign = 'left'
    let lx = 6
    for (const item of legendItems) {
      ctx.fillStyle = item.color
      ctx.fillRect(lx, 4, 10, 4)
      ctx.fillStyle = C.text
      ctx.fillText(item.label, lx + 12, 11)
      lx += ctx.measureText(item.label).width + 24
    }

    dirtyRef.current = false
  }, [])

  // Animation loop — only redraws when new data arrives
  useEffect(() => {
    const loop = () => {
      if (dirtyRef.current) draw()
      animRef.current = requestAnimationFrame(loop)
    }
    animRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(animRef.current)
  }, [draw])

  // Resize
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const observer = new ResizeObserver(() => {
      canvas.width  = canvas.offsetWidth
      canvas.height = height
      dirtyRef.current = true
    })
    observer.observe(canvas)
    canvas.width  = canvas.offsetWidth || 800
    canvas.height = height
    return () => observer.disconnect()
  }, [height])

  // Imperative API
  useImperativeHandle(ref, () => ({
    addBar(frame) {
      barsRef.current.push({
        time:        frame.time,
        wt1:         frame.wt1,
        wt2:         frame.wt2,
        rsi_mfi:     frame.rsi_mfi,
        rsi:         frame.rsi ? (frame.rsi - 50) * 2 : null, // center RSI on zero
        stoch_color: frame.stoch_color,
        buy_dot:     frame.buy_dot,
        sell_dot:    frame.sell_dot,
        gold_dot:    frame.gold_dot,
        bull_div:    frame.bull_div,
        bear_div:    frame.bear_div,
        entry:       frame.entry,
        exit_signal: frame.exit_signal,
      })
      dirtyRef.current = true
    },
    clear() {
      barsRef.current = []
      dirtyRef.current = true
    },
  }))

  return (
    <div style={{ position: 'relative', width: '100%', height }}>
      <canvas
        ref={canvasRef}
        style={{ display: 'block', width: '100%', height }}
      />
    </div>
  )
})

export default MCBPane

// --- Helpers ---
function drawDot(ctx, x, y, r, color, ring = false) {
  ctx.beginPath()
  ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.fillStyle = color
  ctx.fill()
  if (ring) {
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 0.8
    ctx.stroke()
  }
}

function drawTriangle(ctx, x, y, size, color, dir) {
  ctx.beginPath()
  ctx.fillStyle = color
  if (dir === 'up') {
    ctx.moveTo(x, y - size)
    ctx.lineTo(x - size, y + size * 0.5)
    ctx.lineTo(x + size, y + size * 0.5)
  } else {
    ctx.moveTo(x, y + size)
    ctx.lineTo(x - size, y - size * 0.5)
    ctx.lineTo(x + size, y - size * 0.5)
  }
  ctx.closePath()
  ctx.fill()
}
