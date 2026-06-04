/**
 * format.js — display helpers. Pure formatting; no computation of P&L/signals.
 */
import { C } from './styles.js'

export const pnlColor = (v) =>
  v == null ? C.text : v > 0 ? C.green : v < 0 ? C.red : C.text

export const fmtUsd = (v) =>
  v == null ? '—' : `${v >= 0 ? '+' : '−'}$${Math.abs(Number(v)).toFixed(2)}`

export const fmtNum = (v, d = 3) =>
  v == null ? '—' : Number(v).toFixed(d)

export const fmtAnnPct = (v, d = 1) =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}%`

export const fmtTime = (t) => {
  if (!t) return '—'
  return String(t).slice(0, 16).replace('T', ' ')
}

// Short session id for compact display.
export const shortId = (id) => (id ? String(id).slice(0, 8) : '—')
