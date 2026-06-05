/**
 * CollectorHealth.jsx — always-on header strip rendering GET /api/health.
 * Tri-state per table: fresh (green) / stale (red) / empty (amber, benign
 * sit-out) + near-limit (amber, staleness/threshold > 0.8). Foregrounds the
 * funding tables; click to expand full per-DB detail. Pure render.
 */
import { useState } from 'react'
import { C } from '../shared/styles.js'

const FUNDING_TABLES = ['funding_rates', 'funding_signals', 'funding_alerts', 'paper_trades']

function tableState(entry) {
  if (!entry) return { kind: 'unknown', color: C.muted, ratio: null }
  if (entry.error === 'empty table') return { kind: 'empty', color: C.amber, ratio: null }
  if (entry.error) return { kind: 'error', color: C.red, ratio: null }
  const ratio = entry.threshold_seconds ? entry.staleness_seconds / entry.threshold_seconds : 0
  if (entry.is_stale) return { kind: 'stale', color: C.red, ratio }
  if (ratio > 0.8) return { kind: 'near-limit', color: C.amber, ratio }
  return { kind: 'fresh', color: C.green, ratio }
}

const short = (t) => t.replace('funding_', 'f.').replace('paper_', 'p.').replace('order_book_snapshots', 'ob')

export default function CollectorHealth({ health }) {
  const [open, setOpen] = useState(false)
  const tables = health?.tables || {}
  const chips = FUNDING_TABLES.map((t) => ({ name: t, ...tableState(tables[t]) }))
  const worst = chips.some((c) => c.kind === 'stale' || c.kind === 'error') ? C.red
    : chips.some((c) => c.kind === 'near-limit') ? C.amber : C.green

  return (
    <div style={{ position: 'relative' }}>
      <div onClick={() => setOpen((o) => !o)} title="collector freshness — click for full detail"
        style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', padding: '3px 9px', borderRadius: 4, border: `1px solid ${C.border}` }}>
        <span style={{ width: 8, height: 8, borderRadius: 4, background: worst }} />
        <span style={{ fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.5px' }}>health</span>
        {chips.map((c) => (
          <span key={c.name} style={{ display: 'flex', alignItems: 'center', gap: 3 }} title={`${c.name}: ${c.kind}`}>
            <span style={{ width: 7, height: 7, borderRadius: 4, background: c.color }} />
            <span style={{ fontSize: 10, color: C.muted }}>{short(c.name)}</span>
          </span>
        ))}
        <span style={{ fontSize: 9, color: C.muted }}>{open ? '▴' : '▾'}</span>
      </div>
      {open && <HealthDetail health={health} />}
    </div>
  )
}

function HealthDetail({ health }) {
  const dbs = health?.databases || {}
  return (
    <div style={{
      position: 'absolute', right: 0, top: 30, zIndex: 50, width: 480, maxHeight: 440, overflowY: 'auto',
      background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, boxShadow: '0 6px 24px rgba(0,0,0,0.5)', padding: 10,
    }}>
      {Object.keys(dbs).length === 0 && <div style={{ color: C.muted, fontSize: 12 }}>No health data.</div>}
      {Object.entries(dbs).map(([db, info]) => (
        <div key={db} style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 10, color: C.accent, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>{db}</div>
          {Object.entries(info.tables || {}).map(([t, e]) => {
            const st = tableState(e)
            const pct = st.ratio != null ? `${Math.round(st.ratio * 100)}%` : ''
            return (
              <div key={t} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, padding: '2px 0' }}>
                <span style={{ width: 7, height: 7, borderRadius: 4, background: st.color, flexShrink: 0 }} />
                <span style={{ flex: 1, color: C.text }}>{t}</span>
                <span style={{ color: C.muted, width: 90, textAlign: 'right' }}>{e?.error ? e.error : st.kind}</span>
                <span style={{ width: 44, textAlign: 'right', color: st.kind === 'near-limit' ? C.amber : C.muted }}>{pct}</span>
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}
