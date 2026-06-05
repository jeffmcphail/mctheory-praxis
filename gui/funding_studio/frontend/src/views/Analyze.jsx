/**
 * Analyze.jsx — the analytical layer (54c). Multi-select sessions:
 *   1 selected  -> deep-dive (equity curve + metrics + skip breakdown)
 *   2+ selected -> comparison (metric table + same-window equity overlay)
 * Operational Monitor/Control are unchanged; this is where analysis lives.
 * Pure render of the backend's rollup + equity series.
 */
import { useState, useEffect } from 'react'
import { getJSON } from '../api.js'
import { C, S, STATUS_COLOR } from '../shared/styles.js'
import { fmtUsd, fmtTime, shortId } from '../shared/format.js'
import EquityCurve from '../components/EquityCurve.jsx'
import ComparisonTable from '../components/ComparisonTable.jsx'
import SkipBreakdown from '../components/SkipBreakdown.jsx'

const PALETTE = ['#40c4ff', '#ffb74d', '#26a69a', '#b388ff', '#80cbc4', '#ef5350']
const toPoints = (equity) => (equity || []).map((p) => ({ time: Math.floor(p.exit_timestamp / 1000), value: p.cum_pnl_usd }))

export default function Analyze({ sessions }) {
  const [selected, setSelected] = useState([])
  const [bundles, setBundles] = useState({})

  const toggle = (id) => setSelected((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]))

  useEffect(() => {
    let cancelled = false
    const need = selected.filter((id) => !bundles[id])
    if (!need.length) return
    Promise.all(need.map(async (id) => {
      const [detail, equity, trades] = await Promise.all([
        getJSON(`/api/sessions/${id}`),
        getJSON(`/api/sessions/${id}/equity`),
        getJSON(`/api/sessions/${id}/trades`),
      ])
      return [id, { detail, equity, trades }]
    })).then((pairs) => { if (!cancelled) setBundles((prev) => ({ ...prev, ...Object.fromEntries(pairs) })) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [selected]) // eslint-disable-line react-hooks/exhaustive-deps

  const items = selected
    .map((id, i) => ({ id, color: PALETTE[i % PALETTE.length], bundle: bundles[id] }))
    .filter((it) => it.bundle)

  return (
    <div style={{ padding: 16, display: 'flex', gap: 16, alignItems: 'flex-start' }}>
      <div style={{ ...S.panel, width: 290, flexShrink: 0, overflow: 'hidden' }}>
        <div style={{ ...S.sectionTitle, padding: '10px 14px', borderBottom: `1px solid ${C.border}` }}>
          Select sessions <span style={{ color: C.muted, fontWeight: 400 }}>({selected.length})</span>
        </div>
        <div style={{ maxHeight: 600, overflowY: 'auto' }}>
          {sessions.length === 0 && <div style={{ padding: 14, color: C.muted, fontSize: 12 }}>No sessions.</div>}
          {sessions.map((s) => {
            const on = selected.includes(s.session_id)
            const idx = selected.indexOf(s.session_id)
            const col = on ? PALETTE[idx % PALETTE.length] : C.border
            return (
              <div key={s.session_id} onClick={() => toggle(s.session_id)}
                style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '7px 12px', cursor: 'pointer',
                  borderBottom: `1px solid ${C.panel2}`, background: on ? 'rgba(64,196,255,0.07)' : 'transparent' }}>
                <span style={{ width: 11, height: 11, borderRadius: 3, flexShrink: 0, border: `1px solid ${col}`, background: on ? col : 'transparent' }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12 }}>
                    {s.mode === 'paper_replay' ? '⟲' : '●'} {shortId(s.session_id)}
                    <span style={{ ...S.badge(STATUS_COLOR[s.status] || C.sub), marginLeft: 6, fontSize: 9, padding: '0 5px' }}>{s.status}</span>
                  </div>
                  <div style={{ fontSize: 10, color: C.muted }}>
                    {s.replay_start ? `${s.replay_start}→${s.replay_end}` : 'live'} · {fmtTime(s.created_at).slice(5)}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        {items.length === 0 && <Empty loading={selected.length > 0} />}
        {items.length === 1 && <DeepDive item={items[0]} />}
        {items.length >= 2 && <Compare items={items} />}
      </div>
    </div>
  )
}

function Empty({ loading }) {
  return (
    <div style={{ border: `1px dashed ${C.border}`, borderRadius: 6, padding: '36px 16px', textAlign: 'center', color: C.muted }}>
      <div style={{ fontSize: 28, opacity: 0.3, marginBottom: 8 }}>▦</div>
      <div style={{ fontSize: 13 }}>{loading ? 'Loading…' : 'Select a session to analyze, or 2+ to compare.'}</div>
    </div>
  )
}

function Panel({ title, children }) {
  return (
    <div style={{ ...S.panel, overflow: 'hidden', marginBottom: 12 }}>
      <div style={{ ...S.sectionTitle, padding: '8px 14px', borderBottom: `1px solid ${C.border}` }}>{title}</div>
      {children}
    </div>
  )
}

function DeepDive({ item }) {
  const { detail, equity, trades } = item.bundle
  const { session, rollup } = detail
  const points = toPoints(equity)
  const terminal = points.length ? points[points.length - 1].value : null
  const tie = terminal != null && rollup && Math.abs(terminal - rollup.net_pnl_usd) < 0.005
  return (
    <div>
      <Panel title="Session">
        <div style={{ padding: '10px 14px' }}>
          <div style={{ fontSize: 13, fontWeight: 700 }}>
            {session.mode === 'paper_replay' ? '⟲ Replay' : '● Live'} {shortId(session.session_id)}
          </div>
          <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
            {session.replay_start ? `window ${session.replay_start}→${session.replay_end}` : 'live'}
            {' · '}net {rollup ? fmtUsd(rollup.net_pnl_usd) : '—'} · {rollup?.exits ?? 0} exits · {rollup?.entries ?? 0} entries
          </div>
        </div>
      </Panel>
      <Panel title="Equity curve — cumulative realized P&L (USD)">
        {points.length ? (
          <>
            <EquityCurve seriesList={[{ label: shortId(session.session_id), color: item.color, points }]} height={280} />
            <div style={{ padding: '6px 14px', fontSize: 11, color: tie ? C.green : C.red, borderTop: `1px solid ${C.panel2}` }}>
              {tie
                ? `✓ terminal ${fmtUsd(terminal)} == rollup net ${fmtUsd(rollup.net_pnl_usd)} (ties out)`
                : `⚠ terminal ${fmtUsd(terminal)} ≠ rollup net ${rollup ? fmtUsd(rollup.net_pnl_usd) : '—'}`}
            </div>
          </>
        ) : <div style={{ padding: 16, color: C.muted, fontSize: 12 }}>No exits — no curve.</div>}
      </Panel>
      <Panel title="Skip-reason breakdown"><SkipBreakdown trades={trades} /></Panel>
    </div>
  )
}

function Compare({ items }) {
  const windows = items.map((it) => `${it.bundle.detail.session.replay_start}→${it.bundle.detail.session.replay_end}`)
  const anyReplay = items.some((it) => it.bundle.detail.session.replay_start)
  const sameWindow = anyReplay && windows.every((w) => w === windows[0])
  const overlay = items
    .map((it) => ({ label: shortId(it.bundle.detail.session.session_id), color: it.color, points: toPoints(it.bundle.equity) }))
    .filter((s) => s.points.length)
  const tableItems = items.map((it) => ({ session: it.bundle.detail.session, rollup: it.bundle.detail.rollup, color: it.color }))
  return (
    <div>
      <Panel title={`Comparison — ${items.length} sessions`}><ComparisonTable items={tableItems} /></Panel>
      <Panel title="Equity overlay">
        {sameWindow && overlay.length
          ? <EquityCurve seriesList={overlay} height={300} />
          : (
            <div style={{ padding: 16, color: C.muted, fontSize: 12 }}>
              {anyReplay
                ? 'Overlay needs same-window replay sessions (different windows = different time axes). The metric table above still compares them.'
                : 'Overlay is for replay windows; live sessions have no fixed window.'}
            </div>
          )}
      </Panel>
    </div>
  )
}
