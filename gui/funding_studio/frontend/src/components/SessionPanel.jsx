/**
 * SessionPanel.jsx — per-session detail. Fetches the booked rows
 * (detail/trades/exits/positions) and re-fetches on each WS frame (so it fills
 * in when a replay completes or a live tick books). Pure render of API data.
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import { getJSON } from '../api.js'
import { C, S } from '../shared/styles.js'
import { useSessionSocket } from '../hooks/useSessionSocket.js'
import DataTable from '../shared/DataTable.jsx'
import SessionHeader from './SessionHeader.jsx'
import RollupBar from './RollupBar.jsx'
import { fmtUsd, fmtNum, fmtTime, pnlColor } from '../shared/format.js'

export default function SessionPanel({ sessionId, onStop }) {
  const [detail, setDetail] = useState(null)
  const [trades, setTrades] = useState([])
  const [exits, setExits] = useState([])
  const [positions, setPositions] = useState([])
  const busy = useRef(false)

  const refetch = useCallback(async () => {
    if (busy.current) return
    busy.current = true
    try {
      const [d, t, x, p] = await Promise.all([
        getJSON(`/api/sessions/${sessionId}`),
        getJSON(`/api/sessions/${sessionId}/trades`),
        getJSON(`/api/sessions/${sessionId}/exits`),
        getJSON(`/api/sessions/${sessionId}/positions`),
      ])
      setDetail(d); setTrades(t); setExits(x); setPositions(p)
    } catch { /* transient; next frame/poll retries */ }
    finally { busy.current = false }
  }, [sessionId])

  useEffect(() => {
    setDetail(null); setTrades([]); setExits([]); setPositions([])
    refetch()
  }, [sessionId, refetch])

  const { lastState, connected } = useSessionSocket(sessionId, refetch)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <SessionHeader session={detail?.session} lastState={lastState} connected={connected} onStop={onStop} />
      <RollupBar rollup={detail?.rollup} session={detail?.session} />

      <Panel title={`Open positions (${positions.length})`}>
        <DataTable
          columns={[
            { key: 'asset', label: 'Asset', bold: true },
            { key: 'signal_datetime', label: 'Entry', render: (r) => fmtTime(r.signal_datetime) },
            { key: 'hold_days', label: 'Hold (d)', align: 'right' },
            { key: 'intended_size_usd', label: 'Notional', align: 'right', render: (r) => `$${fmtNum(r.intended_size_usd, 0)}` },
            { key: 'intended_direction', label: 'Direction', render: (r) => <span style={{ color: C.muted }}>{r.intended_direction}</span> },
          ]}
          rows={positions} getKey={(r) => `${r.asset}-${r.signal_timestamp}`}
          empty="No open positions (all settled, or none entered)."
        />
      </Panel>

      <Panel title={`Decisions (${trades.length})`}>
        <DataTable
          columns={[
            { key: 'asset', label: 'Asset', bold: true },
            { key: 'signal_datetime', label: 'Window', render: (r) => fmtTime(r.signal_datetime) },
            { key: 'decision', label: 'Decision', render: (r) =>
                r.decision === 'enter'
                  ? <span style={S.badge(C.green)}>enter</span>
                  : <span style={S.badge(C.muted)}>skip</span> },
            { key: 'p_profitable', label: 'P', align: 'right', render: (r) => fmtNum(r.p_profitable, 3) },
            { key: 'hold_days', label: 'Hold (d)', align: 'right' },
            { key: 'skip_reason', label: 'Skip reason', render: (r) =>
                <span style={{ color: C.muted, fontStyle: 'italic', whiteSpace: 'normal' }}>{r.skip_reason || '—'}</span> },
          ]}
          rows={trades} getKey={(r) => `${r.asset}-${r.signal_timestamp}`}
          empty="No decisions yet."
        />
      </Panel>

      <Panel title={`Closed positions (${exits.length})`}>
        <DataTable
          columns={[
            { key: 'asset', label: 'Asset', bold: true },
            { key: 'exit_datetime', label: 'Exit', render: (r) => fmtTime(r.exit_datetime) },
            { key: 'hold_days', label: 'Hold (d)', align: 'right' },
            { key: 'funding_events_count', label: 'Events', align: 'right' },
            { key: 'funding_payments_usd', label: 'Funding', align: 'right',
              render: (r) => fmtUsd(r.funding_payments_usd), color: (r) => pnlColor(r.funding_payments_usd) },
            { key: 'net_pnl_usd', label: 'Net P&L', align: 'right', bold: true,
              render: (r) => fmtUsd(r.net_pnl_usd), color: (r) => pnlColor(r.net_pnl_usd) },
          ]}
          rows={exits} getKey={(r) => `${r.asset}-${r.signal_timestamp}`}
          empty="No closed positions yet."
        />
      </Panel>
    </div>
  )
}

function Panel({ title, children }) {
  return (
    <div style={{ ...S.panel, overflow: 'hidden' }}>
      <div style={{ ...S.sectionTitle, padding: '8px 14px', borderBottom: `1px solid ${C.border}` }}>{title}</div>
      <div style={{ maxHeight: 280, overflowY: 'auto' }}>{children}</div>
    </div>
  )
}
