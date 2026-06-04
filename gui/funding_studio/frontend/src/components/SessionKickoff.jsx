/**
 * SessionKickoff.jsx — start a paper_live or paper_replay session.
 * Replay knobs: window / gate(0.50,0.70) / assets / notional.
 * Live knobs: notional only; gate shown fixed-informational at 0.70.
 * config gate ON + non-editable (verified strategy); no hold-days knob.
 */
import { useState } from 'react'
import { C, S } from '../shared/styles.js'

const ASSETS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX']

export default function SessionKickoff({ onStart }) {
  const [mode, setMode] = useState('paper_replay')
  const [notional, setNotional] = useState(500)
  const [start, setStart] = useState('2025-01-01')
  const [end, setEnd] = useState('2025-04-01')
  const [gate, setGate] = useState('0.70')
  const [assets, setAssets] = useState([...ASSETS])

  const toggleAsset = (a) =>
    setAssets((p) => (p.includes(a) ? p.filter((x) => x !== a) : [...p, a]))

  const replayValid = mode !== 'paper_replay' || (assets.length > 0 && start && end)

  const submit = () => {
    const config_overrides = { max_notional_per_asset_usd: Number(notional) }
    if (mode === 'paper_replay') {
      onStart({
        mode, replay_start: start, replay_end: end, gate: Number(gate),
        assets: assets.length === ASSETS.length ? null : assets,
        config_overrides,
      })
    } else {
      onStart({ mode, config_overrides })
    }
  }

  return (
    <div style={{ ...S.panel, padding: 16 }}>
      <div style={{ ...S.sectionTitle, marginBottom: 12 }}>Start a session</div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
        <span style={S.chip(mode === 'paper_replay')} onClick={() => setMode('paper_replay')}>⟲ Paper Replay</span>
        <span style={S.chip(mode === 'paper_live')} onClick={() => setMode('paper_live')}>● Paper Live</span>
      </div>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-end' }}>
        {mode === 'paper_replay' && (
          <>
            <Field label="Window from">
              <input type="date" style={S.input} value={start} onChange={(e) => setStart(e.target.value)} />
            </Field>
            <Field label="Window to">
              <input type="date" style={S.input} value={end} onChange={(e) => setEnd(e.target.value)} />
            </Field>
            <Field label="Gate (replay)">
              <select style={S.select} value={gate} onChange={(e) => setGate(e.target.value)}>
                <option value="0.70">P &gt; 0.70 (live gate)</option>
                <option value="0.50">P &gt; 0.50 (headline)</option>
              </select>
            </Field>
          </>
        )}
        <Field label="Notional / asset (USD)">
          <input type="number" style={{ ...S.input, width: 120 }} value={notional} step={50} min={0}
                 onChange={(e) => setNotional(e.target.value)} />
        </Field>
        <button style={S.btn(C.green, !replayValid)} disabled={!replayValid} onClick={submit}>
          ▶ Start {mode === 'paper_replay' ? 'Replay' : 'Live'}
        </button>
      </div>

      {mode === 'paper_replay' && (
        <div style={{ marginTop: 12 }}>
          <div style={{ ...S.label, marginBottom: 6 }}>Assets</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {ASSETS.map((a) => (
              <span key={a} style={S.chip(assets.includes(a))} onClick={() => toggleAsset(a)}>{a}</span>
            ))}
          </div>
        </div>
      )}

      <div style={{
        marginTop: 14, paddingTop: 12, borderTop: `1px solid ${C.border}`,
        display: 'flex', gap: 20, flexWrap: 'wrap', fontSize: 11, color: C.muted,
      }}>
        <span>Live gate: <b style={{ color: C.sub }}>P &gt; 0.70</b> (fixed)</span>
        <span>config gate: <b style={{ color: C.green }}>ON</b> (verified strategy)</span>
        <span>kill-switch: <b style={{ color: C.sub }}>env-controlled</b> (shown per running session)</span>
        <span>direction: <b style={{ color: C.sub }}>long spot / short perp</b></span>
      </div>
    </div>
  )
}

function Field({ label, children }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <span style={S.label}>{label}</span>
      {children}
    </div>
  )
}
