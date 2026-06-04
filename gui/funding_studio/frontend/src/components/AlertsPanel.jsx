/**
 * AlertsPanel.jsx — the funding_alerts ledger. The empty state is itself
 * informative ("no alerts in window" = sit-out), so it is a real rendered
 * state, not a blank.
 */
import { C, S } from '../shared/styles.js'
import { fmtNum, fmtTime } from '../shared/format.js'

export default function AlertsPanel({ alerts }) {
  const list = alerts || []
  return (
    <div style={{ ...S.panel, overflow: 'hidden' }}>
      <div style={{ ...S.sectionTitle, padding: '10px 14px', borderBottom: `1px solid ${C.border}` }}>
        Alerts <span style={{ color: C.muted, fontWeight: 400 }}>({list.length})</span>
      </div>
      {list.length === 0 ? (
        <div style={{ padding: '20px 16px', textAlign: 'center' }}>
          <div style={{ fontSize: 13, color: C.green, fontWeight: 600 }}>No alerts in window</div>
          <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>
            Sit-out — nothing has crossed the P &gt; 0.70 gate.
          </div>
        </div>
      ) : (
        <div style={{ maxHeight: 240, overflowY: 'auto' }}>
          {list.map((a) => (
            <div key={`${a.asset}-${a.timestamp}`} style={{
              padding: '8px 14px', borderBottom: `1px solid ${C.panel2}`,
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              <div>
                <span style={{ fontWeight: 700 }}>{a.asset}</span>
                <span style={{ color: C.muted, fontSize: 11, marginLeft: 8 }}>{fmtTime(a.datetime)}</span>
              </div>
              <span style={{ color: C.green, fontWeight: 600 }}>P={fmtNum(a.p_profitable, 3)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
