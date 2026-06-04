/**
 * DataTable.jsx — generic scrollable table used for positions / decisions /
 * exits. Columns are a spec; rendering is pure presentation.
 *
 * columns: [{ key, label, align?, bold?, render?(row)->node, color?(row)->str }]
 */
import { C, S } from './styles.js'

export default function DataTable({ columns, rows, empty = 'No rows', getKey }) {
  if (!rows || rows.length === 0) {
    return (
      <div style={{ padding: 16, color: C.muted, fontSize: 12, textAlign: 'center' }}>
        {empty}
      </div>
    )
  }
  return (
    <table style={S.table}>
      <thead>
        <tr>
          {columns.map((c) => (
            <th key={c.key} style={{ ...S.th, textAlign: c.align || 'left' }}>
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={getKey ? getKey(row, i) : i}
              style={{ background: i % 2 ? C.panel2 : 'transparent' }}>
            {columns.map((c) => (
              <td key={c.key} style={{
                ...S.td,
                textAlign: c.align || 'left',
                color: c.color ? c.color(row) : C.text,
                fontWeight: c.bold ? 700 : 400,
              }}>
                {c.render ? c.render(row) : (row[c.key] ?? '—')}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
