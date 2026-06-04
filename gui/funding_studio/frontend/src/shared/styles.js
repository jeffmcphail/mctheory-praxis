/**
 * styles.js — dark-theme tokens + shared inline-style objects, lifted from the
 * mcb_studio frontend so both studios share one visual idiom.
 */

export const C = {
  bg:     '#131722',
  panel:  '#1e2430',
  panel2: '#1a2035',
  bg2:    '#0f1520',
  border: '#2a3248',
  text:   '#d1d4dc',
  sub:    '#9ba8bf',
  muted:  '#5d6b8a',
  accent: '#40c4ff',
  green:  '#26a69a',
  red:    '#ef5350',
  amber:  '#ffb74d',
}

export const S = {
  panel: {
    background: C.panel,
    border: `1px solid ${C.border}`,
    borderRadius: 6,
  },
  sectionTitle: {
    fontSize: 10,
    fontWeight: 700,
    color: C.muted,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  label: {
    fontSize: 10,
    fontWeight: 600,
    color: C.muted,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  select: {
    background: C.bg,
    border: `1px solid ${C.border}`,
    borderRadius: 4,
    color: C.text,
    padding: '5px 8px',
    fontSize: 13,
    cursor: 'pointer',
    minWidth: 120,
    outline: 'none',
  },
  input: {
    background: C.bg,
    border: `1px solid ${C.border}`,
    borderRadius: 4,
    color: C.text,
    padding: '5px 8px',
    fontSize: 13,
    width: 130,
    outline: 'none',
  },
  btn: (color, disabled) => ({
    background: disabled ? C.border : color,
    border: 'none',
    borderRadius: 4,
    color: disabled ? C.muted : '#fff',
    padding: '7px 18px',
    fontSize: 13,
    fontWeight: 600,
    cursor: disabled ? 'not-allowed' : 'pointer',
    whiteSpace: 'nowrap',
  }),
  chip: (active) => ({
    padding: '4px 12px',
    borderRadius: 3,
    fontSize: 12,
    cursor: 'pointer',
    userSelect: 'none',
    border: `1px solid ${active ? C.accent : C.border}`,
    background: active ? 'rgba(64,196,255,0.12)' : C.bg,
    color: active ? C.accent : C.sub,
  }),
  badge: (color) => ({
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: 3,
    fontSize: 11,
    fontWeight: 600,
    border: `1px solid ${color}`,
    color: color,
    background: 'transparent',
  }),
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: 12,
  },
  th: {
    padding: '6px 12px',
    background: C.panel2,
    color: C.muted,
    fontWeight: 600,
    textAlign: 'left',
    fontSize: 10,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    whiteSpace: 'nowrap',
    position: 'sticky',
    top: 0,
    borderBottom: `1px solid ${C.border}`,
  },
  td: {
    padding: '6px 12px',
    borderBottom: `1px solid ${C.panel2}`,
    whiteSpace: 'nowrap',
    color: C.text,
  },
}

// Status → accent color, shared by SessionList + SessionHeader badges.
export const STATUS_COLOR = {
  created:     C.sub,
  running:     C.accent,
  completed:   C.green,
  stopped:     C.amber,
  error:       C.red,
  interrupted: C.red,
}
