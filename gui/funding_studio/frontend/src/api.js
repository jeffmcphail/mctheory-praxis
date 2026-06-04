/**
 * api.js — tiny fetch wrapper. Relative paths go through Vite's proxy to the
 * funding_studio backend (:8002). The frontend ONLY calls these endpoints; it
 * holds no trade logic and never touches a DB directly.
 */

export async function getJSON(path) {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`GET ${path} → HTTP ${r.status}`)
  return r.json()
}

export async function postJSON(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  })
  if (!r.ok) throw new Error(`POST ${path} → HTTP ${r.status}`)
  return r.json()
}
