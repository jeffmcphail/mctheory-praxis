/**
 * useSessionSocket.js — WebSocket lifecycle for one session's live state,
 * mirroring the mcb_studio direct-to-backend pattern (ws://host:8002, not via
 * the Vite proxy). Exposes the latest state frame + connection flag, and
 * invokes onFrame for each frame (e.g. so the session panel can re-fetch the
 * booked rows when the executor ticks or a replay finishes).
 *
 * The frontend never interprets trade logic from frames — it just renders the
 * phase/summary the backend reports.
 */
import { useEffect, useRef, useState } from 'react'

const WS_BASE = `ws://${window.location.hostname}:8002`

export function useSessionSocket(sessionId, onFrame) {
  const [lastState, setLastState] = useState(null)
  const [connected, setConnected] = useState(false)
  const cbRef = useRef(onFrame)
  cbRef.current = onFrame

  useEffect(() => {
    if (!sessionId) {
      setLastState(null)
      setConnected(false)
      return
    }
    setLastState(null)
    const ws = new WebSocket(`${WS_BASE}/api/ws/sessions/${sessionId}`)
    ws.onopen = () => setConnected(true)
    ws.onmessage = (e) => {
      let frame
      try { frame = JSON.parse(e.data) } catch { return }
      setLastState(frame)
      cbRef.current?.(frame)
    }
    ws.onerror = () => setConnected(false)
    ws.onclose = () => setConnected(false)
    return () => ws.close()
  }, [sessionId])

  return { lastState, connected }
}
