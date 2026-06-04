# RETRO — Cycle 54b: Funding Studio Frontend (Control Panel + Live Monitor)

Predecessor: Cycle 54 backend foundation (`c8c70c5`, `4f3e312`, `243414f`,
`33facdd`). 54b ships the React/Vite frontend for STARTING and MONITORING
Engine-7 sessions. Paper-first; the frontend ONLY calls the backend API — it
holds zero trade logic (never recomputes P&L / signals / sizing / gating in
JS; it fetches + renders).

## Commits

| SHA | What |
|---|---|
| `c8c70c5` … `33facdd` | Cycle 54 backend foundation (predecessor) |
| `e7722b2` | Cycle 54b: signal/alert read endpoints + `gui/funding_studio/frontend` |

(Commits stay local — Jeff pushes to origin at the next session start.)

## RECON (R0b–R4b)

- **R0b** — `mcb_studio` frontend is React 18 + Vite 5 + `lightweight-charts`,
  pure inline-style, dark theme. Reused as PATTERNS (not copied): its
  `useBacktestSocket` → `useSessionSocket`, `KickoffForm` → `SessionKickoff`,
  `BatchRunner` table → `SessionList`, `StatsBar` → `RollupBar`, `TradeLog` →
  a generic `DataTable`. The WS client connects DIRECT to the backend port
  (not via the Vite proxy) — mirrored.
- **R1b** — the control panel was fully covered by existing endpoints; the live
  monitor needed the GLOBAL signal/alert feed (see below).
- **R2b** — component map: an always-on regime panel (SignalFeed + AlertsPanel)
  + a per-session panel (header / rollup / positions / decisions / exits) that
  fills in when a session is selected. Useful with NO session running — the
  common case in this sit-out regime.
- **R4b** — Vite on 5174, proxy `/api`+`/ws` → 8002, `package-lock.json`
  committed, `node_modules` gitignored. No port/tooling collision with mcb
  (5173/8001).

## The 2 read-only endpoints (the one scoped backend addition)

`GET /api/signals` + `GET /api/alerts` in `main.py`. **GLOBAL, not
session-scoped** — `funding_signals`/`funding_alerts` are the monitor's output,
keyed `(asset, timestamp)` with no `session_id`. Read-only MAIN via `?mode=ro`,
missing-table → `[]`, NULL-tolerant `min_pct_positive`. Zero trade logic
(reporting over monitor rows). Essential because in the current sit-out regime
a fresh live session books nothing, so without the feed the live monitor would
render blank — the feed is what shows "what the monitor sees" (per-asset P vs
the gate + funding favorability).

## Fork 2 — no live gate knob (the load-bearing UX decision)

Gate is a **REPLAY research knob only** (0.50 / 0.70, Cycle 53's two streams).
For LIVE it is shown fixed-informational at 0.70 and is **never editable**.
Rationale: the monitor already gates alerts at 0.70 upstream; the executor just
consumes `funding_alerts`. A live gate knob would be either inert or an
**unverified second filter** — the exact Cycle-53 strategy-drift class (booking
trades the verified strategy doesn't endorse). So: replay knobs =
window / gate / assets / notional; live knobs = notional only; `config gate`
ON + non-editable; no hold-days knob; kill-switch shown as STATE only
(env-driven, re-read each poll; toggling deferred).

## Operational / analytical split (54b vs 54c)

- **54b (operational — "is it alive + what's it doing now"):** live WS state +
  `last_run_at` freshness · SignalFeed regime · open positions · rollup
  HEADLINE numbers · decisions + exits tables (per-trade `net_pnl_usd` visible).
- **54c (analytical — "analyze over time"):** charted equity CURVE
  (`lightweight-charts`) · deep skip-reason breakdown · cross-session comparison
  · collector-health strip · on-demand live rollup.

## Verification

- `npm run build` clean (45 modules). Frontend serves on 5174; proxy → 8002.
- **End-to-end through the UI** (Playwright driving headless system Edge —
  `playwright-core` + `channel:'msedge'`, no browser download): kicked a replay
  from the form → WS lifecycle → rollup **+$29.96 (84/84)** + decisions(84) +
  exits(84) + positions(0, settled). SignalFeed renders the live sit-out regime
  (ETH 0.664 highest, SOL 0.192 / −19.9% funding, all sub-0.70). Alerts shows
  the intentional "No alerts in window" state. Live session: start → **0
  bookings** (sit-out) → stop → stopped.
- **R3 isolation re-verified through the UI driver:** after UI-kicked replays,
  MAIN `crypto_data.db` `paper_trades` = 0 — replay rows live only in the
  per-session harness DBs. Structural isolation holds through the new code path.
- Safety-belt: frontend grep clean (no exchange / order / external-fetch
  tokens — it only calls its own backend); the backend endpoints are clean.

## Gotcha — IPv4/IPv6 localhost bind (note for next session)

Vite dev binds `localhost` → `::1` (IPv6). A health check to `127.0.0.1:5174`
fails ("unable to connect") even though the server is up — use `localhost:5174`
(or `vite --host`). Separately, the harness's **background shells start in a
different cwd than foreground**, and the cwd drifts between calls — use
absolute paths for background launches and `git -C <repo>` for git ops. Both
cost a couple of round-trips this cycle; noted so the next launch/test doesn't.

## Cycle 54b status

- **Control panel + live monitor SHIPPED**, curl + UI-tested.
  `gui/funding_studio/frontend` (React/Vite, port 5174); backend gained the 2
  read-only signal/alert endpoints.
- **54c queued:** analyze view (equity curve, deep skip breakdown,
  cross-session comparison) + collector-health strip + on-demand live rollup.
- **Real money STILL GATED (unchanged):** D8-live, `max_daily_loss_pct` wiring,
  SOL conservative sizing, human-in-the-loop, and retiring/coexisting the
  PraxisFundingExecutor scheduled task.
