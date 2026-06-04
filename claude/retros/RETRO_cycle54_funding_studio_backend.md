# RETRO — Cycle 54: Live-Trading Control GUI (Backend Foundation)

Engine 7 funding-carry **Funding Studio** — RECON + session schema + executor
session-threading + shared replay engine + FastAPI backend. PAPER-first; zero
exchange credentials; the backend drives the VERIFIED executor and never
reimplements trade logic. Frontend (54b/54c) and real money stay queued.

## Commits (the whole arc)

| SHA | What |
|---|---|
| `c8c70c5` | Cycle 54 prep: quarantine stale `src/praxis/gui/mcb_studio` duplicate (R0) |
| `4f3e312` | Cycle 54 foundation: first-class trading sessions + executor `session_id` threading (Step 1) |
| `243414f` | Cycle 54 backend: shared `funding_replay` engine + `meta.py` health factor-out + `gui/funding_studio` backend (Steps 2-4) |

## RECON decisions (R0–R5)

**R0 — canonical `mcb_studio`.** Two copies existed; `gui/mcb_studio/` is the
superset (batch engine + lockfile + correct `parents[3]` rooting; the root
launcher targets it on 8001). `src/praxis/gui/mcb_studio/` was a stale,
**non-runnable** partial (its `parents[3]` resolves to `src/praxis`, so
`from engines… import` could never work). Resolution: model on `gui/mcb_studio/`;
new GUI at `gui/funding_studio/`; **quarantined** the src copy first (`c8c70c5`,
separate commit) after confirming nothing imports it.

**R1 — executor session seam.** `funding_executor.py` is the SOLE writer of the
paper tables (two INSERT paths). Threaded `session_id` via the same DI pattern as
`now_func`: constructor param → `decide()`/`compute_exit()` dicts →
`persist()`/`persist_exit()` INSERTs. 5 touch-points, zero new code paths.

**R2 — session_id REQUIRED (NOT NULL).** Schema contract enforced universally.
Mechanical correction: SQLite can't `ADD COLUMN … NOT NULL` without a sentinel
default (which would defeat the contract), so the 0-row paper tables got a
**guarded 0-row rebuild** (cycle54a/b). The scheduled + CLI paths open an
**ambient session** via `funding_executor.main()` (`--trigger-source`, default
`cli`; the bat passes `scheduled`) so every booked row is attributable.
`run_once()` refuses a `None` session (fail-loud — `INSERT OR IGNORE` would
otherwise SILENTLY drop a NOT NULL violation and lose the row).

**R3 — replay DB isolation: REVERSED from the original main-DB lean.** Evidence
forced the reversal: the executor's input reads (`load_pending_alerts` + the
risk queries) are **unscoped** by session/mode, so replay's synthetic
`funding_alerts/signals/rates` in the main DB would be picked up and booked by
the LIVE scheduled executor — Cycle 53's drift-as-data-contamination. Decision:
**live → main DB**, **replay → isolated per-session harness DB** (preserving the
Cycle 53 pattern). The executor's existing `db_path` param IS the seam;
`trading_sessions.harness_db_path` records where a replay's rows live.
**Rationale: structural > procedural for isolation** — a physical DB boundary
can't drift, whereas a `WHERE mode=…` filter is one forgotten clause away from
contamination, forever, on real-money-adjacent infra.

**R4 — one replay engine, two callers.** The Cycle 53 script wasn't a per-window
loop but a pipeline (one harness, single `run_once()` at a terminal clock).
Lifted `regenerate_predictions` + `build_harness` + `run_executor` + `DDL`
verbatim into `engines/funding_replay.py` (105 ins / 386 del proves the move).
The **window + clock are parameters** (defaults = Cycle 53's values, so the
Cycle 53 script's numbers are byte-unchanged) — that's what makes it GUI-ready.

**R5 — session controller.** Per-session `asyncio.Task` + `run_in_executor`
(blocking SQLite off the event loop); 60s live cadence; in-memory registry;
`mark_interrupted` on startup; `EXECUTOR_VERSION` + `last_run_at` captured for
freshness-as-verification.

## Key findings

**Scheduled-task liveness (Step 1).** During the post-Step-1 pause, the
PraxisFundingExecutor scheduled task fired **twice for real** (00:20 + 08:20
LOCAL), booking ambient `trigger_source='scheduled'` sit-out sessions — proving
**production was executing the on-disk (uncommitted) code** while git HEAD was
still `c8c70c5`. Lesson: changes to an already-live, scheduler-driven path must
be **committed promptly to close the prod-vs-git gap** (done: `4f3e312`),
distinct from the GUI backend which is off the live path (no scheduler drives
it → safe to hold uncommitted until end-to-end verified).

**Cross-path point 1 — GUI-scoped startup sweep.** The backend's startup
`mark_interrupted_running_sessions` is scoped to `trigger_source='gui'`. The
scheduled task also creates `running` rows (for the ~1s its process is mid-run);
the GUI must NOT mark those — wrong-owner + race. Scheduled-zombie cleanup is a
separate concern owned by the scheduled path (noted, not solved here).

**Cross-path point 2 — concurrent `run_once()` on MAIN.** A GUI live session and
the scheduled task both `run_once()` on the main DB. For PAPER this is benign:
the `paper_trades` PK + `INSERT OR IGNORE` dedups each alert to exactly one row
(attribution is racy — whichever executor wins). This becomes a **HARD
prerequisite for the real-money cycle**: two live executors must not drive one
real account → "retire/coexist the PraxisFundingExecutor scheduled task" stops
being optional once real orders exist.

**Replay clock semantics — settle-past-all-holds.** The one-shot replay injects
`now = window_end + 30d` (past the 14d max hold) so every in-window position
reaches its exit and the rollup is **complete P&L** (entries == exits, 0 open).
The alternative (`now = window_end`, leaving end-of-window positions open) is the
stepped-clock "watch it stream" view, **deferred to 54b/c**.

**Freshness-aware live loop.** The live loop reconstructs the executor each 60s
poll, so it re-reads the kill-switch env / config rather than freezing it at
session start (long-lived-collector constant-staleness lesson). Emergency stop
via `EXECUTOR_KILL_SWITCH` takes effect on the next poll.

## Two design confirmations (per review)

**Independent R3-isolation verification (verification of record).** After the
end-to-end replay booked 84 entries / 84 exits into its harness DB, an
**independent query of the MAIN `crypto_data.db` confirmed `paper_trades` = 0 and
`paper_position_exits` = 0** — the replay's rows are physically absent from MAIN
and invisible to the live executor. This is the structural-over-procedural
payoff: no `WHERE` clause stands between replay scratch and production data.

**Live-session rollup asymmetry — intentional, by design for 54b.** Replay
PERSISTS `pnl_rollup_json` on the session row (its harness DB is
ephemeral/cleanable, so the headline must survive). A live session leaves
`pnl_rollup_json` NULL; its rows persist durably in MAIN, so the analyze view
**computes the rollup on-demand** (`compute_rollup` over MAIN `WHERE session_id`).
This is deliberate (headline-in-row when the detail is ephemeral; compute-on-read
when the detail is durable), not an accidental gap. The `GET /{id}` endpoint
already falls back to on-demand compute when `pnl_rollup_json` is NULL.

## Verification (end-to-end, gate 0.70, window 2025-01-01..2025-04-01)

1. Migrations apply cleanly + idempotent; harness DDL ↔ migrated schema match
   (no central init_db — the migrations + harness DDL are the schema source).
2. Executor threads `session_id` onto both paper tables; None-guard fails loud;
   ambient helpers write a well-formed `trading_sessions` row (smoke ALL PASS).
3. Scheduled-path run (`--trigger-source scheduled`) sits out + exits 0.
4. `funding_replay` extraction: pure move, `import` resolves, Cycle 53 config-gate
   smoke green; health snapshot byte-identical (in-process; MCP reflects on
   relaunch).
5. **Replay end-to-end via API**: WS streamed `regenerating → building_harness →
   running_executor → done`; rollup = **84 entries / 84 exits, net +$29.96**
   (funding +$63.56 − TC $33.60; TC = 84×2×4bps×$500 ✓). All read endpoints
   (trades/exits/positions) dispatched mode-aware to the harness DB.
6. Live smoke: loop ticked (`last_run_at` advanced), 0 alerts → 0 bookings,
   explicit `/stop` → `stopped`.
7. R3 isolation: MAIN paper tables = 0 post-replay (independent verification).
8. Safety-belt grep on the backend: clean (intent tokens; FastAPI inbound
   `@app.post` is not an outbound call). Server torn down, port 8002 free.

## Cycle 54 status

**Backend FOUNDATION complete + curl/WS-testable.** A session kicks off, drives
the verified engine, and booked P&L flows back through the API with structural
live/replay isolation.

- **NO frontend yet** — 54b (control panel + live monitor) and 54c (analyze view
  + collector-health strip) are queued.
- **Real money still gated** on: D8-live (the config gate firing on real
  alerts), `max_daily_loss_pct` wiring, SOL sizing, human-in-the-loop, and
  retiring/coexisting the scheduled task (cross-path point 2).
- The PraxisFundingExecutor scheduled task and the GUI **coexist** for now
  (paper, dedup-protected).
