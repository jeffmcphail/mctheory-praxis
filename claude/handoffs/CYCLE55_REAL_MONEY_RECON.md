# Cycle 55 — Real-Money Readiness RECON (R0–R6)

> **RECON-ONLY. ZERO CODE THIS CYCLE.** Deliverable = survey + proposals +
> surfaced risks, then a **HARD PAUSE**. We decide the build sequence together
> before anything is built. Companion: `REAL_MONEY_ORDER_PATH_DESIGN.md` (R5/R6).

## Boundaries (stated back, non-negotiable)
- No order-placement code written this cycle (design only).
- No exchange credentials touched — ever, by Claude. Jeff's hands only.
- No live-trade authorization by Claude, autonomously, ever.
- Human-in-the-loop is **per-trade** — not a session opt-in that then auto-fires.
- Fail-safe: ambiguous/unset mode → **paper / no-trade**. Real money is opt-in,
  never default, never inferred.
- Paper-first holds until the real path is built (later), armed (Jeff), fired
  (Jeff, per-trade).
- **Precondition (not a Cycle-55 task):** D8-live — the config gate firing on a
  **real** production alert in paper — has NOT happened (0 alerts in a month;
  ADA closest, P~0.48, funding +10.9%). Real money cannot proceed until the live
  gate has demonstrably fired correctly in paper. This cycle prepares the
  machinery; it does not remove that gate.

## Two findings that reshape the premise (read before R0)
1. **A working real-money order path already exists** —
   `engines/carry_executor.py` + `scripts/run_carry.py --live` — same strategy,
   naive, no two-leg atomicity, market orders, reads `BINANCE_API_KEY/SECRET`,
   bypasses all Cycle 51–54 safety. The brief's "the executor is paper-only;
   no order path exists" is **incorrect**. Details + disposition in the design
   doc (Risk #1). **This must be quarantined before the real build.**
2. **The verified executor (`engines/funding_executor.py`) genuinely has no
   order path** and forbids `import ccxt` (safety-belt grep). So there are **two
   parallel executors**: the audited paper one (Cycle 51–54) and the legacy
   live one. The real-money work must converge on **one**.

---

## R0. MODE ARCHITECTURE — the spine

**Current state (surveyed):**
- `FundingExecutor.__init__` (`engines/funding_executor.py:207`) params:
  `db_path, defaults_override, force_hold_days, now_func, session_id,
  enforce_config_gate`. **No `mode` param.**
- Mode exists only at the **session** level: `trading_sessions.mode` (NOT NULL),
  `VALID_MODES = ("paper_live", "paper_replay")`
  (`engines/trading_session.py:43`). The table docstring already anticipates
  `real_*` modes "in the real-money cycle."
- `paper_trades` (PK `asset, signal_timestamp`) and `paper_position_exits` carry
  `session_id` (NOT NULL, Cycle 54a/b) but **no direct `mode` column** — a row's
  mode is reachable only by JOIN to `trading_sessions.mode`.
- `paper_replay` rows are **physically isolated** in per-session harness DBs
  (`harness_db_path`); they never touch MAIN. (Cycle 54 reversed to structural
  isolation: "structural > procedural".)
- Grep for `real_live|real_money|mode='real'|real_trades`: **0 hits.** No real
  scaffolding exists yet.

**Proposal:**
- Add `mode: str = "paper"` to the executor ctor; thread from session →
  executor. Values `paper | real`. The order-placement call is a **no-op stub**
  in `paper`; the real `create_*_order` is reachable **only** when
  `mode == "real"` **and** behind the per-trade confirm (R3).
- Extend `VALID_MODES` with a real value (`real_live`); keep `paper_replay`
  replay-only (never real).
- **Fail-safe:** if mode can't be resolved → treat as `paper`. The executor
  refuses to construct a real-order client unless mode is *explicitly* `real`.

**The hard problem — row-level distinction (real-live vs paper-live both in
MAIN):** today the only distinguisher is `session_id → trading_sessions.mode`
(a JOIN) plus physical isolation for replay. Real-live rows would land in the
**same `paper_trades` / `paper_position_exits` tables** as paper-live. Two
options (a genuine fork — Jeff decides):
- **(a) `mode` column** on both paper tables (denormalized from the session),
  default `'paper'`. Row-level, no-JOIN distinction; cheap; defense-in-depth.
- **(b) Separate `real_trades` / `real_position_exits` tables** (or a separate
  real DB). Stronger structural isolation; **matches the Cycle 54 replay
  precedent** (structural > procedural) and avoids putting real money in tables
  literally named `paper_*`.
- **Lean: (b), structural** — consistent with the project's own most recent
  isolation decision — possibly **plus** (a)'s `mode` column as belt-and-braces.
  Either way: **fail-safe = a row with indeterminate mode is read as paper.**

---

## R1. `max_daily_loss_pct` WIRING

**Current state (confirmed):** `max_daily_loss_pct: 0.02` sits in `DEFAULTS`
(`funding_executor.py:143`) but `apply_risk_checks` (419) **only evaluates the
USD breaker**: `daily_loss_remaining = max_daily_loss_usd − daily_loss;
daily_loss_ok = daily_loss_remaining > 0` (443–444). The **pct is never read.**
"Carried, unwired" since Cycle 52 — verified. There is **no equity / balance
concept anywhere** in the executor; `_daily_loss_so_far_usd` just sums realized
`net_pnl_usd` losses booked today from `paper_position_exits`.

**The hard problem — pct OF WHAT?** A pct needs a denominator the executor
doesn't have. For real money, pct-of-account-equity means the executor must
**know equity → an authenticated exchange balance query (a live call)** → so
paper can't truly exercise the real denominator. That's the asymmetry.

**Proposal (dissolves the asymmetry):**
- Make the denominator **allocated capital**, a config knob
  (`account_base_usd` / `allocation_usd`) present in **both** modes — **not**
  whole-account equity. Then:
  `daily_loss_ok = daily_loss < allocation_usd × max_daily_loss_pct`, evaluated
  identically in paper and real. **Paper fully tests the real wiring.**
- The live **balance query becomes a SANITY CHECK at real-session start**, not
  the denominator: refuse to arm if `allocation_usd` exceeds a configured
  fraction of the actual futures-wallet balance. This is the *only* place a live
  read enters, it's read-only, and it gates **arming**, not per-loss math.
- Keep the **USD breaker too** — evaluate `min(usd_breaker, pct_breaker)`
  (whichever is tighter trips first). Fail-safe: if `allocation_usd` is unset →
  fall back to the USD breaker alone (never an unbounded pct).

**Surface:** pct-of-allocation (recommended) vs pct-of-equity (needs the live
read, can't be paper-tested) is Jeff's call. Recommended path makes the breaker
testable end-to-end in paper.

---

## R2. EXECUTOR EXCLUSIVITY — the double-submission gate

**Current state (confirmed):**
- The Windows scheduled task **`PraxisFundingExecutor`** runs
  `python -m engines.funding_executor --trigger-source scheduled` 3×/day
  (00:20 / 08:20 / 16:20 LOCAL), opening an ambient `mode="paper_live"` session
  (`services/funding_executor_service.bat:33`, `register_..._task.ps1`). It is
  **paper, no exchange API.** `MultipleInstances IgnoreNew` stops it
  double-running **itself** — but says nothing about a concurrent GUI executor
  (different process).
- The GUI **live loop** (`controller.py:_live_loop`) constructs a fresh
  `FundingExecutor` each poll and runs `run_once()` on **MAIN**, every
  `LIVE_POLL_SECONDS = 60`.
- So **two executors already `run_once()` on MAIN concurrently.** The
  controller docstring (`controller.py:18–24`) explicitly flags this as
  "benign for paper (PK + INSERT OR IGNORE dedups to one row), session
  attribution racy" and "a HARD prerequisite to resolve before the real-money
  cycle: two live executors must not drive one real account."
- **PK dedup does NOT stop a real exchange order** — it dedups DB rows after the
  fact; an order is already placed.

**Proposal (lean as briefed):**
- **Real mode is GUI-only.** The scheduled `PraxisFundingExecutor` stays
  **paper forever** — never goes real. (Document this as a hard invariant; the
  bat already says "NO EXCHANGE API CALLS.")
- **Single-owner lock** (DB row or lockfile) that a `mode == "real"` executor
  must **acquire or refuse to place**. Advisory locks aren't enough across
  processes — use an atomic DB claim (e.g. an `INSERT` into a
  `real_executor_lock` table with a unique constraint, or `INSERT OR IGNORE`
  + verify ownership) with a heartbeat + stale-owner takeover only after a
  human-set timeout. **Fail-safe: cannot acquire → do not place (no-trade),
  not "place anyway".**
- **Client-order-id idempotency** (design doc §2.6) as the second layer: stops
  *one* executor double-submitting on a retry; the lock stops *two* executors
  each submitting once. Both are needed.

**The cross-path question (answered):** while a real GUI session runs, **does
the paper scheduled task keep booking paper rows on MAIN?** Yes — and that's
fine, *provided* real and paper rows are mode-tagged (R0). The scheduled task
places no orders (paper), so there's no order interference; the only shared
surface is the DB, where mode-tagging + the lock (real-only) keep them
non-interfering. **Confirm in the build:** the scheduled paper task must never
attempt to acquire or be blocked by the real lock (it simply doesn't touch it).

---

## R3. HUMAN-IN-THE-LOOP CONFIRM — the core safety property

**Current state (surveyed):** alerts are sparse (0/month; ADA closest at
P~0.48). The GUI is a thin control surface: `POST /api/sessions` →
`controller.start_live` → `_live_loop` polls `run_once()` and streams frames
over `WS /api/ws/sessions/{sid}`. **Today `run_once()` books decisions with no
human in the loop** — fine for paper, unacceptable for real.

**Proposal — propose-and-hold handshake (per-trade):**
1. In `mode == "real"`, when an alert passes all 11 checks, the executor does
   **not** place. It writes a **`pending_confirmation`** record (asset, both
   legs, sizes, current funding, P, gate-state, all 11 check outcomes, expiry)
   and **HOLDS**.
2. The GUI renders the pending proposal and shows the human **exactly what they
   are authorizing**: asset, **both legs** (long spot qty + short perp qty),
   per-leg notional, current funding (ann + per-window), P vs gate, config-gate
   state, estimated slippage/fees (R6), and the real-mode funding floor check.
3. Human clicks an explicit **per-trade Confirm**. Only then does the executor
   transition `ARMED → place` (design doc state machine §2.7).
4. **(a) Timeout → NO-TRADE.** If no confirm within `confirm_ttl` (e.g. the
   funding window, or a few minutes — Jeff sets it), the proposal **EXPIRES**
   to no-trade. **Never auto-confirm on timeout.**
5. **(c) Pending state is persisted** (a `pending_confirmations` table, not
   in-memory) so a backend restart neither loses it nor auto-fires it. On
   restart, pending rows are re-surfaced as **still-pending** (or expired if
   past TTL) — **never** auto-confirmed. (Mirrors
   `mark_interrupted_running_sessions`' fail-safe: a dead process is never
   trusted as alive.)

**The hard problems (addressed):**
- (a) timeout → no-trade, never auto-confirm — ✔ explicit expiry.
- (b) the proposal shows the full authorization payload — ✔ both legs + funding
  + P + gate.
- (c) restart-safe pending state — ✔ persisted table, re-surfaced as pending.

**Testable in paper now (the scaffold):** build the propose → hold → confirm
loop in **paper** with a paper "confirm" that books a paper row instead of
placing an order. This exercises the entire handshake (including timeout +
restart-safety) with **zero** real-order risk, and is a clean pre-D8-live build.

---

## R4. PER-ASSET SIZING MAP

**Current state (confirmed):** a single notional —
`max_notional_per_asset_usd: 500.0` (`DEFAULTS`) — is used both as the proposed
size in `apply_risk_checks` (`proposed_size`, line 431) and as the booked size
in `decide` (line 515). One number for all six assets (BTC, ETH, SOL, XRP, ADA,
AVAX).

**Proposal:**
- Replace the scalar with a **per-asset notional map** in config, e.g.
  `notional_per_asset_usd: {BTC: …, ETH: …, SOL: <down>, XRP: …, ADA: …,
  AVAX: …}`, with the existing scalar as the **default fallback** for any asset
  not in the map (back-compatible).
- Thread it the same way the scalar threads (DEFAULTS → ctor override →
  `apply_risk_checks` / `decide`), keyed by `alert["asset"]`.
- **SOL sized down** (Cycle 53: SOL basis −8 bps/hold mean, 14 bps std — the
  noisiest leg).
- **Conservative first-real sizing overall**, well below the paper $500/asset —
  see R6 cost-discovery sizing (**$25–$100**). The map makes "everything is a
  parameter" literal.

**The hard problem:** sizing interacts with R6's cost floor and R1's allocation
denominator — the per-asset notional must stay coherent with `allocation_usd`
(sum of intended notionals ≤ allocation) and with the min-notional / lot-size
filters of each Binance market (a $25 notional may be below an asset's min-order
size; the map must respect exchange minimums or skip). Surface per-asset
exchange minimums as a real-mode pre-trade check.

---

## R5. EXCHANGE INTEGRATION — see `REAL_MONEY_ORDER_PATH_DESIGN.md`

Full design (CCXT survey, the long-spot/short-perp path, the **two-leg
atomicity** hard problem with the fill-confirm + unwind ladder, the corrected
"borrow" framing, fee model, entry/exit symmetry, idempotency, state machine)
is in the companion doc. **Headline:** the order path already exists in
`carry_executor.py` and demonstrates the exact atomicity hazard the design
prevents. **No order code, no credentials, no live calls were produced.**

---

## R6. PAPER-vs-REAL GAP — see `REAL_MONEY_ORDER_PATH_DESIGN.md` §3

**Headline number:** the verified **+4.65 Sharpe is paper** (`funding − 8 bps`,
funding read from exact historical data). Real round-trip is **4 taker fills
≈ ~30 bps** (≈3–4× the model), plus slippage, partial fills, funding drift.
**At ADA-like funding (~11%) on a 7-day hold, gross funding ≈ ~21 bps — real
costs plausibly eat the entire edge or invert it** (paper +13 bps → real −9 to
−15 bps). Same lesson as Cycle 53 D7/D8, now in cost space. **First real trades:
size as cost-discovery ($25–$100), expect ~breakeven-to-negative, judge on
execution quality not P&L, and require a real-mode funding floor above the paper
gate.** Full table + sizing guidance in the design doc.

---

## Proposed build sequence (for review — NOT a commitment; we decide together)

Ordered by "safety machinery before any order primitive," paper-testable first:

0. **Quarantine the legacy live path** (`carry_executor.py` / `run_carry.py`) —
   prerequisite, decide fate (Risk #1).
1. **R0 mode scaffold** (paper|real, no-op stub, fail-safe to paper) + **row
   distinction** (structural `real_*` and/or `mode` column).
2. **R1 `max_daily_loss_pct` wiring** (allocation denominator; paper-testable).
3. **R4 per-asset sizing map** (paper-testable).
4. **R3 propose→hold→confirm handshake** in **paper** (paper "confirm";
   persisted pending state; timeout→no-trade; restart-safe). The big safety
   build, zero order risk.
5. **R2 single-owner lock** + scheduled-task-stays-paper invariant.
6. *(gated on D8-live + Jeff's go)* **R5 real order path** — atomic two-leg with
   unwind ladder — behind everything above.

Steps 1–5 are all **paper-testable with no exchange call.** Step 6 is the only
one that touches a real order and is gated last.

---

## HARD PAUSE

No code was written. Everything is surfaced. **We review the full picture —
especially R0 (mode + row distinction), R2 (exclusivity), R3 (confirm flow),
R5 (two-leg atomicity), R6 (the gap), and the legacy-live-path finding (Risk
#1) — and decide the build sequence together before anything is built.**
