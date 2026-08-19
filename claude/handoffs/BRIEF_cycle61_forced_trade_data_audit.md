# BRIEF — Cycle 61: Forced-Trade Data Audit

**Estimated Scope:** M (30 min – 2 hr)
**Estimated Cost:** none (no API calls; read-only queries against `crypto_data.db`)
**Mode:** Code (measurement only)
**Retro:** `claude/retros/RETRO_cycle61_forced_trade_data_audit.md`

---

## Context

Chat has written two methodology documents shipped with this brief:

- `docs/FORCED_TRADE_SCREEN.md` — a three-question design-stage filter, retroactively
  validated against Exp 1, Cycle 60, Cycle 59 and Engine 7. It kills all three
  turnover-death failures, passes the one structurally sound strategy, and flags exactly
  the defect that inflated it.
- `docs/FORCED_TRADE_TAXONOMY.md` — 16 scenarios where participants are *compelled* to
  trade, triaged by mechanism, data availability and professional crowding. Three P1
  candidates.

This brief does **not** build or test a strategy. It answers the empirical questions the
taxonomy raises, so that a scenario × regime grid can be designed on measured event
counts rather than guesses.

**Why this matters more than it sounds:** forced-trade events are rare by construction. A
scenario × regime grid multiplies scenarios by 12 regime classes. If a scenario yields 15
events per year, most cells hold one or two observations and are not estimable. **The
event counts determine whether the grid is buildable at all** — and if it isn't, we need
to know that before designing it, not after.

---

## Objective

Measure, for each P1 scenario, how many events exist and whether they are cleanly
identifiable from data Praxis already holds.

**No strategy. No P&L. No Sharpe. No backtest.** Counts, detectability, and feasibility.

---

## Tasks

### T1 — A2, liquidation cascades: are they identifiable without a liquidation feed?

We have no liquidation feed. The hypothesis is that cascades are detectable as bursts of
one-sided *aggressive* flow. `trades` carries `side` and `is_buyer_maker`, so aggressor
side is known rather than inferred.

Build a **detector as a parameterised function**, not a fixed rule:

```
cascade candidate at time t IF
    one-sided aggressive volume over window W  >  K × trailing median
AND signed imbalance over W                    >  I  (fraction, e.g. 0.8)
AND concurrent price move                      >  M  (in trailing vol units)
```

All of `W`, `K`, `I`, `M` are parameters with sensible defaults; nothing hard-coded.

Report:
- event count per year and per asset, at 2–3 threshold settings (show sensitivity)
- duration distribution of detected events
- concurrent `order_book_snapshots` depth vs the trailing norm (does book depth collapse during candidates? that is the corroborating signature)
- **an honest false-positive assessment.** Cross-check a sample of detections against
  known market-wide events. If detections do not concentrate around known stress dates,
  say so plainly — a detector that fires uniformly is finding volume bursts, not cascades.

`trades` covers BTC + ETH only, from 2026-04-29. **State the usable span explicitly**; if
it is ~4 months, that caps the study and needs saying up front rather than discovering later.

### T2 — F1, token unlocks: do `circulating_supply` jumps mark unlock events?

`market_data` carries `circulating_supply` and `total_supply`.

- How many assets have a usable `circulating_supply` series, over what span, at what
  sampling frequency?
- Detect discrete jumps (parameterised threshold, e.g. >1% single-period increase).
  Count events per asset per year.
- Are jumps **discrete cliffs** (consistent with vesting) or **smooth drift** (consistent
  with emissions)? This is the load-bearing question: only cliffs are compulsion. Report
  the distribution of jump sizes and state which pattern dominates.
- Spot-check 3–5 detected jumps against public unlock records and report whether the
  dates line up. If `market_data` is daily and coarse, say how precisely an event can be dated.

### T3 — D1, leveraged tokens: what is the candidate universe?

`engines/xsec_reversal/universe.py` holds `DEFAULT_LEVERAGED_PATTERNS`
(`UPUSDT`/`DOWNUSDT`/`BULLUSDT`/`BEARUSDT`) — the list Cycle 60 *excluded*. Here it is
the candidate list.

- How many such symbols exist in the Binance archive (reuse the Cycle 60 enumeration path)?
- What is their date coverage, and how many are delisted?
- Do we have, or can we cheaply collect, the matching **underlying** series for each
  leveraged token? The strategy needs both legs, and without the underlying there is no study.

### T4 — Open interest: how bad is the gap?

`docs/REGIME_MATRIX.md` class F ("Funding / positioning") specifies *"Funding rates + OI"*.
There is **no OI table** in `crypto_data.db`.

- Confirm the absence and report what `compute_funding_regime` in
  `engines/regime_engine.py` actually uses today. Does it silently degrade to funding-only,
  or does it error?
- Is OI available from the venues we already poll via CCXT, at what cadence, and with what
  history? A forward-only OI collector is cheap; a historical backfill may not be
  available at all — say which.

### T5 — Regime axis feasibility

For each P1 scenario, using the event counts from T1–T3:

- Compute events per regime class, using `RegimeEngine` on the same window.
- Report the **cell occupancy distribution**: how many (scenario × regime) cells have
  0, 1–2, 3–9, 10+ events?
- State plainly whether a grid is buildable, and at what regime granularity. If 12 classes
  is too fine, report occupancy under a coarser collapse (e.g. 3 vol levels × 3 liquidity
  levels) and say which granularity the data can actually support.

---

## Constraints

- **Read-only.** No writes to `crypto_data.db`.
- New code goes in `engines/forced_trade/` with `--validate` / `--verbose` levels
  defaulting to maximum.
- Everything a parameter. No hard-coded thresholds.
- Do **not** tune any detector threshold to make event counts look better. Report
  sensitivity across settings; the spread across thresholds *is* the finding.
- If a task is blocked by missing data, **stop and report the blocker.** Do not substitute
  a proxy without flagging it — a silently substituted proxy is how Cycle 57's basis-blind
  P&L happened.

## Explicitly NOT in scope

- No strategy, P&L, Sharpe, or backtest
- No scenario × regime grid design (Cycle 62, and only if T5 says it is buildable)
- No ML (that is tier 3, and it is gated on a validated scenario existing)
- No external data purchases

---

## Hand back

The retro at `claude/retros/RETRO_cycle61_forced_trade_data_audit.md` containing:

1. Event counts per P1 scenario, with threshold sensitivity
2. Detectability verdict per scenario, including honest false-positive assessment
3. The OI gap assessment and whether it is cheaply closable
4. **The cell-occupancy table from T5 and a plain buildable / not-buildable verdict**
5. Any blockers, stated rather than worked around

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-08-18 (Chat: praxis_main_current)*
*Changes: Initial brief. Cycle 61 data audit for the forced-trade program — event counts,
detectability, and regime-grid feasibility for the three P1 scenarios.*
