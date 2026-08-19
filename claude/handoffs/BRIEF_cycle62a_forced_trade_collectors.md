# BRIEF — Cycle 62A: Forced-Trade Collectors

**Estimated Scope:** M–L (2–4 hr)
**Estimated Cost:** none (public endpoints)
**Mode:** Code (implementation + execution)
**Retro:** `claude/retros/RETRO_cycle62a_forced_trade_collectors.md`
**Predecessor:** Cycle 61 (`400cd64`) — forced-trade data audit
**Companion:** `BRIEF_cycle62b_d1_mechanism_prereg.md` (runs in parallel; independent)

---

## Why this is urgent, and why it precedes the analysis

Cycle 61 established that every forced-trade blocker is a **collector gap, not an analysis
gap** — and that some of the missing data is **not recoverable later**.

From the Cycle 61 T4 venue probe:

| Venue | Open-interest history wall |
|---|---|
| Binance | **hard 30 days** — `since` beyond returns `-1130 startTime is invalid` |
| Bybit | **hard 200 rows** — `since` at −200d, −400d, −730d all return the same 200 rows from 2026-02-01 |

Nothing before ~2026-02 is retrievable from either venue at any granularity. **Every day
without collection is a permanent hole in a series that cannot be reconstructed.** Start
collecting before analysing.

---

## T1 — Binance forced-order (liquidation) stream

The single most decisive addition in this cycle.

Cycle 61 could not compute a false-positive rate for scenario A2 because **a forced
liquidation and a large discretionary market order are identical in the tape**. The
detector found real, clustered, cross-asset-correlated sub-minute flow bursts — dispersion
index 3.79–8.51, cross-asset lift 8–12× — but with no book-stress signature and
*below-chance* concentration on large-move days (lift 0.29 BTC / 0.47 ETH). Whether those
are liquidations is currently unanswerable by construction.

Binance publishes forced orders on a WebSocket stream (`!forceOrder@arr`, all-symbol).
**That converts A2 from inference to observation.**

Requirements:

- New table `liquidations`: `symbol`, `timestamp` (ms), `datetime`, `side`, `price`,
  `quantity`, `quote_qty`, `order_status`, and `venue` (for future multi-venue use).
- **ms timestamps, append-only, single writer** — per the Rule 35 schema standard.
- Reconnect with backoff. **Log gaps explicitly** rather than silently resuming; a gap that
  leaves no trace is indistinguishable from a quiet market.
- **MUST non-zero-exit when a run writes 0 rows in a window where N>0 was expected.** This
  is the standing collector rule: distinguish "ran cleanly with N rows" from "ran cleanly
  with 0 rows when N>0 was expected."
- Report the first 24h of capture: event count, symbol distribution, size distribution.

## T2 — Open interest collector

Repairs regime class F, which T4 showed silently collapsed from five states to three.

- New table `open_interest`: `asset`, `venue`, `timestamp` (ms), `datetime`,
  `open_interest`, `open_interest_value`.
- Binance + Bybit via ccxt `fetchOpenInterest` (both confirmed `True` in the Cycle 61 probe;
  740 / 785 active linear swaps pollable).
- Cadence: hourly is sufficient for a 7-day change feature. **State the choice and why.**
- **Seed once** with whatever history the 30-day / 200-row walls allow, then forward-only.
  **Record the seed boundary date in the retro** so later analysis knows exactly where
  backfill ends and live capture begins — that boundary is a permanent property of the
  dataset.
- Scope: the majors already in `funding_rates`, plus any asset added under T3.

## T3 — Extend `market_data` to an unlock-bearing universe

`market_data` holds **427 rows across ADA / BTC / ETH / SOL / XRP** — five mega-caps whose
supply moves by block subsidy, emission, burn and escrow. Scenario F1 is about tokens with
**VC and team vesting cliffs**, and this universe contains none. Cycle 61 measured zero
supply jumps clearing 1%; the largest anywhere was 0.41%.

- Identify what API `market_data` currently sources from, and whether its asset list is
  config-driven or hard-coded.
- Extend to **≥20 tokens that plausibly carry vesting schedules** — post-2021 launches with
  large locked allocations. **Report the list and the selection rule.** Do not hand-pick by
  eye; a stated rule is auditable and a hand-picked list is not.
- **Requirement:** whatever is added must expose **both** `circulating_supply` and
  `total_supply`, or it does not serve F1. Verify per asset; do not assume.
- If the current provider cannot supply that universe, **stop and report the blocker**
  rather than silently substituting a different supply source. A silently substituted proxy
  is how the Cycle 57 basis-blind P&L happened.

## T4 — Multi-asset OHLCV (repairs regime class K)

Cycle 61 T5 found class K (cross-sectional dispersion) **uncomputable**: it needs ≥3
universe assets, and every OHLCV table in `crypto_data.db` holds exactly BTC and ETH. K
appears in `RegimeState.missing` on 100% of evaluations.

- Extend `ohlcv_daily` (and `ohlcv_4h` if cheap) to the T3 universe.
- Afterwards, **confirm** `compute_dispersion_regime` returns a non-zero state and that K
  stops appearing in `missing`. Verify the acting layer, not the schema.

## T5 — Fix the class F silent degradation (bug; affects PAST results)

Cycle 61 T4, demonstrated by calling `compute_funding_regime` across a grid rather than by
reading the source:

| | |
|---|---|
| Declared states for class F | `[-2, -1, 0, 1, 2]` |
| Reachable **with** OI | `[-2, -1, 0, 1, 2]` |
| Reachable **without** OI | `[-1, 0, 1]` |
| Raises an exception? | **No** |
| Appears in `RegimeState.missing`? | **No** |

Mechanism: `oi_change_7d` initialises to `0.0` and is only overwritten when
`oi_values is not None`; states ±2 require `abs(oi_change_7d) > 0.10`, so they are
unreachable. Neither production caller supplies OI (`engines/cpo_training.py:176`,
`engines/funding_rate_strategy.py:359`).

**Every regime feature vector Praxis has ever produced looked complete and was not.**

- When `oi_series is None`: add `'F'` to `RegimeState.missing` **and** log a warning.
- Add a test asserting F appears in `missing` when OI is absent, and does **not** when OI is
  supplied.
- The generalisable rule: **a degraded axis that does not announce its degradation is worse
  than an absent one.**

## T6 — Record which regime classes are actually live

Update `docs/REGIME_MATRIX.md` with a **status column per class**, using the Cycle 61 T5
measurements (only 8 of 12 axes carry information):

| Class | Cycle 61 finding |
|---|---|
| E (microstructure) | computable, constant over sample |
| F (funding/positioning) | 3 of 5 states reachable; 90.2% on state 0 — **this cycle repairs** |
| H (cross-asset corr) | computable, constant over sample |
| K (dispersion) | uncomputable, 2-asset universe — **this cycle repairs** |
| L (rv/iv spread) | **permanent stub** — returns 0 when `dvol is None`; no options data |

An axis documented as 3-state but always returning 0 is worse than an axis documented as
absent. Record which are live given current collectors, which are degenerate on current
data, which are permanent stubs, and which this cycle repairs.

---

## Constraints

- **Task Scheduler registration is Jeff's hands** (admin). Provide the exact
  `schtasks` / `Register-ScheduledTask` command; do not attempt registration.
- Do not modify existing collectors' behaviour. **Additive only.**
- Everything a parameter. `--validate` / `--verbose` with levels, defaulting to maximum.
- No `.env`, no `*.db`, nothing under `data/` in the delta.
- Scheduled-task `.bat` files read current on-disk code every fire, so **edits go live
  before git commit** — commit promptly after verifying anything on a scheduled path.

## Explicitly NOT in scope

- No strategy, backtest, P&L or Sharpe
- No scenario × regime grid design
- No historical backfill attempts beyond the documented walls

---

## Hand back

Retro containing:

1. First-capture counts for T1 (liquidations) and T2 (open interest)
2. The T3 universe list **and the selection rule that produced it**
3. Confirmation that K is computable and that F announces its degradation
4. **The OI seed boundary date** — where backfill ends and live capture begins
5. The exact scheduled-task commands for Jeff to run
6. Any blockers, stated rather than worked around

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-08-19 (Chat: praxis_main_current)*
*Changes: Initial brief. Cycle 62A collectors — liquidation stream, open interest,
unlock-bearing market_data universe, multi-asset OHLCV, plus the class F silent-degradation
fix and a regime-class status audit.*
