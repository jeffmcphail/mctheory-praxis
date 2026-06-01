# Cycle 50 -- 44a-D1+D2 thin slice: Binance × Bybit cross-venue funding spreads

**Predecessor:** Cycle 49 RECON (atlas Exp 13 revival hypothesis #1).
Thin-slice variant per Cycle 49's recommended Option 2:
Binance × Bybit only, OKX deferred to Cycle 51 if D2c lands Sharpe ≥
+1.0. Hyperliquid + Deribit out of scope (different cadences /
mechanisms per Cycle 49 RECON disconfirmation).

**Mode:** Implementation, two-stage pipeline. ~3-5h Code time total.

## Stage D1: Data layer

- **D1a:** schema migration on funding_rates -- add `venue` column to
  PK via recreate-table pattern; existing rows backfill to
  'binance'. Migration script: scripts/migrations/cycle50_funding_rates_add_venue.py.
  Idempotent on re-run. init_db() in engines/crypto_data_collector.py
  updated to match for fresh DBs.
- **D1b:** extend collect_funding_rates with `--venue` arg (default
  'binance'). Add `_fetch_bybit_funding` direct-REST helper for
  Bybit's GET /v5/market/funding/history (backward-paginated by
  endTime cursor). INSERT OR IGNORE preserves idempotency under
  new PK.
- **D1c:** new scripts/backfill_bybit_funding.py modeled on
  scripts/backfill_funding_history.py. 6 assets × ~24-30 months
  (2023-01-01 onward to support train + validation windows).
- **D1d:** extend funding_collector_service.bat with nested loop:
  6 assets × 2 venues = 12 invocations per scheduled run. Memory
  #12 FAIL_COUNT aggregation hardening preserved.
- **D1e:** verify funding_rates health monitoring still works
  (17h threshold unchanged; MAX(timestamp) across venues).
- **D1 pause point** at end of D1: row counts per venue, smoke-test
  the extended collector.

## Stage D2: Strategy + reproduction

- **D2a:** engines/funding_spread_strategy.py with feature set
  (~10-15 features) per Cycle 49 RECON recommendation. Feature
  list resolved during cycle via Cycle 50 D2a feature discussion
  pause point -- **Option B confirmed: 10 features, drop
  bybit_basis_pct** (cross-venue spread features carry the cross-
  venue information directly; binance_basis is the higher-info
  basis feature; 30-min Bybit perp fetcher avoidance for thin-slice
  spirit).
- **D2b:** 288-config grid (4 thresholds × 4 holds × 3 pct-pos
  × 6 assets).
- **D2c:** phase2/3/4 reproduction at 4-cell matrix [P>0.50 /
  P>0.70 × taker 16bps / maker 7bps]. Output to
  outputs/funding_spread_repro/{taker,maker}/{cpo, p050, p070}.
  SUMMARY.md with the 4-cell result.
- **D2d:** validation reproduction (train 2023 → test 2024 at
  maker P>0.70) IF D2c clears the disconfirm threshold (Sharpe
  ≥ +1.0 in at least one cell).

## D2 pause point + thresholds (per brief)

After D2c, surface 4-cell headline numbers before any atlas update.

- maker > 2× taker Sharpe: re-shape Cycle 51 OKX-vs-executor
  prioritization (execution path matters more than universe expansion)
- both cells (taker, maker) < +1.0: disconfirm + redirect to
  44d / 44b / 44c
- maker P>0.70 in +1.0..+3.0: "publishable but caveat-heavy"
- maker P>0.70 ≥ +3.0: atlas-comparable to Exp 13, unblocks Cycle 51
  OKX + atlas Exp 14 entry

## Out of scope

- OKX leg (Cycle 51 if positive)
- Hyperliquid + Deribit (Cycle 49 disconfirmed)
- Live monitor for spread strategy (gated)
- Atlas Exp 14 entry (Cycle 51 if positive)
- Executor integration (44c; deferred)

## Acceptance

1. D1: funding_rates schema extended; Bybit backfilled across train +
   OOS; collector bat updated and verified.
2. D1 pause point: per-venue row counts surfaced.
3. D2: 4-cell matrix computed.
4. D2 pause point: headline numbers before atlas update.
5. SUMMARY.md captures per-config breakdown.
6. Single commit + push + SHA insertion follow-up at end.
7. Retro captures decision points and findings.

## Mid-flight surface triggers

- Per-venue basis feature complexity (D2a) -- already triggered;
  resolved to Option B.
- maker > 2× taker (D2c trigger) -- did NOT fire; maker only
  20-26% better than taker, both deeply negative.
- both <+1.0 (D2c disconfirm) -- FIRED. All 4 cells negative
  Sharpe; redirect to 44c (executor) per user lean.
- maker +1.0..+3.0 marginal (D2c bybit_basis re-add candidate) --
  not applicable; below +1.0.
- D2d validation: SKIPPED per disconfirm-gate (cleared).
