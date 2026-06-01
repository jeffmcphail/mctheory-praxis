# Retro: Cycle 50 -- 44a-D1+D2 thin slice: Binance × Bybit cross-venue funding spreads -- DISCONFIRMED

**Brief:** `claude/handoffs/BRIEF_funding_spread_thin_slice.md`
**Date:** 2026-06-01
**Mode:** Implementation, two-stage pipeline. Took ~3 h Code time + ~1 h compute.
**Status:** DONE -- with cross-venue funding-spread hypothesis DISCONFIRMED on the Binance × Bybit thin slice.
**Predecessor:** Cycle 49 RECON (atlas Exp 13 revival hypothesis #1).
**Commit:** `d0d2aa8`

---

## Summary

Built the cross-venue funding-spread infrastructure (D1) and ran the 4-cell taker/maker × P>0.50/P>0.70 reproduction (D2c). Result: **every cell deeply negative Sharpe** (taker P>0.50 = −11.33, taker P>0.70 = −6.14, maker P>0.50 = −8.40, maker P>0.70 = −4.91). Best single per-model in the best cell: ETH_SPREAD/maker/P>0.70 at Sharpe **−8.88** over 57 trading days. **No model in any cell was OOS-positive.**

Brief's "both <+1.0 across all 4 cells = disconfirm + redirect" threshold fired unambiguously. D2d validation skipped per the brief's "if D2c succeeds" gate. Atlas Exp 13 revival hypothesis #1 updated to DISCONFIRMED status with the structural finding (ADA didn't trade in any cell despite being Exp 13's top-Sharpe asset).

**Net infrastructure value retained:** the D1 schema migration, Bybit data layer, and collector extension are not wasted. Cycle 51 (44c executor integration) can use Bybit as a backup data source for Exp 13's single-venue carry, and the cross-venue plumbing is reusable for future execution-arb explorations even though the cross-venue carry strategy itself is disconfirmed.

Net change:
- `engines/crypto_data_collector.py` (D1a + D1b): funding_rates schema migrated to (asset, venue, timestamp) PK; collect_funding_rates extended with `_fetch_binance_funding` + `_fetch_bybit_funding` helpers and a `--venue` CLI arg
- `scripts/migrations/cycle50_funding_rates_add_venue.py` (new, D1a)
- `scripts/backfill_bybit_funding.py` (new, D1c) -- 22,452 rows added for 6 assets × 2023-01-01..2026-06-01
- `services/funding_collector_service.bat` (D1d): nested 6×2 loop, FAIL_COUNT aggregation across 12 calls
- `engines/funding_spread_strategy.py` (new, D2a, ~450 lines)
- `scripts/run_cpo.py` (D2a): funding_spread strategy dispatch added
- `outputs/funding_spread_repro/` (D2c outputs, ~7 MB across taker/maker × p050/p070 subdirs + SUMMARY.md)
- `TRADING_ATLAS.md` (D3 partial): Exp 13 revival hypothesis #1 marked DISCONFIRMED with the 4-cell matrix + structural finding inline

---

## Execution log

### D1 -- data layer (~1.5 h)

D1a schema migration: cycle50 migration script created funding_rates_new with the new compound PK `(asset, venue, timestamp)`, INSERTed all 22,458 existing rows tagged `venue='binance'`, dropped old table, renamed. Per-asset counts preserved exactly (6 × 3743). init_db updated to match for fresh DBs. Idempotent (re-run detects new schema and exits 0).

D1b collector extension: `_fetch_binance_funding` (CCXT path -- the prior Cycle-40 behavior) + `_fetch_bybit_funding` (direct REST against `GET /v5/market/funding/history`, backward-paginated by endTime cursor, max 200/page). **One bug surfaced + caught by user mid-cycle:** pagination loop had `end_ms = last_ms - 1` where `last_ms` was undefined (intended `last_ts - 1`). D1d's smoke trigger only hit page 1 of 1 (7 days = 21 events, well under the 200/page limit), so the bug stayed latent. User caught on review; fixed; verified with `--days 90` (270 events = 2 pages = exercises the pagination path) before D2.

D1c Bybit backfill: 6 assets × 3742 events each = 22,452 new rows for 2023-01-01..2026-06-01 (134 s wall-clock; zero duplicates; same Cycle 21.5 seconds-aligned ms representation). Per-venue OOS-window stats showed real cross-venue spreads:

```
asset  Binance ann  Bybit ann  signed Δ
BTC      +4.43%      +4.45%    ≈ 0  (parity-bound)
ETH      +4.04%      +4.29%    -0.25%
SOL      -0.60%      +1.02%    -1.62%
XRP      +2.30%      +3.50%    -1.20%
ADA      +3.30%      +3.68%    -0.38%
AVAX     -0.33%      +1.78%    -2.11%
```

D1d bat update: nested 6×2 loop. Smoke trigger via `Start-ScheduledTask`: LastResult 0, log showed "Collection complete (all 12 calls succeeded)." Memory #12 FAIL_COUNT hardening preserved.

D1 pause point: surfaced symmetric per-venue counts (12 (asset, venue) pairs × 3744 rows = 44,928 total). User approved D2.

### D2 -- strategy + reproduction (~1.5 h setup + ~30 min compute)

D2a feature discussion (mid-cycle pause): the brief's 11-feature spec required Bybit perp bar fetching (a new CCXT venue layer) for the `bybit_basis_pct` feature. Surfaced three options A/B/C; user confirmed **Option B (10 features, drop bybit_basis_pct only)**. Rationale documented in funding_spread_strategy.py docstring: cross-venue spread features carry the cross-venue information directly; binance_basis is the higher-info basis feature; 30-min Bybit perp fetcher avoidance for thin-slice spirit.

D2a strategy module: ~450 lines mirroring funding_rate_strategy.py. Key differences:
- Funding loaded from DB (not CCXT) for both venues
- Spot + Binance perp via CCXT (existing pattern reused)
- Hold simulation: `run_funding_spread_hold` — direction = sign(binance - bybit); position receives direction × spread per 8h event; 4-leg TC (entry pair + exit pair across two venues)
- **Thin-slice simplification documented:** cross-venue perp basis P&L assumed zero. Real deployment would have small but nonzero cross-venue basis drift; if D2c had been positive, Cycle 51 would have refined this. Since D2c was disconfirmed, the simplification didn't matter.

D2b config grid: 48 configs/asset × 6 assets = 288 total. Thresholds 3/5/8/12% ann, holds 3/7/14/30 days, pct-pos 0.5/0.65/0.8.

D2c 4-cell matrix:
| Cell | Sharpe | Cum | WinDays | Pos models |
|---|---:|---:|---:|---:|
| taker / P>0.50 | −11.33 | −2.15% | 6/448 | 0/5 |
| taker / P>0.70 | −6.14 | −0.48% | 1/448 | 0/3 |
| maker / P>0.50 | −8.40 | −0.85% | 16/448 | 0/5 |
| maker / P>0.70 | −4.91 | −0.19% | 4/448 | 0/3 |

Phase3 base rates 1.0–3.6% (vs Exp 13's 31–45%) -- the RF discriminated training-set noise with high AUC but lost OOS. Calibration broken: at taker/P>0.50, the [0.80, 1.01) bin had 0% actual win rate vs 1.7% base.

D2d (validation 2023→2024) SKIPPED per disconfirm gate.

### D3 (partial) -- atlas update

`TRADING_ATLAS.md` Exp 13 entry's revival_hypotheses item #1 updated with a 30-line DISCONFIRMED block:
- 2026-06-01 status marker
- 4-cell matrix
- Per-model best-of-best (ETH_SPREAD/maker/P>0.70 = −8.88)
- Maker vs taker ratio (not >2×)
- Phase3 base-rate divergence vs Exp 13
- ADA structural finding (carry-forward to executor design)
- "What is NOT disconfirmed" (the underlying signal exists; only this formulation failed)
- References to SUMMARY.md + brief + retro

atlas_sync.py to be run as part of this cycle's commit pipeline (updates praxis_meta.db embedding for the modified Exp 13 entry).

---

## Acceptance criteria

| # | Criterion | Status |
|:-:|---|:-:|
| 1 | D1 schema extended; Bybit backfilled; collector bat verified | ✅ |
| 2 | D1 pause point: per-venue row counts surfaced | ✅ |
| 3 | D2 4-cell matrix computed | ✅ |
| 4 | D2 pause point: headlines before atlas update | ✅ |
| 5 | SUMMARY.md captures per-config breakdown | ✅ |
| 6 | Single commit + push + SHA insertion follow-up | ✅ standard pattern |
| 7 | Retro captures decision points + findings | ✅ this file |

---

## Decision points (chronological)

1. **D1a migration recipe choice.** SQLite's ALTER TABLE can't change PK; followed the existing `cycle21_funding_rates_to_v2.py` pattern (recreate-table + INSERT-from-old + drop + rename). Idempotent re-run guard.

2. **Bybit data path: CCXT vs direct REST.** Brief specified direct REST; followed brief. CCXT would have been ~20 lines shorter but the brief's reasoning (explicit auditability, no dependency on CCXT version surface) holds.

3. **Pagination bug (D1d smoke didn't catch).** User caught `last_ms` typo on review. The 7-day smoke fetched 21 events (1 page), so the bug stayed latent. Fixed and verified with `--days 90` (270 events, 2 pages, exercises pagination). Lesson: future cycles touching paginated fetchers should include an explicit multi-page smoke test, not just a single-page success.

4. **Feature set: 10 vs 11 (Option B).** Per-cycle discussion -- bybit_basis_pct dropped to avoid a new CCXT venue layer. Documented in the strategy module docstring. The brief's "if D2c lands in +1.0 to +3.0 marginal zone, re-add bybit_basis_pct as Cycle 51 candidate" trigger did NOT fire (D2c far below +1.0); the feature re-add is not necessary.

5. **Cross-venue basis P&L simplification.** Hold simulation assumes synced perp prices across venues (zero cross-venue basis P&L). Documented in funding_spread_strategy.py. This is an OPTIMISTIC assumption; the realistic version would be more negative. Since the optimistic simulation is already negative, refinement is unnecessary for the disconfirmation -- but if a future cycle revives a cross-venue formulation, the basis model is needed.

6. **D2d skipped (disconfirm gate).** Brief's "if D2c clears the disconfirm threshold" was explicit. Could a future cycle want validation-period evidence to strengthen the disconfirm? Possible -- but the OOS evidence is already comprehensive (all 30 model×cell combos negative).

7. **Maker scenario only marginally better than taker.** 26%/20% improvement in Sharpe ratio, not the >2× threshold the brief flagged. Implication noted: "execution path matters" framing doesn't apply; the structural issue is the spread economics themselves, not execution quality.

8. **Bybit collector cadence (open question for Cycle 51).** User asked whether to (a) keep running 8h, (b) scale to daily, or (c) stop. User leans (a). Code endorses (a) — 12 calls/8h is negligible cost; Bybit data may be useful for Cycle 51 executor design (e.g. Bybit as execution backup if Binance liquidity degrades) and for any future cross-venue research with a different formulation.

---

## Structural finding -- ADA didn't trade (carry-forward to Cycle 51)

ADA was the single highest-Sharpe asset in atlas Exp 13 (`ADA_FUNDING` at Sharpe +7.21, +10.5% cum, 186 trading days) and showed the largest spread tails in Cycle 49 RECON (p99 = 29.45% ann, p90 = 13.79%). **Yet in Cycle 50's spread strategy, ADA did not trade in any of the 4 cells.**

Interpretation: ADA's funding edge lives in **single-venue volatility** — bursts of high positive funding that the Exp 13 RF correctly identifies — rather than in **cross-venue spread divergence**. The spread strategy's RF predicted ADA's spread events would not be profitable enough to clear the gate; the strategy correctly sat out.

Why this matters for Cycle 51 (executor integration):
- The executor should commit single-venue (Binance) for the Exp 13 carry. Bybit is data-only / backup.
- If a future cycle revisits cross-venue, ADA likely needs a different feature/return formulation than the other 5 assets — possibly a "volatility-conditioned spread" feature that captures ADA's spread-tail-during-high-funding-vol regime.

Logged in the atlas DISCONFIRMED block under Exp 13 for future researchers.

---

## What this cycle does NOT do

- Does NOT atlas-update Exp 13's main body (verdict + headline figures stay as-is; only the revival_hypotheses block updated)
- Does NOT create a new Exp 14 entry (negative experiments don't get separate entries in this atlas's convention; they're noted under the source hypothesis)
- Does NOT touch the funding_signals / funding_alerts tables or the live monitor
- Does NOT remove the Bybit collector (continues at 8h cadence per user lean (a))
- Does NOT run D2d validation reproduction (skipped per gate)
- Does NOT touch the TEAMS_WEBHOOK_URL / PRAXIS_ALERT_URL alerting infrastructure
- Does NOT change Exp 13's deployment recommendations (gate P>0.70, 6-asset universe, etc. unchanged)

---

## Open items / Cycle 51+ inputs

- **Cycle 51 redirect: 44c -- Real-money executor integration.** Engine 7 single-venue carry is verified (Cycle 40), deployed (Cycle 41), monitored (Cycle 41-43), alerting (Cycle 43-45). Natural next step: close the loop with execution. The Cycle 50 cross-venue infra is not wasted — Bybit becomes execution backup; cross-venue plumbing reusable for future execution-arb (mechanically different from this disconfirmed carry).
- **Carry-forward: bear-regime accumulation (44d).** Passive; ~14 more days to reach 30d of funding_signals data. No Code action needed.
- **Carry-forward: LSTM v2 (44b).** Atlas-flagged "likelihood: low." Defer indefinitely; revisit if executor work surfaces an LSTM-shaped need.
- **Carry-forward: 44h-refactor / engines/_paths.py helper module.** From Cycle 48 retro; still unstarted. The 17 CWD-anchored constants from Cycles 46-47 still each carry their own `Path(__file__).resolve().parent.parent` boilerplate.
- **Carry-forward: per-venue funding_rates health monitoring (D1e follow-up).** Cycle 50 D1e noted the existing 17h threshold continues to work but doesn't distinguish per-venue. Future cycle could add per-venue staleness checks.
- **Carry-forward: 44q sidecar-table monitoring audit.** From Cycle 47 retro; still queued.
- **Bybit collector cadence (Cycle 51 decision).** Code endorses (a) -- keep running 8h cadence.
- **TEAMS_WEBHOOK_URL fallback removal (46a).** After user's .env migrated. No urgency.
- **44q non-DB path audit.** From Cycle 47 retro (models/lstm, models/crypto, data/training Path constants). Still queued.
- **44q sidecar audit / bat unhappy-path test (46j).** Still queued.

The standing queue is getting long. Cycle 51 should pick one or two and ship; the rest stay queued. My lean: 51 = 44c primary, with 44h-refactor as a small concurrent cleanup if time allows.
