# Cycle 40 -- Engine 7 (Funding Carry) full-universe paper reproduction

**Predecessor:** Cycle 39 RECON (Track E reframe -> Track A). Atlas
Exp 13 (id=11) CONFIRMED POSITIVE; headline Sharpes +4.65 primary /
+10.78 validation marked suspect-LOW by Cycle 37; this cycle escalates
to verified-or-refuted on a reconstructed paper-trading run.

**Mode:** Implementation. ~3 stage pipeline: (1) historical funding
data acquisition for the 4 missing assets, (2) phase2/3/4 reproduction
on the 6-asset universe, (3) atlas verdict reconciliation + commit.

**Risk:** medium. Real compute (~1-2h) on real data; no live trading;
no executor changes. Failure modes are mostly recoverable (re-fetch,
re-run).

**Scope cap:** ~3-5h end-to-end, distributed as:
- D1 historical data fetch (4 assets x 36mo): ~30-60 min wall, mostly
  Binance REST + sleep cadence
- D2 phase2/3/4 reproduction (6 assets, train 2024 -> OOS 2025-Q1
  through 2026-Q1): ~1-2h compute
- D3 atlas reconciliation + retro + commit: ~30-60 min

## Why this cycle exists

Cycle 39 RECON established three load-bearing facts:

1. **No surviving funding-carry phase3 artifact on disk.** The
   `phase3_models_funding.joblib` file is a misnamed Exp 10 TA artifact
   (64 TA models, zero *_FUNDING keys). Likely a filename collision
   during the post-2026-04-24 SSD-recovery, or an earlier naming bug.
   Worth recording as a small disk-hygiene finding in the retro;
   doesn't change cycle scope but explains why "model exists on disk"
   in prior project memory was misleading.

2. **No surviving phase4 paper-trading output on disk.** Atlas's
   headline Sharpe figures (+4.65 / +10.78) are not reproducible
   without re-running phase2/3/4. Cycle 37's suspect-LOW classification
   on Exp 13 escalates to "unverifiable without re-run" -- this cycle
   is the re-run.

3. **No live paper-trading process exists.** PraxisFundingCollector
   is a pure data collector. There's no monitor, no alert, no signal
   log. The cycle's "live-vs-paper reconciliation" framing from
   pre-Cycle-39 doesn't apply; this is purely paper reproduction.

## What this cycle produces

A reconstructed paper-trading run that confirms-or-refutes atlas Exp
13's headline numbers on the full 6-asset universe (BTC, ETH, SOL,
XRP, ADA, AVAX; BNB excluded per atlas). Three possible outcomes:

- **Reproduction within atlas tolerance** (Sharpe within +/-15% of
  +4.65 primary, +/-15% of +10.78 validation): atlas Exp 13 verdict
  upgraded from suspect-LOW to verified; cycle ends with atlas
  note + retro.

- **Reproduction materially below atlas** (e.g. Sharpe < +3.5 primary,
  or signs of overstated headline numbers): atlas Exp 13 verdict
  re-opens for review; cycle ends with detailed delta analysis and
  a Cycle 40b decision point on whether to revise downward.

- **Reproduction materially above atlas** (Sharpe > +5.5 primary):
  worth investigating for any methodological drift (e.g. different
  training data; pre-filter change; OOS extension). Atlas updated
  with the higher number AND a note about the verification context.

## Implementation tasks

### D1: Historical funding-rate data acquisition for missing assets

The 4 missing assets (SOL, XRP, ADA, AVAX) need their funding-rate
history backfilled to match atlas's training and OOS windows:

- Training: 2024-01-01 to 2024-12-31 (1097 events per asset at 8h
  cadence)
- OOS primary: 2025-01-01 to 2026-03-26 (full atlas window)

Binance Futures REST API endpoint: `/fapi/v1/fundingRate?symbol=SOLUSDT
&startTime=...&endTime=...&limit=1000`. Limit is 1000 per call;
14 months of 8h funding = 1278 events per asset = 2 calls per asset.
Plus 1.0s sleep cadence to avoid rate limiting.

Code path:
- Check whether `engines/funding_collector.py` (or wherever the live
  collector lives) has a `--historical` or `--backfill` mode. If yes,
  use it. If no, write a small one-shot script
  `scripts/backfill_funding_history.py` that does the same fetch +
  insert as the live collector does for new events.
- Per Rule 35 (memory #14): every temporally-indexed table needs
  INTEGER ms timestamp + ISO datetime + natural-key PK. Verify the
  funding_rates schema already conforms (BTC/ETH rows do, so this is
  a sanity check, not a migration).
- Insert with `INSERT OR IGNORE` on natural-key PK so re-runs are
  idempotent.

Validation after fetch:
- All 6 assets present in funding_rates
- Per-asset gap analysis: `SELECT date(datetime), COUNT(*) FROM
  funding_rates WHERE asset=? GROUP BY date(datetime) HAVING
  COUNT(*) != 3 ORDER BY date(datetime) LIMIT 50` - flag any
  incomplete days (expected: occasional 0-2-event days when Binance
  has missed events; flag systematic gaps for investigation).
- Per-asset stats: mean funding rate, positive_share, range over
  the OOS window. Document in retro.

### D2: Phase2/3/4 reproduction

Use existing CPO framework. Per atlas Exp 13:

- Framework: Engine 7 (Event/Signal) + Engine 3 (Allocation)
- Training: 2024-01-01 to 2024-12-31
- OOS primary: 2025-01-01 to 2026-03-26
- Configs: 36 per asset x 6 assets = 216 configs total (atlas says
  "7 assets x 36 configs" = 252; we're at 6 assets so 216)
- Feature set: 11 hand-crafted features (per atlas test_conditions)
- Pre-filter: none; gate is RF P > 0.70
- Risk management: Kelly-sized within configurable max-leverage;
  long-only structural exposure

Specific code paths to identify before running:
- Find Engine 7's CPO entry point. Memory mentions
  `engines/funding_carry_strategy.py` or equivalent; verify path.
- Find the feature-generation function for the 11 hand-crafted
  features. Confirm against atlas's listed features (annualized
  funding, percentile rank, sustained-positive flag, basis, OI change,
  volatility level, trend strength, etc.).
- Verify the run_cpo.py invocation shape matches what was used in
  Exp 13's original 2026-03-26 run.

Run sequence:
- Phase2: ~10-30 min (216 configs x ~13 months of 8h events =
  ~280k strategy-events; per-event compute is small)
- Phase3 RF training: ~5-15 min (6 models, ~13k samples each)
- Phase4 portfolio backtest at gate P>0.70: ~5-15 min
- Also produce a P>0.50 result for direct atlas comparison (Sharpe
  +4.65 headline)

Outputs go to `outputs/funding_carry_repro/`:
- `phase2_returns.parquet`
- `phase3_models.joblib` (this time correctly named!)
- `phase4_portfolio_p050.parquet` (gate 0.50 for atlas headline match)
- `phase4_portfolio_p070.parquet` (gate 0.70 for recommended-live
  parameters)
- `SUMMARY.md` with per-gate result tables

### D3: Atlas reconciliation + retro + commit

Compare reproduction outputs to atlas:

| Comparison | Atlas | Reproduction | Within +/-15%? |
|---|---|---|---|
| Sharpe primary (P>0.50) | +4.65 | TBD | TBD |
| Cum primary (P>0.50) | +1.27% | TBD | TBD |
| Max DD primary (P>0.50) | -0.15% | TBD | TBD |
| Sharpe primary (P>0.70) | +4.45 | TBD | TBD |
| Per-model Sharpe BTC | +5.86 | TBD | TBD |
| Per-model Sharpe ETH | +6.58 | TBD | TBD |
| Per-model Sharpe SOL | +3.69 | TBD | TBD |
| Per-model Sharpe XRP | +5.27 | TBD | TBD |
| Per-model Sharpe ADA | +7.21 | TBD | TBD |
| Per-model Sharpe AVAX | +1.98 | TBD | TBD |

The 6 per-model Sharpes are the most informative single check: if
they all reproduce within +/-20% of atlas, the headline numbers
are solid. If the dispersion is wider, something has drifted.

Per-gate atlas updates:
- If reproduction confirms: add Cycle 40 verification note to atlas
  Exp 13 entry, mark verdict "POSITIVE -- CONFIRMED + VERIFIED";
  update Cycle 37 suspect-LOW classification.
- If reproduction refutes: surface delta analysis in retro; pause
  before atlas edit; user decision on whether to revise the entry.

### Possible secondary atlas updates (in scope if findings warrant)

If D2 surfaces the validation Sharpe +10.78 also reproducing
(requires train 2023 -> test 2024 with frozen params, per atlas
addendum), this can also be re-verified. **However:** atlas's 2023
training window requires 2023 funding rates, which are NOT in our
current data. Binance Futures launched perp funding in 2019, so 2023
data is available via REST but we don't have it locally.

**Decision per Code:** if D1's fetch can extend to 2023 cheaply
(~5 extra min wall-clock per asset), include 2023 in the backfill
and run validation reproduction. If 2023 has data-quality issues
(missing assets, gaps, schema changes on Binance's side), document
and skip; revisit in Cycle 41.

### Commit + push

Single commit covering:
- Backfill script (`scripts/backfill_funding_history.py` or similar)
- Reproduction outputs (`outputs/funding_carry_repro/` tree)
- Atlas Exp 13 update (verification note)
- Retro file `claude/retros/RETRO_engine7_repro.md`
- Brief `claude/handoffs/BRIEF_engine7_repro.md`

Commit message template:
```
Cycle 40: Engine 7 (Funding Carry) full-universe paper reproduction --
[CONFIRMED / REFUTED / PARTIAL]

Reproduced atlas Exp 13 phase2/3/4 on full 6-asset universe (BTC, ETH,
SOL, XRP, ADA, AVAX) against the historical funding data backfilled
this cycle. [Headline result vs atlas baseline.]

Atlas update: Exp 13 verdict [updated to CONFIRMED + VERIFIED /
revised / unchanged]; Cycle 37 suspect-LOW classification [resolved /
[etc]].

Backfill: [N assets x N months of funding rates fetched from Binance
Futures REST; idempotent insert via natural-key PK; zero gaps
post-fetch].

[Per-gate result table for the commit body.]
```

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | D1 completes; funding_rates table contains all 6 assets with full atlas window coverage |
| 2 | D2 produces phase2/3/4 outputs in `outputs/funding_carry_repro/` |
| 3 | Per-asset, per-gate Sharpe + cum return + max DD all captured in SUMMARY.md |
| 4 | D3 produces an explicit atlas-vs-reproduction comparison table |
| 5 | Atlas Exp 13 updated with verification note (positive or negative) |
| 6 | Retro fills in with per-asset reproduction numbers, delta analysis, and Cycle 40 verdict |
| 7 | Single commit + push, follow-up commit if SHA insertion needed |
| 8 | The misnamed `phase3_models_funding.joblib` finding (from Cycle 39) recorded in retro as a small disk-hygiene observation |

## Pause points

Two natural pause points where Code should surface findings before
proceeding:

- **After D1 completes:** report fetch results (per-asset row counts,
  any gaps, any schema surprises). Pause for "proceed?" before D2.
  This is cheap insurance against burning 1-2h on a bad data state.

- **After D2 phase4 outputs:** before atlas reconciliation, surface
  the headline reproduction numbers (Sharpe, cum, DD, per-model
  results) for Claude review. If reproduction is materially off
  atlas, we agree on the delta-analysis approach before D3.

## Out of scope

- No live trading
- No executor integration (separate Cycle 41+ candidate)
- No live monitor deployment (separate Cycle 41+ candidate)
- No universe extension beyond atlas's 6 (atlas explicitly
  excluded BNB)
- No new asset classes (e.g. equity perps, FX swaps)
- No LSTM v2 (separate Cycle 41+ candidate, gated on this cycle's
  outcome)

## Notes for Code

- The misnamed `phase3_models_funding.joblib` from Cycle 39 should
  be deleted or renamed before D2 runs. Otherwise D2's phase3 output
  may collide or be misinterpreted by future cycles.
- Idempotent backfill via `INSERT OR IGNORE` on (asset, timestamp)
  natural key. Re-runs MUST be safe.
- Memory #5: max validation + verbose output on the backfill script.
  Add `--validate` and `--verbose` args; default to max validation;
  relax as confidence increases.
- Memory #4: secrets via python-dotenv; never assume raw environment
  variables. Binance API endpoint is public for funding rates (no
  API key needed for /fapi/v1/fundingRate), but if there's a CCXT
  wrapper using authenticated calls, follow the env-loading pattern.
- The 2026-04-24 SSD loss means some assumed-recovered files may be
  in unexpected states. Verify file integrity (joblib load test,
  parquet read test) before assuming a file is usable.
- Cycle 36c's pattern for outputs/{experiment}/ tree is the template
  -- per-cap or per-gate subdirectories with phase4_portfolio.parquet
  + summary stats; aggregated SUMMARY.md at top level; plots optional
  but encouraged.
- If 2023 validation reproduction is included, structure as a
  separate subdirectory under outputs/funding_carry_repro/ so primary
  (train 2024 -> test 2025+) and validation (train 2023 -> test 2024)
  results are cleanly separated.

