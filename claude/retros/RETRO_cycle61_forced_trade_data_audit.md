# Retro: Cycle 61 -- Forced-Trade Data Audit

**Brief:** `BRIEF_cycle61_forced_trade_data_audit.md`
**Date:** 2026-08-19
**Duration:** ~4.5 hours
**Status:** COMPLETE (all five tasks answered; two answered as blockers, which is the answer)

---

## Summary

Measured event counts, detectability and regime-grid feasibility for the three P1
forced-trade scenarios. No strategy, no P&L, no Sharpe, no backtest was written, and
`crypto_data.db` was never opened for writing (enforced by SQLite `mode=ro`, not by
convention).

**The headline is that the scenario x regime grid is not buildable as specified, and the
binding constraint is different for each scenario.** A2 has enough events but they are not
cascades; F1 has the right mechanism but essentially no covered universe; D1 has a huge
event count on a product that no longer exists. Separately, the regime axis itself is
narrower than `docs/REGIME_MATRIX.md` claims: **only 8 of 12 classes carry any information
on the data we hold**, and one of the four dead ones (F) dies silently, in a way that
leaves no trace in `RegimeState.missing`.

| Task | Question | Answer |
|---|---|---|
| T1 | A2 cascades identifiable without a feed? | Events yes, cascades **no** -- 60-second flow bursts with no book-stress signature |
| T2 | Do supply jumps mark unlocks? | **Blocked** -- `market_data` covers 5 mega-caps with no vesting cliffs |
| T3 | D1 candidate universe? | 50 tokens, all legs available, **all delisted since 2024-02** |
| T4 | How bad is the OI gap? | Absent; class F **silently** loses 2 of 5 states; forward-collectable, barely backfillable |
| T5 | Is a scenario x regime grid buildable? | **BTC only, and only at a 3x3 collapse.** ETH no. Other scenarios cannot be placed at all |

---

## T1 -- A2 liquidation cascades

### Usable span (stated up front, as the brief asks)

`trades` holds **BTC and ETH only**, `2026-04-29 19:24` to `2026-08-19 06:52` UTC =
**111.5 days**, 87,647,607 BTC ticks + 75,377,231 ETH ticks. Every per-year figure below
is a **3.3x extrapolation** from that window and is printed next to the raw count, never
instead of it. A four-month single-regime window caps this study regardless of anything
else measured.

Aggressor side is known, not inferred: `side='buy'` <=> `is_buyer_maker=0`, **0 violations
in 17.4M sampled rows** across start/middle/end windows. The two columns are fully
redundant.

### Threshold sensitivity (W=60s, L=1440 windows; the spread IS the finding)

| setting | K | I | M | BTC events | BTC /yr | ETH events | ETH /yr |
|---|---|---|---|---|---|---|---|
| loose | 3 | 0.60 | 2 | 2092 | 6855 | 1142 | 3742 |
| base | 5 | 0.80 | 3 | **229** | 750 | **72** | 236 |
| strict | 10 | 0.90 | 4 | 26 | 85 | 5 | 16 |

**80x spread on BTC, 228x on ETH** between loose and strict. There is no plateau -- the
count is a smooth function of the thresholds, which is what a detector with no natural
event class looks like. Nothing was tuned; these three settings were fixed before any
output was inspected.

### Window-size sensitivity -- the most informative single table

Same K/I/M, varying W:

| W | BTC events | ETH events |
|---|---|---|
| 30s | 1052 | 437 |
| 60s | 229 | 72 |
| **300s** | **1** | **1** |
| **900s** | **0** | **0** |

**The phenomenon does not survive a five-minute window.** Duration confirms it: at the base
setting every percentile from p10 to p90 is exactly 60 seconds, max 120s (BTC) / 60s
(ETH). Every detected event is one or two single windows.

A liquidation cascade of the kind A2 describes -- the sort that moves a market for minutes
to hours and leaves a reversion to trade -- would be found at W=300s. It is not. What the
detector finds has a characteristic timescale **below one minute**.

### Order-book corroboration (depth collapse vs matched random control)

Snapshot cadence is excellent: 944,882 BTC snapshots, median gap 10.0s, p99 10.3s. The
corroboration is not resolution-limited.

Ratio of the statistic during events to its own trailing 24h median, next to the same
ratio at random non-event times of matched duration:

| asset | setting | metric | event median | control median | **event/control** |
|---|---|---|---|---|---|
| BTC | base | depth | 1.06 | 1.03 | **1.02** |
| BTC | base | spread_bps | 1.00 | 1.00 | **1.00** |
| BTC | strict | depth | 1.01 | 1.04 | **0.97** |
| ETH | base | depth | 1.03 | 1.03 | **1.00** |
| ETH | base | spread_bps | 1.00 | 1.00 | **1.00** |

**No depth collapse and no spread widening.** The corroborating signature the brief named
is absent. (Both raw ratios sit slightly above 1 because a short-window mean is compared to
a 24h median of a right-skewed series -- which is precisely why the random control is the
number to read.)

### Honest false-positive assessment

**No false-positive rate is computable.** There is no liquidation feed, so no detection can
be labelled. A forced liquidation and a large discretionary market order are identical in
the tape. Everything below is circumstantial and is reported as such.

Also stated plainly: **the study window (2026-04-29 to 2026-08-19) cannot be cross-checked
against a named external event calendar here.** No liquidation feed, no event feed, and no
external source was purchased (out of scope). The brief asks to "cross-check a sample of
detections against known market-wide events"; the honest report is that the corroboration
available is internal, and it is given below rather than substituted with a proxy and
presented as validation.

**Evidence FOR the detections being real market-wide stress:**

- *Temporal clustering.* Dispersion index (var/mean of daily counts; Poisson null = 1.0) is
  **3.79 to 8.51** across settings. At the strict setting 92.3% of BTC events fall on the
  10 busiest days. The detector is not firing uniformly.
- *Cross-asset concordance.* BTC and ETH detectors run on completely separate order flow.
  At the base setting, 5.7% of BTC events have an ETH event within 300s against a 0.48%
  chance rate = **lift 11.85**; the reverse direction gives **lift 8.36**. Independent
  detectors agreeing 8-12x more than chance is genuine evidence of common shocks.
- Both assets share the same busiest day, `2026-08-16`, at every setting.

**Evidence AGAINST them being liquidation cascades:**

- *No book stress* (table above).
- *They do not concentrate on the big days.* Restricted to the 112 daily bars inside the
  event window, detections land on the 10 largest absolute-return days at **2.6% (BTC) and
  4.2% (ETH) against an 8.9% chance rate -- lift 0.29 and 0.47, i.e. BELOW chance.**
- *They do not last.* Nothing survives W=300s.

**Verdict:** the detector reliably finds clustered, cross-asset-correlated, sub-minute
one-sided flow bursts. Those are real and measurable. They are **not** the multi-hour
deleveraging events scenario A2 is built on, and calling them cascades would be exactly the
substitution the brief warns about. Direction agreement is 99.3-100%, which is mechanical
given conditions I and M and is reported as a diagnostic, not as support.

---

## T2 -- F1 token unlocks: BLOCKED on universe coverage

### The gate: what `market_data` actually covers

| asset | rows | span (days) | median gap | max gap | gaps > 2d | distinct values |
|---|---|---|---|---|---|---|
| ADA | 18 | 82.0 | 4.0d | 13.0d | 11 | 13 |
| BTC | 109 | 110.0 | 1.0d | 2.0d | 0 | 100 |
| ETH | 109 | 110.0 | 1.0d | 2.0d | 0 | 109 |
| SOL | 108 | 110.0 | 1.0d | 2.0d | 0 | 108 |
| XRP | 83 | 84.0 | 1.0d | 2.0d | 0 | 8 |

**Five assets, all mega-caps, maximum 110 days, zero NULL supply.** This is the blocker.
The taxonomy's claim that `circulating_supply` jumps reveal unlocks is mechanically
correct, but the covered universe is five coins whose supply changes are block subsidy,
staking emission, burn and escrow -- **not VC vesting cliffs**. F1 is a scenario about
tokens with cliff schedules, and this table contains none of them.

### Jump detection

| setting | threshold | ADA | BTC | ETH | SOL | XRP | total |
|---|---|---|---|---|---|---|---|
| loose | > 0.20% | 2 | 0 | 0 | 1 | 3 | **6** |
| base | > 1.00% | 0 | 0 | 0 | 0 | 0 | **0** |
| strict | > 3.00% | 0 | 0 | 0 | 0 | 0 | **0** |

**Zero jumps clear 1%.** The largest single-period supply increase anywhere in the table is
0.41%.

### Cliff vs drift -- the load-bearing question

**DRIFT dominates** (median cliff share of positive supply growth = 0.197, on the loose set;
computed on loose because base and strict are empty and a share from an empty numerator
would read as DRIFT by arithmetic rather than by measurement).

- **BTC**: +0.241% over 110 days spread across 99 increasing days. Block subsidy. Pure drift.
- **SOL**: +1.160%, 61 increases and 46 *decreases*. Emission net of burn. Drift.
- **ETH**: -0.005%, 108 decreases and 0 increases. Post-merge net burn. Not an unlock series at all.
- **XRP**: **the one cliff-shaped series** -- 75 flat periods and 7 discrete increases, with
  a trailing background MAD of exactly zero. Cliff share 0.679. This is the monthly escrow
  release signature: scheduled, contractual, and the closest thing to F1 compulsion in this
  universe. Three of its steps were detected; all are under 0.37%.

### Dating precision

Daily at best, and only for BTC/ETH/SOL/XRP. **ADA is effectively 4-day** (median gap 4.0d,
max 13.0d, 11 gaps over 2 days), so an ADA event cannot be dated better than to a
multi-day window.

### Spot-check against public unlock records: NOT performed, deliberately

The brief asks to spot-check 3-5 detected jumps against public unlock records. **The check
would be meaningless on this data and would manufacture false confidence.** The six
detected increases are on ADA, SOL and XRP, and are emission or escrow events, not vesting
unlocks -- there is no VC unlock record to match them against. Reporting the blocker is
the honest deliverable; matching an emission tick to an unlock calendar would be the
"silently substituted proxy" the brief names.

**F1 is not testable from Praxis data as it stands.** It needs a supply series over a
universe of tokens that actually have vesting schedules, which `market_data` does not
collect. This is a collector gap, not an analysis gap.

---

## T3 -- D1 leveraged tokens: universe exists, product does not

### The candidate universe

Enumerated **3,690 symbols** from the S3 bucket listing (the survivorship-bias-free Cycle 60
path, `--keep-leveraged --keep-stables --quote ""`). Note that
`data/external/xsec/symbols_all.txt` is the **post-filter** Cycle 60 universe --
`cmd_symbols` applies `filter_symbol_names` before writing, so reading that file would have
reported a universe of zero.

- 64 symbols parse as `<BASE><UP|DOWN|BULL|BEAR><QUOTE>`
- **60 are genuine** (the implied underlying actually trades): 50 USDT-quoted + 10 BUSD
- 20 UP + 20 DOWN + 5 BULL + 5 BEAR on USDT

### The second leg is available

Every genuine candidate has its underlying in the same archive. The four
non-genuine parses (`JUPUSDT`, `JUPUSDC`, `SYRUPUSDT`, `SYRUPUSDC`) are real spot assets,
not leveraged tokens. Four bare `BULL`/`BEAR` symbols have no base in their name; Binance
listed those as 3x BTC products, so BTC is used as the underlying and the assumption is
flagged in the output rather than made silently.

### The finding that decides it

| | |
|---|---|
| genuine USDT leveraged tokens | 50 |
| **delisted** | **50 / 50** |
| most recent data for ANY of them | **2024-02** |
| archive frontier (calibrated from BTCUSDT) | 2026-07 |
| total symbol-months of 1d data | 950 |
| implied scheduled daily resets | **~28,500 token-days** |

**The entire product class is dead on Binance.** D1 is the only P1 scenario that is not
event-rare -- ~28,500 scheduled resets is orders of magnitude more than A2 or F1 -- but
every one of those events is in the past, on an instrument that can no longer be traded on
this venue. Any D1 work is a historical study of a delisted product unless an equivalent
product on a live venue is identified first.

### Live defect found in Cycle 60

`DEFAULT_LEVERAGED_PATTERNS` in `engines/xsec_reversal/universe.py` is matched with `in`,
not `endswith`:

- `JUPUSDT` (Jupiter) contains `UPUSDT`
- `SYRUPUSDT` (Maple SYRUP) contains `UPUSDT`

Both are ordinary spot assets and both were **silently deleted from the Cycle 60 study
universe**. This is exactly the failure the tokenized-equity comment in the same file warns
about for a naive `endswith("BUSDT")` rule -- the guard was written for one pattern family
and not applied to the other. Impact on Cycle 60 is small (2 symbols of ~655) but it is a
real universe-construction bug, and its fix is structural: require the implied underlying
to trade before treating a name as a leveraged token.

---

## T4 -- Open interest: absent, silently degrading, cheap to start, hard to backfill

### 1. Confirmed absent

16 tables scanned by name and column: **zero** OI-shaped tables, **zero** OI-shaped
columns. `docs/REGIME_MATRIX.md` class F specifies "Funding rates + OI"; only funding
exists.

### 2. It does not error. It silently narrows the state space.

Demonstrated by calling `compute_funding_regime` directly across a grid, not argued from
reading the source:

| | |
|---|---|
| declared states for class F | `[-2, -1, 0, 1, 2]` |
| reachable **with** OI | `[-2, -1, 0, 1, 2]` |
| reachable **without** OI | `[-1, 0, 1]` |
| **lost** | **`[-2, 2]`** |
| raises an exception? | **No** |

**Mechanism:** `oi_change_7d` is initialised to `0.0` and only overwritten when
`oi_values is not None`. States +/-2 require `abs(oi_change_7d) > 0.10`, so they are
unreachable. Class F collapses from a five-state axis to a three-state one. Nothing raises,
nothing logs, and **`RegimeState.missing` does not list F** -- so every downstream consumer
sees a class that looks fully computed.

Neither production caller supplies OI: `engines/cpo_training.py:176` and
`engines/funding_rate_strategy.py:359` both omit `oi_series`. Every regime feature vector
ever produced by Praxis has had a three-state class F.

Measured impact in T5: class F is additionally **90.2% (BTC) / 93.8% (ETH) concentrated on
state 0**, so in practice the axis carries almost nothing.

### 3. Collectable? Yes forward. Barely backwards.

ccxt 4.5.51; both venues report `fetchOpenInterest` and `fetchOpenInterestHistory` = True
and both return live values.

| | Binance | Bybit |
|---|---|---|
| active linear swaps pollable | 740 | 785 |
| 5m history | 1000 bars (~3.5d) | 200 bars (~16h) |
| 1h history | 744 bars (31d) | 200 bars (~8d) |
| 1d history | 31 bars | 200 bars |
| **retention wall** | **hard 30 days** -- `since` beyond returns `-1130 startTime is invalid` | **hard 200 rows** -- `since` at -200d, -400d, -730d all return the same 200 rows from 2026-02-01 |

**A forward-only OI collector is cheap and should be trivial to add** (one poll per symbol
per interval, same venues already polled). **A historical backfill is effectively
impossible**: nothing before ~2026-02 is retrievable from either venue at any granularity.
Any study needing OI-conditioned regime before that date is blocked on data that no longer
exists.

---

## T5 -- Scenario x regime grid feasibility

### The arithmetic, first

- Full joint product of all 12 classes: **5,832,000 cells**
- Marginal (class, state) pairs: **45 cells per scenario**

A 111-day window on two assets cannot populate 5.8M cells. Occupancy is therefore reported
marginally (per axis) and at two-axis collapses. Presenting a "12-class grid" as 12 cells
would be exactly the post-hoc pooling the taxonomy warns about, so the arithmetic is stated
rather than glossed.

### Axis degeneracy -- the measurement the matrix does not supply

How many states each class **actually takes** over the sample (BTC, 112 daily evaluations,
90-day trailing window):

| class | name | declared | observed | values | modal share | degenerate |
|---|---|---|---|---|---|---|
| A | trend | 5 | 5 | -2..2 | 0.393 | |
| B | vol_level | 4 | 4 | 0..3 | 0.491 | |
| C | vol_trend | 3 | 3 | -1..1 | 0.518 | |
| D | serial_corr | 5 | **2** | 0,1 | 0.625 | |
| E | microstructure | 3 | **1** | -1 | 1.000 | **YES** |
| F | funding_positioning | 5 | **3** | -1..1 | 0.902 | (near) |
| G | liquidity | 4 | 3 | 0..2 | 0.643 | |
| H | cross_asset_corr | 3 | **1** | 2 | 1.000 | **YES** |
| I | volume_participation | 4 | 3 | 0..2 | 0.786 | |
| J | term_structure | 3 | 3 | -1..1 | 0.464 | |
| K | dispersion | 3 | **1** | 0 | 1.000 (100% uncomputable) | **YES** |
| L | rv_iv_spread | 3 | **1** | 0 | 1.000 | **YES** |

**Only 8 of 12 axes carry information.** Effective joint grid drops from 5,832,000 nominal
to 9,720 (BTC) / 10,368 (ETH); effective marginal from 45 to 30. Causes:

- **K (dispersion)** needs >= 3 universe assets. **Every OHLCV table in `crypto_data.db`
  holds only BTC and ETH** (`ohlcv_1m`, `ohlcv_4h`, `ohlcv_daily`, `info_bars` -- all 2
  assets). K is uncomputable from this database, full stop. It is the one degenerate class
  that *does* announce itself, appearing in `missing` on 100% of evaluations.
- **L (rv_iv)** is an explicit stub -- `compute_rv_iv_regime` returns 0 when `dvol is None`,
  and no options data exists. Permanently 0, never listed in `missing`.
- **E** and **H** are computable but constant over this sample.
- **F** is the T4 finding.

### Cell occupancy (scenario A2, base setting: 229 BTC / 72 ETH events)

Marginal, 45 cells:

| bucket | BTC cells | % | ETH cells | % |
|---|---|---|---|---|
| 0 | 15 | 33.3% | 20 | 44.4% |
| 1-2 | 1 | 2.2% | 4 | 8.9% |
| 3-9 | 5 | 11.1% | 0 | 0.0% |
| **10+** | **24** | **53.3%** | **21** | **46.7%** |

Joint B (vol level) x G (liquidity), 4x4 = 16 cells:

| bucket | BTC | ETH |
|---|---|---|
| 0 | 8 (50.0%) | 11 (68.8%) |
| 1-2 | 3 | 1 |
| 3-9 | 1 | 2 |
| **10+** | **4 (25.0%)** | **2 (12.5%)** |

Collapsed 3x3 (extreme states merged; grouping written out explicitly in
`DEFAULT_COLLAPSE_3` so the merge is auditable):

| bucket | BTC | ETH |
|---|---|---|
| 0 | 3 (33.3%) | 4 (44.4%) |
| 1-2 | 1 | 1 |
| 3-9 | 0 | 2 |
| **10+** | **5 (55.6%)** | **2 (22.2%)** |

### Buildable / not-buildable

| asset | granularity | cells | empty | estimable (>=10) | % | **buildable** |
|---|---|---|---|---|---|---|
| BTC | marginal (12 axes) | 45 | 15 | 24 | 53.3% | **YES** |
| BTC | joint B x G (4x4) | 16 | 8 | 4 | 25.0% | no |
| BTC | **collapsed B x G (3x3)** | 9 | 3 | 5 | 55.6% | **YES** |
| ETH | marginal (12 axes) | 45 | 20 | 21 | 46.7% | no |
| ETH | joint B x G (4x4) | 16 | 11 | 2 | 12.5% | no |
| ETH | collapsed B x G (3x3) | 9 | 4 | 2 | 22.2% | no |

**Plain verdict.** A scenario x regime grid is buildable for **BTC only**, for **A2 only**,
at **one-axis-at-a-time or a 3x3 collapse of two axes**, on **8 informative axes rather
than 12**. ETH does not have the events for it at any granularity. No finer conditioning is
supportable, and 12-class conditioning is not close.

**And that verdict is generous**, because it takes the A2 event counts at face value --
while T1 concludes those events are sub-minute flow bursts, not cascades. The grid is
buildable for a scenario whose events are not the ones the scenario is about.

**The other two P1 scenarios cannot be placed on the grid at all**, which is a finding
rather than an omission:

- **F1** -- regime needs intraday OHLCV. `crypto_data.db` has `ohlcv_1m` for BTC and ETH
  only, while `market_data` covers ADA/BTC/ETH/SOL/XRP. The three non-BTC/ETH assets cannot
  be regime-assigned from this database, and BTC/ETH have no unlock events.
- **D1** -- no event series exists yet (T3 is a universe audit, not a collection run) and
  the underlying klines live in the Binance archive rather than `crypto_data.db`. Its event
  count is bounded by construction at one reset per token per day.

---

## Changes Made

### Files Added

| File | Purpose | Lines |
|------|---------|-------|
| `engines/forced_trade/__init__.py` | Package docstring, module map, read-only contract | 31 |
| `engines/forced_trade/common.py` | Read-only DB access (Rule 34), Rule 35 time helpers, ASCII table formatting, noisy-logger pinning | 213 |
| `engines/forced_trade/cascade.py` | T1 detector: bucket cache builder, parameterised firing rule, event merging, two validators | 419 |
| `engines/forced_trade/corroborate.py` | T1 corroboration: book depth vs matched random control, concentration, cross-asset concordance, extreme-day overlap | 346 |
| `engines/forced_trade/unlocks.py` | T2: coverage report, cliff-vs-drift jump detector, growth decomposition | 263 |
| `engines/forced_trade/leveraged.py` | T3: structural leveraged-token classification, substring-FP detection, archive coverage | 205 |
| `engines/forced_trade/oi_audit.py` | T4: schema scan, degradation probe, caller audit, live venue probe | 220 |
| `engines/forced_trade/occupancy.py` | T5: fixed-window regime series, axis degeneracy, marginal/joint/collapsed occupancy | 336 |
| `engines/forced_trade/run_audit.py` | CLI (`t1-cascades` .. `t5-occupancy`, `all`) | 757 |

**2,790 lines. No existing file was modified.** All 9 files are ASCII-only (Rule 20,
byte-level verified) and all pass `ast.parse` (Rule 19).

### Key Decisions

- **`mode=ro` URI, not discipline.** Every connection is read-only at the driver level;
  a write attempt raises `OperationalError: attempt to write a readonly database`.
  Verified explicitly rather than assumed.
- **One scan, cached.** `trades` is scanned once per asset into 10-second buckets
  (963,049 BTC buckets, ~5 min) and cached to parquet. The cache holds **raw aggregates
  only** -- no threshold is applied while building it, so caching cannot influence any
  reported count. Every threshold and window setting is then evaluated in memory.
- **`MAX(timestamp)` as the single min/max aggregate.** SQLite guarantees bare columns take
  their value from the row producing the sole min/max aggregate, which gives per-bucket
  close price in one pass instead of two. Documented in the SQL, and `--validate`
  re-derives 25 random buckets from source independently anyway (0 price mismatches,
  0 flow mismatches on both assets).
- **`rolling(L).stat().shift(1)`, always.** Trailing statistics never include the
  observation being tested. A detector normalising by a median containing its own spike
  would fire *less* often on the largest events.
- **Random control alongside every event statistic.** "Median depth ratio 1.06 during
  events" is unreadable on its own. Control windows are drawn from the same span with
  durations resampled from the observed event durations, excluding +/-900s around any
  detection.
- **Direction agreement is a diagnostic, never a filter.** Adding it to the firing rule
  would have improved the numbers and destroyed their meaning.
- **`sweep_params()` as single source of truth.** The base K/I/M was initially duplicated in
  three places (sweep table, window sweep, T5 occupancy). An occupancy table computed from
  different thresholds than the count table it explains would be silently wrong.

---

## Test Results

### Passed

- `python -m engines.forced_trade.run_audit --coverage --network all` completes end-to-end
- `--validate` (default on): side convention **0 violations / 17.4M rows**; bucket cache
  **0 price mismatches, 0 flow mismatches** on 25 independently re-derived buckets per asset
- Read-only enforcement verified: `CREATE TABLE` on the audit handle raises
  `OperationalError`
- ASCII-only: 0 non-ASCII bytes across all 9 files (Rule 20)
- `ast.parse` clean on all 9 files (Rule 19)
- Rule 34 honoured: fresh connection per read pass **and** `isolation_level=None`
- All artifacts written to `outputs/forced_trade/` (35 CSV/JSON files)

### Known Issues / Limitations

- **111.5-day span, 2 assets.** Every A2 count annualises at 3.3x. The window is a single
  regime; nothing here supports an out-of-sample claim.
- **No liquidation feed** -> no computable false-positive rate for T1, by construction.
- **No external event calendar** for 2026-04 to 2026-08 was available in this cycle, so the
  brief's "cross-check against known market-wide events" is answered with internal
  corroboration (cross-asset concordance, clustering, book stress) and the limitation is
  stated rather than papered over.
- **T2 blocked** on universe coverage, not on method.
- Cross-asset concordance is O(n^2) in event count (prefix `max` per query). Fine at 2k
  events, would need an interval tree at 100k.

---

## Failures & Debugging Trail

### 1. `validate_side_convention` used a query shape that cost ~25 minutes

First implementation used `SELECT side, is_buyer_maker, COUNT(*) ... GROUP BY side,
is_buyer_maker`. No covering index exists, so SQLite builds a temporary B-tree over ~90M
rows per asset. The run was killed after ~20 minutes still inside the check.

**Fix:** count *violations* instead --
`WHERE (side='buy') <> (is_buyer_maker=0)`. No grouping, no sort, one streaming pass, same
guarantee, and it reports the number that actually matters. Also added
`--validate-full-scan` (opt-in, every row) versus the default sampled mode (14 days across
start/middle/end windows via the `(asset, timestamp)` index). Sampled mode: **36s for both
assets**, 17.4M rows, 0 violations.

### 2. The cliff detector classified the purest cliffs as drift

XRP has 75 flat periods and 7 discrete supply steps -- background MAD is **exactly zero**.
The scorer computed `(jump - median) / MAD`, guarded division by zero, and returned NaN ->
`is_cliff = False`. **The most cliff-like series in the table scored as not-a-cliff.**

**Fix:** three explicit cases -- `MAD > 0` scores normally; `MAD == 0` scores `+inf` (a
discrete step out of a perfectly flat background is the purest possible cliff); `MAD` NaN
is `scorable=False` and kept distinct from "measured and rejected". XRP's three steps now
classify correctly, which is what surfaced the escrow-release signature.

### 3. The extreme-day overlap test was vacuous and would have been read backwards

First version ranked the top-10 absolute daily-return days over the whole `ohlcv_daily`
history (2023-11 onward, 1012 days) while events can only exist in the 111-day tick window.
Result: **0 hits, chance 1.0%, lift 0.00** -- guaranteed by construction, and it reads as
"the detector avoids stress days".

**Fix:** restrict the daily frame to the event window before ranking. Corrected result:
BTC 6/229 on top days vs 8.9% chance (lift 0.29), ETH 3/72 (lift 0.47). Still below chance,
but now that is a *measurement* rather than an artifact -- and it is one of the strongest
pieces of evidence against the cascade interpretation.

### 4. T3 could not read the Cycle 60 symbol list

`data/external/xsec/symbols_all.txt` is written **after** `filter_symbol_names`, so it
contains zero leveraged tokens by construction -- the exclusion list the taxonomy points to
as "a ready-made candidate list" is not in that file. Re-enumerated from the S3 origin with
the exclusions inverted.

### 5. Structural parse alone was not enough to define the universe

The first classifier trusted the name split, which admitted `JUPUSDT` (J+UP+USDT) and
`SYRUPUSDT` (SYR+UP+USDT) as leveraged tokens. Added the requirement that the implied
underlying must itself trade -- which both separates real tokens from coincidental
spellings *and* is what identifies the Cycle 60 defect.

### 6. `-vv` produced 471 KB of ccxt option-chain JSON

Rule 25 asks for maximum verbosity from our code; inheriting `DEBUG` into `ccxt` and
`urllib3` buried every finding. `setup_logging` now pins a `NOISY_LOGGERS` list to WARNING
independently of `-v`.

### 7. Pre-existing crash in `run_xsec.py symbols` (NOT fixed -- out of scope)

`engines/xsec_reversal/run_xsec.py:81` prints `client.rejected_symbols`, which contains
5 non-ASCII bucket prefixes (Chinese-character symbol names in the Binance bucket). On
Windows cp1252 stdout this raises `UnicodeEncodeError` and the command exits non-zero.
The output file is written before the print, so the effect is cosmetic plus a bad exit
code. Left alone -- Rule 7, it is another engine's file and not in this brief. Noted below
for Chat.

---

## Commits

- `400cd64` -- Cycle 61: forced-trade data audit (measurement only, read-only)

---

## Open Items for Chat

1. **A2 is not what the taxonomy says it is.** The detector finds sub-minute one-sided flow
   bursts with no book-stress signature that do not concentrate on big-move days. The
   taxonomy's flagship P1 candidate should be re-described, or the study should be scoped
   to "sub-minute flow imbalance" (a different, more crowded game -- Q2 accessibility
   applies) rather than "liquidation cascades". **A liquidation feed would settle this**;
   Binance publishes a forced-order WebSocket stream, which is a forward-only collector.
2. **F1 needs a collector decision, not an analysis decision.** `market_data` covers 5
   mega-caps. F1 requires supply series on tokens with vesting schedules. Extend
   `market_data`'s asset list, or accept F1 as blocked.
3. **D1 needs a venue decision before anything else.** All 50 Binance leveraged tokens are
   delisted since 2024-02. Is a historical-only study of a dead product worth a cycle, or
   should we first look for an equivalent live product (and note that the Canada CEX-perp
   constraint from Cycle 56 applies to whatever replaces it)?
4. **The class F degradation is a live bug affecting past results**, not just future ones.
   Every regime feature vector Praxis has produced has had a 3-state class F while the
   matrix documents 5. Worth a small fix (log a warning, or add F to `missing`, when
   `oi_series is None`) plus a note on which past cycles consumed regime features.
5. **Class L is a permanent stub** and class K is uncomputable with a 2-asset universe.
   Should `REGIME_MATRIX.md` record which classes are actually live given current
   collectors? An axis documented as 3-state but always returning 0 is worse than an axis
   documented as absent.
6. **A forward-only OI collector is cheap** (both venues, ~740/785 symbols, ccxt supported)
   and would make class F whole within ~30 days of running. Historical backfill is
   impossible before ~2026-02. Recommend starting collection now regardless of whether the
   grid gets built, because the data is not recoverable later.
7. **`DEFAULT_LEVERAGED_PATTERNS` substring bug** in `engines/xsec_reversal/universe.py`
   deleted JUPUSDT and SYRUPUSDT from the Cycle 60 universe. Small impact, real bug. Fix
   pattern available in `engines/forced_trade/leveraged.py:split_leveraged`.
8. **`run_xsec.py:81` UnicodeEncodeError** on non-ASCII bucket prefixes (item 7 in the
   debugging trail). One-line fix, not made here per Rule 7.

---

## Artifacts

`outputs/forced_trade/` (35 files):

- `t1_threshold_sensitivity.csv`, `t1_window_sensitivity.csv`, `t1_events_BTC_base.csv`,
  `t1_events_ETH_base.csv`, `t1_book_corroboration.csv`, `t1_concentration.csv`,
  `t1_cross_asset_concordance.csv`, `t1_side_convention.json`,
  `t1_cache_validation_{BTC,ETH}.json`
- `t2_coverage.csv`, `t2_jump_sensitivity.csv`, `t2_growth_decomposition.csv`,
  `t2_jumps_{base,loose}.csv`, `t2_verdict.json`
- `t3_candidates.csv`, `t3_coverage.csv`, `t3_substring_false_positives.json`
- `t4_schema_scan.json`, `t4_degradation.json`, `t4_caller_audit.json`,
  `t4_venue_probe.json`
- `t5_regime_series_{BTC,ETH}.csv`, `t5_events_with_regime_{BTC,ETH}.csv`,
  `t5_axis_degeneracy_{BTC,ETH}.csv`, `t5_effective_grid_{BTC,ETH}.json`,
  `t5_marginal_occupancy.csv`, `t5_joint_occupancy.csv`, `t5_collapsed_occupancy.csv`,
  `t5_verdict.csv`

`data/external/forced_trade/` (cache, regenerable):

- `trade_buckets_BTC_10s.parquet` (963,049 buckets)
- `trade_buckets_ETH_10s.parquet` (962,051 buckets)
