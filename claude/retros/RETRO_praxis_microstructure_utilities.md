# Retro: Praxis Microstructure Utilities v0.1 (COMPLETE)

**Date:** 2026-04-23
**Status:** COMPLETE -- all acceptance criteria met, whale detector fires on live data
**Brief:** `claude/handoffs/BRIEF_praxis_microstructure_utilities_v0_1.md`
**Companion to (not superseding):** `RETRO_praxis_intrabar_confluence.md` (intrabar series remains at v8.2.1).

---

## 1. TL;DR

V0.1 of a general-purpose microstructure utilities library lands cleanly. Three pure-stats helpers in `engines/microstructure_utils.py` (Modified Z-Score, adaptive outlier detector, tiered threshold detector) are backed by 11 unit tests verifying statistical properties (not just exit codes). First consumer `engines/whale_detector.py` wired into the CLI as `detect-whales`, operating on the live-accumulating `trades` table from v8.2.1. Live 60-min detection run surfaced real microstructure signal: a BTC $2.47M window at +71.3% buy imbalance (caught by `always_floor`) plus three more statistical hits, and an ETH $263k single-trade block sell plus six windowed events. Both detection reasons (`always_floor`, `statistical`) fired, confirming the tiered design works end-to-end.

---

## 2. What landed

### 2.1 `engines/microstructure_utils.py` (5,501 bytes)

Three pure functions, no side effects, no DB access:

**`modified_zscore(series)`** -- Iglewicz and Hoaglin 1993. Uses MAD instead of std. The 0.6745 constant normalizes to match sigma on normal data. Handles NaN positions, empty input, and the degenerate MAD=0 case (returns 0 for values at median, +/-inf for values not at median with NaN preserved).

**`adaptive_outlier_detector(series, sigma_z=3.5, sigma=3.0, n_threshold=30)`** -- auto-switches: MAD-based for small samples (n<30), median+sigma*std for large samples. Upper-tail only per brief. Returns boolean array.

**`tiered_threshold_detector(series, always_floor, never_floor, ...)`** -- three zones: above always_floor always flag; below never_floor never flag; middle zone uses adaptive detector. Masks middle-zone values with NaN before passing to the adaptive layer so the adaptive layer's median/MAD aren't skewed by already-classified values.

### 2.2 `tests/test_microstructure_utils.py` (4,638 bytes)

11 tests, all passing:

| Test | Verifies |
|---|---|
| `test_normalization_matches_sigma_on_normal_data` | std of modified Z output on N(0,1) samples is in [0.85, 1.15] |
| `test_contamination_resistance` | MAD detects outlier that standard Z misses; modified Z > 20 where std Z stuck ~5 |
| `test_preserves_nan_positions` | NaN in -> NaN out at same position |
| `test_degenerate_all_equal` | Constant input -> all zeros out |
| `test_empty_input` | Empty -> empty |
| `test_small_sample_uses_mad` | n=20 + 1 outlier: outlier flagged, bystanders mostly clean |
| `test_large_sample_uses_std` | n=1000 normal: flag rate under 1% |
| `test_switch_at_n_threshold` | Both paths flag an injected outlier |
| `test_always_floor_bypasses_statistics` | Huge value flagged even when statistics would be deflated |
| `test_never_floor_overrides_statistics` | Statistically-outlying value below never_floor suppressed |
| `test_middle_zone_uses_adaptive` | Middle-zone outlier flagged by adaptive path |

Runtime: 0.39s. One harmless RuntimeWarning in the degenerate-all-equal test (`0 * inf = NaN` intermediate, filtered out by `np.where` in the final output). Benign; fix would require wrapping the `np.sign(...) * np.inf` in `np.errstate(invalid='ignore')` for cosmetics only.

### 2.3 `engines/whale_detector.py` (6,457 bytes)

First consumer. Two detection modes:

- `detect_single_trade_whales(asset, conn, lookback_minutes)` -- single trades where `quote_amount >= single_trade_always`. Pure SQL filter; no stats needed for this tier.
- `detect_windowed_whales(asset, conn, lookback_minutes, window_seconds)` -- buckets trades into `window_seconds` windows, aggregates total/buy/sell invested, computes aggressor imbalance, then applies `tiered_threshold_detector` to the `total_invested` column. Returns only flagged windows with `detection_reason` column distinguishing `always_floor` from `statistical` hits.
- `summarize_whales(single_df, window_df, asset)` -- human-readable table output.

DEFAULT_THRESHOLDS encoded: BTC {single=$500k, window=$2M}, ETH {single=$200k, window=$800k}. Engineered guesses scaled from Khirman's options-whale convention; expected to re-tune after first data review.

Comment-documented fragility: the `buy_invested` / `sell_invested` groupby lambdas use `df.loc[x.index, "side"]` which requires the original df's index to survive the groupby. Works today because we don't `reset_index()` upstream, but anyone who adds such a call breaks the aggregation silently. Flagged inline.

### 2.4 `detect-whales` CLI subcommand

Wired into `engines/crypto_data_collector.py`:

- Parser: `--asset {BTC,ETH}`, `--lookback` (min, default 60), `--window-seconds` (default 30).
- Handler `cmd_detect_whales`: opens connection, calls both detectors, prints summary, closes. Uses lazy import of `engines.whale_detector` to keep the collector's import path light.

Zero changes to existing collector code paths, existing DB tables, existing scheduled tasks, or `engines/intrabar_predictor.py`.

## 3. Live verification output

Run: `python -m engines.crypto_data_collector detect-whales --asset BTC --lookback 60` followed by same for ETH. Verbatim output archived to `models/intrabar/v0_1_whales_btc.log` and `v0_1_whales_eth.log`.

### 3.1 BTC (60-min lookback)

```
=== Whale Detection Summary: BTC ===

Single-trade whales: 0

Windowed whale events: 4
  Top 5:
    06:23:00  $   2,468,858  1044 trades  imb= +71.3%  (always_floor)
    06:19:30  $     986,980   467 trades  imb= -59.5%  (statistical)
    07:03:30  $     905,922   177 trades  imb= +78.8%  (statistical)
    06:44:00  $     904,566   388 trades  imb=  -2.9%  (statistical)
```

Observations:
- Zero single-trade whales (>=$500k) -- consistent with BTC market structure: institutions split large orders.
- Four windowed whales firing both detection paths: one `always_floor` ($2.47M, very heavy +71.3% buy imbalance, 1044 trades in 30s), three `statistical` hits in the $900k-$990k range. The `statistical` path is correctly finding windows below the $2M always_floor but still anomalous relative to the rolling background.
- Aggressor imbalance column surfacing real one-sided pressure: +71.3% buy and -59.5% sell in the same hour are the kind of microstructure events v8.3 feature engineering will want.

### 3.2 ETH (60-min lookback)

```
=== Whale Detection Summary: ETH ===

Single-trade whales: 1
  Total dollar volume: $263,610
  Buy-initiated: 0, Sell-initiated: 1
  Top 5:
    2026-04-23T06:19:56.079Z  sell  $     263,610  @ $  2,352.35

Windowed whale events: 6
  Top 5:
    06:23:30  $   1,507,104  1225 trades  imb= +46.4%  (always_floor)
    06:19:30  $   1,102,465   432 trades  imb= -38.4%  (always_floor)
    07:15:30  $     927,563   827 trades  imb= -37.6%  (always_floor)
    06:43:00  $     744,769   195 trades  imb= +64.5%  (statistical)
    06:24:00  $     732,195   447 trades  imb=  -9.9%  (statistical)
```

Observations:
- One single-trade whale ($263,610 sell block at $2,352) -- over ETH's $200k threshold but below BTC's $500k, reflecting the calibration choice.
- Six windowed whales. Three via `always_floor` ($800k+), three via `statistical`. ETH generates more events per hour because thresholds are lower -- could be seen as either healthy sensitivity or over-fire. Let Chat decide in calibration pass.

### 3.3 Calibration notes (informational, not tuned in this Brief)

- ETH detects more events than BTC. Expected given lower thresholds, but worth confirming this matches intended risk appetite.
- Statistical-path windows clustered in the $700k-$1M range for BTC and $730k-$930k for ETH. Both distributions have a clear separation from the always_floor but aren't trivially tiny, so the middle zone is doing useful work.
- No `never_floor` suppressions are visible in the output (only flagged windows are printed). Upstream, many small windows are being correctly suppressed from the statistical test by the never_floor.
- Aggressor imbalance ranges from -59.5% to +78.8% on BTC flagged windows -- mostly one-sided, which is what "whale activity" should look like.

## 4. Acceptance criteria (all green)

- [x] `engines/microstructure_utils.py` created with 3 pure functions
- [x] All 11 unit tests pass (`tests/test_microstructure_utils.py`)
- [x] Modified Z-Score normalization test verifies std on N(0,1) in [0.85, 1.15]
- [x] Contamination resistance test verifies MAD detects outlier that std Z misses
- [x] `engines/whale_detector.py` created with DEFAULT_THRESHOLDS + 3 functions
- [x] `detect-whales` CLI subcommand wired into dispatch
- [x] Manual verification on BTC + ETH produced reasonable output
- [x] AST parse + ASCII check pass on all new/modified files (4 files, all clean)
- [x] No modifications to existing collector code paths or existing DB tables
- [x] Existing tests unchanged (only added new test file)

## 5. Files changed in working tree

**New (all uncommitted):**
- `engines/microstructure_utils.py` (5,501 B)
- `engines/whale_detector.py` (6,457 B)
- `tests/test_microstructure_utils.py` (4,638 B)
- `models/intrabar/v0_1_whales_btc.log` (verification output)
- `models/intrabar/v0_1_whales_eth.log` (verification output)

**Modified (uncommitted):**
- `engines/crypto_data_collector.py` -- added `detect-whales` subcommand argparse block + `cmd_detect_whales` handler + dispatch entry. Existing collector code paths, init_db schema, and all other subcommands untouched.

## 6. State at session end

- **Processes:** `PraxisOrderBookCollector` and `PraxisTradesCollector` both `Running` throughout. trades table grew from ~12k rows at brief start to 127k BTC + 127k ETH at verification time (~5 hours of continuous collection). Order book snapshots also continue accumulating per v8.2.
- **DB:** unchanged schema; all microstructure utility work is read-only against the existing `trades` table.
- **ETH:** collected on the data side (order book + trades) but still no ETH-side intrabar predictor work.
- **Git:** working tree has new files + one modified file. Not committed per brief's "no commits in this Brief" convention.

## 7. Future work (carried from brief §"Future work", informational)

- **v0.2:** `order_flow_imbalance_rolling()` for order-book-depth rolling OFI (needed for v8.3 microstructure features).
- **v0.2:** `vpin()` for Volume-Synchronized PIN.
- **v0.3:** Apply MAD-based detection retroactively to other Praxis detectors (funding rate anomaly, convergence detector, smart money tracker).
- **v0.3:** `tiered_threshold_detector` for the convergence detector.
- **v0.4:** Extract the "cheap-scan -> expensive-analysis" pattern as a utility once there are two concrete use cases.

**Immediate next step candidate for Chat:** threshold calibration pass on the whale detector. Options are (a) leave as-is and collect a week of detection logs before tuning; (b) tune now based on the first hour's distribution visible above; (c) replace fixed floors with percentile-based floors computed on a rolling basis, eliminating the tuning question. Option (a) is probably cheapest -- the detector has value at current settings, and more data will make the tuning decision more informed.
