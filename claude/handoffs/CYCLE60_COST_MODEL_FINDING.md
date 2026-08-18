# CYCLE 60 AMENDMENT — the OHLC spread estimator is invalid on 4h crypto bars

**Status:** AMENDMENT to `CYCLE60_XSEC_REVERSAL_PREREG.md`. Written AFTER first
results existed. The original pre-registration is deliberately **left unedited** —
amending a pre-registration in place after seeing results destroys the very
tamper-evidence it exists to provide. This document records what changed and why.

**Trigger:** §5 of the pre-registration required that "if the two estimators
disagree materially in T3, the cost conclusion for that tier is NOT ROBUST."
They disagreed everywhere, and an anchor test showed both are unusable.

---

## 1. What was measured

`run_xsec diagnose` on the full panel (600 symbols, 4h, 2021-01 → 2026-06),
comparing estimator output against symbols whose true Binance spot effective
spread is known to be ~1–2 bps:

| Symbol | Corwin-Schultz | Abdi-Ranaldo | True (approx) |
|---|---|---|---|
| BTCUSDT | 37.21 bps | 0.00 | ~1–2 bps |
| ETHUSDT | 51.39 bps | 0.00 | ~1–2 bps |
| BNBUSDT | 44.87 bps | 0.00 | ~1–2 bps |
| XRPUSDT | 60.89 bps | 0.00 | ~1–2 bps |
| SOLUSDT | 76.13 bps | 0.00 | ~2–3 bps |

Universe-wide: Corwin-Schultz med 83.2 bps (p25 64.2, p95 176.0);
Abdi-Ranaldo p25 0.00, **med 0.00**, p95 149.9.

## 2. Three distinct failures

1. **Magnitude.** Corwin-Schultz is inflated 19–38× on the anchors. Its
   identifying assumption is that the high-low range decomposes into volatility
   (which scales with the observation interval) plus spread (which does not). On
   high-volatility 4h crypto bars the volatility term swamps the spread term.
   Negative estimates — frequent under these conditions — are floored at zero per
   the original paper, which biases the *average* upward.

2. **It is measuring the wrong quantity.** The anchor estimates rank
   SOL > XRP > ETH > BNB > BTC, which is the **volatility** ordering of those five
   assets, not their spread ordering (all five are within ~1 bp of each other).
   Corwin-Schultz has degenerated into a volatility proxy here.

3. **Cross-sectional inversion — the fatal one.** In the tiered backtest the
   estimator assigned **T1_liquid the HIGHEST cost (103.7 bps/rebalance) and
   T3_illiquid the LOWEST (90.1)** — backwards from market reality. Mechanism:
   thinly-traded bars frequently have high == low, so log(H/L) → 0 and the
   estimator reports a *tight* spread precisely where the asset is least liquid.
   Since this experiment **is** a per-tier alpha-versus-cost race, an inverted
   cost curve invalidates the comparison outright, not merely its scale.

4. **Abdi-Ranaldo is degenerate.** Over half of all estimates floor at zero
   (p25 = med = 0.00), i.e. the estimator returned negative values more often
   than not.

## 3. Consequence for the Phase-3 probe

The single-config probe (f=6, h=6, q=0.2, demean, IS 2021-01 → 2024-06) reported
net Sharpe −12.1 / −10.8 / −9.5. **Those net figures are void.** The arithmetic
was internally consistent (85 bps spread + 10 bps fee × 1.54 turnover ≈ 81
bps/rebalance ≈ −295% annualised, matching the −347% reported) — the engine
computed correctly from an invalid input.

**The GROSS result stands and is unaffected**, since it uses no cost model:

| Tier | Gross Sharpe | Gross bps/reb | Mean IC |
|---|---|---|---|
| T1_liquid | 1.117 | 8.51 | −0.0282 |
| T2_mid | 2.305 | 17.28 | −0.0539 |
| T3_illiquid | 3.116 | 22.03 | −0.0763 |

Monotone in both gross alpha and |IC|, in the predicted direction, with the
negative IC confirming reversal. This is the pre-registered hypothesis pattern,
and it reconciles Fayez (2026): that study's ten liquid majors correspond to
**T1**, our weakest tier.

## 4. Amendment to the cost model

`spread_model` now defaults to **`tiered_fixed`**: an assumed effective spread per
liquidity tier, anchored on published Binance spot market structure —

    T1_liquid 3 bps · T2_mid 15 bps · T3_illiquid 40 bps

These are **assumptions, not measurements**, and are therefore never to be reported
at a single point. The backtest gains `--sensitivity`, sweeping a multiplier
(e.g. 0.5×, 1×, 2×, 4×) and reporting where each tier's net result changes sign.

**Amended success criterion (replaces §7 item 1 for cost purposes):** a tier
qualifies only if its net Sharpe stays above the 0.85 noise floor across a
*plausible range* of spread assumptions — not merely at the most favourable one.

The OHLC estimators are retained in the codebase solely so this finding is
reproducible. `estimate_spread_bps` now refuses `tiered_fixed`, and the backtest
prints a prominent warning if an OHLC estimator is selected.

## 5. A constraint that survives any spread model

At a 10 bps/side taker fee with turnover ≈ 1.6 per daily rebalance, **fees alone
cost 16 bps per rebalance** — before any spread. Breakeven spreads:

| Tier | Gross bps | Max tolerable spread |
|---|---|---|
| T1_liquid | 8.51 | **negative** — cannot clear fees at any spread |
| T2_mid | 17.28 | 1.60 bps |
| T3_illiquid | 22.03 | 7.54 bps |

So at this rebalance frequency the result does not hinge on the spread estimate
at all. The remaining levers are **longer holding periods** (the pre-registered
grid includes h=12 and h=24, cutting rebalance count 2–4×) and **maker-side or
lower-tier fees**. This is recorded now, before the grid runs, so it cannot be
retrofitted as an explanation afterwards.

## 6. Optional rigour upgrade (not yet done)

The Cycle 59 discipline — buy the quotes rather than estimate them — applies here
too. The Binance archive may expose real quote or trade data for spot
(`aggTrades`, possibly `bookTicker`). Downloading a *sample* (≈15 symbols spanning
the tiers × 3 months) would let us calibrate the per-tier constants empirically
instead of assuming them. Cheap, bounded, and it would convert §4's assumptions
into measurements.

## 7. Unchanged

Universe construction, survivorship handling (169/600 delisted — gate passed),
the causal signal path, the power analysis (SE ≈ 0.43, noise floor ≈ 0.85), the
IS/OOS split, the decision tree, and the standing caveat that a negative result
here stays provisional until Cycle 59 validates the framework.

---

# AMENDMENT 2 — three engine defects found in the first grid run

**Status:** written after the first IS grid run (2026-08). The grid output is
**VOID**; the defects below were found by reading the run log, not by a test.

## D1. `--train-end` never sliced the data — THE SEALED OOS WINDOW WAS CONSUMED

`cmd_backtest` computed `is_close` / `oos_close`, printed their ranges, and then
ran the backtest on the **full panel**. Proof from the run's own log: it built
the universe over **2007 rebalances**; 12,042 full-panel bars / 6 = 2007, while
the IS slice (7,657 bars) would give 1,276.

**Consequence.** The run advertised as "IS only, OOS sealed" in fact selected the
best of 288 trials on a sample that *includes* 2024-07 → 2026-06. The Phase 3 /
Phase 4 separation in the pre-registration was violated.

**Honest accounting.** Selection was on full-sample aggregates; no parameter was
tuned to OOS-specific behaviour, and no OOS-only metric was ever inspected. The
contamination is real but bounded. **The OOS window is downgraded from "clean
holdout" to "weak holdout"**, and any future OOS number must be reported with
this caveat attached. It cannot be repaired — the data ends 2026-06, so there is
no untouched later period to promote.

**Fix.** `--train-end` now slices `close/high/low/dvol`, and a new `--oos` flag
selects the post-split slice. The run prints which slice is active.

## D2. Rebalance frequency was pinned to `args.holding`, not the grid's `h`

`reb_every = args.rebalance_every or args.holding` (default 6) applied to every
config, so a config with `h=24` recomputed positions every 6 bars while measuring
24-bar forward returns — a **4x overlap**.

**Measured effect (synthetic panel, H=24, reb 6 vs 24):** the reported *Sharpe*
is essentially unchanged (−1.690 vs −1.694), while the *annualised return*
inflates ~1.8x. **A prediction made before measuring — that Sharpe would inflate
~2x — was WRONG and is recorded here as such.**

**The real damage is the standard error.** Overlapping returns are
autocorrelated, so the effective independent sample is ~N/4. SE(Sharpe) rises
from ≈0.43 to ≈0.86, and the pre-registered **0.85 noise floor becomes ≈1.7 for
4x-overlapped configs**. Both headline winners were overlapped:

| Tier | best config | reported net SR | applicable floor | verdict |
|---|---|---|---|---|
| T2_mid | f=12, **h=24**, q=0.1 | 1.666 | ~1.7 (4x overlap) | **does NOT clear** |
| T3_illiquid | f=24, **h=12**, q=0.1 | 1.686 | ~1.2 (2x overlap) | clears, pending DSR |

**Fix.** Rebalancing now follows each config's holding period (non-overlapping by
construction); universes are built and cached per rebalance frequency.

## D3. The Deflated Sharpe step was skipped — twice

The call passed `kurtosis=`; the implementation takes `kurt=`. The first fix used
a string replace with `stats['n']` while the file contains `stats["n"]`, so the
replace was a **silent no-op** and was shipped without asserting the match. The
multiple-testing correction — pre-registered as success criterion #2 against a
288-trial grid — has therefore **never actually run**.

**Fix.** Corrected with an assertion, plus a test that greps the call site.

## Status of the pre-registered criteria after these defects

| # | Criterion | Status |
|---|---|---|
| 1 | Net Sharpe > noise floor | **UNRESOLVED** — floor mis-specified for overlapped configs |
| 2 | DSR > 0.95 over 288 trials | **NEVER RAN** |
| 3 | Mean IC negative | ✅ met (−0.061 / −0.074), unaffected by D1–D3 |
| 4 | demean beats none | to be re-checked on the corrected grid |
| 5 | survives IS→OOS | **compromised by D1** (weak holdout only) |
| 6 | capacity ≥ $100k | ✅ met (T2 $1.5M, T3 $541k) |

## What still stands

The **gross tier ordering** — Sharpe 1.12 / 2.31 / 3.12 and IC −0.028 / −0.054 /
−0.076, monotone in the predicted direction — is computed without costs, without
the split, and at h=6 where rebalance *equals* holding, so it is untouched by all
three defects. That remains the cycle's real finding.

## Required next step

Re-run the grid IS-only with the corrected engine before any verdict:

    run_xsec backtest --grid --train-end 2024-06-30 --sensitivity "0.5,1,2,4"

and read the result against the *per-config* floor, not a single global one.
