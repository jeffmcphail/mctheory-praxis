# CYCLE 60 PRE-REGISTRATION — Tiered Cross-Sectional Reversal × Crypto

**Status:** PRE-REGISTERED. Written and committed BEFORE any real data was collected.
**Atlas cell:** `MEAN_REVERSION × CRYPTO` — currently EMPTY. Also the first
cross-sectional (relative-value) experiment on crypto in the atlas; every prior
crypto experiment (2, 7, 8, 9, 10, 13, 14) is single-asset time-series.
**Engine:** `engines/xsec_reversal/`
**Depends on:** nothing. Runs in parallel with Cycle 59 (Chan replication), which is
blocked on Kibot data.

---

## 1. Hypothesis (declared before results exist)

> Short-horizon cross-sectional reversal in crypto is **real but liquidity-tiered**:
> absent in the liquid majors, present in the illiquid tail, and bounded by capacity.

This is a *reconciliation* hypothesis. Two credible bodies of evidence conflict:

**Positive evidence — the effect exists**
- Kakushadze (2018, *Cryptoasset Factor Models*): the leading common factor in daily
  close-to-close cryptoasset returns is the prior day's momentum, with a **negative**
  coefficient — i.e. a cross-sectional mean-reversion effect.
- Kakushadze & Yu (2019, *Altcoin-Bitcoin Arbitrage*): builds a dollar-neutral
  long-short on that effect, and stresses it requires a **sizable cross-section** —
  explicitly warning it makes little sense on a few names.
- Four-factor study over 1,160 coins (2014–2022): finds a strong **size** effect, a
  distinctive **reversal** effect, and a significant **illiquidity premium**.
- Krauss-style ML stat-arb on 40 coins at 120-min horizon: **+7.1 bps/day net of
  15 bps per half-turn** — a rare net-positive after realistic costs.

**Negative evidence — the effect is gone**
- Fayez (2026), *Failure of Cross-Sectional Alpha Screening on Cryptocurrency
  Perpetual Futures*: **ten** USDT perps, Jul 2022–Apr 2026, purged walk-forward.
  Naive linear IC = −0.0097 (net Sharpe −3.22); XGBoost ranker IC = +0.0243
  (t = 3.55) yet **net Sharpe −2.91, max DD −95.6%**.

**The reconciliation:** the failure used *ten liquid majors*. The positive literature
locates the effect in the **long tail** (size effect, illiquidity premium, "sizable
cross-section"). Both can be true at once. That is the testable claim.

**Economic mechanism (why this is not another TA pattern).** The atlas's own
principle — *"structural/mechanical signals with clear economic mechanisms are
categorically different from pattern-based signals"* — is satisfied here: forced
deleveraging and liquidation cascades push prices away from fair value for
non-informational reasons, and someone is **compelled** to trade at a bad price.
Same non-zero-sum shape as funding carry (Exp 13, the atlas's one confirmed edge).
This is NOT a TA-indicator experiment and does not fall under the atlas's
UNIVERSAL PRINCIPLE prohibition.

---

## 2. Data

| Item | Value |
|---|---|
| Source | `data.binance.vision` archive (**S3 bucket listing**, NOT `exchangeInfo`) |
| Market | Binance spot, USDT quote |
| Interval | 4h primary; 1d as the Kakushadze-faithful control |
| Window | 2021-01 → 2026-06 (~5.5 yr) |
| Universe | all archive symbols passing filters; expect ~200–400 incl. delisted |
| Cost | **estimated** from OHLC (no quotes in klines) — see §5 |

### 2.1 Survivorship bias — the load-bearing design decision

Binance's REST API and the official `fetch-all-trading-pairs.sh` helper return
**only currently-listed symbols**. Building the universe that way deletes every
delisted coin — and in the **illiquid tier specifically**, delisted coins are
disproportionately the ones that collapsed. That bias would **manufacture the exact
edge this experiment is trying to detect**, in the exact tier the hypothesis is about.

**Fix:** enumerate from the S3 bucket listing, which retains archives for delisted
pairs. (Binance's own README uses `ADABKRW` — long delisted with the BKRW
stablecoin — as its download example.) `run_xsec.py symbols` does this and prints a
warning against regenerating the list from the API.

**Verification requirement:** `collect` reports the count of symbols with data
earlier in the window but **none in the final bar**. A nonzero count is positive
evidence the fix is working. **If that count is 0, STOP** — the universe is
survivor-only and every result is invalid.

### 2.2 Exclusions (parameters, not hard-coded)
- **Leveraged tokens** (`UPUSDT`/`DOWNUSDT`/`BULLUSDT`/`BEARUSDT`): their daily
  rebalancing mechanics create *artificial* mean reversion. Including them is a
  fake-positive generator.
- **Stablecoin/pegged pairs**: no cross-sectional dispersion; pure ranking noise.

### 2.3 Known data traps encoded in the loader
- **ms → µs timestamp switch on 2025-01-01.** Binance changed spot archive
  timestamps to microseconds. Naive `unit='ms'` parsing sends 2025+ data to the year
  ~55000 and silently empties every panel. The parser detects the unit by
  **magnitude** (threshold 1e14), not by date.
- **Column-count guard.** A layout change raises immediately rather than
  silently mis-mapping columns.
- **Optional header row** in newer archives — sniffed, not assumed.

---

## 3. Strategy specification

At each rebalance `t`, **independently within each liquidity tier**:

1. **Formation return** `r_form(i) = close_i(t)/close_i(t−L) − 1` (causal).
2. **Residualize** — remove the common factor:
   `demean` (subtract cross-sectional mean = equal-weight index proxy) primary;
   `beta` (trailing-beta adjusted) secondary; `none` as a **pre-registered control**
   that tests whether any apparent edge is merely beta dispersion.
3. **Rank** the residual ascending; **LONG the bottom quantile** (biggest losers),
   **SHORT the top quantile** (biggest winners). Reversal direction.
4. **Dollar-neutral**, equal-weight within each leg, gross exposure 1.0.
5. **Enter after `execution_lag_bars` (default 1)**, hold `H` bars, rebalance.
   Lag ≥ 1 guarantees a signal can never earn the return of the bar that produced it.
   (Kakushadze's own design is analogous: signal from prior close-to-close, trade the
   subsequent open-to-close.)

### 3.1 Tiering
Tiers are formed **within each rebalance date** by **trailing** dollar ADV, across
eligible symbols only. Ranking on full-sample ADV would leak the future (a coin that
*later* became liquid would be pre-promoted), and since the hypothesis **is about tier
membership**, that leak would corrupt the headline result rather than a footnote.
Default: 3 equal-count tiers, T1_liquid → T3_illiquid.

---

## 4. Pre-registered grid

| Parameter | Values |
|---|---|
| `formation_bars` | 3, 6, 12, 24 |
| `holding_bars` | 3, 6, 12, 24 |
| `quantile` | 0.1, 0.2, 0.3 |
| `residualize` | demean, none |
| tiers | 3 |

**96 configs × 3 tiers = 288 trials.** Declared here, in advance, so the
multiple-testing denominator is honest.

---

## 5. Transaction costs — honest limitation

Binance klines contain **OHLCV only, no quotes**. Unlike Cycle 59 (where per-bar
bid/ask was purchased), spreads here are **ESTIMATED**, via two standard
microstructure estimators:

- **Corwin & Schultz (2012)** — effective spread from consecutive high-low ranges
  (range reflects volatility, which scales with time, plus spread, which does not).
- **Abdi & Ranaldo (2017)** — close vs high-low mid-range across consecutive
  periods; generally more robust in illiquid samples, i.e. the tier we care about.

Default `spread_model = max_of_estimators` (conservative). One-way cost =
`spread/2 + fee_bps_per_side + extra_slippage_bps_per_side`, with fee default
10 bps (Binance spot taker, base tier).

**Robustness requirement:** if the two estimators disagree materially in T3, the
cost conclusion for that tier is **NOT ROBUST** and must be reported as such.

---

## 6. Power analysis (computed before running — do not skip)

The sampling SE of an **annualized** Sharpe estimated from N periods:

    SE(SR_ann) ≈ sqrt(periods_per_year / N)

| Sample | N | SE | 2-SE band |
|---|---|---|---|
| 4h, 1 yr | 2,190 | 1.00 | ±2.00 |
| 4h, 3 yr | 6,570 | 0.58 | ±1.15 |
| **4h, 5.5 yr (this study)** | **12,045** | **0.43** | **±0.85** |
| 1d, 5.5 yr | 2,007 | 0.43 | ±0.85 |

**Consequences, binding:**
1. The 5.5-year 4h window resolves a true Sharpe of ~0.85+ at 2 SE. A 1-year window
   could not resolve even 1.0 — so **short-window results are not interpretable** and
   will not be reported as edges.
2. **Any per-tier Sharpe below ~0.85 is indistinguishable from zero** and must not be
   called an edge, regardless of how attractive it looks.
3. This is *before* the deflated-Sharpe haircut for 288 trials, which raises the bar
   further.

*This calibration was discovered the hard way: the engine's null test initially
"failed" with an apparent Sharpe of 2.73 on random data. That value was 2.26 SE from
zero — pure noise. The test threshold was wrong; the engine was correct. Recording it
so the same mistake is never made when reading real results.*

---

## 7. Success criteria (declared before any result)

A tier shows a **CONFIRMED EDGE** only if ALL hold:

1. **Net Sharpe > 0.85** (the 2-SE resolution limit from §6) after estimated costs.
2. **Deflated Sharpe Ratio > 0.95** against the declared 288-trial count, using the
   realised skew/kurtosis of the winning config (not normal defaults).
3. **Mean IC is negative** — the reversal direction must be present in the raw
   rank correlation, not just in the P&L. A positive-IC "reversal" edge is a red flag.
4. **`residualize=demean` beats `residualize=none`** — otherwise the result is beta
   dispersion, not relative value.
5. **Survives the IS/OOS split** (train ≤ 2024-06-30, test 2024-07-01 → 2026-06),
   with parameters frozen from IS. A sign flip between periods = overfit.
6. **Capacity ≥ $100k book** at 1% ADV participation. Below that it is a negative
   result for practical purposes — pre-declared, not rationalized after the fact.

---

## 8. Decision tree (every branch decisive)

| Outcome | Interpretation |
|---|---|
| **A.** No tier passes | Cross-sectional reversal is dead in crypto. Fills the atlas cell NEGATIVE. Consistent with Fayez (2026), extended to the illiquid tail. |
| **B.** Only T3_illiquid passes | **Hypothesis confirmed.** The edge is capacity-constrained and lives where institutions can't fit. Report the capacity ceiling as the headline constraint. |
| **C.** T1 passes too | Contradicts Fayez (2026). Suspect a bug or a cost model that is too generous **before** believing it. Re-audit before any celebration. |
| **D.** Gross positive everywhere, net negative everywhere | The atlas's TC-bound pattern again (Exp 1). Framework sound, edge uncapturable. |

**Standing caveat:** a **negative** result here remains **PROVISIONAL** until Cycle 59
(framework validation) lands. An unvalidated instrument cannot distinguish "no edge"
from "broken measurement" — that is precisely the confound Cycle 59 exists to break.
A **positive** result is informative either way.

---

## 9. Phasing

- **Phase 1 — COMPLETE (this delivery).** Engine built; 21 tests pass on synthetic
  panels, including a **power test** (recovers a planted reversal: gross Sharpe > 1.0,
  IC < −0.02) and a **null test** (finds nothing in random data, tested against the
  correct noise floor). No real data touched.
- **Phase 2 — collection.** `symbols` → `collect`. **Gate:** the delisted-symbol count
  must be > 0 (§2.1) and per-symbol bar coverage must be sane.
- **Phase 3 — IS only.** Run the grid on train ≤ 2024-06-30. Do **not** look at OOS.
- **Phase 4 — OOS once, parameters frozen.** Apply §7 in full. Record the verdict in
  the atlas (`MEAN_REVERSION × CRYPTO`) whichever way it goes.

---

## 10. What would make this experiment invalid

Stated up front so they cannot be rationalized away later:

- Universe rebuilt from `exchangeInfo` or `fetch-all-trading-pairs.sh` (survivor-only).
- Zero delisted symbols in the collected panel.
- Leveraged tokens left in the universe.
- Tiers assigned from full-sample ADV instead of trailing.
- `execution_lag_bars = 0` in the headline result.
- Reporting a Sharpe below the §6 noise floor as an edge.
- Any parameter tuned after seeing OOS results.
