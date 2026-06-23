# Cycle 58 — Info-Bar Directional Model: PRE-REGISTERED VALIDATION PROTOCOL

**Status:** PRE-REGISTRATION. Written BEFORE any model fit. The numbers in
§6–§7 are fixed now so they cannot be moved after seeing results. Reviewed by
Jeff at the HARD PAUSE; the deployment-grade run is gated on that review + the
§7 data-maturity precondition.

**Amended Cycle 58b (2026-06-23 — still pre-registration; no fit, no results):**
added the Deflated Sharpe Ratio multiple-testing haircut (§6.4 +
`engines/infobar_lstm/deflated_sharpe.py`), restated the success criterion in
DEFLATED terms (§6.3), added the Mechanism A-vs-B success lens (§0), and made the
data-maturity regime wording operational (§7). These are all PRE-registration
fixes — locked before any results exist.

**Null hypothesis (the base case we must reject to deploy):**
> H0 — "An info-bar directional model has **no durable edge net of realistic
> spot cost, out-of-sample, across regimes.**"

The atlas already supports H0 for *every* prior directional/microstructure
attempt (Exp 1, 2, 3, 7, 17). The burden of proof is on the alternative, and the
bar for rejecting H0 is set deliberately high below.

This protocol is the **judge**. This cycle builds and verifies the judge. The
**subject** (data) must grow before the verdict means anything (§7).

---

## 0. The bet: Mechanism A vs Mechanism B (the success lens)

atlas Exp 1 (SP500 pairs — the verified TC-bound anchor) pre-registered the
dollar-bar revival hypothesis and decomposed it into two mechanisms. We adopt the
same lens, because an info-bar directional model is the same bet in a new venue:

- **Mechanism A — fewer trades → less total TC.** Moving from time bars to
  information bars drops the bar count ~10–30×, so fewer bars → fewer position
  changes → less cumulative transaction cost. This is **MECHANICAL and
  RELIABLE. It is NOT the question.**
- **Mechanism B — per-bar gross alpha rises via event concentration.** Event-time
  sampling is meant to concentrate information so each bar carries more signal,
  raising gross alpha **per bar**. This is **MAGNITUDE-UNCERTAIN, could go either
  way. This is the entire bet.**

Therefore: **SUCCESS = Mechanism B holds** — gross alpha per bar survives
event-time sampling by enough to clear the (now-lower) cost. **Turnover reduction
(A) is NECESSARY-NOT-SUFFICIENT**: cheaper churn is not an edge.

Run A already showed A alone does not carry it: the momentum/MA baseline still
churned **~40 legs/day on dollar bars** (gross_cum −0.045 → net −4.4 @40 bps).
A helps; it does not decide.

**Reporting requirement (binds §6.3):** the validation reports the
**gross-alpha-per-bar** finding EXPLICITLY (gross Sharpe and gross return per
bar, with turnover separately), so a Mechanism-A cost win can never be mistaken
for a Mechanism-B edge. **Net Sharpe alone is insufficient evidence of an edge.**

---

## 1. Scope of runs

| Run | What | Gate |
|---|---|---|
| **A. Plumbing/leakage audit** | Pipeline executes data→features→labels→splits→costs→benchmarks on real bars; automated leakage self-checks; **no model fit** | none — done this cycle (`engines/infobar_lstm/run_audit.py`) |
| **B. Diagnostic dry-run** | One bar-config, one small LSTM, purged walk-forward on the 8 weeks; report with the loud "one regime, not an edge" caveat | **Jeff reviews this protocol first** |
| **C. Deployment-grade run** | Full pre-committed grid (§5), nested walk-forward, the §6 success criterion | **§7 data-maturity gate + Jeff sign-off** |

Run A is the honest deliverable of this cycle. Runs B and C are downstream of
the pause.

---

## 2. Bar-construction integrity (audited — `engines/info_bars/bars.py`)

Signed off in `CYCLE58_INFOBAR_LSTM_AUDIT.md` §P0.6:
- Builders are streaming; trades pushed in `(timestamp, trade_id)` order.
- Close on a **fixed** threshold — **no EWMA/adaptive threshold, no full-sample
  normalization** in the bar layer.
- Overshoot attributed to the closing bar; partial bars never persisted.
- ⇒ Bar construction is **causal** and is not a leakage source. The pipeline
  consumes only **closed** bars ordered by `end_timestamp`.

## 3. Labels — triple-barrier, event-time (pre-committed)

Fixed-clock horizons are mismatched to event bars (audit §P0.2). Labels are
**triple-barrier in BAR SPACE** (AFML Ch. 3):

- For bar `i`: causal volatility `sigma_i` = EWMA(span=**50**) of past bar
  log-returns (uses bars ≤ i only).
- Upper barrier `+pt·sigma_i`, lower `−sl·sigma_i`, with **pt = sl = 1.0**
  (symmetric, pre-committed).
- Vertical barrier `H = 20` bars ahead (pre-committed).
- Label = +1 if upper hit first, −1 if lower hit first, 0 if vertical hit first,
  scanning bars `i+1 … i+H` using each bar's `high`/`low` (intrabar touch).
- **Executable variant (P1):** map {−1,0,+1} → {FLAT, FLAT, LONG}. The model is
  also scored as 3-class for diagnostics, but the *economic* score is on the
  long/flat executable strategy only.
- **Base-rate guard (Exp 7 lesson):** report the label distribution. If any
  class < 10% the config is flagged degenerate and accuracy/AUC are not reported
  as headline (per-class precision/recall only).

## 4. Temporal separation — purged, embargoed, expanding walk-forward

- Bars ordered by `end_timestamp`. **Expanding-window** walk-forward: fold k
  trains on `[0 … t_k)`, tests on `[t_k … t_k + step)`.
- **Purge:** remove from each train fold any bar whose triple-barrier label
  window (`i … i+H`) overlaps the test fold — prevents overlapping-label
  leakage across the boundary (AFML Ch. 7).
- **Embargo:** additional `H` bars dropped after each test fold before the next
  train extension.
- **The final held-out segment is touched exactly once**, at the very end.
- **8-week reality:** this yields only a **handful of short folds**. That is
  fine for a leakage/plumbing check; it is **NOT** an edge claim. Stated loudly
  in every Run-B output.
- **Sample uniqueness (AFML Ch. 4):** report average label uniqueness; if low,
  the effective sample is far below raw bar count (audit §P0.5). Run C uses
  sequential-bootstrap or uniqueness-weighted sampling.

## 5. Hyperparameters / researcher degrees of freedom (PRE-COMMITTED GRID)

Bar-type + threshold are now **also hyperparameters**. The full Run-C search
space is fixed here so multiple-comparisons inflation is visible:

- **Bar configs** (BTC + ETH): `dollar`∈{$1M,$5M}, `vrb`∈{500k}, `vib`∈{500k},
  `volume`∈{BTC:100/500, ETH:1000/5000}. → **7 configs/asset × 2 assets = 14**.
- **Model HPs:** hidden∈{24,32,48}, layers∈{1,2}, seq_len∈{16,32}, dropout∈{0.2,
  0.3}, lr∈{1e-3,3e-4}. → **3·2·2·2·2 = 48**.
- **Total candidate configs = 14 × 48 = 672.** Any reported OOS result must be
  accompanied by "N of 672 configs tried" so the reader can deflate for multiple
  comparisons — formalized as the **Deflated Sharpe Ratio (§6.4)**. Nested
  walk-forward fixes the tune-vs-evaluate leak; it does **not** fix the
  **selection bias** of reporting the best-of-672 (the top config can clear a
  threshold by luck), which is exactly what the DSR corrects.
- **Tuning discipline:** if HPs are tuned, use **nested** walk-forward — tune on
  an inner walk-forward within each train fold, judge on the untouched outer
  test fold. No HP is ever chosen by looking at outer-test results.
- **Dry-run (Run B):** fixes ONE config — **BTC `dollar` $1M, hidden=32,
  layers=1, seq_len=32, dropout=0.3, lr=1e-3** — and does NOT tune. It exists to
  exercise plumbing, not to select.

## 6. Cost, turnover, benchmarks, and the SUCCESS CRITERION (pre-committed)

### 6.1 Cost + turnover
- Cost charged **per position change** (entry + exit) at a spot round-trip
  baseline, reported at **20 / 40 / 60 bps** RT (maker-biased; P1 §1.2).
- Execution fill = **next bar's open** after the signal bar (causal, conservative;
  P1 §1.3).
- **Turnover reported loudly:** position changes per day and per 100 bars. The
  atlas pass/fail line (Exp 1): **`gross_alpha > TC × turnover`**.

### 6.2 Benchmarks (all OOS, net of cost)
1. **Buy-and-hold** the asset over the same test window — and its **own Sharpe**
   (the beta control).
2. **Naive momentum/MA** baseline at **comparable turnover** (long when bar-close
   > causal MA, else flat).
3. **Null at the same trade rate** — random long/flat entries matched to the
   model's trade frequency, averaged over many seeds (distribution, not a point).
4. **Beta isolation:** evaluate **up-trend and down-trend sub-periods
   separately**. A long/flat model must not look good purely by being long in an
   up-window, and must not bleed in a down-window.

### 6.3 SUCCESS CRITERION — what would justify real capital (Run C)
Deployment is justified **only if ALL hold** (pre-committed, strict). Judged on
**DEFLATED, net-of-cost, OOS** numbers — raw Sharpe is never the headline:

1. **Deflated Sharpe Ratio ≥ 0.95** (§6.4): ≥95% confidence the winning config's
   true Sharpe is positive AFTER the 672-trial selection haircut, non-normality,
   and finite sample length — computed on the daily OOS net returns at the
   **40 bps** RT baseline.
2. **Beats buy-and-hold's own Sharpe** in BOTH the up and the down sub-period
   (skill, not beta).
3. **Mechanism B explicit (§0):** the **gross alpha-per-bar** (gross Sharpe /
   gross return per bar) is reported and is positive enough to clear cost on its
   own. A Mechanism-A turnover/cost win does **not** count; net Sharpe alone is
   insufficient.
4. **Beats the turnover-matched null** at p < 0.05 (above its 95th pct).
5. **Cost headroom:** `gross_alpha > 2 × (TC × turnover)` at 40 bps — the edge
   survives a doubling of cost/turnover (Exp 1 margin of safety).
6. **Regime robustness:** criteria 1–5 hold across the data-maturity span (§7),
   not one 8-week window.

Anything short of all six ⇒ **fail to reject H0 ⇒ do NOT deploy.** A marginal,
single-regime, or raw-Sharpe-only positive is explicitly NOT sufficient (the
carry +4.65→negative flip is the cautionary precedent).

### 6.4 Deflated Sharpe Ratio — the multiple-testing haircut (Cycle 58b)

Reporting the best-of-672 is a selection problem nested walk-forward does NOT
fix: with 672 trials the top config can clear a Sharpe bar by luck. The
**Deflated Sharpe Ratio** (Bailey & López de Prado 2014;
`engines/infobar_lstm/deflated_sharpe.py`) is the pre-committed correction —
the native tool for exactly this, by the same author the pipeline is built on.

- **DSR** = P(the selected config's true Sharpe > 0) after deflating by the
  **expected maximum Sharpe under the null** across N trials, and correcting for
  return **skew/kurtosis** and **sample length**.
- **Inputs (PER-OBSERVATION units):** observed daily Sharpe (= annualized ÷
  √365), `n_obs` = number of OOS daily returns, skew + (non-excess) kurtosis of
  the daily net returns, the **variance of the per-day Sharpes across trials**
  (`trial_sharpe_stats`), and **N** = trial count. Mixing an annualized SR with a
  per-day `n_obs` silently corrupts the DSR.
- **N reported twice:** at **N = 672 (conservative, the verdict)** AND at a
  correlation-adjusted **effective N** (`effective_independent_trials` on the
  trial-return correlation matrix; Galwey 2009) for context — correlated configs
  (same bars / overlapping HPs) make the effective count < 672.
- The §6.3 #1 bar (DSR ≥ 0.95) is judged on the **conservative N = 672** figure.

## 7. DATA-MATURITY GATE (the precondition that makes a verdict count)

The 8-week dry-run is **diagnostic only**. The deployment-grade verdict (Run C)
does not count until:

- **Minimum tick history:** at least **~12 months** of `trades` for the target
  asset(s) feeding `info_bars` — enough to span materially different conditions.
  (Today: ~8 weeks, 2026-04-29→06-22.)
- **Regime span (operational):** calendar span **≥ ~12 months** AND the OOS
  folds must **straddle ≥ 1 documented vol/trend regime change** — a dated, named
  transition (e.g. a realized-vol regime shift or a trend→range break) identified
  BEFORE scoring. Regime *count* is hard to pre-specify cleanly; "the OOS folds
  cross at least one documented regime boundary" is the testable version. Bar
  count is **not** regime coverage (audit §P0.5).
- **Either** path to get there:
  - **(a) Accumulate-forward:** keep the live collectors running; re-evaluate
    when (≥12 months AND the OOS folds straddle ≥1 documented regime change) is
    met. Cost: time (months). Risk: none.
  - **(b) Paid tick backfill:** source multi-year historical trade data
    (e.g., a tick-data vendor) for BTC/ETH, backfill `trades`, rebuild
    `info_bars`. Cost: money + ingest engineering. Benefit: a multi-regime
    verdict now instead of in a year.
  - **Decision for Jeff** (at the pause): (a) vs (b). The audit recommends
    framing (b)'s cost against the value of a year saved on a strategy whose
    prior (atlas) is negative — i.e., only pay for backfill if the dry-run
    plumbing looks clean enough to be worth de-risking quickly.

Until the gate is met, **no deployment-grade claim is made and no capital is
risked.** The pipeline simply waits for the subject to grow.

## 8. Leakage audit sign-off (item-by-item — completed in Run A)

Each item is asserted by an automated check in `run_audit.py` / `tests/`:

| # | Surface | Control | Checked by |
|---|---|---|---|
| 1 | Bar construction | fixed-threshold, causal, closed-bars-only | audit §P0.6 + `info_bars` tests |
| 2 | Features ≤ bar-close | every feature uses bars ≤ i; as-of joins ts ≤ end_ts | `test_feature_causality` |
| 3 | Label volatility | EWMA uses past bar returns only | `test_label_locality` |
| 4 | Label target | strictly bars i+1…i+H | `test_label_locality` |
| 5 | Scaler | fit on TRAIN fold only, applied to test | `test_scaler_train_only` |
| 6 | Split | purge overlapping labels + embargo H | `test_walkforward_purge` |
| 7 | Execution fill | next bar open (strictly after signal) | `costs` unit |
| 8 | Final holdout | touched once | orchestrator invariant |

A leak typically reveals itself as implausibly high accuracy on the dry-run; the
structural checks above are the primary defense, the model number is the
secondary smell test.

---

## 9. Decision summary for Jeff (at the pause)

1. **Is the pipeline trustworthy and the protocol rigorous enough** that, once
   tick history matures, running it yields a verdict we can believe? (Review
   Run A output + this protocol.)
2. **Authorize the diagnostic dry-run (Run B)?** It produces a plumbing result,
   not an edge — but it exercises the LSTM fit end-to-end.
3. **Data-maturity path: accumulate-forward (a) vs paid tick backfill (b)?**
   (§7.)
4. Reminder of boundaries: no real capital, no order code, no
   credentials/keys, no trained-model deployment — all gated.
