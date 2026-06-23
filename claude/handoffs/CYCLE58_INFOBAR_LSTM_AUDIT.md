# Cycle 58 — Info-Bar Directional LSTM: P0 Audit + P1 Executability Mapping

**Mode:** RECON / AUDIT (read-only) + design. No training, no order code, no
credentials. Track B deliverable, surfaced at the HARD PAUSE.
**Companion docs:** `CYCLE58_VALIDATION_PROTOCOL.md` (P2 pre-registration),
`CYCLE58_INFOBAR_LSTM_AUDIT.md` (this file). Track A shipped separately
(commit `3fdb02e`).

---

## TL;DR (the decisions this audit forces)

1. **The existing `lstm_predictor.py` is a 100% daily TIME-BAR model. It does
   not touch `info_bars` at all.** Building the info-bar directional model is a
   **from-scratch repoint**, not a tweak. (Central P0 question — answered.)
2. **The data foundation is real but it is ONE ~8-week regime.** `info_bars` is
   correctly built and causal; but every bar config is confined to
   2026-04-29 → 2026-06-22 (the `trades` tape span), BTC+ETH only. Bar *count*
   is large (vrb BTC ~88k); regime *coverage* is one. **This cycle can prove the
   pipeline and catch leakage; it cannot establish a durable edge.**
3. **The atlas prior is strongly negative and specific.** Directional ML on
   price (Engine 2) has failed across 7+ experiments / 4 asset classes; the
   binding constraint is the **transaction-cost bound** (Exp 1: a *real* +5.3
   bps gross signal went **−5.6 bps/day net**). The only confirmed edge is
   Engine 7 funding carry — structural, not ML pattern-finding.
4. **The executable strategy is LONG/FLAT spot only** (Canada-legal venues; no
   easy retail shorting). "Down" predictions collapse to FLAT, halving
   expressible positions and loading the result with latent long beta that must
   be controlled for.

The pipeline + protocol are still worth building: they are durable
infrastructure for *any* microstructure idea, and they are the apparatus that
can reject the null honestly once tick history matures. But go in eyes-open:
the null ("no durable edge net of cost OOS") is the base case, and 8 weeks
cannot overturn it.

---

## P0 — AUDIT

### P0.1 Code-vs-intent: does `lstm_predictor.py` consume `info_bars`?

**No. It is a daily time-bar model, end to end.** Evidence (`engines/lstm_predictor.py`):

| Aspect | Finding | Line(s) |
|---|---|---|
| Data source | `SELECT ... FROM ohlcv_daily WHERE asset=?` | 66–69 |
| Bar clock | Daily calendar bars | `load_all_data` |
| Sequence | `SEQUENCE_LENGTH = 60` **days** lookback | 48 |
| Horizons | `PREDICTION_HORIZONS = [1, 7, 30]` **days** | 47 |
| Min data gate | "Need 200+ daily candles" | 75–76 |
| `info_bars` references | **zero** | — |

It is a competent two-model ensemble (multi-horizon quantile LSTM via pinball
loss + XGBoost quantamental + a Hurst/Volterra regime block + a Polymarket
comparator + a z-score "confluence" overlay). All of it operates on **daily
records merged from `ohlcv_daily` + `fear_greed` + `funding_rates` (daily AVG) +
`onchain_btc`**. None of it is information-bar-aware.

**Gap to repoint:** essentially total. The repoint is not "change a table name":

- **Loader** (`load_all_data`) must become a bar-sequence loader keyed on
  `(asset, bar_type, threshold_value)`, ordered by `end_timestamp`, reading the
  `info_bars` microstructure columns (`buy_quote/sell_quote/imbalance_quote/
  tick_count/quote_volume`), not daily OHLCV.
- **Features** (`compute_features`) are daily-TA centric (RSI-14d, MA-200d,
  Bollinger, multi-timescale Hurst on daily closes). They must be replaced by
  **per-bar, event-time, causal** features (signed order-flow imbalance, bar
  duration, volume/flow ratios, causal returns, frac-diff price).
- **Labels** (`build_lstm_sequences`) use a **fixed-clock horizon** (`future_close
  = records[i + horizon - 1]["close"]`, horizon in days). On event bars this is
  conceptually mismatched (see P0.2) — needs **triple-barrier / event labels**.
- **Split** (`TRAIN_SPLIT = 0.80`, single chronological cut, lines 49/533/860)
  is **not** walk-forward and has **no purge/embargo** → leakage on overlapping
  labels (P0.6).

**Verdict:** treat `lstm_predictor.py` as a *reference for architecture idioms
only* (PyTorch LSTM, pinball loss, early stopping). The info-bar model is a new
module (`engines/infobar_lstm/`), not an edit to this file. Leave the daily
model in place; it is a separate, still-valid experiment surface.

### P0.2 What it predicts / label method

Current code (time-bar):
- **Target:** binary up/down (`build_lstm_sequences`, label = `1 if future_close
  > current_close`) AND a multi-horizon quantile-return head (`build_multi_horizon
  _sequences`, returns at 1/3/7/14/30d). Direction **and** magnitude.
- **Horizon:** fixed clock (days).
- **Assets:** BTC, ETH (`SUPPORTED_ASSETS`).
- **Label method:** **fixed-horizon**. This is the conceptual mismatch the brief
  flags: on **event-time** (information) bars, a fixed *clock* horizon is the
  wrong pairing because bars do not advance in clock time. The right pairing is
  **event-based / triple-barrier** labels measured in **bar space** (horizon =
  N bars, barriers scaled by causal volatility). The new pipeline uses
  triple-barrier (see protocol P2).

### P0.3 Features → available data map

The info-bar model's features must be computable **point-in-time from data ≤
bar-close**. Inventory of what exists (verified by DB recon):

| Source table | Rows | Span | Usable for info-bar features? |
|---|---|---|---|
| `info_bars` | 340,502 | 8 wk | **YES — primary.** signed `imbalance_quote`, `buy_quote`, `sell_quote`, `tick_count`, `quote_volume`, OHLC, start/end ts (bar duration) |
| `trades` | 88.5M | 8 wk | YES (bars built from it; raw tape if finer features needed) |
| `order_book_snapshots` | 921,233 | 8 wk | YES — 10-level depth + `order_imbalance_top10`, `spread_bps`. **As-of join only** (point-in-time; nearest snapshot ≤ bar-close) |
| `funding_rates` | 45,684 | **2023→2026** | YES as a slow contextual feature (as-of ≤ bar-close); multi-year |
| `fear_greed` | 954 | 2023→2026 | YES (daily, as-of join; slow) |
| `market_data` | 187 | sparse | marginal (btc_dominance etc.); sparse, low priority |
| `onchain_btc` | 416 | 2025→2026 | BTC-only, daily; slow contextual, low priority |
| `ohlcv_1m` | 674k | ~7.5 mo | not needed (bars supersede) |

**Stationarity handling (brief callout):** raw price levels are non-stationary →
do NOT feed `close` as a level. Use (a) **log returns** over k bars, (b)
**fractional differencing** of log-price (AFML Ch. 5) to keep memory while
achieving stationarity, (c) flow/volume features normalized to causal rolling
means. All scaling fit **train-fold only**.

**Features needing data we lack / cannot compute point-in-time:** none required,
but two cautions — (i) `order_book_snapshots` must be as-of joined (nearest
snapshot with ts ≤ bar end_ts), never the surrounding/nearest-in-time snapshot
(that leaks future); (ii) `funding_rates`/`fear_greed` are slow daily series —
on 8-week event bars they are near-constant and contribute little; include but
expect ~0 importance.

### P0.4 Architecture & complexity (proposed, small by design)

The new model is intentionally small (the examples:params and regime-coverage
limits below demand it):

- Input: sequence of last **L bars** (L≈16–32), each a vector of ~10–14 causal
  features.
- Backbone: 1–2 layer LSTM, hidden 24–48, dropout. Head → triple-barrier class
  {−1,0,+1} (or long/flat binary for the executable variant).
- Param count target: **~10–25k**. (For hidden=32, ~12 features, 2 layers ≈
  ~18k params.)

### P0.5 Examples vs REGIME (the headline risk)

Per-config bar counts (verified) and the regime-coverage framing:

| slice | n_bars | examples:params (~18k) | regime coverage |
|---|---|---|---|
| BTC vrb 500k | 87,906 | ~4.9 : 1 | **1 regime (~8 wk)** |
| BTC dollar $1M | 61,658 | ~3.4 : 1 | **1 regime** |
| BTC vib 500k | 47,976 | ~2.7 : 1 | **1 regime** |
| ETH vrb 500k | 43,136 | ~2.4 : 1 | **1 regime** |
| ETH dollar $1M | 32,723 | ~1.8 : 1 | **1 regime** |
| BTC volume 500 | 1,799 | ~0.1 : 1 | **1 regime** |

Two distinct problems, do not conflate them:
- **examples:params** is *thin-to-OK* for the larger configs (≈3–5:1) and
  hopeless for the coarse ones (volume-500). But raw count overstates it: bars
  are autocorrelated, sequence windows overlap, and **triple-barrier labels
  overlap** (concurrent labels) → *effective* independent samples are far fewer
  (AFML Ch. 4 sample-uniqueness; the protocol reports average uniqueness).
- **regime coverage = ONE** for every config. This is the binding limit. We
  just watched carry go +4.65→negative across a regime flip; an 8-week
  directional model has **no second regime to test generalization against**.
  *Bar count is not regime coverage.* This is why the dry-run is diagnostic
  only and deployment-validation is gated on data maturity (P2 gate).

### P0.6 Leakage pre-scan (existing code + the surfaces to control)

`lstm_predictor.py` (time-bar) leakage findings — relevant because the new
pipeline must NOT inherit them:

- **Split:** single chronological 80/20 cut (`TRAIN_SPLIT`), **no walk-forward,
  no purge/embargo**. With fixed-horizon labels the last `horizon` train rows'
  labels look into the test region → boundary leakage. **New pipeline: purged +
  embargoed expanding walk-forward.**
- **Per-sequence normalization** (`build_lstm_sequences` divides by
  `records[i-1]["close"]`) is **per-window, causal** → OK in isolation, but
  there is **no train-only feature scaler**; any cross-sequence standardization
  must be fit train-only. **New pipeline: scaler fit on train fold only.**
- **Bar construction:** AUDITED CLEAN. `engines/info_bars/bars.py` builders are
  streaming, push trades in `(timestamp, trade_id)` order, close on a
  **fixed** threshold (no EWMA/adaptive, no full-sample normalization),
  overshoot attributed to the closing bar, partial bars never persisted. The
  bar layer is causal and not a leakage source. (One thing to preserve: the
  live collector's 30-s safety lag — backfill has no lag — is fine for research
  on closed bars.)
- **Target leak risk (new):** triple-barrier labels are *future by design* (not
  leakage), but features must be strictly ≤ bar-close and the volatility used to
  size barriers must be causal (EWMA of past bar returns only).
- **As-of joins (new):** order-book / funding / fear-greed joins must use the
  last value with ts ≤ bar end_ts, never nearest.

### P0.7 Atlas triage — prior directional-ML / microstructure / TC evidence

Searched `TRADING_ATLAS.md`. The relevant graveyard and the one survivor:

- **Exp 1 — the canonical TC-bound tale (Chan CPO + Burgess pairs, RF on SP500
  pairs).** Gross signal *genuine*: 57.6% win, +5.3 bps/trade, RF AUC 0.86–0.90.
  **Net −5.6 bps/day** after 4 bps RT × 2.7 trades/day. Quote: *"No amount of
  feature engineering can fix gross_alpha < TC."* Viability law:
  **`gross_alpha > TC × trades_per_day`**. This is the single most important
  prior for an info-bar directional model — **the cost gate, not the AUC, is the
  verdict.**
- **Exp 2 (TA crypto):** Sharpe −0.94; model-type rankings invert year-to-year.
  *"Any risk management tuned while observing OOS results is overfit by
  definition."*
- **Exp 3 (TA futures):** Test A +1.70 Sharpe, Test B −1.38 → *"regime-dependent,
  not structural."*
- **Exp 7 (MCb CPO):** AUC 0.999 at 1–2% base rate = **degenerate** (predicts 0
  always). *"The AUC of 0.999 is the diagnostic, not a good sign."* → directly
  relevant to triple-barrier class balance: a near-all-zero label set yields a
  fake-perfect classifier. The protocol must report base rate + per-class
  metrics, not accuracy/AUC alone.
- **Exp 8/9 (TSMOM ETH/SOL):** weak positive (Sharpe +0.55 / +0.25 validation);
  asset-selection (ETH+SOL not BTC) flagged as *"a soft form of selection
  bias."* → the bar-type/threshold/asset choices are researcher DoF that must be
  pre-committed (protocol).
- **Exp 17 (1-min momentum + OFI):** **confirmed negative**, 31% win, −827 bps /
  88 trades. *"1-minute **time** bars at low-vol periods contain mostly noise
  that the composite scorer mistakes for signal."* → this is the *motivation*
  for info bars (event sampling should be less noisy than fixed 1-min clock) AND
  a caution (microstructure direction in crypto has already failed once).
- **Engine 5 (microstructure / OFI):** order-flow imbalance added only +2.65%
  AUC on equities, dwarfed by trend/serial-corr; insufficient in crypto
  composite.
- **Engine 7 (funding carry):** the **only** confirmed structural edge
  (Sharpe 4–11, max DD −0.03%) — *"Structural strategies don't need regime
  features."* Not ML pattern-finding.

**7-engine taxonomy placement:** the info-bar directional LSTM is **Engine 2
(momentum/trend / directional) × Engine 5 (microstructure)** — precisely the two
engines the atlas has found weak or TC-killed. It is *not* Engine 7. The honest
framing: we are re-entering a documented graveyard with a better *data
foundation* (event bars + signed flow) than the prior attempts (time bars), and
the burden of proof is to beat the TC bound OOS across regimes — which 8 weeks
cannot discharge.

---

## P1 — EXECUTABILITY MAPPING (score the strategy you can RUN)

### P1.1 Legal-venue reality: LONG/FLAT spot only

Canada retail constraints (the operating reality): no easy retail **spot
shorting**, leverage/derivatives restricted. The tradeable instrument is **spot
on a Canada-legal venue** (Kraken Pro / Coinbase Advanced). Therefore:

- Model output {−1, 0, +1} maps to **{FLAT, FLAT, LONG}** — a "down" prediction
  is **not a short**, it collapses to **flat**. Only the +1 (up) class is
  expressible. This **halves** the strategy's degrees of freedom and means the
  model can only earn from the up-leg.
- **Consequence — latent long beta:** a long/flat model in an up-trending window
  looks brilliant from beta alone. **Skill must be isolated** from beta:
  benchmark against buy-and-hold's *own* Sharpe, and evaluate up *and* down
  sub-periods separately (a long/flat model should not beat B&H in a pure
  up-window just by being long, and must not bleed in a down-window).

### P1.2 Realistic spot costs (the gate that actually decides)

Spot maker/taker on the legal venues (round-trip ≈ entry + exit), conservative:

| Venue (spot) | Maker | Taker | RT (taker) | RT (maker) |
|---|---|---|---|---|
| Coinbase Advanced | ~0.0–0.4% | ~0.05–0.6% | up to ~120 bps | ~tens of bps |
| Kraken Pro | ~0.16–0.25% | ~0.26–0.40% | ~52–80 bps | ~32–50 bps |

Plan with a **conservative ~30–60 bps round-trip** baseline (maker-biased) and
report sensitivity at 20/40/60 bps. Cross-reference the Cycle 57 cost finding
(carry: 9 bps@7d-roll already marginal; 20 bps killed it). For a directional
strategy that changes position more often than a multi-day carry, **turnover ×
cost is the whole game** (Exp 1). The protocol REPORTS turnover loudly and
treats `gross_alpha > TC × turnover` as the pass/fail line, exactly as the atlas
demands.

### P1.3 Event-clock vs wall-clock execution mapping

Info bars close in **event time**; orders execute in **wall time**. The mapping:

- A bar closes when its threshold (e.g., $1M traded) is crossed — at an
  *irregular* wall-clock instant. The signal is generated **at bar-close**.
- **Executable mapping:** signal at bar `i` close (wall-time `t_i = end_ts`) →
  place a spot order **after** `t_i`. Realistic fill assumption: next-trade /
  next-snapshot price at `t_i + latency` (latency ~ seconds), **not** the bar's
  own close price (that would be same-instant, optimistic). The dry-run uses the
  **next bar's open** as the executable fill proxy (strictly after the signal),
  which is conservative and causal.
- **Hold:** position held until an exit rule (opposite signal / triple-barrier
  vertical in bar space) closes it — again at a later wall-clock instant.
- **Cost is per position CHANGE**, charged at each entry and exit at the
  spot-cost baseline above. Bars that fill fast (high activity) cluster signals
  in wall time → potential turnover spikes; the turnover report captures this.

This wall-clock mapping is itself a small modeling surface and is specified in
the protocol so the dry-run's cost accounting is not optimistic.

---

## Run A results (plumbing + leakage audit — EXECUTED this cycle, no fit)

`python -m engines.infobar_lstm.run_audit` on the pre-registered config
**BTC `dollar` $1M**:

- **Data:** 61,664 bars, 2026-04-29 → 2026-06-23, 54.2 days (~7.7 weeks, **ONE
  regime**), ~675k bars/year (~47 s/bar). 61,382 usable (99.5%).
- **Labels (pt=sl=1σ, H=20):** −1 **49.8%** / 0 **0.1%** / +1 **50.1%**. On this
  fast event clock the **vertical barrier almost never wins** — the 3-class
  problem is effectively **binary**. Implication for Run B/C: widen pt/sl,
  shorten H, or use the long/flat **binary** framing (the executable variant
  already does). The actionable +1 class has a healthy ~50% base rate (no Exp-7
  degeneracy in the class we act on).
- **Sample uniqueness:** 0.329 → **effective N ~20.2k** from 61.4k raw
  (overlapping labels). examples:params **raw 8.3:1, effective 2.7:1** — thin;
  mandates a small model.
- **Walk-forward:** 4 expanding folds, each ~12k test bars, **purged +
  embargoed** (H=20).
- **Leakage self-checks: ALL PASS** — feature causality (prefix-invariance),
  label locality (window-local), scaler train-only (test-perturbation
  invariant), walk-forward purge. *This is the cycle's verifiable result: the
  pipeline is leakage-free by construction and by test.*
- **Economic machinery** (200-bar MA BASELINE — **not** the model): 40
  legs/day turnover; **gross_cum −0.045 → net −4.41 @40 bps** — cost dominates,
  the atlas TC bound made vivid on this data. The baseline (Sharpe −35.7) does
  **not** beat the turnover-matched null (p95 −29.2). Beta isolation, up/down
  sub-periods, buy-hold-own-Sharpe, and the turnover-matched null all compute.
- **Model: NOT FIT.** `model.py:train_one_fold` raises unless
  `authorized_run_b=True` — the hard pause is enforced in code.

## What this audit changes about the plan

- The work is a **from-scratch info-bar pipeline** (`engines/infobar_lstm/`),
  not an `lstm_predictor.py` edit. (Build is P2.)
- The **only honest claim available this cycle** is "the pipeline is leakage-free
  and the protocol is rigorous" — established by construction + audit, *without*
  needing a model number. The model fit is the **gated** next step.
- The verdict that matters (edge net of cost OOS across regimes) is **blocked on
  data maturity**, not on code. The decision for Jeff (at the pause) is the
  data-maturity gate + accumulate-forward vs paid tick backfill — see the
  protocol's gate section.
