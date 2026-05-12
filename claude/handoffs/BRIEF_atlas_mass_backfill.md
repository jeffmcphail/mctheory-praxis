# Cycle 35 -- Atlas mass backfill: remaining 13 trading experiments

**Predecessor:** Cycle 34 (`647d50c` + retro `2f0d2ef`, Info Bars v0.1
shipping). Cycles 33 + 33.5 (atlas schema extension; MCP serialization
fix; Exps 1 and 13 backfilled as design validation).

**Mode:** Hybrid. Drafting-heavy brief; Code applies markdown edits
+ runs atlas_sync + commits. No code or schema changes. No collector
changes.

**Risk:** low. Pure content work. atlas_sync is idempotent;
non-backfilled experiments stay unchanged (md_hash stable).

## Why now

Cycle 33 shipped the atlas schema extension and backfilled 2 reference
experiments (Exp 1 NEGATIVE + Exp 13 POSITIVE) as design validation.
Cycle 34 shipped Info Bars v0.1 -- the substrate that lets revival
hypotheses for Engine 2 NEGATIVE experiments reference dollar bars
*concretely* (real $1M-threshold bars sitting in `crypto_data.info_bars`,
not abstract possibility).

Cycle 35 brings the 13 remaining trading experiments to the same
structured shape: `test_conditions`, `revival_hypotheses`,
`regime_state_at_test`, `computational_engine`. After this lands, the
Atlas DB is queryable along the dimensions that matter for Research
Agent triage and Cycle 36+ revival re-runs.

## What

Three classes of edit, all in `TRADING_ATLAS.md`:

1. **Two pre-existing cleanups** (Q1 strict-canonical decision):
   - **Exp 1**: split the combined-letter regime bullet
     `B, C, F, H, J, K: <=0.4% -- noise for equities` into 6 individual
     canonical bullets so `regime_state_at_test` produces clean keys
     (`B_<class>`, `C_<class>`, etc.) instead of sanitized fallback
     `b,_c,_f,_h,_j,_k`.
   - **Exp 13**: rewrite the regime bullets to use canonical
     `F (Funding/positioning)` form. Move "F = +1, +2", "F = 0",
     "F = -1, -2" state details from the bullet KEY (which produces
     sanitized fallback keys) into the bullet DESCRIPTION text.
     Single canonical F-bullet, not three compound-form bullets.

2. **Exp 3 result_class fix**: markdown clearly states "regime-dependent,
   not structural" + "failed validation" but never adds a final
   `**Result**` row to the attribute table. Atlas DB result_class is
   NULL. Add `| **Result** | **NEGATIVE** (regime-dependent, failed
   validation) |` to the attribute table so atlas_sync's classifier
   parses it.

3. **The 13 backfills**: add `Computational engine` row to each
   experiment's attribute table + three new structured sections
   (`**Test conditions:**`, `**Active regimes during test:**`,
   `**Revival hypotheses:**`) before the `---` separator that ends each
   experiment's block.

## Pre-work for Code (look up canonical regime class names)

The strict-canonical decision (Q1) means regime bullets must use
`<Letter> (<Class Name>):` form. Code should query the regime_classes
table from `data/praxis_meta.db` to get the 12 canonical class labels:

```python
import sqlite3
conn = sqlite3.connect("data/praxis_meta.db")
classes = conn.execute(
    "SELECT class_letter, class_name FROM regime_classes ORDER BY class_letter"
).fetchall()
for letter, name in classes:
    print(f"{letter}: {name}")
```

Confirm the 12 classes are A-K + L (or A-K + some letter -- whatever
the actual scheme is). Use the exact names from the DB in the new
markdown -- do not paraphrase. If a class name in the DB differs from
my guesses in the brief content below, use the DB version.

I'll write the regime bullets below using **placeholder labels** like
`B (Vol level)`. Code MUST replace these with the exact class names
from regime_classes before committing.

## Out of scope

- New experiments. This is backfill; not run-anything-new.
- PREDICTION_MARKET_ATLAS.md changes. Different file, different
  cycle.
- Retroactively running the regime engine over historical test
  windows to fill `regime_state_at_test` with measured data for
  experiments that didn't natively track it. Most experiments will
  have `regime_state_at_test` keys like `general: not_measured` --
  honest documentation that the data wasn't captured at the time.
- Engine-filter parameter for `atlas_search` (deferred TODO --
  comes after this cycle when `computational_engine` is fully
  populated).
- Re-running any of the 13 experiments. Revival re-runs are
  Cycle 36+.

## Template shape (verbatim per experiment)

```markdown
| **Computational engine** | <N> (<Engine name>); <secondary clause if any> |

[... existing experiment prose continues ...]

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | <type> |
| Frequency | <freq> |
| Universe | <assets> |
| TC | <bps> |
| Feature set | <description> |
| Pre-filter | <description or 'none'> |
| Risk management | <description> |
| Computational engine | Engine <N> (<name>); <secondary clause> |

**Active regimes during test:**

<one or two sentences of framing if needed>

- <Letter> (<Class name>): <state and behavior, or 'not measured'>
- <Letter> (<Class name>): ...
...

<optional one-line context closer>

**Revival hypotheses:**

<optional one-line framing>

1. **<Title>** -- likelihood: <low/medium/high>. <description>
2. **<Title>** -- likelihood: <low/medium/high>. <description>
...
```

The `Computational engine` row goes inside the existing attribute
table, NOT in the test_conditions block (though it appears in both --
that's the brief's intent for cross-reference convenience).

## Where to insert in each experiment's markdown

For each experiment:
- The `**Computational engine**` row gets appended to the existing
  attribute table at the top (after `| **Result** | ... |` or
  equivalent).
- The three new sections (`**Test conditions:**`,
  `**Active regimes during test:**`, `**Revival hypotheses:**`) get
  inserted BEFORE the closing `---` separator that ends each
  experiment's block.
- If an experiment lacks a closing `---` (because it's followed
  directly by the next `### N.` heading), insert before that heading.

---

## EXPERIMENT 1 (already backfilled in Cycle 33) -- small cleanup only

Current `regime_state_at_test` has key `b,_c,_f,_h,_j,_k` (sanitized
fallback). Find the bullet in the markdown:

```
- B, C, F, H, J, K: <=0.4% -- noise for equities
```

Replace with 6 individual canonical bullets. Use the DB-fetched class
names for B through K. Example using placeholder labels:

```
- B (<Vol level>): <=0.4% AUC lift -- noise for equities
- C (<Vol-of-vol>): <=0.4% AUC lift -- noise for equities
- F (<Funding/positioning>): <=0.4% AUC lift -- noise for equities
- H (<class name>): <=0.4% AUC lift -- noise for equities
- J (<class name>): <=0.4% AUC lift -- noise for equities
- K (<class name>): <=0.4% AUC lift -- noise for equities
```

(Code: replace `<Class name>` with the actual class name from
regime_classes table for each letter. If any letter doesn't exist in
the table, use a best-effort label and flag it in the retro.)

No other changes to Exp 1.

---

## EXPERIMENT 13 (already backfilled in Cycle 33) -- regime canonical cleanup

Current `regime_state_at_test` has keys like
`f_=_+1,_+2_(positive_funding_sustained)` (sanitized fallback). Find
the regime bullets in the markdown (the
`**Active regimes during test:**` section). Replace:

```
- F = +1, +2 (positive funding sustained): trades nearly
  every day; this is the alpha-bearing regime
- F = 0 (flat funding): mostly does not trade
- F = -1, -2 (negative funding): does not trade (correctly --
  the structural carry is absent)
- A (Trend): not directly conditioned but observed alignment;
  positive funding tends to coincide with positive trend in
  crypto bull regimes
- B (Vol level): no strong dependence; the carry mechanism is
  vol-agnostic when long-only
```

with:

```
- F (<Funding/positioning>): regime-conditioned by design.
  Trades aggressively in F=+1,+2 (positive funding sustained;
  alpha-bearing regime), mostly does not trade in F=0 (flat
  funding), and correctly does not trade in F=-1,-2 (negative
  funding; the structural carry is absent).
- A (<Trend>): not directly conditioned but observed alignment;
  positive funding tends to coincide with positive trend in
  crypto bull regimes.
- B (<Vol level>): no strong dependence; the carry mechanism is
  vol-agnostic when long-only.
```

(Code: replace `<class name>` with DB-canonical names.)

Result after re-sync: `regime_state_at_test` keys become
`F_funding_positioning`, `A_trend`, `B_vol_level` (or whatever the
canonical names sanitize to -- all single canonical letters now).

---

## EXPERIMENT 3 result_class fix (small)

Find the attribute table at the top of Exp 3 (after the heading
`### 3. TA_STANDARD × FUTURES (ES, NQ, YM, RTY, CL, GC, SI, NG)`).
After the existing `| **TC** | 1.0 bps/leg × 2 legs = 2 bps round-trip |`
row, ADD a new row:

```
| **Result** | **NEGATIVE** (regime-dependent, failed cross-period validation) |
```

This will be picked up by the atlas_sync result_class classifier and
populate `result_class = 'NEGATIVE'` in the DB on next sync. (Per
Cycle 32 retro, this resolves the NULL value that was folded into
the NEGATIVE breakdown manually.)

---

## EXPERIMENT 2 -- TA_STANDARD x CRYPTO (NEGATIVE, Engine 2)

### Attribute table addition

After `| **Result** | **NEGATIVE** (-53.9% cum, Sharpe -0.94) |`:

```
| **Computational engine** | 2 (Momentum/Trend); composes with Engine 3 (Allocation) |
```

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, 24/7 |
| Universe | BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE (8 assets) |
| TC | 4 bps round-trip |
| Feature set | 8 TA types (RSI, MACD, BOLL, EMA_CROSS, STOCH, ATR_BREAK, VOL_BREAK, VWAP_REV) x 338 signal configs |
| Pre-filter | training-period top-performing TA types kept; unstable across years |
| Risk management | equal weight, P>0.50 gate, per-model cap |
| Computational engine | Engine 2 (Momentum/Trend); composes with Engine 3 (Allocation) for portfolio construction |

**Active regimes during test:**

Native regime ablation was not run for this experiment (predates the
12-class regime engine instrumentation). The OOS window (2025 full
year + 2026 Q1) spans both bull and corrective crypto regimes, which
is consistent with the experiment's primary finding: model-type
rankings invert across years, suggesting the strategy is heavily
regime-dependent even though no specific class is identified.

- general: not_measured -- recommend re-running with regime engine
  features (Class A trend, Class B vol level, Class D serial
  correlation) as ablation candidates

**Revival hypotheses:**

1. **Information-driven bars (dollar bars)** -- likelihood: medium. The annual model-type inversion (MACD <-> STOCH <-> RSI swapping which is profitable) is consistent with TA indicators sampling at wrong-frequency for the underlying market activity rhythm. Dollar bars at $1M threshold give consistent economic-activity granularity that may stabilize TA indicator behavior across regimes. Substrate is now available (Cycle 34 info_bars table, BTC + ETH at $1M and $5M thresholds; volume bars at 100/500 BTC and 1000/5000 ETH). Predicted lift: stabilization of model-type rankings, not necessarily positive absolute alpha.
2. **MCb-style composite gating** -- likelihood: low. Already explored in Exp 7 -- multi-indicator confluence helps signal quality but doesn't reach portfolio-positive even at GUI level on BTC 2024 bull, and CPO integration architecturally failed. Layering MCb over standard TA doesn't fix the underlying signal weakness.
3. **Triple-barrier labeling on TA signals** -- likelihood: low. Exp 10 already attempted this and produced INCONCLUSIVE (leverage runaway). Even if the leverage fix lands, the underlying TA-on-crypto signal is the binding constraint, not exit timing.
4. **Accept the verdict** -- likelihood: high. The cumulative evidence (this experiment + Exp 3 futures + Exp 4 FX + Exp 17 1-min momentum) is conclusive: standard TA has no persistent edge across asset classes. Future Engine 2 work should focus on either (a) structurally-grounded momentum like TSMOM (Exp 8/9) or (b) Engine 7 features (Engine 7 / Exp 13 funding carry is the confirmed-edge precedent).
```

---

## EXPERIMENT 3 -- TA_STANDARD x FUTURES (NEGATIVE folded, Engine 2)

(After the Result row added in the small fix above.)

### Attribute table addition

After the new `| **Result** | **NEGATIVE** ... |` row:

```
| **Computational engine** | 2 (Momentum/Trend); composes with Engine 3 (Allocation) |
```

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly (futures hours: equity index 24h via electronic; commodities ~22h) |
| Universe | ES, NQ, YM, RTY (equity index futures) + CL, GC, SI, NG (commodity futures) |
| TC | 2 bps round-trip (yfinance approximation) |
| Feature set | 8 TA types x 8 assets x config grid |
| Pre-filter | training-period top-performing TA types kept; selection inverted between Test A and Test B |
| Risk management | equal weight, per-model cap |
| Computational engine | Engine 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

**Active regimes during test:**

Native regime ablation not run. Test A (2025 train, Q1 2026 OOS)
selected one set of TA types and appeared positive; Test B (H1 2025
train, H2 2025 OOS) selected a completely different set and failed.
The cross-period TA-type inversion is the diagnostic finding.

- general: not_measured -- the strategy's regime sensitivity is implicit in the cross-period instability of pre-filter selection, but specific class contributions weren't measured

**Revival hypotheses:**

1. **Information-driven bars on futures** -- likelihood: medium. Same logic as Exp 2: if regime-dependence is a sampling-frequency issue, info bars stabilize the input distribution. Futures don't have the same retail microstructure as crypto, so the magnitude of improvement may be smaller, but the mechanism is the same. Caveat: info bars not yet built for futures; Cycle 34 only covered BTC + ETH crypto. Pre-requisite work: extend info_bars infrastructure to futures asset class.
2. **Regime D (serial correlation) hard filter** -- likelihood: medium. Cross-period TA-type inversion is exactly what serial-correlation regime detection is designed to flag. Hurst exponent at ~0.5 means TA signals (which assume either trend or mean-reversion) will inconsistently work. Hard-gate trading to only Hurst >0.55 (trend regime) or <0.45 (MR regime) for the appropriate signal class.
3. **VWAP_REV-only on equity index futures** -- likelihood: low-medium. Exp 11 (triple-barrier re-run on futures) showed VWAP_REV is the standout signal; futures' natural daily-VWAP reference may give that signal more grip than other TA types. Worth isolating before declaring all-TA-on-futures dead.
4. **Accept the verdict** -- likelihood: medium. Less conclusive than crypto/FX TA because the small sample sizes (one test A, one test B) means we can't say with the same confidence as Exp 2. But the underlying logic (TA has no persistent edge) is consistent.
```

---

## EXPERIMENT 4 -- TA_STANDARD x FX_G10 (NEGATIVE, Engine 2)

### Attribute table addition

After `| **TC** | 0.5 bps/leg × 2 legs = 1 bps round-trip (tightest of all classes) |`:

```
| **Result** | **NEGATIVE** (-0.31% cum, Sharpe -0.59 over Q1 2026 OOS) |
| **Computational engine** | 2 (Momentum/Trend); composes with Engine 3 (Allocation) |
```

(Exp 4 also has no explicit `**Result**` row; while we're here, add it
for parser cleanliness. This is small drift from the brief's "only Exp
3 needs Result fix" but the same problem; mention in retro.)

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, FX market hours (~24h Mon-Fri) |
| Universe | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP |
| TC | 1 bps round-trip (tightest of all asset classes tested) |
| Feature set | 8 TA types x 8 pairs |
| Pre-filter | EMA_CROSS, ATR_BREAK, STOCH, MACD kept |
| Risk management | equal weight, per-model cap |
| Computational engine | Engine 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

**Active regimes during test:**

Native regime ablation not run. FX markets are notably efficient with
narrow spreads; the experiment isolates TC as a non-issue (1 bps RT,
lowest of any tested class) which makes the negative result a stronger
TA-has-no-edge signal than crypto or futures.

- general: not_measured -- particularly relevant FX dimensions (Class
  F funding/carry differentials; Class A trend per pair; central bank
  policy regime) weren't measured

**Revival hypotheses:**

1. **Carry + TA composite (Engine 7 + Engine 2)** -- likelihood: medium. FX has a structural carry signal (interest rate differentials) that's been documented for decades. Use carry as a structural conditioner (Engine 7-style) and TA as timing within the carry-positive direction. This is analogous to how funding rate carry (Exp 13) works on crypto perps but with central-bank-driven base rates instead of perpetual funding.
2. **News-event filter (high-frequency around scheduled releases)** -- likelihood: low. FX intraday alpha is concentrated around NFP, FOMC, ECB events; this is a different strategy entirely (event-driven, not systematic TA), would be its own experiment, not a TA revival.
3. **Information-driven bars on FX** -- likelihood: low. FX is dominated by institutional flow; sampling-frequency issues are smaller than crypto. Info bars unlikely to surface signal where 1 bps TC already can't.
4. **Accept the verdict** -- likelihood: high. Combined with Exps 2/3, the three-asset-class consensus is "standard TA has no persistent edge anywhere." FX is the strongest "even with negligible TC" data point.
```

---

## EXPERIMENT 7 -- TA_COMPOSITE x CRYPTO MCb (PARTIAL, Engine 2)

### Attribute table addition

After `| **TC** | 10 bps round-trip (0.1%/leg, Binance spot maker) |`:

```
| **Result** | **PARTIAL** (GUI level promising for Anchor & Trigger 15m BTC; CPO integration architecturally failed) |
| **Computational engine** | 2 (Momentum/Trend); composes with Engine 3 (Allocation) |
```

(Exp 7 also has no explicit `**Result**` row -- add it.)

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | 15-minute and 4-hour time bars |
| Frequency | per-candle close, 24/7 |
| Universe | BTC, ETH, SOL tested; BNB/XRP/ADA/AVAX/DOGE pending |
| TC | 10 bps round-trip (Binance spot maker) |
| Feature set | WaveTrend (WT1/WT2), RSI+MFI, Stochastic RSI composite |
| Pre-filter | multi-indicator confluence (only fires on agreement; base_rate ~1-2%) |
| Risk management | none beyond signal-based entry/exit |
| Computational engine | Engine 2 (Momentum/Trend); composes with Engine 3 (Allocation) for portfolio construction. CPO integration architecturally incompatible due to low base_rate. |

**Active regimes during test:**

Native regime ablation not run. The 2024 BTC bull run (40K -> 100K)
is the dominant regime in the GUI backtest period; cross-year
validation (2022 bear, 2023 choppy) is the explicit "is the edge
real or regime-conditioned" test, not yet run.

- general: not_measured -- bull-regime tailwind almost certainly inflates the 2024 results

**Revival hypotheses:**

1. **Cross-year validation (2022 bear, 2023 choppy) of GUI Anchor & Trigger 15m BTC** -- likelihood: high. The explicit "next step" in existing analysis. 2024 BTC was bull; if the edge survives 2022 bear and 2023 choppy at GUI level (no CPO), the structural claim is supported. Pure data run; no new infrastructure required.
2. **Daily regime classifier as binary gate (atlas-principle direction)** -- likelihood: high. Build a daily binary classifier ("favorable regime for Anchor & Trigger today?") using yesterday's MCb features as features, not CPO. This is the architectural fix for the base_rate issue: instead of trying to predict per-3-day-window whether the signal will be profitable (base_rate 1-2%), predict per-day whether *today is a good day to leave the strategy on*. Base rate becomes 30-50% (most days the strategy is at least neutral), making the RF trainable.
3. **Multi-asset transfer (ETH, SOL, etc.)** -- likelihood: medium. Easy parallel test of "is the edge BTC-specific or transferrable?". Cheaper than cross-year validation; uses existing GUI infrastructure.
4. **Triple-barrier exits on Anchor & Trigger signals** -- likelihood: medium. The 15m Anchor signal fires every 2-3 weeks per asset; current exits are time-based (hold N hours). Triple-barrier (TP/SL/timeout) could improve exit P&L without changing the entry-signal frequency. Doesn't fix the base_rate issue for CPO but doesn't depend on CPO either; pairs naturally with #1 above.
```

---

(Brief continues with Exps 8-17. Document is intentionally long for
verbatim-paste workflow; Code can chunk through experiments
sequentially.)
## EXPERIMENT 8 -- MOMENTUM x CRYPTO TSMOM (PARTIAL, Engine 2)

### Attribute table addition

After `| **Gate** | P > 0.60 (RF probability threshold) |`:

```
| **Result** | **PARTIAL** (WEAK POSITIVE; primary +1.91% Sharpe +0.545; validation +0.46% Sharpe +0.245) |
| **Computational engine** | 2 (Momentum/Trend); composes with Engine 3 (Allocation) |
```

(Exp 8 also has no explicit `**Result**` row -- add it.)

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, 24/7 |
| Universe | ETH, SOL (BTC excluded -- mean-reverts too fast for momentum) |
| TC | 4 bps round-trip |
| Feature set | TSMOM_4H (4-48h lookback, 4-24h hold) + TSMOM_DAILY (24-168h lookback, 12-48h hold) |
| Pre-filter | asset selection (ETH+SOL only) is itself a pre-filter, motivated by Exp 8 OOS observation -- a soft form of selection bias |
| Risk management | equal weight, P>0.60 gate, per-model cap |
| Computational engine | Engine 2 (Momentum/Trend); composes with Engine 3 (Allocation) |

**Active regimes during test:**

Native regime ablation not run. Cross-year calibration drift (5.9%
lift 2024->2025 vs. 0.5% lift 2023->2024) is the primary regime-
sensitivity diagnostic. BTC vs ETH/SOL divergence (BTC momentum mean-
reverts fast; ETH/SOL momentum persists) is itself a microstructure-
regime difference rather than a market-state regime.

- A (<Trend>): not measured directly; momentum strategies are by construction trend-conditioned
- B (<Vol level>): not measured; momentum is documented to fail in low-vol regimes (no signal-to-noise)
- D (<Serial correlation>): not measured; the most relevant class for momentum-vs-mean-reversion regime detection

**Revival hypotheses:**

1. **Information-driven bars (volume bars or dollar bars)** -- likelihood: medium-high. Cross-year calibration drift is consistent with time-bar sampling-frequency issues -- a momentum lookback of "past 24 hours" captures different amounts of economic activity in low-vol vs. high-vol regimes. Volume bars normalize this. Substrate available (Cycle 34): BTC + ETH at multiple thresholds. ETH-specific volume bars at 1000 / 5000 ETH thresholds are the obvious starting point.
2. **Vol-scaled momentum (TSMOM-VS)** -- likelihood: medium. Standard TSMOM design: score = past_return / past_vol. Theoretically grounded (Moskowitz-Ooi-Pedersen "Time Series Momentum"); more robust across vol regimes than raw N-period returns. Stated as a "next step" in the existing analysis.
3. **Bear market (2022) validation** -- likelihood: needs-data. Momentum strategies are known to fail in bear markets; without bear validation we don't actually know the strategy's worst-case behavior. Required before any production deployment.
4. **Re-include BTC with separate model (not portfolio cancellation)** -- likelihood: low. BTC's mean-reversion is fast, ETH/SOL is persistent; including BTC in the same portfolio cancels both signals. Running BTC as a separate (mean-reversion) strategy and ETH/SOL as the momentum strategy might recover both edges, but this is two experiments not one.
```

---

## EXPERIMENT 9 -- MOMENTUM x CRYPTO Triple Barrier (PARTIAL, Engine 2+7)

### Attribute table addition

After `| **RF AUC** | 0.879-0.902, base_rate 27-39% |`:

```
| **Result** | **PARTIAL** (+2.43% cum, Sharpe +0.778, Max DD -2.49%; improved over Exp 8 by triple-barrier exits) |
| **Computational engine** | 2 (Momentum/Trend) primary + 7 (Event/Signal for triple-barrier labels); composes with Engine 3 (Allocation) |
```

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, 24/7 |
| Universe | ETH, SOL |
| TC | 4 bps round-trip |
| Feature set | TSMOM_4H + TSMOM_DAILY x 1728 (signal config x barrier config) combinations |
| Pre-filter | as Exp 8: asset selection ETH+SOL only |
| Risk management | equal weight, P>0.60 gate, per-model cap, triple-barrier TP/SL/timeout exits |
| Computational engine | Engine 2 (Momentum/Trend) primary; Engine 7 (Event/Signal) for triple-barrier labels; composes with Engine 3 (Allocation) |

**Active regimes during test:**

Same diagnostic as Exp 8 (no native ablation). Improvement from
Sharpe +0.55 to +0.78 vs Exp 8 isolates the triple-barrier exit
mechanism as the lift source. Non-monotonic calibration (0.60-0.65
-1.1%, 0.65-0.70 +5.2%, 0.70-0.80 -8.3%) confirms the underlying RF
remains regime-sensitive even with improved exits.

- A (<Trend>): not measured
- B (<Vol level>): not measured
- D (<Serial correlation>): not measured

**Revival hypotheses:**

1. **Info bars + triple-barrier (the Financial Innovation 2025 paper recipe)** -- likelihood: high. This is the canonical LSTM v2 setup. Move from hourly time bars to dollar bars (BTC + ETH info_bars available Cycle 34); keep the triple-barrier label discipline but apply it in bar-index space (not wall-clock space). Predicted lift: stabilize the non-monotonic calibration across regimes -- the same dollar-bar-event experiences should produce similar feature distributions regardless of when in calendar time they occur.
2. **Volume bars specifically** -- likelihood: medium-high. Dollar bars normalize by dollar volume; volume bars normalize by base-asset volume. For momentum strategies where the underlying physics is "more activity = more conviction = stronger persistence", volume bars may be more direction-stable than dollar bars. Both available in Cycle 34 substrate.
3. **LSTM v2 architecture (Cycle 37+ scope)** -- likelihood: medium for ultimate use, but the RF here works architecturally. Replacing RF with LSTM is a bigger architectural commitment that should happen alongside info-bar + triple-barrier integration, not separately.
4. **Bear market (2022) validation** -- likelihood: needs-data. Same as Exp 8.
```

---

## EXPERIMENT 10 -- TA x CRYPTO Triple Barrier (INCONCLUSIVE, Engine 2+7)

### Attribute table addition

After `| **Pre-filter kept** | STOCH, RSI, EMA_CROSS, VOL_BREAK, BOLL (5/8 types) |`:

```
| **Result** | **INCONCLUSIVE** (-83.78% portfolio is leverage-runaway artifact, not signal failure; individual models like ADA_STOCH +117% confirm signal works) |
| **Computational engine** | 2 (Momentum/Trend) primary + 7 (Event/Signal for triple-barrier labels); composes with Engine 3 (Allocation -- failed via leverage runaway) |
```

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, 24/7 |
| Universe | 8 crypto assets x 5 TA types kept (40 individual models) |
| TC | 4 bps round-trip |
| Feature set | 8 TA types x 338 signal configs x 72 barrier configs |
| Pre-filter | training-period top 5/8 TA types kept (STOCH, RSI, EMA_CROSS, VOL_BREAK, BOLL) |
| Risk management | equal weight (5% per-model cap) -- but no PORTFOLIO-level leverage cap; 35-40 models simultaneously above gate produced 175% gross daily |
| Computational engine | Engine 2 + Engine 7 labels; composes with Engine 3 (Allocation FAILED via portfolio leverage runaway) |

**Active regimes during test:**

Not measured. The catastrophic portfolio result masks any
regime-sensitivity analysis. 2025 was broadly corrective for crypto,
which is the worst possible environment for a 175%-leveraged long
basket, but even in a flat regime the construction would have been
unstable.

- general: not_measured -- the leverage construction failure is independent of regime and dominates the result

**Revival hypotheses:**

1. **Hard portfolio leverage cap (e.g., max total_weight = 0.5)** -- likelihood: very high. Explicitly stated as the fix in the existing markdown. Trivial code change; re-run with this cap will produce a coherent portfolio result for the first time. Individual-model results (ADA_STOCH +117%, BTC_STOCH +28.8%) suggest the underlying signal does have content; the portfolio failure was purely a construction artifact. Expected re-run outcome: ambiguous-to-modestly-positive at portfolio level, since the constituent models that worked (~5-10) had real edge, and the rest were noise.
2. **Reduce model count to top-K by training Sharpe** -- likelihood: high. Alternative to portfolio cap: pre-filter the 40 models to top 10-15 by training-period Sharpe before portfolio construction. Tighter selection, similar effective gross-leverage outcome, simpler decision rule.
3. **Info bars (after fixing leverage)** -- likelihood: medium. Once the leverage issue is fixed and we can read the underlying signal quality cleanly, info bars become a natural next test for the same Exp-2 reasons.
4. **Accept partial-revival with leverage cap, treat as "confirmed PARTIAL"** -- likelihood: high follow-on. After leverage fix, the experiment moves out of INCONCLUSIVE limbo into a definite category (most likely PARTIAL or NEGATIVE depending on what surfaces). That's the immediate-value revival path even if the strategy doesn't end up production-deployable.
```

---

## EXPERIMENT 11 -- TA x FUTURES Triple Barrier (PARTIAL, Engine 2+7)

### Attribute table addition

After `| **Pre-filter kept** | VWAP_REV, ATR_BREAK, STOCH (3/8 types) |`:

```
| **Result** | **PARTIAL** (+2.93% cum, Sharpe +1.528, Max DD -2.53% -- but only 47-day OOS) |
| **Computational engine** | 2 (Momentum/Trend) primary + 7 (Event/Signal for triple-barrier labels); composes with Engine 3 (Allocation) |
```

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly (futures hours) |
| Universe | ES, NQ, YM, RTY (equity index) + CL, GC, SI, NG (commodity) |
| TC | 2 bps round-trip |
| Feature set | 8 TA types x 8 futures x 7920 configs |
| Pre-filter | training top 3/8 TA types kept (VWAP_REV, ATR_BREAK, STOCH) |
| Risk management | equal weight, P>0.50 gate, triple-barrier exits |
| Computational engine | Engine 2 + Engine 7 labels; composes with Engine 3 (Allocation) |

**Active regimes during test:**

Native regime ablation not run. 47-day OOS window (Q1 2026) is far
too small to assess regime sensitivity meaningfully. Calibration
lift at 0.65-0.70 was +14.1% (strongest of any experiment), but
sample size doesn't separate signal from noise.

- general: not_measured -- 47 days is below the threshold for meaningful regime analysis

**Revival hypotheses:**

1. **Extend OOS window to 200+ days** -- likelihood: very high. The single largest defect of this experiment is sample size. Run the same configuration on 2025 H2 data (or 2025 full year if accessible) -- pure data accumulation, no infrastructure changes needed. Result determines whether the +14.1% calibration lift is real or sampling noise.
2. **Single-strategy focus on VWAP_REV** -- likelihood: medium. Pre-filter selected VWAP_REV as the strongest signal; isolating it (drop the other 7 TA types entirely) tests whether the futures TA edge is VWAP-specific or distributed. Cheap test.
3. **Volume-bar-equivalent on futures (contract-count bars)** -- likelihood: medium. Futures have well-defined contract sizes; the natural "info bar" unit is contracts (or notional dollars). Info bars infrastructure (Cycle 34) was built for crypto trades; extending to futures is its own infrastructure work but uses the same writer/builder framework.
4. **Per-asset triple-barrier tuning** -- likelihood: medium. Gold (GC) is documented as the main drag; the TP/SL barriers were fit on aggregate. Per-asset volatility-aware barrier sizing (e.g., barriers scaled by ATR) could improve the per-asset signal-to-noise.
```

---

## EXPERIMENT 12 -- TA x FX_G10 Triple Barrier (INCONCLUSIVE, Engine 2+7)

### Attribute table addition

After `| **Pre-filter kept** | EMA_CROSS, RSI, MACD, STOCH, BOLL (5/8 types) |`:

```
| **Result** | **INCONCLUSIVE** (-0.39% cum, Sharpe -0.471; same leverage issue as Exp 10 at smaller magnitude; 52-day OOS too short) |
| **Computational engine** | 2 (Momentum/Trend) primary + 7 (Event/Signal for triple-barrier labels); composes with Engine 3 (Allocation) |
```

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, FX market hours |
| Universe | 8 G10 pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP) |
| TC | 1 bps round-trip |
| Feature set | 8 TA types x 8 pairs x 7920 configs |
| Pre-filter | training top 5/8 TA types kept |
| Risk management | equal weight, per-model cap -- same lack-of-portfolio-leverage-cap as Exp 10, smaller magnitude on FX |
| Computational engine | Engine 2 + Engine 7 labels; composes with Engine 3 (Allocation) |

**Active regimes during test:**

Not measured. 52-day OOS too short for regime analysis. Mixed
individual results (22/40 models positive by Sharpe; EMA_CROSS on
AUDUSD Sharpe +4.07, USDJPY +3.06) suggest pair-specific edge exists
but is diluted by 40-model portfolio.

- F (<Funding/positioning>): for FX, funding is interest rate differentials; carry-bearing pairs likely have a different regime profile than non-carry pairs (AUDUSD vs. EURUSD have very different carry profiles)
- general: not_measured

**Revival hypotheses:**

1. **Hard portfolio leverage cap + extended OOS** -- likelihood: very high. Both fixes are obvious and cheap. Combine the Exp 10 portfolio fix with the Exp 11 OOS-extension; the resulting 200+ day clean-leverage run will give a real verdict.
2. **Restrict to top-performing pairs (AUDUSD, USDJPY)** -- likelihood: medium. Top per-pair results (EMA_CROSS Sharpe +4.07 and +3.06) suggest the edge is concentrated in specific pairs, likely carry-related. AUDUSD and USDJPY are both carry pairs; pure G10 carry sensitivity may be the underlying structural signal.
3. **Carry overlay** -- likelihood: medium. Combine with the rate-differential signal: TA as timing within carry-positive direction. This is the FX equivalent of what funding rate carry (Exp 13) does for crypto -- structural signal as gate, TA as timing.
4. **Info bars on FX** -- likelihood: low. FX is dominated by institutional flow; sampling-frequency issues are smaller than crypto. Info bars unlikely to surface signal where 1 bps TC already can't.
```

---

(Brief continues with Exps 14-17 in part 3.)
## EXPERIMENT 14 -- GRID BOT x CRYPTO (NEGATIVE, Engine 1)

### Attribute table addition

After `| **OOS** | 2025-01-01 → 2026-03-27 (448 days) |`:

```
| **Result** | **NEGATIVE** (CONFIRMED -- Sharpe -10.79; calibration lift +7% insufficient for asymmetric payoff that requires 70%+ WR) |
| **Computational engine** | 1 (Cointegration/Mean-Reversion -- grid bots are range-bound MR); composes with Engine 3 (Allocation) |
```

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | hourly, 24/7 |
| Universe | BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE |
| TC | 4 bps one-way (per grid crossing) |
| Feature set | 36 configs (4 spacings x 3 range widths x 3 hold durations) |
| Pre-filter | none -- all 8 assets feed CPO |
| Risk management | equal weight per model -- but RF calibration too weak (max +7% lift) to filter losing regimes effectively |
| Computational engine | Engine 1 (range-bound mean reversion); composes with Engine 3 (Allocation) |

**Active regimes during test:**

Not measured directly, but the failure mode IS regime-based: grid
bots are profitable in choppy (Class A: low-trend) regimes and fail
catastrophically in trending regimes. The 2025 OOS window was broadly
declining (BTC -25%), which is the worst possible environment for
long-only grids.

- A (<Trend>): the dispositive class. RF cannot reliably detect trend regime ahead of hold periods; calibration lift only +7% at P>0.65 (vs +41% needed; reference funding carry Exp 13)
- B (<Vol level>): not directly measured but relevant; high-vol trending markets are worst-case for grids

**Revival hypotheses:**

1. **External regime detector (not CPO RF) as hard gate** -- likelihood: medium. The atlas verdict explicitly notes "may still be viable as standalone strategy with real-time regime monitoring (e.g., triggered by VIX proxy, manually managed)." Use a dedicated regime classifier (ADX-based trend detection, vol regime detection, or human-in-the-loop) as a hard gate. Only run grid bots in confirmed-choppy regime. The CPO RF cannot do this; a focused single-purpose regime model can.
2. **Adaptive grid spacing (vol-aware)** -- likelihood: low. Re-architects the strategy substantially; doesn't address the binding "RF can't detect trend regime well" issue.
3. **Asymmetric grids with stop-out** -- likelihood: low. Allows grid bots to escape trending environments by abandoning positions when range breaks. Effectively makes the strategy "grid until breakout, then exit" which is a different strategy from pure grid.
4. **Accept the verdict as CPO-architectural** -- likelihood: high. The atlas principle is clear: grid bots have insufficient CPO-learnable signal to overcome TC and asymmetric drawdown. Future grid work belongs outside the CPO framework (Engine 1-pure or manual operation), not as a CPO revival.
```

---

## EXPERIMENT 15 -- VRP x BTC/ETH (INCONCLUSIVE, data-blocked, Engine 4)

### Attribute table addition

After `| **OOS** | 2024 (synthetic DVOL) |`:

```
| **Result** | **INCONCLUSIVE** (BLOCKED on data infrastructure -- Deribit geo-block + limited free API history; synthetic DVOL kills the signal) |
| **Computational engine** | 4 (Volatility/Options); composes with Engine 3 (Allocation) |
```

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | daily |
| Frequency | daily, derived from options chain snapshots |
| Universe | BTC, ETH (7d + 30d tenor each = 4 nominal models, but tenor collapse bug = 2 effective) |
| TC | 10 bps round-trip (options bid-ask wider than spot) |
| Feature set | 21-config grid; VRP = IV - GARCH-forecast RV; surface features (skew, butterfly, term structure, vol-of-vol, IV rank) |
| Pre-filter | none |
| Risk management | delta-hedged via variance swap approximation; VRP-gated entry |
| Computational engine | Engine 4 (Volatility/Options); composes with Engine 3 (Allocation) |

**Active regimes during test:**

Cannot be measured with synthetic DVOL data. The synthetic
implementation reduces VRP features to "GARCH_forecast vs
rolling_RV vs GARCH_forecast" -- all derived from the same
information source, so the RF has no genuine forward-looking
implied vol signal to learn from. Regime-relevant classes for vol
strategies cannot be assessed.

- B (<Vol level>): the dispositive class for vol strategies; not measurable with synthetic DVOL
- C (<Vol-of-vol>): also dispositive; not measurable

**Revival hypotheses:**

1. **Real IV data source: Amberdata, Kaiko, or Tardis.dev** -- likelihood: very high (purely a data question). Once real historical IV is available, the strategy is testable. Paid APIs but the cost is dwarfed by the value of confirming/refuting an Engine 4 edge.
2. **CBOE BVOL as alternative free source** -- likelihood: medium. BVOL is Bitcoin volatility index; free and may have better historical coverage than Deribit free API.
3. **Tenor collapse bug fix** -- likelihood: required (not optional). Independent of data, BTC_7d_VOL and BTC_30d_VOL produce identical results because `model.tenor` is never used in feature computation. Must fix before any real-data run.
4. **Run from non-geo-blocked machine for Deribit live** -- likelihood: low. Solves geo-block but not the historical-depth problem (free API only ~16 days history). Useful for forward-tested paper trading; insufficient for training.
5. **VRP strategy is structurally sound infrastructure-wise** -- likelihood: not a revival but a confirmation. `engines/garch_model.py`, `engines/vol_surface.py`, `engines/vol_strategy.py` are complete and correct per the atlas notes. The strategy IS ready to test the moment real data exists. This is the experiment with the highest "data availability gates everything else" sensitivity.
```

---

## EXPERIMENT 16 -- CROSS-DEX x ARBITRUM (NEGATIVE, Engine 6)

### Attribute table addition

After `| **Execution** | Flash loan atomic arb ... |`:

```
| **Result** | **NEGATIVE** (CONFIRMED -- zero executable depth at retail scale on Arbitrum L2; pools showing spreads have ~$50-100 real liquidity) |
| **Computational engine** | 6 (On-Chain/DeFi); standalone (does not compose with Engine 3) |
```

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | event-driven (per-block on-chain state) |
| Frequency | continuous, 15s scanner interval |
| Universe | WETH, WBTC, USDC, USDCe, USDT, ARB, GMX, LINK on Uniswap V3 + SushiSwap V3 + PancakeSwap V3 |
| TC | flash loan fee (~0.05%) + gas + slippage |
| Feature set | cross-venue price differences with V3 concentrated-liquidity swap math |
| Pre-filter | staleness detection; arb-eligible pair filtering (22 of 219 pools) |
| Risk management | atomic execution via flash loan (zero capital risk if executed successfully); failure case is gas cost + transaction revert |
| Computational engine | Engine 6 (On-Chain/DeFi); standalone -- arbitrage is single-tx atomic |

**Active regimes during test:**

Not regime-dependent in the traditional sense. The failure mode is
structural / microstructure: pools showing apparent spreads are
themselves the cause of the spread (thin liquidity that makes
arb-execution impossible). This is a market-structure verdict, not a
state-dependent one.

- E (<Microstructure>): the dispositive class; Arbitrum L2 pools are too thin at retail trade sizes for spatial arb to execute
- G (<Liquidity>): closely related; the per-pool $50-100 liquidity quote is the binding constraint

**Revival hypotheses:**

1. **Different chain with thicker pools (Ethereum L1, Solana)** -- likelihood: low. The efficiency arguments scale UP on more-liquid L1s; if Arbitrum is fully efficient at retail, L1 is more so. The MEV / private-orderflow ecosystems on L1 actively compete for any visible spread.
2. **Cross-chain arbitrage (bridge latency exploit)** -- likelihood: low-medium. Bridges introduce real latency (minutes to hours); price-discovery lag across chains exists but bridge costs + bridge risk usually dominate. Would be its own experiment (different infrastructure, different risk profile), not a revival of the same Engine 6 strategy.
3. **MEV-aware execution (sandwich, JIT liquidity)** -- likelihood: low. Different game entirely -- requires private mempool access, flashbots-style infrastructure, builder relationships. The Engine 6 framework supports the math but the execution layer is fundamentally different.
4. **Accept the verdict: retail spatial arb is dead** -- likelihood: very high. The atlas verdict is structural ("fully efficient at retail"). The infrastructure built (dex_scanner, dex_quoter, flash_executor) remains useful as Engine 6 substrate for any future on-chain strategy, but spatial arb specifically is closed.
```

---

## EXPERIMENT 17 -- SHORT-TERM MOMENTUM x CRYPTO (NEGATIVE, Engine 2)

### Attribute table addition

After `| **Intended use** | Entry signals for flash loan looping (leveraged Aave positions) |`:

```
| **Result** | **NEGATIVE** (CONFIRMED -- 31% win rate, -827.5 bps over 88 trades; -2,482 bps at 3x leverage) |
| **Computational engine** | 2 (Momentum/Trend) + 5 (Order Book / Microstructure features) -- order flow imbalance is a microstructure input to the 5-component composite |
```

### Sections to add before the closing `---`

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | 1-minute time bars |
| Frequency | per-minute |
| Universe | BTC/USDT, ETH/USDT, SOL/USDT, ARB/USDT, WBTC/USDT |
| TC | implicit in TP=50bps / SL=30bps barrier sizing |
| Feature set | 5-component composite: volume spike + price velocity (3/5/10 min) + consecutive candles + range breakout + order flow imbalance |
| Pre-filter | none |
| Risk management | TP=50bps, SL=30bps, hold 5-30 min |
| Computational engine | Engine 2 (Momentum/Trend) for price-based signals + Engine 5 (Order Book / Microstructure) for OFI; composes with Engine 3 if scaled |

**Active regimes during test:**

72h backtest is too short for meaningful regime analysis. The 31%
win rate is symptomatic of "signals actively wrong about direction
~50% of the time" -- a strong indication that 1-minute time bars at
low-vol periods contain mostly noise that the composite scorer
mistakes for signal.

- general: not_measured -- 72h sample too small; signal-vs-noise issue dominates regime considerations

**Revival hypotheses:**

1. **Information-driven bars (especially volume-imbalance bars)** -- likelihood: medium. This is the canonical use case the Financial Innovation 2025 paper targets: 1-minute time bars at low-vol periods have no signal, but volume-imbalance bars only close when actual aggressor-side activity exceeds a threshold. Substrate available (Cycle 34): VIB bars at $500k threshold for BTC + ETH. The 5-component composite reformulated against VIB bars (instead of 1-min time bars) is a clean test of whether the signal is real-but-time-bar-noised vs. structurally absent.
2. **Structural conditioner overlay (funding rate, OI change)** -- likelihood: medium. The 5-component composite is all TA-derived (price + volume technical). Layering Engine 7 features (funding rate level, OI change rate, basis) tests whether the 1-min momentum has *conditional* alpha when structural regime is favorable.
3. **Triple-barrier labeling on the same signals** -- likelihood: low. The directional accuracy is the problem (31% WR); better exit logic doesn't help when the entries are wrong half the time. Improving exits would help a 55% WR signal; it can't rescue 31%.
4. **Accept verdict: 1-min retail crypto momentum has no edge** -- likelihood: high. Consistent with Exp 2 (all TA on crypto), Exp 4 (TA on FX), Exp 3 (TA on futures). Adds the "high-frequency intraday" data point to the broader TA-has-no-edge conclusion.
```

---

## Step ordering for Code

1. **Read regime_classes table** from `data/praxis_meta.db`. Print the
   12 class letters + names so the substitutions below are exact:

   ```python
   import sqlite3
   conn = sqlite3.connect("data/praxis_meta.db")
   for row in conn.execute(
       "SELECT class_letter, class_name FROM regime_classes ORDER BY class_letter"
   ).fetchall():
       print(row)
   ```

2. **Substitute placeholder regime labels** throughout this brief's
   markdown content. Where the brief writes `B (<Vol level>)`, replace
   `<Vol level>` with the actual name from the DB. Where it writes
   `<Class name>` (less specific guess), do the same lookup.

3. **Apply the two cleanups first** (Exp 1 + Exp 13). These edit
   existing structured sections; smaller scope, lower risk.

4. **Apply Exp 3 result_class fix** (single attribute-table row add).
   Same for Exps 4, 7, 8, 11 which also lack explicit `**Result**`
   rows -- add them as we go. The brief content shows the exact row
   to add for each.

5. **Apply the 11 fresh backfills** (Exps 2, 3, 4, 7, 8, 9, 10, 11,
   12, 14, 15, 16, 17 -- wait that's 13, recount: Exps 2, 3, 4, 7,
   8, 9, 10, 11, 12, 14, 15, 16, 17 = 13. Yes 13.) Mechanical paste
   from the brief content; one experiment at a time; verify each
   parses correctly before moving on.

6. **Pre-flight test parser** before committing: run
   `python -m engines.atlas_sync --validate --no-embed` and confirm:
   - 0 errors during parsing
   - All 13 experiments now have `test_conditions`, `revival_hypotheses`,
     `regime_state_at_test`, `computational_engine` populated when
     printed via verbose mode
   - Exp 1 and Exp 13's regime_state_at_test keys are now canonical
     (no sanitized fallback like `b,_c,_f,_h,_j,_k` or
     `f_=_+1,_+2_(positive_funding_sustained)`)

7. **Run atlas_sync for real**:

   ```powershell
   python -m engines.atlas_sync
   ```

   Expected: 15 updated TRADING entries (13 fresh backfills + Exp 1 +
   Exp 13 because their full_markdown changed via the regime cleanups),
   0 added, 0 removed. 15 embeddings regenerated, 20 PMA entries
   skipped (no changes).

8. **MCP verification** (user will do):
   - `praxis:atlas_get(2)` should now show populated 4 new fields
   - `praxis:atlas_get(12)` (Exp 14 grid bot) should show
     `computational_engine = 1`
   - `praxis:atlas_get(13)` (Exp 15 VRP) should show
     `computational_engine = 4`
   - `praxis:atlas_get(1)` and `praxis:atlas_get(11)` (Exp 13)
     should show canonical regime keys
   - `praxis:atlas_search("dollar bars revival")` should now surface
     the experiments where dollar bars are listed as revival hypotheses

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | All 13 fresh-backfill experiments have populated `test_conditions`, `revival_hypotheses`, `regime_state_at_test`, `computational_engine` in atlas DB |
| 2 | Exp 1's `regime_state_at_test` keys are canonical (`B_<...>`, `C_<...>`, `F_<...>`, `H_<...>`, `J_<...>`, `K_<...>` -- no `b,_c,_f,_h,_j,_k` sanitized fallback) |
| 3 | Exp 13's `regime_state_at_test` keys are canonical (`F_funding_positioning`, `A_trend`, `B_vol_level` -- no `f_=_+1,_+2_...` sanitized fallback) |
| 4 | Exp 3's `result_class` is now `NEGATIVE` (was NULL pre-cycle) |
| 5 | Exps 4, 7, 8, 11 also have `result_class` populated correctly (they were missing explicit `**Result**` rows; same drift as Exp 3) |
| 6 | atlas_sync re-run reports 15 updated / 0 added / 0 removed on TRADING_ATLAS.md (the 13 fresh backfills + Exp 1 + Exp 13 regime cleanups) |
| 7 | 15 embeddings regenerated; 20 PMA entries skipped (no changes) |
| 8 | `praxis:atlas_search("dollar bars revival")` returns relevant experiments with new revival_hypotheses content; `praxis:atlas_search("portfolio leverage cap")` returns Exps 10 and 12 |
| 9 | All `regime_state_at_test` keys across the 15 backfilled trading experiments follow canonical `<Letter>_<class_name>` form (no compound-letter or compound-state fallback) |
| 10 | All 13 fresh-backfill experiments have a `Computational engine` row in their attribute table |

## Out of scope reminder

- No re-runs of any experiment (that's Cycle 36+).
- No PMA changes (different file).
- No `atlas_search` engine-filter parameter (deferred TODO).
- No new info_bars threshold backfills (Cycle 36 can add additional thresholds when revival re-runs identify needs).
- No retroactive regime engine runs (the "not_measured" honest signal stands).

## Commit message (use verbatim; remember the heredoc gotcha)

Use `git commit -F <file>` with the message in a file to avoid the
heredoc `$` escaping issue from Cycle 34 commit 808f19e. Save the
following to a temp file (e.g. `/tmp/cycle35_commit.txt`):

```
Cycle 35: Atlas mass backfill -- remaining 13 trading experiments

Brings the 13 trading experiments not covered by Cycle 33's design
validation (Exps 1 and 13) to the same structured shape:
computational_engine integer, test_conditions JSON dict,
revival_hypotheses JSON list, regime_state_at_test JSON dict.

Per the Q1 strict-canonical decision (Cycle 35 design phase),
also cleans up two pre-existing experiments whose
regime_state_at_test had sanitized fallback keys:

- Exp 1: split the combined-letter B, C, F, H, J, K bullet into
  six canonical single-letter bullets so keys become
  B_<class>, C_<class>, ... instead of b,_c,_f,_h,_j,_k.
- Exp 13: rewrite regime bullets to use canonical F (Funding/
  positioning) form. F-state details (+1,+2 / 0 / -1,-2) moved
  from bullet KEY to bullet DESCRIPTION so the dict key
  collapses to canonical F_funding_positioning instead of
  f_=_+1,_+2_(positive_funding_sustained).

Also folds in 5 small Result-row fixes for experiments whose
markdown clearly stated their verdict but lacked an explicit
**Result** row in the attribute table (Exps 3, 4, 7, 8, 11 --
Exp 3 was the original Cycle 32 retro callout; the others
surfaced during the mass-backfill read-through).

Revival hypotheses for each experiment are calibrated to that
experiment's specific failure mode. Engine 2 NEGATIVE
experiments now reference Cycle 34's info_bars infrastructure
concretely (dollar bars and volume bars at specific thresholds)
where the failure mode is consistent with sampling-frequency /
input-distribution instability. Engine 4 VRP references the
data infrastructure path. Engine 6 cross-DEX references the
chain / venue choice. Engine 1 grid bot references external
regime detection. Engine 2+7 triple-barrier reruns reference
the canonical info-bars + triple-barrier-in-bar-index-space
recipe per the Financial Innovation 2025 paper.

atlas_sync re-run after edits: 15 TRADING entries updated
(13 fresh + Exp 1 + Exp 13 cleanups), 0 added, 0 removed.
15 embeddings regenerated.

After this commit, the 15-experiment TRADING_ATLAS is fully
backfilled along the Cycle 33 structured schema. The Research
Agent can navigate by computational_engine, query revival
hypotheses programmatically, and surface experiments whose
revival hypothesis sets reference specific Cycle 34 info-bar
thresholds. Cycle 36+ can pick the highest-likelihood revival
hypotheses across the backfilled atlas and run them.

Out of scope: re-running any experiment, PMA changes,
atlas_search engine-filter parameter (deferred TODO).
```

Then commit:

```powershell
git commit -F /tmp/cycle35_commit.txt
```

## Post-cycle status

After Cycle 35 lands:

- 15 trading experiments fully backfilled along Cycle 33 schema
- All `regime_state_at_test` keys canonical (no fallback shapes)
- `atlas_search` queries against revival concepts will surface
  relevant experiments by semantic content + structural fields
- Cycle 34 info_bars substrate has its primary downstream
  consumer wired up (atlas references point at specific
  thresholds in the live table)
- Next: Cycle 36 picks the highest-likelihood revival hypothesis
  (or hypotheses) and runs them as actual research work
