# Cycle 53 (D7+D8) — Paper-Executor Backtest Replay + Config-Gate Fix

_Generated: replay clock = 2026-04-30T00:00:00+00:00; OOS 2025-01-01..2026-03-26; notional $500; TC 8 bps round-trip._

## D7 — decomposition vs atlas Exp 13 (pre-fix baseline)

**Criterion 1 — funding-term exact match (< 1e-6 USD/trade):**

- gate 0.50: PASS — 702 trades, max |Δ| = 6.661e-16 USD, mean = 5.652e-17
- gate 0.70: PASS — 243 trades, max |Δ| = 6.661e-16 USD, mean = 6.878e-17

### gate P>0.50 (executor books every alert — no config gate)

```
  asset  shared     exec%    atlas%    basis%     resid%  | zeroed  zero% execOnly%
  ---------------------------------------------------------------------------------
  BTC        74   +2.8768   +3.3549   +0.4780  -8.67e-17  |      3   3.9%   +0.1874
  ETH       103   +4.4964   +5.0411   +0.5446  -1.13e-15  |     23  18.3%   -0.7349
  SOL        56   +1.3188   +1.5507   +0.2319  +6.07e-16  |      4   6.7%   +0.7823
  XRP       112   +3.6962   +4.0532   +0.3569  +1.39e-15  |     59  34.5%   -0.9850
  ADA       160  +10.6447  +10.5102   -0.1345  -1.47e-15  |     26  14.0%   -0.3064
  AVAX       78   +0.1411   +0.8733   +0.7322  -8.67e-17  |      4   4.9%   +0.1555
  ---------------------------------------------------------------------------------
  ALL       583  +23.1741  +25.3833   +2.2092  -3.47e-16  |    119  17.0%   -0.9012
```

Per-asset basis term over the shared set (criterion 2; bps/hold):

| asset | n | basis mean (bps) | basis std (bps) |
|---|---|---|---|
| BTC | 74 | +0.646 | 1.862 |
| ETH | 103 | +0.529 | 1.650 |
| SOL | 56 | +0.414 | 2.893 |
| XRP | 112 | +0.319 | 3.525 |
| ADA | 160 | -0.084 | 3.887 |
| AVAX | 78 | +0.939 | 5.515 |

### gate P>0.70 (executor books every alert — no config gate)

```
  asset  shared     exec%    atlas%    basis%     resid%  | zeroed  zero% execOnly%
  ---------------------------------------------------------------------------------
  BTC        36   +3.1378   +3.5080   +0.3703  -6.51e-16  |      0   0.0%   +0.0000
  ETH        44   +1.6488   +2.0672   +0.4184  +6.07e-16  |      0   0.0%   +0.0000
  SOL         8   +0.2197   +0.3516   +0.1320  +0.00e+00  |      4  33.3%   +0.7823
  XRP        42   +2.5572   +3.1786   +0.6215  -3.47e-16  |      4   8.7%   +0.1305
  ADA        75   +8.9799   +9.4541   +0.4742  +2.17e-15  |      0   0.0%   +0.0000
  AVAX       30   +0.2500   +0.9218   +0.6718  +8.67e-17  |      0   0.0%   +0.0000
  ---------------------------------------------------------------------------------
  ALL       235  +16.7933  +19.4814   +2.6881  +3.47e-15  |      8   3.3%   +0.9128
```

Per-asset basis term over the shared set (criterion 2; bps/hold):

| asset | n | basis mean (bps) | basis std (bps) |
|---|---|---|---|
| BTC | 36 | +1.028 | 2.390 |
| ETH | 44 | +0.951 | 1.722 |
| SOL | 8 | +1.650 | 3.571 |
| XRP | 42 | +1.480 | 3.367 |
| ADA | 75 | +0.632 | 3.851 |
| AVAX | 30 | +2.239 | 5.246 |

## Criterion 6 — live-monitor config-gate (the gap)

`funding_monitor` gated on argmax-P > gate ALONE; the per-config hard thresholds (`min_funding_ann` / `min_pct_positive`) lived only in `run_funding_single_day` (backtest path). So the monitor alerted — and the executor booked — trades atlas zeroed. Since Cycle 41 the live system has therefore been running a DIFFERENT (unverified) strategy than atlas's Sharpe +4.65 measured; it has not bitten only because `funding_alerts` has stayed at 0 rows in the current sit-out regime.

- gate 0.50: **17.0%** (119/702) atlas-zeroed days the pre-fix system books; executor-only P&L = -0.9012% (net negative — atlas's gate was correctly skipping losers).
- gate 0.70: **3.3%** (8/243) atlas-zeroed days the pre-fix system books; executor-only P&L = +0.9128% (net positive here, but taken without the discipline that justified them).

## D8 — config-gate fix (monitor D8a + executor D8b)

Both layers now enforce atlas's Condition 1 + 2 (`ann_rate ≥ min_funding_ann AND pct_positive ≥ min_pct_positive`): the monitor suppresses the alert (D8a) and the executor's 11th risk check skips it (D8b, `config_thresholds_not_met`). Post-fix decomposition — the atlas-zeroed leak goes to 0 booked / 0.0000% execOnly:

### gate P>0.50 (post-fix)

- D8a: alerts emitted 702 → 583 (monitor suppressed 119 config-fail signals)
- D8b: with all alerts emitted, executor entered 583 / skipped 119 (`config_thresholds_not_met`)
```
  asset  shared     exec%    atlas%    basis%     resid%  | zeroed  zero% execOnly%
  ---------------------------------------------------------------------------------
  BTC        74   +2.8768   +3.3549   +0.4780  -8.67e-17  |      3   3.9%   +0.0000
  ETH       103   +4.4964   +5.0411   +0.5446  -1.13e-15  |     23  18.3%   +0.0000
  SOL        56   +1.3188   +1.5507   +0.2319  +6.07e-16  |      4   6.7%   +0.0000
  XRP       112   +3.6962   +4.0532   +0.3569  +1.39e-15  |     59  34.5%   +0.0000
  ADA       160  +10.6447  +10.5102   -0.1345  -1.47e-15  |     26  14.0%   +0.0000
  AVAX       78   +0.1411   +0.8733   +0.7322  -8.67e-17  |      4   4.9%   +0.0000
  ---------------------------------------------------------------------------------
  ALL       583  +23.1741  +25.3833   +2.2092  -3.47e-16  |    119  17.0%   +0.0000
```

### gate P>0.70 (post-fix)

- D8a: alerts emitted 243 → 235 (monitor suppressed 8 config-fail signals)
- D8b: with all alerts emitted, executor entered 235 / skipped 8 (`config_thresholds_not_met`)
```
  asset  shared     exec%    atlas%    basis%     resid%  | zeroed  zero% execOnly%
  ---------------------------------------------------------------------------------
  BTC        36   +3.1378   +3.5080   +0.3703  -6.51e-16  |      0   0.0%   +0.0000
  ETH        44   +1.6488   +2.0672   +0.4184  +6.07e-16  |      0   0.0%   +0.0000
  SOL         8   +0.2197   +0.3516   +0.1320  +0.00e+00  |      4  33.3%   +0.0000
  XRP        42   +2.5572   +3.1786   +0.6215  -3.47e-16  |      4   8.7%   +0.0000
  ADA        75   +8.9799   +9.4541   +0.4742  +2.17e-15  |      0   0.0%   +0.0000
  AVAX       30   +0.2500   +0.9218   +0.6718  +8.67e-17  |      0   0.0%   +0.0000
  ---------------------------------------------------------------------------------
  ALL       235  +16.7933  +19.4814   +2.6881  +3.47e-15  |      8   3.3%   +0.0000
```

## D8b executor smoke test

- negative case (ann_rate 3.0 < min 5): decision=`skip`, reason=`config_thresholds_not_met (ann_rate=3.0 vs min 5; pct_pos=0.950 vs min 0.50)` → PASS
- positive case (thresholds met): decision=`enter` → PASS

## Column definitions

Split into the **shared trade set** (atlas AND executor both booked) and the **atlas-zeroed leak**:

- **shared** — # gate-passing days atlas actually traded.
- **exec%** — executor cumulative P&L (funding − TC) over the shared set.
- **atlas%** — atlas Exp 13 net return over the shared set (funding + basis − TC); reproduces phase4_model_stats exactly.
- **basis%** — basis drag, summed INDEPENDENTLY from regenerated spot/perp price returns (NOT atlas−exec), so resid% is a genuine check.
- **resid%** — atlas% − exec% − basis% over the shared set; ~1e-14 confirms the funding+basis−TC decomposition is exact.
- **zeroed / zero%** — gate-passing days atlas zeroed via its hard config gate. Pre-fix the executor books these; post-fix execOnly→0.
- **execOnly%** — executor P&L booked on atlas-zeroed days.
