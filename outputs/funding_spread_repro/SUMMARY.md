# Cycle 50 D2c — Cross-venue funding spread (Binance × Bybit) — DISCONFIRMED

**Date:** 2026-06-01
**Strategy:** `engines/funding_spread_strategy.py` (Cycle 50 D2a)
**Brief:** `claude/handoffs/BRIEF_funding_spread_thin_slice.md`
**Retro:** `claude/retros/RETRO_funding_spread_thin_slice.md`

---

## Headline: 4-cell matrix — every cell negative-Sharpe

| Cell | Sharpe | Cum | AnnRet | MaxDD | WinDays | Avg M/d | Pos/Total |
|---|---:|---:|---:|---:|---:|---:|:--:|
| **taker / P>0.50** | **−11.33** | −2.15% | −1.21% | −2.15% | 6/448 (1.3%) | 1.08 | 0/5 |
| **taker / P>0.70** | **−6.14** | −0.48% | −0.27% | −0.48% | 1/448 | 0.23 | 0/3 |
| **maker / P>0.50** | **−8.40** | −0.85% | −0.48% | −0.85% | 16/448 (3.6%) | 1.08 | 0/5 |
| **maker / P>0.70** | **−4.91** | −0.19% | −0.10% | −0.19% | 4/448 | 0.23 | 0/3 |

**Brief threshold:** "Below +1.0 across all 4 cells = disconfirm + redirect."
**Verdict:** all 4 cells are well below 0.0 (let alone +1.0). **Disconfirmed.**

## Per-model Sharpes (every cell, every model)

| Asset | taker/p050 | taker/p070 | maker/p050 | maker/p070 |
|---|---:|---:|---:|---:|
| BTC  | −17.55 (86d) | −28.99 (11d) | −14.15 (86d) | −18.51 (11d) |
| ETH  | −14.04 (198d) | −14.50 (57d) | −7.39 (198d) | **−8.88** (57d, "best") |
| SOL  | −18.39 (165d) | −24.15 (34d) | −11.82 (165d) | −18.02 (34d) |
| XRP  | −20.38 (19d) | (no trade) | −11.43 (19d) | (no trade) |
| ADA  | (no trade) | (no trade) | (no trade) | (no trade) |
| AVAX | −10.997 (12d) | (no trade) | −9.95 (12d) | (no trade) |

The **best single per-model in the best cell** is `ETH_SPREAD / maker / P>0.70` at **Sharpe −8.88 over 57 trading days**. No model, no cell, no asset crosses 0.0 OOS. ADA didn't trade in any cell.

## Phase3 base-rate + AUC (taker; maker similar shape)

| Model | AUC | Base rate (% of (day, config) labeled profitable) |
|---|---:|---:|
| BTC_SPREAD  | 0.9877 | 3.5% |
| ETH_SPREAD  | 0.9843 | 3.6% |
| SOL_SPREAD  | 0.9878 | 3.2% |
| XRP_SPREAD  | 0.9990 | 1.4% |
| ADA_SPREAD  | 0.9970 | 1.0% |
| AVAX_SPREAD | 0.9968 | 1.3% |

Base rates 1.0–3.6% are **drastically lower than Exp 13's 31–45%**. Very few (day, config) combos in training were actually profitable after TC. The RF achieves high AUC by memorizing those rare events on the training set, but they don't generalize OOS — the calibration is broken:

```
Taker/P>0.50 calibration (sample, n=482, base 1.7%):
   [0.50,0.55)    n=94    actual WR=0.0%   lift= -1.7%
   [0.55,0.60)    n=99    actual WR=2.0%   lift= +0.4%
   [0.60,0.65)    n=114   actual WR=3.5%   lift= +1.8%
   [0.65,0.70)    n=71    actual WR=1.4%   lift= -0.3%
   [0.70,0.80)    n=81    actual WR=1.2%   lift= -0.4%
   [0.80,1.01)    n=23    actual WR=0.0%   lift= -1.7%
```

The high-P bins (>=0.70) have actual WR at or below the 1.7% base rate. **The RF is not discriminating OOS at all** — it's pattern-matching training-set noise.

## Why this strategy failed (analysis)

1. **TC is too high relative to spread magnitudes.** Cycle 49 RECON found median absolute spreads of 3-5% annualized across the 6 assets. The 7-day taker break-even is ~10.4% ann; even 30-day maker is ~0.85% — but the spreads are too noisy/short-lived to capture at 30-day holds.

2. **4-leg execution compounds slippage.** Long+short on each of 2 venues means 4 fills round trip. At 4 bps taker/leg = 16 bps RT. Even at maker rates, 7 bps RT eats most of the expected spread.

3. **Spread "stability" is illusory.** The pct-positive feature (sign consistency over 30d) doesn't predict the next 7-30 days' sign accurately enough to make selective entries profitable.

4. **Cross-venue basis P&L was assumed zero.** The thin-slice simulation ignored cross-venue perp basis drift; in reality this adds noise. If the strategy were positive even with this optimistic simplification, the realistic version would be more negative.

5. **Direction-switching cost.** When spread flips sign, the strategy would need to reverse the position — but our hold-then-exit model doesn't capture this. In a real deployment, spread-flip mid-hold creates whipsaw risk we haven't modeled.

## Maker vs taker — was it materially better?

Maker / taker ratios:
- P>0.50: −8.40 / −11.33 = **0.74** (26% improvement)
- P>0.70: −4.91 / −6.14 = **0.80** (20% improvement)

**Not the >2× threshold** the brief flagged as "execution path matters more than universe expansion." Maker is *consistently* better but only marginally. Both regimes are deeply negative.

## D2d (validation 2023→2024) — SKIPPED

Brief: "if D2c clears the disconfirm threshold" — D2c does not clear; validation reproduction skipped per the brief's gate.

## Disconfirmation finding

The cross-venue funding-spread carry hypothesis (atlas Exp 13 revival #1) **is disconfirmed** at this strategy formulation on the Binance × Bybit universe. The economic intuition (different funding rates → arbitrage) is real, but:
- The magnitudes are too small relative to 4-leg TC
- The persistence is too short to amortize TC over multi-day holds
- The RF cannot extract a discriminating signal OOS even at the most permissive cell (maker, P>0.50, 5 models trading 198/448 days)

**What this does NOT disconfirm:**
- The cross-venue spread *signal* is real and persistent (Cycle 49 RECON confirmed median 3-5% ann spreads). The trade-ability via this strategy formulation is the negative finding.
- A different formulation (e.g. continuous rebalancing, statistical arb with strict mean-reversion, or paired with Exp 13's single-venue carry as a hedge) might be viable.

## Recommended redirect

Per brief: "Below +1.0 across all 4 cells = disconfirm + redirect to 44d/44b/44c."

Cycle 51 candidates (Code's lean: **44c**):

| Cycle 51 candidate | Why now |
|---|---|
| **44c — Real-money executor integration** | Engine 7 single-venue carry (Exp 13) is verified, deployed, monitored, and alerting. The natural next step is closing the loop with execution. The cross-venue infrastructure built in Cycle 50 (schema, 2-venue collector) is not wasted — it provides Bybit as a backup execution venue for single-venue carry. |
| 44d — Bear-regime accumulation analysis | Passive; ~30 days of funding_signals data needed. No Code action required (data accumulates automatically). Defer to whenever 30d has elapsed since Cycle 41 deployment (~2 weeks from now). |
| 44b — LSTM v2 architecture test | Atlas Exp 13 revival hypothesis #4. "Likelihood low" per atlas. With funding-carry verified and execution within reach, LSTM v2 is research-flavored exploration. Defer until execution is in place. |

**My lean: 44c.** Productionizes the verified strategy. Cycle 50's cross-venue plumbing becomes future-useful (Bybit as execution backup; potential for cross-venue *execution arb* on Exp 13 fills, which is mechanically different from this spread carry).

## Output tree

```
outputs/funding_spread_repro/
├── SUMMARY.md             ← this file
├── taker/
│   ├── phase2.log, phase3.log
│   ├── cpo/
│   │   ├── phase2_returns.parquet      (105,120 rows = 6 models × 365 days × 48 configs)
│   │   ├── phase2_features.parquet     (2184 rows)
│   │   ├── phase3_models.joblib        (6 RF models, AUC 0.98-0.999, base 1-3.6%)
│   │   ├── phase3_importances.json
│   │   ├── phase4_portfolio.parquet    (last gate's portfolio; copies in subdirs)
│   │   └── phase4_model_stats.json
│   ├── p050/  phase4.log + phase4_portfolio.parquet + phase4_model_stats.json
│   └── p070/  ditto
└── maker/
    ├── phase2.log, phase3.log
    ├── cpo/  same shape as taker/cpo/ but with tc_bps=1.75
    ├── p050/  ditto
    └── p070/  ditto
```

Total tree ~6 MB. Funding_rates table grew by ~22,000 rows (the Bybit backfill); funding_signals untouched (this cycle did not touch the monitor).
