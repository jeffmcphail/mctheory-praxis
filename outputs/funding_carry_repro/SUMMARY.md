# Cycle 40 — Engine 7 Funding Carry full-universe paper reproduction

**Date:** 2026-05-26
**Brief:** `claude/handoffs/BRIEF_engine7_repro.md`
**Retro:** `claude/retros/RETRO_engine7_repro.md`
**Backfill script:** `scripts/backfill_funding_history.py`
**Verdict:** **CONFIRMED + VERIFIED** — atlas Exp 13 reproduces to ≤0.4% Δ across every comparable metric for both primary and validation windows.

---

## Strategy invocation

```
python scripts/run_cpo.py --strategy funding_rate --feature-mode funding \
    --assets BTC,ETH,SOL,XRP,ADA,AVAX \
    --training-start <2024|2023>-01-01 --training-end <2024|2023>-12-31 \
    --cache-dir data/funding_cache \
    --output-dir outputs/funding_carry_repro[/validation] \
    --tc-bps 4.0 \
    --prob-threshold <0.50|0.70> \
    phase4 --start <2025|2024>-01-01 --end <2026-03-26|2024-12-31>
```

- 6-asset universe (BTC, ETH, SOL, XRP, ADA, AVAX); BNB excluded per atlas
- 36 configs × 6 models = **216 configs**
- TC 4.0 bps one-way (Cycle 39 finding: default `--tc-bps 2.0` had to be overridden to match atlas Exp 13 spec)
- equal-weight allocation, max-leverage 2.0, max-weight 0.05 (gate-limited so not binding)

## Wall-clock

| Stage | Cold? | Time |
|---|---|---:|
| D1 backfill SOL/XRP/ADA/AVAX (2023-01-01 .. 2026-05-26) | cold | 24 s |
| D1 backfill BTC/ETH (2023-01-01 .. 2025-04-29 fill) | cold | 9 s |
| D2 phase2 (216 configs, 2024 training) | CCXT cold | ~5 min |
| D2 phase3 (6 RF models) | — | 45 s |
| D2 phase4 P>0.50 (incl. CCXT 2025–2026 OOS fetch) | cold OOS | ~2 min |
| D2 phase4 P>0.70 | cache warm | ~1 min |
| D2 validation phase2 (2023 training) | cold | ~5 min |
| D2 validation phase3 + phase4 (2024 test) | cache warm | ~2 min |
| **Total D1+D2** | | **~16 min** vs brief's 1.5–2.5 h estimate |

---

## Primary OOS reproduction — atlas headline table

Training 2024-01-01..2024-12-31 → Test 2025-01-01..2026-03-26 (448 trading days).

| Metric | Atlas | Cycle 40 | Δ | Within ±15%? |
|---|---:|---:|---:|:---:|
| Sharpe (P>0.50) | **+4.65** | **+4.6525** | +0.05% | ✅ |
| Cum return (P>0.50) | +1.27% | +1.27% | exact | ✅ |
| Max DD (P>0.50) | −0.15% | −0.15% | exact | ✅ |
| Win days (P>0.50) | 29.9% | 29.9% (134/448) | exact | ✅ |
| Avg models/day (P>0.50) | 1.6 | 1.6 | exact | ✅ |
| Sharpe (P>0.70) | **+4.45** | **+4.4492** | −0.18% | ✅ |
| Cum return (P>0.70) | +0.97% | +0.97% | exact | ✅ |
| Max DD (P>0.70) | −0.03% | −0.03% | exact | ✅ |
| Win days (P>0.70) | 15.8% | 15.8% (71/448) | exact | ✅ |
| Avg models/day (P>0.70) | 0.5 | 0.5 | exact | ✅ |

## Per-model OOS Sharpes (P>0.50)

| Model | Atlas Sharpe | Cycle 40 Sharpe | Δ | Days (atlas / repro) | CumRet (atlas / repro) |
|---|---:|---:|---:|---:|---:|
| ADA_FUNDING | +7.21 | +7.214 | +0.06% | 186 / 186 | +10.5% / +10.51% |
| ETH_FUNDING | +6.58 | +6.582 | +0.03% | 126 / 126 | +5.0% / +5.04% |
| BTC_FUNDING | +5.86 | +5.863 | +0.05% | 77 / 77 | +3.4% / +3.36% |
| XRP_FUNDING | +5.27 | +5.274 | +0.08% | 171 / 171 | +4.1% / +4.05% |
| SOL_FUNDING | +3.69 | +3.687 | −0.08% | 60 / 60 | +1.6% / +1.55% |
| AVAX_FUNDING | +1.98 | +1.981 | +0.05% | 82 / 82 | +0.9% / +0.87% |
| **Mean** | | **+5.100** | | | |

6/6 positive. All per-model active-day counts match atlas exactly.

## Phase 3 model quality (vs atlas Phase 3 RF Quality table)

| Asset | Atlas AUC | Cycle 40 AUC | Atlas base | Cycle 40 base |
|---|---:|---:|---:|---:|
| BTC_FUNDING | 0.987 | **0.9869** | 39.0% | **39.0%** |
| ETH_FUNDING | 0.986 | **0.9860** | 40.6% | **40.6%** |
| SOL_FUNDING | 0.978 | **0.9782** | 38.2% | **38.2%** |
| XRP_FUNDING | 0.979 | **0.9789** | 44.9% | **44.9%** |
| ADA_FUNDING | 0.982 | **0.9817** | 42.3% | **42.3%** |
| AVAX_FUNDING | 0.982 | **0.9819** | 31.1% | **31.1%** |

AUC reproduces to ≤0.001; base rate reproduces exactly to one decimal.

## Calibration reproduction (atlas's "most monotonic ever seen")

| P bin | Atlas n | Atlas WR | Cycle 40 n | Cycle 40 WR | Match |
|---|---:|---:|---:|---:|:---:|
| [0.50, 0.55) | 120 | 21.7% | **120** | **21.7%** | exact |
| [0.55, 0.60) | 124 | 36.3% | **124** | **36.3%** | exact |
| [0.60, 0.65) | 112 | 42.0% | **112** | **42.0%** | exact |
| [0.65, 0.70) | 103 | 45.6% | **103** | **45.6%** | exact |
| [0.70, 0.80) | 156 | 66.7% | **156** | **66.7%** | exact |
| [0.80, 1.01) | 87 | 90.8% | **87** | **90.8%** | exact |

Every bin reproduces exactly. Monotonicity confirmed.

## Validation reproduction (train 2023 → test 2024 at P>0.70)

| Metric | Atlas | Cycle 40 | Δ | Within ±15%? |
|---|---:|---:|---:|:---:|
| Sharpe validation | **+10.78** | **+10.7726** | −0.07% | ✅ |
| Cum return | +16.73% | +16.67% | −0.36% | ✅ |
| Max DD | −0.03% | −0.03% | exact | ✅ |
| Win days | 70.3% | 70.3% (256/364) | exact | ✅ |
| Avg models/day | 3.7 | 3.7 | exact | ✅ |

Per-asset (atlas flagship per-model figures):

| Asset | Atlas cum / Sharpe | Cycle 40 cum / Sharpe |
|---|---|---|
| BTC | +52% / +16.9 | +52.34% / +16.88 |
| ETH | +111% / +16.9 | +111.41% / +16.86 |
| AVAX | +63% / +16.5 | +62.89% / +16.46 |

**Model count semantic.** Atlas reports "Models positive: 7/7"; this 6-asset deployment-universe reproduction is 6/6 (BNB excluded per the entry's "BNB excluded — degenerate base rate" rule applied uniformly to primary + validation). The "all-positive" claim survives; the count differs because of universe size.

---

## Output tree

```
outputs/funding_carry_repro/    (~11 MB, 23 files)
├── SUMMARY.md                                 ← this file
├── phase2.log, phase3.log
├── cpo/
│   ├── phase2_returns.parquet                 (188 KB, 78,840 rows)
│   ├── phase2_features_funding.parquet        (117 KB, 2,184 rows)
│   ├── phase3_models_funding.joblib           (5.45 MB, 6 RF models — correctly named)
│   ├── phase3_importances.json
│   ├── phase4_portfolio.parquet               ← from last phase4 run (P>0.70)
│   └── phase4_model_stats.json
├── p050/  phase4.log + phase4_portfolio.parquet + phase4_model_stats.json
├── p070/  phase4.log + phase4_portfolio.parquet + phase4_model_stats.json
└── validation/
    ├── phase2.log, phase3.log, phase4.log
    └── cpo/  phase2_returns.parquet, phase2_features_funding.parquet,
              phase3_models_funding.joblib (2023-trained),
              phase3_importances.json,
              phase4_portfolio.parquet, phase4_model_stats.json
```

## Disk-hygiene note (carried from Cycle 39)

`outputs/exp10_revival/cpo/phase3_models_funding.joblib` (37.4 MB) **deleted** during D2 setup — Cycle 39 RECON established it was a misnamed Exp 10 `universal_ta` artifact (64 TA models, zero `*_FUNDING` keys). The correctly-named funding-carry model now lives only at `outputs/funding_carry_repro/cpo/phase3_models_funding.joblib`.

## funding_rates DB state after D1

| Asset | Rows | Range |
|---|---:|---|
| BTC | 3 726 | 2023-01-01 .. 2026-05-26T16:00 |
| ETH | 3 726 | 2023-01-01 .. 2026-05-26T16:00 |
| SOL | 3 724 | 2023-01-01 .. 2026-05-26T00:00 |
| XRP | 3 724 | 2023-01-01 .. 2026-05-26T00:00 |
| ADA | 3 724 | 2023-01-01 .. 2026-05-26T00:00 |
| AVAX | 3 724 | 2023-01-01 .. 2026-05-26T00:00 |

Zero gap-days inside each asset's covered window. BTC/ETH end one funding-event later than the 4 backfilled assets because the live `PraxisFundingCollector` ran 16:00 UTC today. Total: 22,348 funding events across the 6-asset universe.
