# Cycle 57 — The Real (Basis-Inclusive) Risk-Adjusted Carry: MEASUREMENT + VERDICT

> **MEASUREMENT cycle. No build, no capital, no keys, no order code.** Read-only
> public price collection into the isolated research DB. P4 (monitor venue-scoping
> fix) committed separately (`298660f`). Scripts: `cycle57_price_collect.py`,
> `cycle57_hl_price_backfill.py`, `cycle57_basis_sharpe.py`.

## P0 — AUDIT VERDICT: the +4.65 is same-venue & funding-dominated; the executor is basis-blind

Exact atlas P&L (`engines/funding_rate_strategy.py:181-188`):
```
spot_ret = (spot_exit - spot_entry)/spot_entry      # long spot
perp_ret = (perp_entry - perp_exit)/perp_entry      # short perp
gross    = clip(spot_ret + perp_ret + total_funding, -0.99, 5.0)   # basis IS included
net      = clip(gross - 2*tc_pct, -0.99, 5.0)                      # 8 bps RT
```
- **The atlas +4.65 is NOT basis-blind** — `spot_ret + perp_ret` is the realized basis
  change over the hold. But it is **same-venue (binance spot vs binance perp)**, where
  basis is structurally tiny, and **endpoint-only** (entry vs exit price). The code comment
  says it outright: *"basis_change (≈0 for perfect hedge)"*. So +4.65 is a **same-venue,
  funding-dominated** number.
- **The executor / paper path IS basis-blind** (`compute_exit`: `funding − TC` only) — the
  +$29.96 Cycle-54c tie-out. That's the number that *looked* basis-blind, and it is.
- **Empirical confirmation of the spine:** in every row below, `pnl_vol ≈ basis_vol` —
  funding contributes ~zero volatility (it's a smooth drip). **The basis term is the entire
  Sharpe denominator.** A basis-blind P&L has a near-zero denominator → an inflated Sharpe.

## P1 — Data (isolated research DB)
HL perp (ccxt + raw candle) is **hard-capped at ~5000 candles ≈ 7 months** (2025-11-26→now);
the public candle endpoint serves no older history (funding history goes to 2023, price does
not). **Binance spot+perp**: full 2024-01-01→now (21,693 hrs). **Coinbase spot**: full 2.5y
(21,683 hrs) — **chosen as the hedge leg** (Kraken's OHLC caps at ~720 candles; Coinbase
paginates). So: **HL cross-venue basis measurable over the recent ~7 months** (covers the
last-90-day crux + H1-2026); **full 2024-26 regime breakdown via the CEX anchor.**

## P2 — Basis (the load-bearing risk), annualized hourly Δbasis vol
Cross-venue (HL-perp vs Coinbase-spot) basis vol, last-90d: **BTC 1.7%, ETH 2.0%, SOL 2.3%,
LTC 4.7%, DOGE 5.0%, LINK 5.8%, XRP 2.8%, UNI 8.7%, AVAX 9.4%, ADA 6.8%.** Same-venue CEX
(binance/binance) basis vol is lower (BTC ~1.0-1.2%) — the cross-venue hedge adds ~0.5-1%
of basis vol on majors (tolerable) but is large on ADA/AVAX/UNI (6-9%, poor hedge). Basis
*drift* is ~0 everywhere (mean-reverting) — basis hurts via **volatility, not drift**.
(HL-perp vs binance-spot gives near-identical results → the measurement is robust to which
major-CEX spot is used.)

## P3 — THE REAL SHARPE

### Hyperliquid cross-venue (HL perp + Coinbase spot) — GROSS (no trading cost)
| Asset | funding_ann (90d) | basis_vol (90d) | **Sharpe full(7mo)** | **Sharpe last-90d** |
|---|---|---|---|---|
| ETH | 4.2% | 2.0% | 2.38 | **1.88** |
| LINK | 9.6% | 5.8% | 1.50 | **1.54** |
| BTC | 2.9% | 1.7% | 2.04 | **1.50** |
| LTC | 7.3% | 4.7% | 0.91 | **1.34** |
| DOGE | 6.4% | 5.0% | 0.75 | **1.16** |
| UNI | 8.0% | 8.7% | 0.71 | **0.76** |
| AVAX | 5.6% | 9.4% | 0.32 | **0.58** |
| XRP | −0.7% | 2.8% | −0.09 | **−0.37** |
| ADA | −3.2% | 6.8% | −0.51 | **−0.48** |
| SOL | −0.9% | 2.3% | −1.07 | **−0.61** |

Gross of cost, the carry still has **positive risk-adjusted quality on HL** for the liquid
majors + LINK/LTC/DOGE even in the last 90 days (Sharpe ~1.1–1.9) — **far below the idealized
+4.65, but not dead.** SOL/XRP/ADA are negative (funding went negative/near-zero).

### CEX same-venue anchor (binance/binance) — the +4.65 reproduction + the regime decay
| Asset | 2024 Sharpe (fund%) | 2025 Sharpe (fund%) | **last-90d Sharpe (fund%)** |
|---|---|---|---|
| BTC | 8.60 (11.9%) | 5.26 (5.1%) | **0.58 (0.6%)** |
| ETH | 6.69 (13.0%) | 4.42 (4.9%) | **0.52 (0.6%)** |
| SOL | 6.47 (13.6%) | 0.18 (0.4%) | **−1.54 (−2.5%)** |
| XRP | 5.11 (14.2%) | 2.60 (3.5%) | **−0.86 (−1.1%)** |
| ADA | 4.49 (14.0%) | 1.98 (4.1%) | **0.28 (0.9%)** |
| AVAX | 3.06 (10.5%) | 0.03 (0.1%) | **0.18 (0.9%)** |

- **The same-venue Sharpe reproduces the +4.65 regime** (BTC 2024-25 ≈ 5-8.6), confirming P0:
  +4.65 was a same-venue, basis≈0, funding-dominated, **2024-flavored** number.
- **The BIS decay, measured on our own data:** CEX carry Sharpe collapsed from ~5-8 (2024)
  to ~0-0.6 (last 90d), **negative for SOL/XRP**. Funding fell from ~12-14% (2024) to ~0.6%
  (BTC, last 90d). **The regime is no longer paying on the CEX majors.**

### Three-way ladder (BTC, ETH) — attributing the degradation (common ~7-month window)
| Asset | leg | funding_ann | basis_vol_ann | **gross Sharpe** |
|---|---|---|---|---|
| BTC | A bin-perp / bin-spot (same-venue) | 1.6% | 0.9% | 1.71 |
| BTC | B bin-perp / cb-spot (cross-CEX, no DeFi) | 1.6% | 1.3% | 1.19 |
| BTC | **C HL-perp / cb-spot (real target)** | 3.7% | 1.7% | **2.04** |
| ETH | A bin-perp / bin-spot (same-venue) | 1.0% | 0.9% | 1.00 |
| ETH | B bin-perp / cb-spot (cross-CEX, no DeFi) | 1.0% | 1.6% | 0.55 |
| ETH | **C HL-perp / cb-spot (real target)** | 5.0% | 2.0% | **2.38** |

- **A→B (cross-venue basis per se):** Sharpe DROPS (BTC 1.71→1.19, ETH 1.00→0.55) — moving the
  spot hedge from the same exchange to a different CEX lifts basis vol (0.9%→1.3-1.6%) with no
  funding benefit. The pure cost of a cross-venue hedge, even between two deep liquid CEXs.
- **B→C (HL perp + HL's ~2-5× funding):** Sharpe RISES (BTC 1.19→2.04, ETH 0.55→2.38). **C beats
  B for both assets** → **HL's funding premium more than pays for its higher idiosyncratic basis
  vol. The HL funding edge is REAL EDGE, not compensation for worse basis/tail risk.** (C even
  beats same-venue A in this window, because HL funding currently dwarfs binance funding.)
- Caveat: in this funding-starved window even the idealized same-venue A is only ~1.0-1.7 (vs
  8.6/6.7 in 2024) — the regime decay shows at every rung.

### Cost model (per leg, per venue) + breakeven hold — the maker/taker swing
The hybrid pays fees on BOTH legs: **HL perp RT** (~9 bps taker / ~3 bps maker) **+ Coinbase
spot RT**, and the **spot leg is the swing factor** (Coinbase maker ~0, taker ~40-60 bps/side):
| All-in RT scenario | composition | ~bps |
|---|---|---|
| OPTIMISTIC | HL taker perp + Coinbase **maker** spot | **~9** |
| MIXED | HL taker + light Coinbase taker / slippage | ~35 |
| REALISTIC CEILING | HL taker + Coinbase **taker** spot (~50/side) | **~100+** |

Funding accrues hourly; cost is paid once per round-trip and amortizes over the hold, so the
**Cycle-56 minHold framing governs**: `minHold = cost_RT × 365 / funding_ann` (HL/CB, last-90d funding):
| Asset | funding (90d) | minHold @9bps | @35bps | @100bps |
|---|---|---|---|---|
| LINK | 9.6% | 3.4d | 13.3d | 38.1d |
| UNI | 8.0% | 4.1d | 15.9d | 45.5d |
| LTC | 7.3% | 4.5d | 17.6d | 50.2d |
| DOGE | 6.4% | 5.2d | 20.1d | 57.4d |
| AVAX | 5.6% | 5.9d | 22.8d | 65.1d |
| ETH | 4.2% | 7.8d | 30.3d | 86.6d |
| BTC | 2.9% | 11.5d | 44.8d | 128.1d |
| SOL / XRP / ADA | ≤ 0 | never | never | never |

- **Maker-spot (~9 bps) is the only viable regime:** high-funding alts (LINK/UNI/LTC/DOGE) clear
  at **3-5-day holds**; majors need 8-12 days; SOL/XRP/ADA never (funding ≤ 0).
- **Taker-spot (~100 bps) kills it:** LINK needs a 38-day hold, BTC ~128 days — untenable for a
  carry that must exit when funding flips. **Maker execution on the spot leg is mandatory.**
- (Net-Sharpe cross-check at a fixed 7-day roll: at 9 bps only LINK/LTC/DOGE stay positive
  (≈+0.2 to +0.7); BTC/ETH go negative; at 20 bps+ everything is negative.)

## VERDICT — does the edge survive risk-adjusted on Hyperliquid, especially the last 90 days?

**Marginal — mostly NO, with a narrow exception.**
- **Gross of cost**, the HL carry retains positive risk-adjusted quality for the liquid
  majors + LINK/LTC/DOGE even now (Sharpe ~1.1–1.9) — and the three-way ladder confirms this is
  **real venue edge**: C (HL) > B (cross-CEX) for both BTC and ETH, so HL's funding premium more
  than pays for its higher basis vol (it isn't basis-risk compensation). The venue is the right
  one; the problem is the regime and the cost, not Hyperliquid.
- **Net of realistic DeFi cost**, the edge is **thin and fragile**: in the last 90 days only
  **LINK / LTC / DOGE** clear a 9-bps round-trip (and only with **long holds + maker execution**);
  BTC/ETH/SOL/XRP/ADA are net-negative. Nothing survives at 20 bps.
- **The regime is the dominant fact:** carry funding has decayed ~10× since 2024 (CEX BTC
  12%→0.6%), exactly the BIS "negative in 2025" finding — measured here on our own data, and
  *worse* in the last 90 days. The +4.65 was a 2024-regime, same-venue, funding-dominated number;
  the real, current, cross-venue, cost-and-basis-inclusive Sharpe is **~0.2–0.7 net for the few
  names that clear, negative for the rest.**

**Recommendation (Jeff's call):** the broad carry does **not** robustly survive risk-adjusted
today → **keep it parked**, with at most a **tiny, LINK/LTC/DOGE-only, long-hold, maker-execution
pilot** if you want a live toehold. The venue unblock did not resurrect a paying regime. **Redirect
the primary real-money effort to the directional candidate (the parked LSTM predictor on a
Canada-legal spot venue)** — the carry's edge is currently too regime-dependent and cost-fragile
to anchor a deployment. Re-run this measurement when funding regimes turn (the machinery now exists).

## Limitations
- HL price history capped at ~7 months (public API) → HL cross-venue Sharpe is recent-window only;
  the full 2024-26 regime arc comes from the CEX anchor (which is the relevant decay story anyway).
- Static always-on position (no selective gating) — the real strategy gates on favorable funding,
  which would lift the mean (skip negative-funding hours); so gross Sharpe here is a conservative
  floor for the selective strategy, but the cost/regime conclusions hold.
- USDC (HL) vs USD (Coinbase) vs USDT (binance) quote differences add a small stable-coin basis
  (minor vs the measured basis vol). Cost sensitivity assumes ≈maker execution.
