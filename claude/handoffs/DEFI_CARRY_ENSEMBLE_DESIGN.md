# DeFi Funding-Carry — ENSEMBLE + COST/RISK DESIGN (Cycle 56, V4 + V5)

> **DESIGN ONLY. No real capital, no order code, no wallet keys, no live calls.**
> The ensemble is BUILT later, gated on the V3 cost gate clearing AND Jeff's go.
> This is the written design the brief asks for, plus the per-(venue,asset)
> cost/risk model. Funding-data collection (Hyperliquid + dYdX, isolated
> research DB) is the only thing this cycle touches; see `CYCLE56_DEFI_VENUE_RECON.md`.

## Boundaries (stated back)
No real capital · no order code · no exchange/wallet credentials or private keys
(ever — Jeff's hands) · output is data + validation + written design · nothing
touches funds. Live deployment gated on a (venue, asset) that clears DeFi costs
AND a funding regime that pays.

---

## V4 — ENSEMBLE DESIGN (gated on V3)

The carry now has **two selection axes** instead of one. The design must beat a
naive baseline out-of-sample or it is overfitting.

### V4.0 The benchmark to beat (define first, OOS)
**Baseline B0:** *equal-weight the 6 majors; short on the single venue with the
best average funding-net-of-fees; hedge long spot on the cheapest reachable
spot venue; fixed notional per asset.* No switching, no asset selection, no
sizing cleverness. **If the multi-asset/multi-venue ensemble cannot beat B0 on
a held-out window, it is fitting noise and we ship B0 (or nothing).** Every
added degree of freedom (venue switching, asset weights, exotic inclusion) must
pay for its complexity against B0 OOS.

### V4.1 Axis 1 — short-leg venue selection (per asset, per window)
Be short where **funding net of fees** is highest for that asset, with
**switching hysteresis** so we don't churn:

```
switch venue  iff  (edge_other − edge_current) × expected_remaining_hold
                   >  round_trip_switch_cost + unhedged_window_cost
```
- `edge = funding_net_of_fees` (annualized, per venue, per asset).
- `round_trip_switch_cost` = close short on venue A (taker + slippage) + open
  short on venue B (taker + slippage) + any spot-leg rebalance.
- `unhedged_window_cost` = expected adverse move during the seconds/minutes the
  position is mid-switch and not delta-neutral (the R5/Cycle-55 two-leg-atomicity
  hazard, now also across venues). Default: **do not switch within a hold unless
  the edge gap is large and the remaining hold is long** — most value is in
  *initial* venue choice, not mid-hold churn.
- Venue eligibility per window: venue live (not halted/dead), funding mechanism
  is clean symmetric (see V5), book depth ≥ N× intended size.

### V4.2 Axis 2 — asset selection + sizing
Allocate across the **carry-bearing tradeable universe** (majors + vetted
exotics that pass the V1 hedge-intersection test), sized by **funding-net-of-cost
AND liquidity/risk**:

```
weight_i  ∝  carry_score_i  ×  liquidity_factor_i  ×  hedge_quality_i
size_i    =  clamp( weight_i × allocation,  exchange_min_i,  depth_cap_i )
```
- `carry_score_i` = mean funding-net-of-fees × positive-share (persistence),
  per asset per chosen venue (the V1 score).
- **Exotics sized DOWN** — thin books, high vol, short history (the SOL
  precedent: SOL basis −8 bps/hold mean, 14 bps std → noisiest of the majors;
  exotics are worse on every axis). Hard cap exotic exposure as a fraction of
  the book.
- `depth_cap_i` = a small fraction of the perp book's depth-within-X% (the JELLY
  lesson: size to book depth, not conviction — ADL/oracle risk concentrates in
  thin markets).
- Respect each venue's per-market **min-notional / lot size** (skip if intended
  size < min).

### V4.3 The long (hedge) leg
- **Default: Kraken or Coinbase Advanced spot** (Canadian-reachable, withdrawable).
  Coinbase Advanced (maker 0 bps, broad listings, withdrawable) is often cheaper
  than Kraken's post-July-2026 40/80 bps entry tier for a small account.
- **Switch to on-chain spot only if** cross-venue basis analysis (V5) shows an
  on-chain spot (e.g. Uniswap/Jupiter, or Hyperliquid's own HyperCore spot for
  the spot∩perp intersection) tracks the chosen perp venue's mark materially
  better than the CEX spot — net of gas, bridging, and self-custody burden.
- **The $30k/12-mo Canadian net-purchase cap** (on non-BTC/ETH) is a hard
  constraint on the long leg: every altcoin spot hedge consumes the cap. This
  bounds how many non-major carries can run concurrently for an Ontario-resident
  retail account. (BC/AB/SK/MB/QC residents are cap-exempt — Jeff's province
  matters here.)

### V4.4 Degrees-of-freedom / overfitting budget (surface explicitly)
The search space is **N venues × M assets × switching policy × per-asset sizing**,
and **exotic history is short** (DeFi ~2023-on; HIP-3 / new listings months).
Controls:
- **Hold out** the most recent window (e.g. last 6 months) for the B0 comparison;
  never tune on it.
- **Cap M** — start with majors + a *handful* of vetted exotics, not the whole
  230-perp universe (most of which fails V1 hedge-availability anyway).
- **Penalize short history** — an exotic with < ~12 months of funding gets a
  shrinkage haircut on its carry_score (toward zero) and a hard size cap.
- **Prefer the simplest policy that beats B0.** Venue switching and exotic
  inclusion are only justified by an OOS improvement that exceeds a complexity
  margin. Report the DoF count alongside any reported Sharpe.
- **Regime flag (load-bearing):** the BIS/SSRN "Crypto Carry" paper reports the
  carry's Sharpe **turned negative in 2025**, and Engine 7 itself has sat out for
  a month (0 alerts). The ensemble must be evaluated across the negative-carry
  regime, not just the bull-funding years — or it will overstate.

---

## V5 — COST / RISK MODEL per (venue, asset)

### V5.1 Fees + gas (short-leg venues)

| Venue | Perp taker / maker | Gas / chain | Funding cadence | Round-trip perp cost* |
|---|---|---|---|---|
| **Hyperliquid** | 4.5 / 1.5 bps (base; →2.4/0 at volume) | **gasless** (own L1; ~1 USDC one-time activation) | hourly | ~9 bps taker, ~3 bps maker-entry |
| **dYdX v4** | 5.0 / 1.0 bps (→2.5 / −1.1 rebate) | **gasless orders**, USDC fees | hourly | ~10 bps taker, ~6 bps maker-entry |
| **GMX v2** | 4–6 bps (imbalance-dependent), ×2 open/close | Arbitrum gas (~sub-$0.10/tx, keeper-refunded) | (funding + 1-dir borrow) | ~8–12 bps + gas |
| Drift | 2.5–3.5 / −0.25 bps | Solana (~$0) | hourly | cheapest — **but HALTED** |
| Aevo | 8 / 5 bps | L2 (~$0) + bridge gas | hourly | ~16 bps — thin |

*Perp leg only. The **spot hedge leg** adds its own round trip (see V5.2).

### V5.2 Spot-hedge (long-leg) cost — the other half of the round trip
| Hedge venue | Spot taker / maker | Notes |
|---|---|---|
| **Coinbase Advanced** | ~10 / 0 bps (volume-dependent) | broadest withdrawable long-tail; best default |
| **Kraken Pro** | 80 / 40 bps entry (→16/6 at $1M+) | **hiked July 9 2026**; expensive for small accounts unless maker |
| NDAX | 20 / 20 bps flat | cheap flat, narrow listings |
| On-chain (Uniswap/Jupiter) | pool fee 5–100 bps + gas + slippage | only path for perp-only exotics; worst on thin tokens |

**All-in round trip (both legs, entry+exit)** is the real cost to beat with
funding. Best case (HL maker short + Coinbase maker spot) ≈ **low-teens bps**;
typical (HL taker + Kraken taker) ≈ **30–90 bps**; on-chain exotic hedge ≈
**100+ bps with slippage**. This is the V3 break-even input.

### V5.3 Cross-venue basis / tracking risk
- **Hybrid (Kraken/Coinbase spot + DeFi perp):** the two legs price off
  **different oracles/books** on different venues/chains → **inter-oracle
  divergence**. The perp settles to its venue's index; the CEX spot moves on its
  own book. Basis can widen exactly when you need to unwind. Worse for exotics.
- **All-DeFi (on-chain spot + DeFi perp, ideally same chain/venue):** tighter
  basis (often same oracle), and Hyperliquid's HyperCore spot can hedge the
  spot∩perp intersection on-venue — but only ~the majors have both, and it means
  **full self-custody** of the spot too.
- **Delta drift:** funding is paid on notional; as price moves, the perp short's
  delta and the spot long's delta diverge → periodic **rebalancing** cost
  (more fills, more fees), heavier on volatile exotics.

### V5.4 DeFi-native risks (absent on a CEX) — and why exotics carry MORE of each
| Risk | Mechanism | Realized example | Exotic amplification |
|---|---|---|---|
| **Smart-contract exploit** | protocol bug drains funds | **Drift ~$285M (Apr 2026, DPRK)** → venue HALTED; GMX **v1** ~$42M (Jul 2025) | newer/forked venues less battle-tested |
| **Oracle / mark manipulation** | thin spot moved to mispricing perp | **Hyperliquid JELLY (Mar 2025)** — oracle override + forced settle + delist | thin exotics are the attack surface |
| **ADL (auto-deleverage)** | a winning short is force-closed at bankruptcy price in a squeeze | Hyperliquid first cross-margin ADL Nov 2025 | concentrated in thin/high-funding markets — i.e. the carry targets |
| **Chain halt** | can't adjust/exit while spot hedge moves | **dYdX halted twice** (Apr 2024, Oct 2025); Solana 7 outages | small chains halt more |
| **Bridge risk** | validator-signed bridge forge/freeze | HL bridge: <25 validators, ~200s dispute | — (venue-level) |
| **Stablecoin-collateral depeg** | USDC/USDT collateral loses peg → margin shock | — | — (venue-level) |
| **Self-custody key security** | leaked private key drains everything | — | more wallets/chains for exotics = bigger key surface |
| **Venue death** | venue winds down, strands positions | **Vertex** wound down (Jul–Aug 2025) | low-liquidity venues die |
| **Regulatory geofence** | front-end blocks the user | **Hyperliquid ToS restricts Ontario**; dYdX left Canada Apr 2023 | — (user-level) |

**Net:** every DeFi-native risk is **worse on exotics** — exactly the assets with
the highest funding. The carry edge and the tail risk are positively correlated.
This is why exotics are sized DOWN, capped to book depth, history-shrunk, and why
**venue intervention (oracle override / ADL / halt) — not ordinary funding-flip —
is the dominant risk** for a delta-neutral DeFi carry: each can decouple the perp
leg from the off-venue spot hedge and turn a "neutral" position directional at an
administratively-set price.

### V5.5 The hedge-intersection gate (restate — the crux)
An asset is **real, tradeable carry only if ALL three hold**:
1. **persistent positive funding** net of fees (V1 carry score), AND
2. a **liquid, legally-holdable spot hedge** reachable by a Canadian (Kraken/
   Coinbase/NDAX, or a deep-enough on-chain pool), AND
3. it **clears all-in round-trip costs** at realistic size (V3 gate).
The highest-funding tail (HIP-3 / meme perps) systematically **fails (2)** and
often **(3)** after slippage. The majors **pass (2)/(3)** but have the
**thinnest funding** (most competed). The viable universe lives in between:
**liquid mid-caps that are both DeFi-perp-listed and Canadian-spot-listed.**

---

**END — DESIGN ONLY. No funds, no orders, no keys. Build gated on V3 + Jeff's go.**
