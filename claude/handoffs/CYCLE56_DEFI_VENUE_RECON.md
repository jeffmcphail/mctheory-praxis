# Cycle 56 — DeFi Funding-Carry Venue + Asset RECON / Data Inventory (V0–V5)

> **RECON + DATA INVENTORY. No real capital, no order code, no wallet keys, no
> live calls.** The only thing this cycle touches is READ-ONLY public funding
> collection (Hyperliquid + dYdX) into an **isolated research DB**. Ensemble +
> cost/risk design is in `DEFI_CARRY_ENSEMBLE_DESIGN.md` (V4/V5). HARD PAUSE after.

## Context (stated back)
Engine 7's **+4.65 Sharpe is CEX-validated** (Binance+Bybit funding) on **6 majors**
(BTC/ETH/SOL/XRP/ADA/AVAX). Canada bans retail **CEX perps/leverage**, so the
**funding-earning short-perp leg** is reachable only via **DeFi perp DEXes**; the
**long spot leg earns no funding** (pure hedge). `funding_rates` already has a
`venue` column (Cycle 50). We are now ALSO **expanding beyond the 6 majors**
because high/persistent funding lives in lower-cap / hotter tokens.

**The crux (brief):** an asset is real carry only if it has **persistent funding
AND a reachable liquid spot hedge AND clears costs** — the V1 hedge-intersection
and the V3 cost gate.

---

## TWO findings that reframe the cycle (read first)

1. **The live monitor's funding read is venue-UNSCOPED (latent bug + hard
   blocker).** `scripts/funding_monitor.py:169` does
   `SELECT timestamp, funding_rate FROM funding_rates WHERE asset=? AND
   timestamp BETWEEN ? AND ?` — **no `venue` filter** — then
   `drop_duplicates("timestamp")`. Consequences:
   - Since Cycle 50 added bybit, the monitor has been reading a **venue-ambiguous
     mix** of binance+bybit (deduped arbitrarily by timestamp). Small effect
     today (CEX majors track closely) but **wrong**.
   - Writing **hourly** DeFi funding (HL/dYdX = 24 rows/day) into production
     `funding_rates` would **corrupt the live signal** (mixed with 3 eight-hourly
     CEX rows/day → garbled annualization → bad `funding_signals`/alerts).
   - The **executor is correctly scoped** (`compute_exit` filters
     `venue='binance'`), so the asymmetry is monitor-only.
   - **→ Prerequisite for ANY production DeFi-venue integration: scope the
     monitor read to a venue (and decide which venue the live signal uses).**
   - **→ This cycle therefore writes DeFi funding to an ISOLATED research DB
     (`data/defi_funding_research.db`), never production `funding_rates`.**
     (Structural isolation — the Cycle 54 replay lesson, applied again.)

2. **The carry edge has been REGIME-NEGATIVE recently.** The BIS/SSRN "Crypto
   Carry" paper (Schmeling–Schrimpf–Todorov) — which formally defines this exact
   cash-and-carry trade and documents historical >10%/yr — reports the carry's
   **Sharpe turned negative in 2025.** Engine 7 itself has **sat out for a month
   (0 alerts)**. So even if a venue/asset *clears costs mechanically*, the
   regime must also *pay*. The V3 gate is necessary, not sufficient; deployment
   is double-gated (clears costs AND regime pays).

---

## V0 — VENUE INVENTORY

All four research streams were web-sourced (cited in the agent reports) and the
load-bearing claims adversarially re-verified. Dollar figures are point-in-time
(June 2026) and volatile.

| Venue | Funding model | Clean carry analog? | Funding history | ccxt? | Taker/maker | Chain / gas | Liquidity (OI) | Status |
|---|---|---|---|---|---|---|---|---|
| **Hyperliquid** ★ | CEX-style **symmetric**, **hourly**; premium+interest, interest ~+11.6% APR baseline *to short* | **YES** | public `/info` `fundingHistory`, no-auth, hourly, ~2023+, 500/req | **yes** (`hyperliquid`; `fetchFundingRateHistory`✓) | 4.5 / 1.5 bps (→2.4/0) | own L1, **gasless** | **~$9.4B (dominant, >70% share)** | live |
| **dYdX v4** ★ | CEX-style **symmetric**, **hourly**; interest 0% on cross majors | **YES** (verify rate is per-hour, ≈8× smaller than CEX 8h) | public Indexer `historicalFunding`, no-auth, hourly, genesis 2023-10-26, 1000/req | **yes** (`dydx` = v4; `fetchFundingRateHistory`✓) | 5.0 / 1.0 bps (→2.5/−1.1) | Cosmos appchain, **gasless orders** | **~$57M (small, declining)**; BTC+ETH ~80% | live, **halted ×2** |
| **GMX v2** | **funding (symmetric) + separate 1-dir borrow fee** (only larger-OI side pays) | **partial** (short = lighter side earns funding & ~0 borrow in longs-crowded regime) | Subsquid GraphQL; REST candles ~6mo | **no** (gmx-python-sdk + Subsquid) | 4–6 bps ×2 | Arbitrum/Avax, gas | ~$57M (only live+liquid of the stretch four) | live |
| **Drift** | CEX-style symmetric, hourly (mechanically best) | yes — **but offline** | data-api `/fundingRates`, ~30d window | no (`driftpy`) | 2.5–3.5 / −0.25 bps | Solana | $0 (was ~$554M) | **HALTED** (Apr 2026 ~$285M hack) |
| **Vertex** | (was) symmetric | n/a | dead | removed | (was 0/2 bps) | — | ~$50 | **DEAD** (→ Kraken Ink "Nado") |
| **Aevo** | symmetric, hourly, no funding fee | yes — but thin | `/funding-history`, ns times, 50/req | no (`aevo-sdk`) | 8 / 5 bps | own OP-stack L2 | ~$6–10M; ~70% SOL | live, declining |

**Hedge (long-spot) venues, Canadian-reachable:** **Coinbase Advanced** (maker
0 bps, broadest withdrawable long-tail — best default), **Kraken Pro** (spot
**hiked to 80/40 bps entry** Jul 9 2026; all 6 majors + HYPE/PENDLE/PEPE/AERO),
**NDAX** (flat 20 bps, narrow), **Shakepay** (BTC/ETH only). Spread-model apps
(Newton/Coinsquare/Wealthsimple/Bitbuy-Express) are 50–200 bps — too dear for
active hedging; several mark coins non-withdrawable (a hedge needs the real coin).

**V0 verdict:** **Hyperliquid is the clear priority short-leg venue** — clean
symmetric hourly funding, free deep public history, in ccxt, gasless, dominant
liquidity. **dYdX v4** is the #2 clean-funding comparison but small/halting.
**GMX v2** is a viable third (favorable funding/borrow for a short in the normal
regime) but no-ccxt + ~6mo history. Drift (offline), Vertex (dead), Aevo (thin)
are out for now.

---

## V1 — ASSET-UNIVERSE DISCOVERY + HEDGE INTERSECTION

**Listed perp universe (live-pulled from Hyperliquid `meta` this cycle): 230
perps.** Beyond the majors, the universe includes ATOM, MATIC, DYDX, BNB, APE,
OP, LTC, ARB, DOGE, INJ, SUI, kPEPE, CRV, LDO, LINK, STX, RNDR, CFX, FTM, GMX,
SNX, BCH, APT, AAVE, COMP, MKR, WLD, kSHIB, UNI, … plus HL-native (HYPE, PURR,
FARTCOIN) and HIP-3 equity/RWA perps (AAPL/TSLA/NVDA…). dYdX v4 has 140–220+
markets but permissionless listing requires ~6 CEX listings — so **dYdX's tail
tends to already be CEX-listed (better hedge intersection, narrower tail)**;
Hyperliquid's HIP-3 tail is broader but largely **un-hedgeable in Canada**.

**The hedge-intersection (the crux):**

| Class | Funding | Canadian spot hedge | Tradeable carry? |
|---|---|---|---|
| **Majors** (BTC/ETH/SOL/XRP/ADA/AVAX) | thin (most competed) | easy (Kraken/Coinbase); BTC/ETH cap-exempt | **yes — but low edge** |
| **Liquid mid-caps** (LINK/AAVE/UNI/DOGE/LTC/ARB/APT/SUI/OP/CRV/MKR/INJ/ATOM/BNB) | moderate | mostly yes (Kraken/Coinbase); consumes $30k cap | **yes — the sweet spot** |
| **Hot/meme/HIP-3 tail** (FARTCOIN/PURR/HIP-3, many memes) | **highest** | **mostly none** (on-chain only → 10–50% slippage) | **mostly NO** (fails hedge gate) |
| **HL-native** (HYPE) | elevated | yes (Kraken/Coinbase list HYPE) + HyperCore spot | yes (but cap-consuming) |

**The structural tension:** funding is richest exactly where the hedge is
hardest (newly-listed, low-cap, on-chain-only) and thinnest where the hedge is
easy (majors). **The viable expanded universe is the liquid mid-caps that are
BOTH DeFi-perp-listed AND Canadian-spot-listed.**

### V1 carry-score ranking (from collected data)
*Hyperliquid, mean annualized funding (cadence-correct ×8760) over 2024-01-01→now,
full window (21,689 hrs) unless flagged. minHold = days to clear a 35 bps all-in
round-trip. Hedge = Canadian-reachable liquid spot (verify per-asset live).*

| Asset | mean_ann | pos_share | minHold@35bps | Canadian spot hedge | Notes |
|---|---|---|---|---|---|
| kPEPE | 24.0% | 0.86 | 5.3d | via **PEPE** spot (Kraken/CB), ×1000 scaling | meme tail |
| HYPE | 22.5% | 0.94 | 5.7d | Kraken/CB + HyperCore | *post-launch ~Dec-2024, shorter history* |
| DOGE | 19.3% | 0.85 | 6.6d | **easy** (all CEX) | **sweet spot** |
| LINK | 17.6% | 0.95 | 7.3d | **easy** | **sweet spot** |
| UNI | 15.5% | 0.92 | 8.3d | **easy** | **sweet spot** |
| LTC | 15.4% | 0.92 | 8.3d | **easy** | **sweet spot** |
| BTC | 14.7% | 0.89 | 8.7d | easy (cap-exempt) | major |
| ETH | 13.2% | 0.86 | 9.7d | easy (cap-exempt) | major |
| SOL | 12.8% | 0.78 | 10.0d | easy | major |
| SUI | 11.9% | 0.78 | 10.8d | CB/Kraken | |
| ARB | 11.7% | 0.80 | 11.0d | CB/Kraken | |
| XRP | 10.8% | 0.81 | 11.9d | easy | major |
| INJ | 10.7% | 0.79 | 11.9d | CB/Kraken | |
| AVAX | 10.6% | 0.77 | 12.1d | easy | major |
| CRV | 10.5% | 0.87 | 12.1d | CB/Kraken | |
| kSHIB | 9.7% | 0.70 | 13.2d | via **SHIB** spot, ×1000 | *partial (missing recent ~2mo)* |
| OP | 9.4% | 0.77 | 13.5d | CB/Kraken | |
| WLD | 9.1% | 0.79 | 14.1d | CB/Kraken (verify) | |
| ADA | 7.9% | 0.73 | 16.2d | easy | major — weakest major |
| BNB | 5.3% | 0.74 | 24.0d | **hard** (Binance left CA; CB no BNB) | low edge + hedge-poor |
| APT | 4.8% | 0.68 | 26.8d | CB | low edge |
| ATOM | 0.6% | 0.67 | 230d | CB | **no usable carry** |
| ~~MKR~~ | ~~32.7%~~ | 0.95 | 3.9d | CB/Kraken | ⚠️ **2024-only partial (5500h) — window-biased HIGH; re-pull before trusting** |
| ~~AAVE~~ | ~~30.3%~~ | 0.97 | 4.2d | easy | ⚠️ **2024-only partial (8500h) — window-biased HIGH; re-pull** |

**Read:** the **liquid, Canadian-hedgeable sweet spot** is DOGE/LINK/UNI/LTC
(+ majors BTC/ETH/SOL) — mid-teens-to-20% funding, clearing 35 bps at 7–10-day
holds, all easily spot-hedged. The meme tail (kPEPE/kSHIB) scores higher but
hedges only via ×1000-scaled PEPE/SHIB spot (workable but volatile). MKR/AAVE
look best but are window-biased 2024 partials — **do not headline them.**

---

## V2 — DATA SCOPE + COLLECTION

### What already exists (production `crypto_data.db`)
| Data | Coverage | Gap for this cycle |
|---|---|---|
| `funding_rates` | **binance + bybit × 6 majors**, 3806 rows each (8-hourly), 2023-01-01 → 2026-06-22 | no DeFi venues; majors only |
| `ohlcv_4h` (price) | **BTC + ETH only**, 2023-11-12+ | **no price for SOL/XRP/ADA/AVAX or any non-major** |
| `ohlcv_1m` | BTC + ETH only, 2025-10-31+ | same |
| `ohlcv_daily` | BTC + ETH only, 954 rows | same |
| `market_data` | BTC/ETH/SOL/XRP/ADA, recent partial | sparse |

**Key V2 gap:** for basis / hedge-quality (V5.3) we need **spot AND perp price
history per (venue, asset)**, and the DB has price for **BTC+ETH only**. Funding
alone (which we can get) is insufficient to judge hedge quality on everything
else — **price-history collection per candidate asset is a required follow-up**
(out of scope to collect fully this cycle).

### What this cycle collects (isolated research DB)
`data/defi_funding_research.db` (schema mirrors `funding_rates`, plus a
`cadence` column = `'hourly'` for DeFi). Via `scripts/cycle56_defi_funding_collect.py`
(standalone, idempotent, public APIs, no creds):
- **Hyperliquid** hourly funding, 2024-01-01 → now, for **24 candidates**
  (6 majors + 14 liquid mid-caps + 4 tail: HYPE/kPEPE/kSHIB/WLD).
- **dYdX v4** hourly funding, 2024-01-01 → now, for the **6 majors**.
- Uniform 2.5y window for clean cross-venue/asset comparison; covers atlas Exp
  13 OOS (2025+) and the 2025 negative-carry regime.

**History-limit note:** DeFi venues are ~2023-on; HIP-3 / new listings are
months — **weaker statistical power on non-majors** (handled in V4.4 via
short-history shrinkage + size caps).

### Collection results
`data/defi_funding_research.db` — **Hyperliquid: 24 assets, 480,824 hourly rows**,
2024-01-01 → 2026-06-22. Most assets full (21,689 hrs). Partials/flags:
- **AAVE 8,500** and **MKR 5,500** — rate-limit/timeout stopped them early; both
  cover **2024 only** (forward-paged from window start) → **window-biased to the
  stronger 2024 regime; re-pull for the full window before relying.**
- **HYPE 13,544** — genuine (token launched ~late 2024), not a failure.
- **kSHIB 19,500** — missing only the recent ~2 months (minor).
- **dYdX v4: 0 rows — every Indexer request returned HTTP 403** (anti-bot on the
  plain urllib client). Retryable with a browser `User-Agent`/ccxt in the build;
  **not blocking** — Hyperliquid is the priority venue and is sufficient for the gate.
- **CEX comparison** uses production `funding_rates` (binance/bybit, 8-hourly,
  n≈2711 over the same window) — untouched, read-only.

---

## V3 — VALIDATION (the gate)

### V3a — ANCHOR (majors first): validate the machinery on knowns
Re-derive the funding-carry return on the **6 majors** from **Hyperliquid /
dYdX** funding (this cycle's data) with each venue's **real fee model**, and sanity-
check it against the known CEX behavior (the +4.65 was Binance funding − 8 bps).
Because the executor/replay already reproduce atlas Exp 13 to the digit on CEX
funding, the anchor here is: *does DeFi-venue funding on the majors produce a
sane, same-sign carry of comparable magnitude, net of DeFi fees?* (Full
atlas-replay re-run on DeFi funding is the reviewed build; this cycle does the
first-pass empirical carry on the collected series.)

### V3b — EXPANDED: top carry-bearing tradeable non-majors
Extend the first-pass carry to the vetted mid-caps, **per venue, with real fees
+ a liquidity/slippage haircut** (exotic books are thin), and **redo the R6
break-even per (asset, venue)**. DeFi fees (HL ~9 bps round-trip taker, ~3 bps
maker; gasless) are **far below the ~30 bps CEX-real estimate** from Cycle 55 —
which **could flip the edge favorable** on assets the CEX cost model zeroed.
The question is whether high exotic funding **survives slippage**, or is illusory.

### THE GATE
> **Does carry clear all-in round-trip costs on the best (venue, asset) combos?**
> - **If nothing clears → STOP.** The edge isn't reachable for a Canadian.
> - **If yes → proceed to V4** (ensemble design), still double-gated on a paying
>   regime (finding #2) and Jeff's go.

### V3 verdict — **GATE CLEARS on the cost axis** (Sharpe axis unresolved)

**V3a anchor — Hyperliquid funding ≈ 2× CEX on the same majors, same window:**

| asset | binance | bybit | **hyperliquid** |
|---|---|---|---|
| BTC | 7.09% | 7.16% | **14.67%** |
| ETH | 7.32% | 7.28% | **13.19%** |
| SOL | 4.97% | 6.18% | **12.82%** |
| XRP | 6.80% | 7.72% | **10.76%** |
| ADA | 7.36% | 7.89% | **7.88%** |
| AVAX | 4.17% | 5.44% | **10.60%** |

The DeFi pipeline reproduces a **sane, same-sign, comparable-magnitude** carry on
the majors — and in fact **higher**: Hyperliquid pays ~2× the CEX funding
(HL's structural +11.6%-APR-to-short interest component + degen-long premium).
Combined with **lower DeFi fees** (gasless, ~9 bps taker round-trip vs the
~30 bps CEX-real estimate from Cycle 55 R6), **the edge is MORE favorable on
Hyperliquid than on the CEX baseline, not less.** This flips the naive worry.

**The gate:** at a realistic **35 bps all-in round-trip**, the **entire liquid
Canadian-hedgeable universe clears at ≤14-day holds** (DOGE 6.6d, LINK 7.3d,
BTC 8.7d, ETH 9.7d, SOL 10.0d, …), even on the *always-short* mean; selective
entry (pos_share 0.77–0.95) makes it easier. **GATE CLEARS. → proceed to V4.**

**The honest caveat (do not over-read):** this measures **gross funding net of
fees** — the raw short-leg carry. It is **NOT the risk-adjusted Sharpe**, which
the BIS/SSRN paper says **went negative in 2025**. Positive funding-mean and
negative Sharpe are **not contradictory**: Sharpe can go negative from
**basis/execution/tail** volatility (the delta-neutral leg's price-divergence,
ADL/squeeze events, slippage) even while mean funding stays positive. Measuring
the Sharpe needs **per-asset spot+perp price history** (the V2 gap — only BTC+ETH
exist; not collected this cycle). **So: the cost gate passes; the Sharpe/regime
question is unresolved pending price-history collection.** Deployment stays
double-gated (clears costs ✓ AND regime pays — TBD).

---

## V4 / V5 — see `DEFI_CARRY_ENSEMBLE_DESIGN.md`
Two-axis ensemble (venue selection w/ hysteresis + asset selection sized by
carry-net-of-cost & liquidity, exotics down), the B0 OOS benchmark, the DoF/
overfitting budget, and the full per-(venue,asset) cost/risk model (fees+gas,
basis/tracking, slippage, and the DeFi-native risk table: exploit/oracle/ADL/
halt/bridge/depeg/key/venue-death — all worse on exotics).

---

## Analysis results (carry scores + V3 gate)

Reproducible via `scripts/cycle56_carry_analysis.py` (reads the research DB +
production CEX funding; pure read + arithmetic). Key outputs are the V1 ranking
table (above), the V3a anchor table, and the V3 gate verdict (above).

**Headline numbers:**
- **Hyperliquid funding ≈ 2× CEX** on majors over 2024-01-01→now (cadence-correct).
- **Cost gate clears**: full liquid Canadian-hedgeable universe clears 35 bps at
  ≤14-day holds; the sweet spot (DOGE/LINK/UNI/LTC + majors) clears at 7–10 days.
- **pos_share 0.77–0.97** on the good names → funding is favorable most of the
  time even across the weak-2025 window (encouraging, not cherry-picked).
- **Caveats**: MKR/AAVE window-biased (2024-only partials); dYdX 403 (no data);
  gross-funding-only (no Sharpe — needs price history).

**Method notes for the build:**
- Annualization is **cadence-correct** (hourly ×8760, 8h ×1095) — the single
  biggest unit-trap when mixing DeFi (hourly) and CEX (8h) funding. dYdX's
  Indexer `rate` is also per-hour (≈8× smaller than CEX 8h quotes) — same trap.
- `minHold_days = cost_frac × 365 / mean_ann_frac` — the gate metric; low-funding
  majors need long holds, high-funding names clear short holds.

---

## Surfaced risks register

| # | Risk | Severity | Disposition |
|---|---|---|---|
| 1 | Monitor funding read is **venue-unscoped** → can't add any DeFi venue to production `funding_rates` without corrupting the live signal | **High (blocker)** | Scope the monitor read to a venue + pick the live-signal venue. Prerequisite to integration. DeFi data isolated in research DB meanwhile. |
| 2 | Carry **regime-negative in 2025** (BIS/SSRN); Engine 7 sitting out | High (economic) | Double-gate deployment: clears costs AND regime pays. Evaluate ensemble across the negative regime. |
| 3 | **Hedge-intersection failure** on the high-funding tail (no Canadian spot) | High | V1 gate: only assets with a reachable liquid spot hedge are tradeable. |
| 4 | DeFi venue **intervention** (oracle override/ADL/halt) decouples perp from off-venue spot hedge | High (tail) | Size to book depth; majors/liquid mid-caps; venue-risk caps; see V5.4. |
| 5 | **No price history** beyond BTC+ETH → can't yet judge basis/hedge quality on most assets | Medium | Collect per-asset spot+perp price history (follow-up). |
| 6 | **$30k/12-mo Canadian cap** on non-BTC/ETH spot bounds concurrent altcoin carries | Medium | Constraint in V4.3 sizing; depends on Jeff's province. |
| 7 | dYdX rate is **per-hour** (≈8× smaller than CEX 8h quotes) → unit-mismatch risk | Medium | Normalize per-venue cadence in all annualization (done in analysis). |

---

## HARD PAUSE — decisions for Jeff
1. **Which venues** to pursue (lean: Hyperliquid primary; dYdX as comparison; GMX v2 maybe).
2. **Which discovered assets** beyond the majors (the V1 carry-vs-hedgeability shortlist).
3. **Does the edge clear** (V3 gate verdict below) — and given the negative-carry
   regime, is it worth building the ensemble now or waiting for funding to pay?
4. The monitor venue-scoping fix (risk #1) — do it as the first integration step.
No build past this cycle's isolated funding collection until review.
