# REAL MONEY ORDER PATH — DESIGN (Cycle 55 RECON, R5 + R6)

> **STATUS: DESIGN ONLY. NO CODE. NO CREDENTIALS. NO LIVE CALLS.**
> Nothing in this document was built. It is a written design + risk surface for
> review. The build is a **later cycle**, gated on R0–R4 done **+** D8-live
> (the config gate firing correctly on a real production alert in paper) **+**
> Jeff's explicit go. Real orders remain **impossible** until that path is
> built (later), armed (Jeff), and fired (Jeff, **per-trade**).

## Boundaries (restated, non-negotiable)

- No order-placement code written this cycle. Design only.
- No exchange credentials touched — ever, by Claude. That is Jeff's hands.
- No live-trade authorization by Claude, autonomously, ever.
- Human-in-the-loop is **per-trade**, not a session-level opt-in that then
  fires autonomously. A real order is impossible without an explicit per-trade
  human act.
- Fail-safe everywhere: ambiguous/unset mode → paper / no-trade. Real money is
  opt-in, never default, never inferred.
- Paper-first holds until the real path is built, armed, and fired.

---

## 0. The premise correction (READ FIRST)

The brief's R5 asks me to "confirm there's no order code anywhere today."

**I cannot confirm that. The opposite is true.** A complete, working
real-money order path already exists in the repo and has since before the
Cycle 51–54 paper program:

| File | What it is |
|---|---|
| `engines/carry_executor.py` | A delta-neutral carry executor (long spot + short perp) that calls `exchange.create_market_order(...)` / `create_limit_order(...)` in its non-paper branch (lines 267–284), loading `BINANCE_API_KEY` / `BINANCE_API_SECRET` from `.env` (lines 168–187). Gated only by `paper=True` **default**. |
| `scripts/run_carry.py` | A CLI that wires `carry_executor` to `funding_monitor` signals. `--live` flips `CarryExecutor(paper=not args.live)`. Subcommands: `enter`, `exit`, `auto`. `auto --live` **auto-enters positions** for every asset above gate, every 8h, for a configured duration. |

So the real-money path is not hypothetical and not the verified executor — it
is a **parallel, older, naive implementation of the same strategy**, reachable
today with one command:

```
python scripts/run_carry.py auto --live          # real Binance orders
python scripts/run_carry.py enter --asset BTC --notional 500 --live
```

It is guarded by a typed-`YES` console prompt (`run_carry.py:92,123,175`) — a
session-level confirm, **not** the per-trade GUI confirm this cycle is meant to
design — and by `CarryExecutor`'s own simpler caps (`max_position_usd`,
`max_total_exposure_usd`, already-have-position). The order primitive itself
(`_execute_order`'s non-paper branch) is unconditionally present and would fire
real orders if `paper=False` and the keys in `.env` are populated. **Whether
those keys are populated is Jeff's domain — I did not and will not read
`.env`.** `.env.example` ships `BINANCE_API_KEY=REPLACE_ME` placeholders.

**Why this matters for the whole cycle:** building a careful, per-trade-confirm,
mode-gated, atomic real path (R0–R5) while this path sits next to it unguarded
is a contradiction. The clean, single-owner real path we design is undermined
if a second, naive `--live` path remains a command away. **Neutralizing /
quarantining `carry_executor.py` + `run_carry.py` is a prerequisite to the
real-money build, not an afterthought.** See the Surfaced Risks Register
(bottom) — this is Risk #1.

`carry_executor.py` is also a **perfect worked example of the R5 hard problem**:
`enter_carry` (lines 357–369) fires spot-buy (leg 1) then perp-short (leg 2)
**sequentially, with no check that leg 1 filled, no fill reconciliation between
legs, no unwind if leg 2 fails, and market orders by default.** It is exactly
the two-leg atomicity hazard this design exists to prevent.

---

## 1. CCXT usage survey

| Usage | Files | Auth? | Order calls? |
|---|---|---|---|
| **Read-only market data** (collectors): `fetch_ohlcv`, `fetch_funding_rate_history`, `fetch_ticker` | `engines/crypto_data_collector.py`, `scripts/funding_monitor.py`, `gui/mcb_studio/backend/data_feed.py`, various strategy modules | No key (`ccxt.binance({"enableRateLimit": True})`) | None |
| **The verified paper executor** | `engines/funding_executor.py` | — | **None — `import ccxt` is explicitly forbidden** (safety-belt grep over the file before every commit; docstring lines 17–26) |
| **The legacy real path** ⚠️ | `engines/carry_executor.py` (+ `scripts/run_carry.py`) | Reads `BINANCE_API_KEY/SECRET` | **Yes — `create_market_order` / `create_limit_order`** |
| Polymarket order code (different venue, different domain) | `engines/market_maker.py`, root `first_trade*.py`, `batch_*.py`, `bump_*.py` | Polymarket CLOB client | `client.create_order(...)` — unrelated to crypto carry |

**Net:** the verified Engine-7 executor has no order path; the collectors are
read-only; but a Binance order path for *this exact strategy* exists in
`carry_executor.py`.

---

## 2. The order path — design

Target: **delta-neutral funding carry** — long spot + short perp at equal
notional, on Binance, via CCXT. Collect funding (short perp receives when
funding > 0). Exit = sell spot + buy-to-close perp.

### 2.1 Symbols & venues
- Spot: `{ASSET}/USDT` on `ccxt.binance` (spot).
- Perp: `{ASSET}/USDT:USDT` (USDⓂ linear perp) on `ccxt.binance({options:{defaultType:'future'}})`.
- Single venue (Binance) per the Cycle 50 structural commit (`EXECUTION_VENUE = "binance"`).

### 2.2 The HARD PROBLEM — two-leg atomicity

There is **no atomic two-leg primitive** on Binance (spot and futures are
separate wallets/matching engines; no cross-product OCO). Between leg 1 and
leg 2 the book is, for a moment, **directional**:

- Buy spot **then** short perp → net **LONG** during the gap (loses if price drops).
- Short perp **then** buy spot → net **SHORT** during the gap (loses if price rises).

If the second leg **fails** and we hold the first, we own an **unhedged
directional position the strategy never intended.** This is the failure mode to
engineer against. Design:

**A. Pre-trade gate (before any order):**
1. Both markets tradeable + not in maintenance; load markets, respect
   `amount_to_precision` / `price_to_precision` / min-notional filters.
2. Snapshot both order books; estimate slippage at intended size; **abort if
   estimated slippage on either leg exceeds a configured ceiling** (e.g. `max_slippage_bps`).
3. Verify margin/collateral for the perp short and cash for the spot buy
   (see §2.3 — this replaces the brief's "borrow availability", which is
   mis-framed for a *perp* short).
4. Verify the position is delta-neutral by construction: `spot_qty == perp_qty`
   at the same asset.

**B. Execution sequence (fill-confirmed, never fire-and-forget):**
1. Place **leg 1** with a **client order id** (idempotency key, §2.6). Use
   marketable / IOC-style sizing to minimize the open window.
2. **Confirm leg 1's actual fill** (poll order status until filled/closed or
   timeout). Read the **actual filled quantity** `q1` — do **not** assume the
   intended quantity.
3. Place **leg 2 sized to `q1`** (the real fill), not the intended size. Same
   idempotency-key discipline.
4. **Confirm leg 2's fill** `q2`.
5. **Reconcile:** assert `|q1 − q2| ≤ tol`. If a residual remains, either
   top-up the short leg to match or unwind the residual (configurable;
   default = unwind residual, fail-safe to flat).

**C. Abort / unwind ladder (the safety net):**
- Leg 1 fails entirely → no position; log + alert; done (clean).
- Leg 1 fills, **leg 2 fails** → **immediately market-unwind leg 1** with a
  `reduceOnly`-style close, return to flat, alert LOUD. Never hold the naked leg.
- Leg 1 **partial**, leg 2 unavailable → unwind the partial.
- Unwind itself fails (e.g. exchange outage mid-sequence) → **page the human**
  (this is the one place the machine cannot self-heal; it must escalate, not
  retry blindly). The pending/holding state (§ R3 / R0) records exactly what is
  open so the human sees the true position.

**D. Which leg first?** Lean: place the **perp short first** (the hedge /
funding leg, usually deepest liquidity on majors), confirm, then buy spot to
match. Rationale: the perp is the leg whose fill quantity should *lead* (it's
the funding-bearing leg and the one with position limits); spot is cash and
trivially sized to match. The window is symmetric in risk either way; minimize
it with size + marketable orders, don't try to eliminate it (you can't).

### 2.3 "Borrow availability" — corrected

The brief lists "borrow availability for the short perp." **A perp short does
not borrow the asset** — perps are derivatives; you post **USDT margin** and
open a short. The real constraints are:
- Sufficient **USDT collateral** in the futures wallet (initial + maintenance margin).
- **Liquidation risk**: if spot and perp diverge (basis blowout), the perp
  short can approach liquidation even though the *combined* position is
  delta-neutral, because the two legs sit in **separate wallets** with separate
  margin. This is the genuine tail risk of cross-wallet delta-neutral carry and
  must be sized for (low leverage, margin buffer, basis-divergence monitor).
- (Securities-borrow / locate *would* matter if we shorted **spot** on margin —
  we do not. The long leg is **cash-funded spot**, the short leg is a **perp**.)

### 2.4 Fee model
- Binance **spot taker ≈ 10 bps** (7.5 bps with BNB), **perp taker ≈ 5 bps**
  (4 bps with BNB); maker tiers lower; VIP tiers lower still.
- A full round trip is **4 taker fills** (entry: buy spot + short perp; exit:
  sell spot + close perp).
- The verified paper model charges **8 bps round-trip total** (`TC_PCT_ONE_WAY
  = 0.0004`, applied once at entry + once at exit, on one leg's notional). See
  §R6 — this is a **material undercount**.

### 2.5 Entry / exit symmetry
Exit mirrors entry with the same atomicity machinery: sell spot + buy-to-close
perp, fill-confirmed, reconciled, with the same unwind ladder. Exit timing
stays the verified atlas-Exp-13 window semantics
(`signal_ts + hold_days*86_400_000 ms`).

### 2.6 Idempotency (ties to R2)
Every order carries a deterministic **client order id** (e.g.
`{session_id}:{asset}:{signal_ts}:{leg}:{entry|exit}`). On a timeout/retry, the
**exchange dedups on the client id** — a resubmission after a dropped response
does not double-fill. This is the real-exchange analogue of the paper PK dedup
(which does **not** protect a real order — R2). It is necessary but **not
sufficient**: it stops *one executor* from double-submitting on retry; it does
**not** stop *two executors* from each submitting once (different ids). That
needs the single-owner lock (R2).

### 2.7 State machine (per trade)
```
PROPOSED ──human-confirm (R3)──▶ ARMED ──pre-trade gate──▶ LEG1_PLACED
   │ (timeout / decline)              │ (gate fail)             │
   ▼                                  ▼                         ▼
 EXPIRED(no-trade)                ABORTED(no-trade)       LEG1_FILLED
                                                               │
                                                LEG2_PLACED ───┤
                                                     │         │(leg2 fail)
                                                     ▼         ▼
                                                  OPEN     UNWINDING ──▶ FLAT
                                                                 │(unwind fail)
                                                                 ▼
                                                            ESCALATE(human)
```
Every transition is persisted (survives a backend restart — R3c) and every
terminal state except `OPEN` is **flat / no naked exposure**.

---

## 3. R6 — paper-vs-real gap (written explicitly)

The verified **+4.65 Sharpe is a PAPER number.** It is computed as
`funding_payments − 8 bps TC`, with funding read from the **historical
on-disk `funding_rates` table** (exact, no execution risk). Real trading is
strictly worse. What paper does **not** model:

| Gap | Paper today | Real | Rough magnitude |
|---|---|---|---|
| **Transaction fees** | 8 bps round-trip (one notional) | 4 taker fills: spot 2×~10 + perp 2×~5 | **~30 bps** taker (≈23 with BNB) — **~3–4× the model** |
| **Slippage / spread** | none (uses the funding rate, no price) | each fill crosses the spread + walks the book | ~1–5 bps/fill majors; **5–20+ bps/fill** on SOL/XRP/ADA/AVAX at size |
| **Partial fills** | none (full size assumed) | re-orders, extra fees, residual unwind | situational |
| **Funding drift** | exact historical rate | rate at *actual fill time*, set at funding settlement, ≠ signal-time estimate | ± several bps of edge |
| **Basis / liquidation** | none (delta-neutral assumed perfect) | cross-wallet margin, divergence risk | tail risk, not a mean cost |
| **Borrow cost** | n/a | **n/a for perp short** (margin, not borrow) | — |

### The number that should set expectations

Worked example at the **current closest signal** (ADA, funding **+10.9%**
annualized, the brief's live example), **7-day hold**:

- Gross funding ≈ `10.9% × 7/365` ≈ **~21 bps** over the hold.
- **Paper** (8 bps TC): net ≈ +13 bps → looks profitable.
- **Real** (~30 bps taker + a few bps slippage): net ≈ **−9 to −15 bps → LOSS.**
- Real with BNB + tight majors (~23 bps): net ≈ **−2 bps → ~breakeven.**

**At ADA-like funding (~11%) and a 7-day hold, real all-in costs plausibly eat
the entire edge — or invert it.** The strategy needs **higher funding**
(meaningfully above ~15–20% annualized for a 7-day hold), **longer holds**
(amortize the fixed 4-fill entry/exit cost over more funding periods), **maker
execution**, or a **lower fee tier** to clear costs. This is the **same lesson
as Cycle 53 D7/D8** (the executor booking trades the atlas zeroed), now in
**cost space**: an optimistic TC made marginal trades look positive that real
costs zero or flip.

### How the first real trades should be sized / expected
1. **Size as cost-discovery, not profit.** First trades are to *measure* real
   slippage + fills + funding-drift on this account/fee tier, not to make money.
   Well below the paper $500/asset (R4) — think **$25–$100 notional**.
2. **Expect ~breakeven-to-negative on the first trades** and judge them on
   *execution quality* (did both legs fill, was slippage within ceiling, did
   unwind never trigger), **not P&L vs the backtest.**
3. **Require a funding cushion**: only arm real trades when funding is high
   enough that even at ~30–40 bps real round-trip the hold clears costs with
   margin. Encode as a real-mode-only `min_funding_ann_real` floor, distinctly
   above the paper gate.
4. **Reconcile after**: compare realized funding + fills against the paper
   model's prediction for the same window; that delta **is** the measured gap,
   and it should re-calibrate the TC assumption before any size-up.

---

## 4. How real money is gated (summary; full designs in the RECON)

- **Mode (R0):** `paper | real`, default **paper**, order path a **no-op stub**
  in paper; the real `create_*_order` call reachable **only** in real mode
  **and** behind the per-trade confirm. Unset/ambiguous → paper.
- **Row distinction (R0):** real-live rows must be distinguishable from
  paper-live rows in MAIN — proposed via a `mode` column (and/or separate
  `real_*` tables; see RECON R0 for the structural-vs-column fork).
- **Exclusivity (R2):** real mode is **GUI-only**; the scheduled task stays
  **paper forever**; a **single-owner lock** a real executor must acquire or
  refuse to place.
- **Per-trade confirm (R3):** propose → **HOLD** (place nothing) → human
  reviews both legs/sizes/funding/P/gate-state → explicit confirm → only then
  place. Timeout → **no-trade**. Pending state persisted.
- **Sizing (R4):** per-asset notional map, SOL sized down, conservative.
- **Precondition:** **D8-live** must fire correctly on a real production alert
  in paper first. This design does not remove that gate.

---

## 5. Surfaced Risks Register

| # | Risk | Severity | Where | Disposition |
|---|---|---|---|---|
| **1** | **Latent live-order path** `carry_executor.py` + `run_carry.py --live`: same strategy, naive, **no two-leg atomicity**, market orders, reads real keys, bypasses all Cycle 51–54 safety. One command from real orders. | **CRITICAL** | `engines/carry_executor.py`, `scripts/run_carry.py` | **Quarantine/neutralize before the real build.** Decide its fate with Jeff: delete, hard-disable (`paper` arg removed / `--live` raises), or fold into the new gated path. |
| 2 | Two-leg atomicity (the R5 hard problem) — directional exposure between legs; naked leg if leg 2 fails. | High | new real path | Designed in §2.2 (fill-confirm + unwind ladder + escalate). |
| 3 | Double-submission across two executors on one real account. | High (catastrophic on real) | scheduled task + GUI both `run_once()` on MAIN | R2: GUI-only real + single-owner lock + client-order-id idempotency. |
| 4 | Cost model undercount (~8 bps vs ~30 bps real) makes marginal trades look profitable. | High (economic) | `TC_PCT_ONE_WAY` | R6: re-calibrate TC; real-mode funding floor; cost-discovery sizing. |
| 5 | `max_daily_loss_pct` unwired + no equity concept; circuit breaker is USD-only. | Medium | `apply_risk_checks` | R1: define denominator (allocation, not whole-account); wire it. |
| 6 | Cross-wallet liquidation on basis divergence (perp short margin separate from spot). | Medium (tail) | exchange structure | Low leverage + margin buffer + basis monitor; size for it. |
| 7 | Real rows named into `paper_trades` / `paper_position_exits` tables. | Medium (clarity/safety) | schema | R0: `mode` column and/or separate `real_*` tables (lean structural per the Cycle 54 replay-isolation precedent). |
| 8 | Pending-confirm state lost on backend restart could auto-fire or be orphaned. | Medium | GUI in-memory session registry | R3c: persist pending-confirm; restart → fail to no-trade, never auto-confirm. |

---

## 6. Open questions for Jeff (decide before the build)

1. **`carry_executor.py` fate** — delete, hard-disable, or absorb? (Risk #1.)
2. **Row model** — `mode` column on the paper tables, or separate `real_*`
   tables? (Structural isolation matches the Cycle 54 replay precedent.)
3. **pct-of-what denominator** — allocated capital (testable in paper) vs
   account equity (needs a live balance query)? (R1.)
4. **Fee tier / BNB** — what tier is the account, BNB-discount on? Changes the
   R6 break-even funding floor.
5. **First-trade notional + funding floor** — confirm cost-discovery sizing
   ($25–$100) and the real-mode `min_funding_ann_real` floor.

---

**END — DESIGN ONLY. No code, no credentials, no live calls were produced.
Build is a later cycle, gated on R0–R4 + D8-live + Jeff's explicit go.**
