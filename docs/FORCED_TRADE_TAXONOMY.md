# Forced-Trade Taxonomy

*Scenarios where market participants are compelled to trade at a bad price.*

**Status:** Cycle 61 deliverable. Enumeration and triage only — no scenario has been
backtested. Data-availability claims are checked against the live `crypto_data.db`
schema; **event counts are NOT yet verified** (that is the Cycle 61 data audit).

**Companions:** `docs/FORCED_TRADE_SCREEN.md` (the filter), `docs/REGIME_MATRIX.md`
(the regime axis, 12 classes A–L).

---

## Inclusion test

From the screen:

> Is there a rule, contract, or mechanism that forces the trade **regardless of the
> trader's opinion**? If a human could choose not to trade, it is not compulsion.

Each scenario is tagged:

- **Trigger type** — `SCHEDULED` (calendar-known in advance) or `TRIGGERED` (fires when observable state crosses a threshold)
- **Data status** — `HAVE` / `PARTIAL` / `MISSING` against the actual Praxis schema
- **Crowding** — how contested the flow is by professionals
- **Priority** — `P1` viable now · `P2` viable with modest data work · `P3` blocked

---

## What Praxis actually has

Verified against `crypto_data.db` (2026-08-18):

| Table | Content | Relevance |
|---|---|---|
| `trades` | Tick-level, **with `side` and `is_buyer_maker`** | Aggressor-side flow — the core liquidation-cascade signal |
| `order_book_snapshots` | 10 levels each side, `spread_bps`, `order_imbalance_top10` | Depth, impact, liquidity at event time |
| `funding_rates` | asset × **venue** × timestamp | Funding compulsion; leverage-state proxy |
| `info_bars` | Event bars with `buy_quote`/`sell_quote`/`imbalance_quote` | Pre-aggregated signed flow |
| `market_data` | market cap, **`circulating_supply`**, **`total_supply`**, btc_dominance | **Unlock detection** — see F1 |
| `ohlcv_1m` / `_4h` / `_daily` | Price/volume | Baseline |
| `onchain_btc` | active addresses, hash rate, difficulty | Miner economics |
| `fear_greed` | Sentiment index | Regime context |

**Critical gaps:**

| Missing | Blocks |
|---|---|
| **Open interest (OI)** | Regime class F specifies "Funding rates + OI"; there is **no OI table**. Leverage-state estimation is degraded to funding-only. |
| Liquidation feed | Direct cascade labelling (must be inferred from flow) |
| Options / DVOL | Regime class L; all gamma/expiry scenarios |
| Quarterly futures | Roll and term-structure scenarios |
| Index constituents, ETF flows | All equity-mandate scenarios |

---

## A. Leverage compulsion

### A1 — Perpetual funding payments
**Mechanism:** the perp contract debits the crowded side every 8 hours. Contractual; opinion-irrelevant.
**Trigger:** `SCHEDULED` (fixed settlement times) · **Data:** `HAVE` · **Crowding:** high, well known
**Screen:** Q1 ✅ (3–12 day holds) · Q2 ✅ · Q3 ⚠️ basis is the denominator — *this is Engine 7*
**Status:** validated, **regime-gated and currently dormant** (BTC funding 0.6% vs 11.9% in 2024). Tripwire armed. **P1 (passive)**

### A2 — Liquidation cascades
**Mechanism:** margin engine force-closes underwater positions at market, regardless of the holder's view. The purest compulsion in crypto — the trader is not consulted.
**Trigger:** `TRIGGERED` (margin thresholds) · **Data:** `HAVE` — cascades are detectable as bursts of one-sided aggressive flow in `trades.side`, with depth from `order_book_snapshots` · **Crowding:** moderate; HFT competes at the millisecond, but the *reversion over minutes-to-hours* is a different game
**Screen:** Q1 — depends entirely on holding period, must be measured · Q2 ✅ strongest mechanism available to us · Q3 — price volatility, straightforward
**Note:** BitMEX recorded a ~$20bn liquidation cascade on 10–11 Oct 2025 with order books at multi-year lows. Events of that scale are rare but enormous.
**Priority: P1 — the flagship candidate.**

### A3 — Auto-deleveraging (ADL)
**Mechanism:** when the insurance fund is exhausted, the venue force-closes *profitable* counterparties to socialise losses. Compulsion applied to the winning side.
**Trigger:** `TRIGGERED` · **Data:** `MISSING` (venues publish ADL events inconsistently) · **Crowding:** low
**Priority: P3** — mechanism is excellent, observability is poor.

---

## B. Mandate compulsion

### B1 — Index add / delete
**Mechanism:** trackers must hold the index; an addition forces buying at any price on a known date.
**Trigger:** `SCHEDULED` · **Data:** `MISSING` (equity constituent data) · **Crowding:** **very high** — textbook strategy, dedicated desks
**Screen:** Q2 fails on the accessibility half. **Priority: P3**

### B2 — ETF creation / redemption baskets
**Mechanism:** authorised participants must deliver/receive the basket to keep the ETF at NAV.
**Trigger:** `TRIGGERED` by flows · **Data:** `MISSING` · **Crowding:** very high (AP-gated — the same gated-leg problem that killed the tokenized-stock idea)
**Priority: P3**

### B3 — Risk-parity / vol-target deleveraging
**Mechanism:** vol-target mandates mechanically cut exposure when realised vol rises. Selling *because* vol rose, not because of a view.
**Trigger:** `TRIGGERED` (vol thresholds) · **Data:** `PARTIAL` — we can compute the vol trigger (regime classes B/C) but cannot observe the flow directly · **Crowding:** high in equities; **materially lower in crypto**
**Screen:** Q2 ⚠️ — mechanism real, our observation is indirect. **Priority: P2**

### B4 — Month / quarter-end rebalancing
**Mechanism:** funds rebalance to target weights on a calendar date irrespective of price.
**Trigger:** `SCHEDULED` · **Data:** `HAVE` (calendar + price) · **Crowding:** high, extensively published
**Priority: P2** — cheap to test since the calendar is free, but expect it to be arbed.

---

## C. Expiry compulsion

### C1 — Futures roll
**Mechanism:** holders of an expiring contract must roll or settle by a fixed date.
**Trigger:** `SCHEDULED` · **Data:** `MISSING` (no quarterly futures series collected) · **Crowding:** high
**Priority: P3** — would need a new collector.

### C2 — Options expiry pinning / gamma
**Mechanism:** dealers hedging short gamma must trade *with* price into expiry; pinning is mechanical.
**Trigger:** `SCHEDULED` · **Data:** `MISSING` (no options data; regime class L also blocked) · **Crowding:** very high
**Priority: P3**

---

## D. Product compulsion

### D1 — Leveraged token daily rebalancing
**Mechanism:** a 3× leveraged token **must** buy more exposure after a rise and sell after a fall, to reset leverage daily. Mechanical, price-insensitive, direction fully predictable from the day's return.
**Trigger:** `SCHEDULED` (daily reset) · **Data:** `PARTIAL` — Binance UP/DOWN tokens are in the archive and we already have the collector path from Cycle 60 · **Crowding:** moderate
**The inversion worth noting:** Cycle 60 **excluded** leveraged tokens from its universe precisely because their rebalancing creates *artificial mean reversion* that would have manufactured a false positive. **What was contamination for that experiment is the signal for this one.** The exclusion list in `engines/xsec_reversal/universe.py` is a ready-made candidate list.
**Priority: P1** — mechanism is unambiguous, direction is knowable in advance, data is one collector away.

### D2 — Covered-call ETF systematic selling
**Mechanism:** the fund must write calls on schedule regardless of pricing.
**Trigger:** `SCHEDULED` · **Data:** `MISSING` (options) · **Priority: P3**

---

## E. Operational compulsion

### E1 — Tax-loss harvesting
**Mechanism:** deadline-driven selling of losers in December; the deadline, not the price, drives it.
**Trigger:** `SCHEDULED` · **Data:** `HAVE` (calendar + returns) · **Crowding:** high, well documented
**Sample-size problem:** one observation per year. **Priority: P3** on sample grounds alone.

### E2 — Fund redemption selling
**Mechanism:** redemptions force sales regardless of view.
**Trigger:** `TRIGGERED` · **Data:** `MISSING` · **Priority: P3**

### E3 — Miner operational selling
**Mechanism:** miners must sell coin to pay electricity and hardware costs — a cash-flow obligation, not a market view. Intensifies when margins compress (post-halving, difficulty spikes, price falls).
**Trigger:** `TRIGGERED` (margin compression) · **Data:** `PARTIAL` — `onchain_btc` gives hash rate and difficulty; miner flow itself needs an on-chain labelling source we lack · **Crowding:** low
**Priority: P2** — genuinely under-studied, but observation is indirect.

---

## F. Crypto-native compulsion

### F1 — Token unlock cliffs
**Mechanism:** vesting contracts release locked supply to VCs and teams on fixed dates. The recipients did not choose the timing, and a large cohort has strong incentive to sell into a fixed, publicly-known date.
**Trigger:** `SCHEDULED` — **calendar known months ahead** · **Crowding:** **low** — largely ignored by traditional professionals
**Data:** `PARTIAL`, and better than expected. `market_data` carries **`circulating_supply` and `total_supply`**; their difference is locked supply, and **jumps in `circulating_supply` reveal unlock events retrospectively** — enough to build and validate the historical study without buying anything. Forward-looking schedules would need an external source (TokenUnlocks/CryptoRank) for live trading.
**Screen:** Q1 ✅ event-driven, naturally low turnover · Q2 ✅ contractual, uncrowded · Q3 — price volatility, straightforward
**Priority: P1** — best combination of clean mechanism, advance knowledge, low crowding, and workable data.

### F2 — Staking unbonding queues
**Mechanism:** unbonding periods create a queue of supply arriving at a knowable time.
**Trigger:** `SCHEDULED` · **Data:** `MISSING` (chain-specific) · **Priority: P3**

### F3 — Exchange delisting forced closure
**Mechanism:** on delisting notice, holders must exit by a deadline; the venue force-closes what remains.
**Trigger:** `SCHEDULED` (notice period) · **Data:** `PARTIAL` — **Cycle 60 already established that the Binance archive retains delisted symbols; 169 of 600 in that universe were delisted.** Delisting dates are derivable from where each symbol's data stops · **Crowding:** low
**Priority: P2** — the survivorship-free universe machinery from Cycle 60 is directly reusable.

### F4 — Bridge / protocol migration deadlines
**Mechanism:** token migrations impose hard swap deadlines.
**Trigger:** `SCHEDULED` · **Data:** `MISSING` · **Priority: P3**

---

## Triage summary

| Priority | Scenario | Why |
|---|---|---|
| **P1** | **A2 Liquidation cascades** | Purest compulsion; full tick + book data already collected |
| **P1** | **F1 Token unlocks** | Scheduled months ahead, low crowding, detectable via `circulating_supply` |
| **P1** | **D1 Leveraged token rebalance** | Mechanical and direction-predictable; Cycle 60's exclusion list is the candidate list |
| P1 (passive) | A1 Funding carry | Validated, dormant, tripwire armed — no work required |
| **P2** | B3 vol-target · B4 quarter-end · E3 miner selling · F3 delisting | Viable with modest data work |
| **P3** | A3, B1, B2, C1, C2, D2, E1, E2, F2, F4 | Blocked on data, crowding, or sample size |

**Three P1 candidates, all crypto-native, all using data we already collect.** That is not
a coincidence: crypto is where compulsion is *observable* (public liquidations, on-chain
supply, exchange-published mechanics) and where professional crowding is thinnest.

---

## Open questions for the data audit

To be answered empirically before any grid is designed:

1. **How many events does each P1 scenario actually produce?** Rarity is the binding
   constraint on a scenario × regime grid. If A2 yields 15 cascades/year and the regime
   axis has 12 classes, most cells are unfillable — and pooling them post hoc to get
   sample size is exactly how a null result gets laundered into a finding.
2. **Can liquidation cascades be reliably identified** from `trades.side` bursts plus
   `order_book_snapshots` depth, without a liquidation feed? What false-positive rate?
3. **Do `circulating_supply` jumps cleanly mark unlock events**, or is the series too
   coarse/lagged to date them?
4. **Does the missing OI table materially degrade regime class F**, and is OI collectable
   from the venues we already use?
5. **What is the realistic holding period** for each P1 scenario — the Q1 input, and the
   number that decides whether any of this survives costs.

---

*Last updated: 2026-08-18 (Chat: praxis_main_current)*
*Changes: Initial enumeration. 16 scenarios across 6 compulsion categories, each tagged
with mechanism, trigger type, data availability against the live schema, crowding, and
screen verdict. Three P1 candidates identified. Data gaps recorded, including the absent
open-interest table that regime class F depends on.*
