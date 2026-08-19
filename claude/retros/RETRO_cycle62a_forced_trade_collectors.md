# Retro: Cycle 62A -- Forced-Trade Collectors

**Brief:** `BRIEF_cycle62a_forced_trade_collectors.md`
**Date:** 2026-08-19
**Status:** COMPLETE. T1 revised — Bybit adopted as the liquidation venue and collecting.

> **Revision (same day).** T1 originally closed as blocked, with Bybit recorded as an option
> deliberately not taken. That rejection assumed the Binance archive existed as a fallback;
> enumerating it proved it does not. With **no backfill available on any path**, the series
> can only be built forward and every unrecorded hour is lost permanently — so Bybit
> `allLiquidation` is now the T1 venue and is live. Two things changed with it: the
> non-comparability warning was carried forward into the code and schema rather than
> dropped, and the "local middlebox" egress claim was **measured and corrected** — the
> block is server-side. Superseded sections are marked in place.

---

## Summary

Five of six tasks landed and are running. **T1's Binance paths are both closed, and the
second one is the decisive finding of the cycle** — which is what forced the venue
substitution.

The live stream is unreachable: Binance's futures WebSocket accepts a connection from this
host and then never sends a frame, on any stream, including one that pushes unconditionally
every second. Not a collector bug, not the sandbox.

The archive pivot — same venue, historical, over a CDN the block does not touch — was then
enumerated rather than assumed, and **the dataset it needs does not exist**. There is no
USD-margined `liquidationSnapshot` prefix at all, and the coin-margined one Binance does
publish **stopped on 2024-10-14**, roughly 18.5 months before our `trades` coverage begins.
**The overlap is zero**, so the A2 false-positive rate cannot be computed from it at any
date. That was the payoff, and it is unavailable — no confusion matrix, in either
direction.

The two regime repairs the brief asked for are both verified at the acting layer, not the
schema: **class F now announces its degradation, and class K is computable for the first
time.** Neither depended on T3, which was the explicit instruction.

| Task | Deliverable | Status |
|---|---|---|
| T1 | Binance forced-order labels | **BLOCKED, both paths** — stream silent from this host; archive dataset does not exist for UM and ends 2024-10-14 for CM (zero overlap with our data) |
| T2 | Open interest collector | **LIVE** — 7,368 rows seeded, 6 assets × 2 venues |
| T3 | Unlock-bearing `market_data` universe | **LIVE** — 25 assets from a stated rule |
| T4 | Multi-asset OHLCV | **LIVE** — 25,238 rows, `ohlcv_daily` 2 → 20 assets |
| T5 | Class F silent-degradation fix | **DONE** — `bbff96d`, 65 tests pass |
| T6 | Regime-class status audit | **DONE** — `docs/REGIME_MATRIX.md` |

Commits: `bbff96d` (T5), `3cea45d` (T1+T2 + schema), `2bd1093` (T3+T4+T6), plus the
archive-findings commit below.

---

## T1 -- Liquidation labels: Binance BLOCKED, Bybit ADOPTED

The collector is complete: parser unit-tested against Binance's documented payload
including the Rule 35.4 seconds-vs-milliseconds foot-gun, reconnect with capped
exponential backoff, gap rows written open-on-disconnect and closed-on-reconnect, and an
exit-code contract that separates "ran cleanly with N rows" from "ran cleanly with 0 rows
when N>0 was expected".

It receives nothing. A 119-second live run connected successfully and captured 0 events,
exiting 2 — which is the contract working, but 0 events across all Binance symbols in two
minutes is not a quiet market.

### What was actually measured

| Endpoint | Stream | Result |
|---|---|---|
| Binance SPOT `stream.binance.com:9443` | `btcusdt@aggTrade` | **347 frames / 20s** |
| Binance PERP `fstream.binance.com` | `btcusdt@aggTrade` | 0 frames / 20s |
| Binance PERP `fstream.binance.com` | `btcusdt@markPrice@1s` | 0 frames / 20s |
| Binance PERP `fstream.binance.com` | `!markPrice@arr@1s` | 0 frames / 20s |
| Binance PERP `fstream.binance.com` | `!forceOrder@arr` | 0 frames / 119s |
| Binance PERP `fstream.binance.com` | combined `/stream?streams=` | 0 frames / 45s |
| Bybit PERP `stream.bybit.com/v5/public/linear` | `publicTrade.BTCUSDT` | **271 frames / 20s** |
| Hyperliquid `api.hyperliquid.xyz/ws` | `trades BTC` | **96 frames / 20s** |

Every `fstream` stream completes the WebSocket upgrade (**HTTP 101**), leaves the socket
open (`state=1`, no close code), and delivers zero bytes. `!markPrice@arr@1s` is the
decisive control: it pushes every second unconditionally, so 0 frames cannot be a quiet
market.

Binance futures **REST** (`fapi.binance.com`) works normally from the same host and
process — which is why T2 works. Reproduced identically inside and outside the tool
sandbox, so it is not a harness artefact.

The signature — upgrade accepted, stream silent, one venue's derivatives only, REST
unaffected — is consistent with Binance geo-restricting derivatives *streaming*, matching
the Canadian CEX-perp constraint Cycle 56 established. **There is no same-venue fallback:**
Binance withdrew the public `GET /fapi/v1/allForceOrders` REST endpoint, so forced orders
are stream-only.

### The archive pivot -- also closed, and measured

The natural fix is the public archive: same venue as our `trades` data, historical rather
than forward-only, and reached over the same CDN Cycle 60 pulled 600 symbols from, so the
`fstream` block is irrelevant to it. It was enumerated rather than assumed, and it cannot
serve this purpose either.

| Prefix | Result |
|---|---|
| `data/futures/um/daily/liquidationSnapshot/` | **DOES NOT EXIST — 0 keys.** There is no USD-margined liquidation dataset in the public archive. |
| `data/futures/cm/daily/liquidationSnapshot/` | Exists — 118 **coin-margined** symbols (`BTCUSD_PERP` + quarterlies) — but the entire dataset **stops at 2024-10-14**. |

`cm/monthly`, `spot/daily` and `option/daily` were also enumerated: no liquidation dataset
anywhere else in the bucket.

**Our `trades` coverage begins 2026-04-29. The overlap with the archive is ZERO** — a gap
of roughly 18.5 months. The cross-check of the Cycle 61 A2 flow-burst detector against
archive labels is not possible at any date, so no confusion matrix and no false-positive
rate can be produced from it. That was the payoff, and it is unavailable.

The dataset is also the wrong market twice over: coin-margined inverse contracts, whereas
our funding, OI and OHLCV are all USD-margined/spot.

### What the archive is, measured anyway — because it bounds any future use

Both of the properties flagged for verification were checked against file contents rather
than taken from the docs:

**1. It IS a sampled lower bound.** Across every file examined, the maximum number of
*distinct* liquidations in any single second was exactly **1**. The archive inherits the
documented "largest order per symbol per 1000ms" throttle. **Event counts and volumes from
it are a floor, not a complete record**, and every magnitude claim downstream would have to
be stated that way.

**2. Every row is written exactly twice.** Not in the brief, and it would have silently
doubled every count:

| File | Rows | Unique | Dup factor | Max distinct/sec |
|---|---|---|---|---|
| BTCUSD_PERP 2024-10-14 | 100 | 50 | 2.00 | 1 |
| BTCUSD_PERP 2024-09-02 | 112 | 56 | 2.00 | 1 |
| ETHUSD_PERP 2024-08-05 | 1146 | 573 | 2.00 | 1 |

The two errors run in **opposite directions** — throttling undercounts, duplication
overcounts — so they do not cancel and neither is safe to ignore. This is exactly what
"inspect the layout before parsing" is for; a parser written against the assumed schema
would have reported double the true event count without a single error.

**Layout** (10 columns, header row present, inspected before any parsing):
```
time,side,order_type,time_in_force,original_quantity,price,average_price,
order_status,last_fill_quantity,accumulated_fill_quantity
1728866962145,SELL,LIMIT,IOC,1,62246.2,62472.0,FILLED,1,1
```
`time` is epoch **milliseconds** (1728866962145 → 2024-10-14) — the ms/us guard in
`xsec_reversal/archive.py` resolves it to `ms`, and the 10-column count is stable across
every file checked.

Field mapping to the `liquidations` schema is clean and unambiguous:

| Archive column | `liquidations` column |
|---|---|
| `time` | `timestamp` (ms, canonical) |
| `side` | `side` |
| `price` | `price` |
| `average_price` | `avg_price` |
| `original_quantity` | `quantity` |
| `accumulated_fill_quantity` | (× `average_price`) → `quote_qty` |
| `order_status` | `order_status` |
| `order_type` | `order_type` |
| — | `venue` = `binance`, `source` = `archive:futures/cm/daily/...` |

### Futures vs spot, stated rather than blurred

These are **futures** liquidations; our price and flow data are **spot**. That asymmetry is
the correct experiment — forced selling happens on futures, the impact lands on spot — and
it is worth stating plainly so the two never blur in a later write-up. It is moot for this
cycle only because the coverage gap bites first.

### The Bybit option — ADOPTED (revised)

> Superseding the original "not taken". The rejection was predicated on the Binance archive
> existing as a fallback; the enumeration above proved it does not.

Bybit's `allLiquidation` topic is reachable and carries the same event class (**129 events
in 300s** across BTC/ETH/SOL, against a 31,763-event `publicTrade` control on the same
connection). It is now the **T1 venue**.

The argument is an asymmetry, not a preference. **No liquidation backfill exists anywhere**
— Binance withdrew the `allForceOrders` REST endpoint, and the archive has no `um` dataset
and a `cm` dataset with zero overlap. So this series can only ever be built *forward*, and
every hour not recorded is gone permanently. The venue choice is reversible; the data loss
is not. Waiting costs something irreplaceable to buy something reversible.

**The original warning stands and is carried forward, not dropped.** It is written into
`engines/bybit_liquidation_collector.py`, `services/bybit_liquidation_collector_service.bat`
and the schema migration, so it cannot be lost by anyone who reads only one of them.

One diagnostic trap worth keeping: an early Bybit probe subscribed to 13 topics in one
request, received `success: true`, and then got nothing. Bybit ACKs and silently drops a
batch that exceeds the per-request arg limit. The first read of that was "Bybit is blocked
too", which was wrong. **A success ACK is not evidence of a working subscription** — only a
control topic that must produce data is. The collector now encodes that lesson directly: it
subscribes in batches of 5 and keeps a `kline.1` control topic whose silence, not the
liquidation count, is what makes a run exit non-zero.

### Bybit is NOT Binance — three reasons, and they do not cancel

Recorded here, in the schema migration and in the collector because a later reader will be
tempted to pool these counts with a Binance-derived number:

| | Difference | Effect on counts |
|---|---|---|
| 1 | **Perp market share.** Binance carries substantially more perpetual OI and volume. | The same cascade produces a different number of events, on different symbols, at different sizes. |
| 2 | **Liquidation engine.** Margin tiers, partial-liquidation rules, ADL and insurance-fund behaviour differ. | The same position is force-closed at a different price, in a different number of pieces. |
| 3 | **Stream throttle — measured.** Binance's `!forceOrder@arr` caps at **one event per symbol per second** (the archive inherits the same cap: max distinct events per symbol-second was exactly 1 in every file). Bybit applies **no cap** — up to **18 distinct events in a single BTCUSDT second** on this host. | Bybit is the *more complete* record and simultaneously the *less comparable* one, by a factor that varies with how bursty the moment was. |

**Consequence:** any Binance-based prior on liquidation event *rates* — per-minute
thresholds, burst-size percentiles, "events per cascade" — does not transfer and must be
re-estimated on Bybit data. What does transfer is the event **class**: these are genuine
forced closures, which is what scenario A2 needs.

Coverage also narrows in kind: Binance offered one all-market topic; Bybit's is **per
symbol**, so coverage is exactly the six-asset funding/OI universe subscribed
(BTC ETH SOL XRP ADA AVAX). Anything outside it is *unobserved* — a coverage boundary, not
a quiet market.

### The side convention is INVERTED between the venues — caught before it corrupted anything

The single highest-risk field in the swap, and it would have failed silently.

- Binance `forceOrder.S` is the **ORDER** side. Closing a long sends a market SELL, so
  **`SELL` ⇒ a long was liquidated**.
- Bybit `allLiquidation.S` is the **POSITION** side. So **`Buy` ⇒ a long was liquidated**.

Same two letters, opposite meanings, both legal members of `{BUY, SELL}`. Copying Bybit's
`S` straight across would have inverted every Bybit row against every Binance row, and no
type check, validator or unit test would have noticed.

The docs were **not** taken on trust — this field is widely mis-documented. It was settled
by physics: force-closing a long sends a market SELL and ticks price *down*; force-closing a
short sends a market BUY and ticks it *up*. Liquidations and `publicTrade` were captured on
one connection (5,286 liquidations against 709,885 trades) and the signed move around each
event measured.

**The first run nearly recorded the answer backwards.** The tape was rallying, so the raw
mean move was *positive* around both sides. Adding a random-time baseline from the same tape
and reporting the **excess over concurrent drift** separated them cleanly and consistently
at every window:

| Window | baseline drift | `S=Buy` excess (n=136) | `S=Sell` excess (n=5150) |
|---|---|---|---|
| ±250ms | +3.897 bps | **−3.008** | **+1.850** |
| ±1000ms | +6.426 bps | **−5.311** | **+17.008** |
| ±3000ms | +11.464 bps | **−9.438** | **+55.903** |

`S=Buy` sits on *negative* excess (a market SELL hit) ⇒ a **long** was liquidated.
`S=Sell` sits on *positive* excess (a market BUY hit) ⇒ a **short** was liquidated. The
documentation is correct, and it is now correct *because it was measured*, not because it
was read. Evidence: `outputs/forced_trade/t1_bybit_side_convention.json`.

Storage decision: `side` holds **one** convention for all venues — the Binance ORDER side,
because that is what the column already means and what the Cycle 61 A2 detector compares
against — and the new **`side_raw`** column keeps whatever the venue actually sent, so the
translation stays auditable and reversible. The map lives in exactly one place
(`engines/liquidation_common.SIDE_MAP`) and `validate` re-derives every stored row against
it, so a drift between the two collectors fails a check instead of silently skewing a study.

### Bybit's price is a BANKRUPTCY price, not a fill price

Binance sends an order price *and* an average fill price; the collector prefers `ap × z`, so
its `quote_qty` is close to true executed notional. Bybit sends only `p`, documented as the
**bankruptcy price** — where margin reaches zero, not where anything traded. A Bybit
`quote_qty` is therefore an approximation with a known directional bias, recorded per row as
**`price_basis`** (`'bankruptcy'` vs `'executed'`) rather than presented as a measurement.
Bybit also sends no order status, order type or average fill price; those stay NULL —
absent, not guessed.

### The egress finding — CORRECTED, and it is not a local middlebox

> The original entry read "points at a **local middlebox**, not a regional restriction."
> **That was wrong.** The reasoning was that geo-blocking would take REST and WS together,
> so a split must be local. The follow-up measurement refutes it.

The fstream endpoint **answers control messages** and withholds only market data:

```
-> {"method":"LIST_SUBSCRIPTIONS","id":1}
<- {"result":[],"id":1}                    server replies
-> {"method":"SUBSCRIBE","params":["btcusdt@aggTrade"],"id":2}
<- {"result":null,"id":2}                  subscribe ACCEPTED
   then: zero market-data frames, socket open, no close code
```

The identical exchange on spot returns the same two replies **and** starts delivering
aggTrade frames immediately. The futures server is alive, parsing our JSON, and selectively
sending everything except the data.

Nothing on the path can do that. All three local candidates were checked and are absent:

- **Windows Firewall** — zero enabled outbound Block rules. (And a firewall block refuses the
  connection; it does not ACK a subscribe.)
- **Antivirus TLS inspection** — the only security product installed is Windows Defender,
  which does not intercept TLS.
- **Proxy** — WinHTTP direct, WinINET disabled, no PAC.

Most decisively, `fstream` / `fapi` / `stream` all present a genuine **DigiCert "GeoTrust TLS
RSA CA G1"** chain for `*.binance.com` (Bybit presents a genuine Amazon chain). There is **no
interception**, so nothing on the path can read a WebSocket frame at all — let alone tell a
subscribe ACK from an aggTrade and drop one of the two.

The suppression therefore happens *inside* the TLS tunnel, at the application layer, which
only Binance can do. It is **server-side** — an entitlement or regional restriction on
futures market data for this egress IP, consistent with the Canadian CEX-perp constraint
Cycle 56 established.

**What follows:** no local change fixes this. Firewall, AV and proxy settings are dead ends.
Only a different egress (VPN, different network) would test it — an infrastructure decision,
not a code one. Binance therefore does **not** become the default; Bybit stays T1.

*Method note:* the original error came from reasoning about what a block "would normally"
look like instead of asking the server a question it had to answer. `LIST_SUBSCRIPTIONS`
settled in one round-trip what a week of signature-matching would not have.

### What landed for T1

- `liquidations` schema, unchanged from spec, plus a **`source` column** distinguishing
  `stream:*` from `archive:*` — added *before* any row is loaded, because the archive's
  two counting errors make provenance-free counts uninterpretable.
- Plus **`side_raw`** and **`price_basis`** (migration
  `cycle62a_liquidations_bybit_columns.py`), and an `(venue, timestamp)` index, because the
  table is now genuinely multi-venue and every honest query filters on venue.
- **`engines/bybit_liquidation_collector.py` — the ACTIVE T1 collector**, with
  `services/bybit_liquidation_collector_service.bat` and
  `services/register_bybit_liquidation_task.ps1`.
- **`engines/liquidation_common.py`** — the side-convention map, `price_basis` map, and the
  shared `report` / `validate` both collectors now use, so the two venues cannot drift into
  different definitions of the same table.
- `engines/liquidation_collector.py` (Binance) stays in place, **unscheduled**, now carrying
  the corrected egress evidence, and writes `side_raw` / `price_basis` too so it validates
  cleanly if the block ever lifts.

**Live verification (400s, 2026-08-19):** 1 connect, 3/3 subscribes ACKed, **400 control
frames**, 4,937 events received → **4,937 rows written**, 0 duplicates, 0 rejected, 0 bad
frames, 0 gaps, all 6 symbols present, exit 0. All **13** validation checks pass, including
the two new cross-venue ones (`side_raw` preserved; every stored row's `side` re-derived
against the declared map). The window happened to catch a short squeeze — $180.7M notional,
BUY 4,805 / SELL 132 on the canonical order-side convention.

---

## T2 -- Open interest: LIVE

7,368 rows seeded across 6 assets × 2 venues. All validation checks pass.

### The walls, measured rather than assumed — and the measurement changed the plan

| Venue | Wall | Behaviour |
|---|---|---|
| Binance | **temporal, ~30 days** | `startTime` beyond returns `-1130 parameter 'startTime' is invalid` — identically at 1h, 4h **and** 1d |
| Bybit | **row-count, 200 rows** | not a date wall |

Bybit's cap being a row count rather than a date is the useful finding, because
**granularity then trades against reach**:

| Bybit timeframe | Rows | Reach |
|---|---|---|
| 1h | 200 | 8.3 days |
| 4h | 200 | 33.2 days |
| 1d | 200 | **199.0 days** (to 2026-02-01) |

Cycle 61 read Bybit's wall as a date ("the same 200 rows from 2026-02-01") because it
probed at daily granularity. Both readings describe the same API; the row-count reading is
the actionable one. Seeding all three granularities took Bybit's reach from 8.3 days to
199 — the finer passes densify the recent window, the coarse pass reaches the floor. Since
the PK is `(asset, venue, timestamp)`, the granularities coexist without conflict, and
`source` records which produced each row (`bybit:1d`, `binance:1h`, …).

### SEED BOUNDARY — where backfill ends and live capture begins

| Venue | Earliest observation | Mechanism |
|---|---|---|
| **binance** | **2026-07-20T15:00:00+00:00** | 30-day temporal wall |
| **bybit** | **2026-02-02T00:00:00+00:00** | 200 rows × 1d |

Seeded 2026-08-19. Everything after that date is live capture. This boundary is a
permanent property of the dataset: nothing before it is retrievable from either venue at
any granularity, ever.

### Cadence: hourly

The feature is `oi_change_7d` — OI now against OI ~7 days ago. Hourly gives 168 points
across that window, ~24× finer than the feature can use, so it is not the binding
constraint on anything. Chosen over coarser because it matches Binance's finest history
granularity, so seeded and live rows are the same kind of measurement rather than two
regimes stitched together; and it is cheap (6 × 2 × 24 = 288 rows/day).

### Notional

Binance's *history* endpoint returns `sumOpenInterestValue`; its *current-value* endpoint
does not. Bybit returns neither. Routing both the seed and the live poll through the
history endpoint means the two paths produce identical fields — **720/720 Binance rows
carry notional, 0/508 Bybit rows do**, recorded as absent rather than synthesised from
price.

### Exit-code contract: staleness, not row count

The brief's rule is non-zero exit when a run writes 0 rows *where N>0 was expected*. For
this collector a row count is the wrong test: right after a seed, or on a re-run inside
the same hourly bucket, **0 new rows is the correct outcome**. What separates that from a
dead feed is whether the newest observation has kept up with the clock. Verified in both
directions — exit 0 when current, exit 2 at a forced 0.001h threshold.

---

## T3 -- Unlock-bearing universe: LIVE

`market_data` held 427 rows across ADA/BTC/ETH/SOL/XRP — five mega-caps whose supply moves
by block subsidy, emission, burn and escrow, none of which is a vesting cliff.

**Provider:** CoinGecko. **Asset list:** was hard-coded in `SUPPORTED_ASSETS`
(`engines/crypto_data_collector.py:49`), now config-driven for the unlock universe via
`config/unlock_universe.json`.

### Why it is a separate list, not a wider `SUPPORTED_ASSETS`

Widening that dict would have widened every collector that iterates it — the market_data,
OHLCV and funding services all invoke `--asset all` — turning a 5-asset CoinGecko cadence
into a 25-asset one and changing existing collectors' behaviour, which the brief forbids.
The unlock universe is its own list; existing collectors keep their scope exactly.

### THE SELECTION RULE

Mechanical and re-runnable, because a hand-picked list is not auditable:

| | Rule |
|---|---|
| R1 | `market_cap_rank <= 300` — an unlock in an illiquid token produces no measurable price response |
| R2 | `circulating_supply > 0 AND total_supply > 0` — F1's hard requirement |
| R3 | `circulating_supply / total_supply <= 0.75` — at least 25% of supply still locked |
| R4 | not in CoinGecko's own `stablecoins` category — fetched programmatically, never by eye |
| R5 | not in the base universe (BTC/ETH/SOL/XRP/ADA/AVAX/BNB) |
| R6 | top 25 survivors by market cap |

**Rejection tally (auditable):** R1 rank 2 · R2 supply missing 0 · R3 float too high 162 ·
R4 stablecoin 45 · R5 base universe 1 · **passed 90, took top 25**.

### On "post-2021 launches"

The brief framed the target as post-2021 launches with large locked allocations. **R3
measures that property directly instead of proxying it by launch date**, and is the better
filter: a 2019 token with 40% still locked can produce an unlock cliff, and a 2022 token
that has fully unlocked cannot. The selected set proves the point — LINK (2017), XLM and
UNI all carry >25% locked supply and would be missed by a launch-date filter.

Genesis dates were to be recorded for auditability, but CoinGecko's free tier 429s hard on
per-coin detail calls and returns `genesis_date: null` for most of these tokens anyway.
The genesis pass is **off by default** rather than silently dropping every asset with a
missing field — which would be exactly the degradation T5 exists to stamp out.

### R5 exists because R3 alone admitted XRP

XRP passes R3 on ~37% "locked" supply that is **escrow release, not a VC cliff** — the
precise mechanism Cycle 61 measured as producing no supply jump above 0.41%. Excluding the
base universe by rule keeps T3 additive rather than re-importing the problem it exists to
fix.

### Collection result

All **25/25** assets stored, **every one carrying both `circulating_supply` and
`total_supply`** — verified per asset against the same `/coins/{id}` response the collector
writes from, not a separate pass against a different endpoint. `market_data` went from
**5 assets to 30**.

Float ratios span 0.233 (HYPE) to 0.748 (LINK), i.e. **25%–77% of supply still locked** —
against a base universe where Cycle 61 measured the largest supply jump anywhere at 0.41%.
F1 now has a universe where an unlock can actually show up.

### Known imperfection, stated

`STABLE` (rank 79) passed R4 because CoinGecko's `stablecoins` category does not list it.
R4 is only as good as the vendor's categorisation. Recorded rather than hand-removed — a
rule edited until it produces the desired list is no longer a rule.

---

## T4 -- Multi-asset OHLCV: LIVE, and K is repaired

**25,238 rows** written. `ohlcv_daily` went from **2 assets to 20**; `ohlcv_4h` likewise.

Deliberately **not dependent on T3**: this collector reads Binance via ccxt REST while T3
reads CoinGecko — different providers, different failure modes. The asset list is layered
so that the base universe (BTC ETH SOL XRP ADA AVAX) alone clears K's floor of three, with
the unlock universe added only where the asset actually lists on Binance. Symbols are
resolved against Binance's live market list, so an unlisted asset is skipped loudly rather
than producing an empty series that later reads as a quiet market.

### K verified at the acting layer

```
assets with >= 60 daily bars: 19
compute_dispersion_regime on 19 real 24h returns -> state=1, dispersion=0.013882
RegimeEngine.compute with a 19-asset universe:
    K state      : 1
    K in missing : False
```

K was in `RegimeState.missing` on 100% of evaluations before this cycle. It now carries a
state and is absent from `missing`.

---

## T5 -- Class F announces its degradation (`bbff96d`)

`compute_funding_regime` initialised `oi_change_7d = 0.0` and only overwrote it when an OI
series of ≥22 observations was supplied. States ±2 require `abs(oi_change_7d) > 0.10`, so
without OI they were unreachable — silently.

| | Reachable states |
|---|---|
| Declared | `[-2,-1,0,+1,+2]` |
| Without OI | `[-1,0,+1]` |
| With OI rising | `[-1,0,+1,+2]` |
| With OI falling | `[-2,-1,0,+1]` |
| With OI (union) | `[-2,-1,0,+1,+2]` |

Verified by probing the classifier across a funding grid, not by reading the source.

**Fix:** when funding is present but OI is not, `F` is appended to `RegimeState.missing`, a
reason is recorded in a new `RegimeState.degraded` map, and a warning is logged.
`compute_funding_regime` now publishes `oi_available` in its raw features, and
`OI_MIN_PAYMENTS = 22` is named rather than buried as a literal.

`degraded` was added because `missing` alone conflates two different things: F without OI
still carries a *usable* 3-state signal, which is not the same as an axis forced to 0. Its
keys are always a subset of `missing`, so any existing reader of `missing` sees the
degradation without knowing about the new field.

**Tests:** 65 pass. The required pair (F in `missing` without OI, absent with OI) plus a
regression guard asserting the ±2 states really are OI-gated — if someone rewrites the
classifier so they are not, that test fails and the warning must be revisited.

One existing test changed meaning: `test_with_funding_data` previously asserted
`"F" not in state.missing` with funding-only input. That assertion encoded the bug, so it
now asserts the corrected contract.

**Blast radius:** `engines/forced_trade/occupancy.py` counts `len(st.missing)`. Its
reported counts will now include F wherever OI is absent — the correct number, but
different from what Cycle 61 printed.

---

## T6 -- Regime-class status audit

`docs/REGIME_MATRIX.md` gains a **Live Status** section separating the design from what
the collectors actually support, with five status values (LIVE / DEGENERATE / DEGRADED /
UNCOMPUTABLE / STUB) and per-class evidence.

| Class | Before | After |
|---|---|---|
| E microstructure | DEGENERATE (constant over sample) | unchanged — not addressed |
| **F funding/positioning** | **DEGRADED, silently** | **LIVE** — OI collected; degradation now announced |
| H cross-asset corr | DEGENERATE (constant over sample) | indirectly improved — universe 2 → 20 |
| **K dispersion** | **UNCOMPUTABLE** | **LIVE** — verified at the acting layer |
| L RV/IV spread | STUB | unchanged — **permanent**; needs a DVOL feed, not a code change |

`python -m engines.atlas_sync` re-run: 12 regime classes and 60 relevance rows still parse
correctly.

---

## Scheduled tasks — for Jeff to run (admin)

Registration was not attempted. Naming matches the existing `PraxisXxxCollector`
convention.

**1. Open interest — hourly (register this one):**
```
schtasks /create /tn "PraxisOpenInterestCollector" /tr "cmd.exe /c \"C:\Data\Development\Python\McTheoryApps\praxis\services\open_interest_collector_service.bat\"" /sc hourly /mo 1 /ru jmcphail /f
```

**2. Multi-asset OHLCV — daily (register this one):**
```
schtasks /create /tn "PraxisOhlcvUniverseCollector" /tr "cmd.exe /c \"C:\Data\Development\Python\McTheoryApps\praxis\services\ohlcv_universe_collector_service.bat\"" /sc daily /st 01:10 /ru jmcphail /f
```

**3. Unlock market data — daily (register this one):**
```
schtasks /create /tn "PraxisUnlockMarketDataCollector" /tr "cmd.exe /c \"C:\Data\Development\Python\McTheoryApps\praxis\services\unlock_market_data_collector_service.bat\"" /sc daily /st 01:40 /ru jmcphail /f
```

**4. Liquidations (Bybit) — hourly. REGISTER THIS ONE.** This is the active T1 collector,
verified live. Every hour it is not running is permanently unrecoverable — there is no
backfill on any path.
```
schtasks /create /tn "PraxisBybitLiquidationCollector" /tr "cmd.exe /c \"C:\Data\Development\Python\McTheoryApps\praxis\services\bybit_liquidation_collector_service.bat\"" /sc hourly /mo 1 /ru jmcphail /f
```
Or, with the matching execution-time-limit and overlap settings already filled in:
```
powershell -ExecutionPolicy Bypass -File "C:\Data\Development\Python\McTheoryApps\praxis\services\register_bybit_liquidation_task.ps1"
```

**5. Liquidations (Binance) — DO NOT REGISTER.** Still unreachable, and now known to be
**server-side** (see the corrected egress finding), so no local change will fix it.
Registering it would produce an hourly stream of exit-2 runs. It stays in place for
whenever the egress changes.
```
schtasks /create /tn "PraxisLiquidationCollector" /tr "cmd.exe /c \"C:\Data\Development\Python\McTheoryApps\praxis\services\liquidation_collector_service.bat\"" /sc hourly /mo 1 /ru jmcphail /f
```

Verify after registering:
```
schtasks /query /tn "PraxisOpenInterestCollector" /fo LIST /v
```

---

## Blockers, stated rather than worked around

1. **T1 has no reachable source of *Binance* liquidation labels — RESOLVED by venue
   substitution, not by unblocking.** Two independent paths, both still closed:
   - *Live stream:* `fstream.binance.com` accepts the WS upgrade, **answers control
     messages**, ACKs a subscribe, and then delivers zero market-data frames. Verified
     **server-side**, not local: no TLS interception (genuine DigiCert chain), no outbound
     firewall block, no third-party AV, no proxy. Only an egress change would test it.
   - *Public archive:* no `um` liquidation dataset exists; the `cm` one ends 2024-10-14
     with **zero overlap** against our 2026-04-29-onward `trades`, and is coin-margined
     rather than USDT-margined.

   Binance withdrew the public `allForceOrders` REST endpoint, so there is no third path.
   **Closed:** Bybit `allLiquidation` is adopted as the T1 venue and is collecting. The
   residual risk is not availability but **comparability** — see the three-reason table
   above. Any Binance-derived prior on event rates must be re-estimated.
2. **CoinGecko free-tier rate limits** made the per-asset genesis-date enrichment
   impractical (429s exhausting a 7-step backoff ladder). Worked around by making genesis
   optional and using the directly-measured overhang instead — but a paid key would make
   the launch-date audit trivial if it is wanted.
3. **R4's stablecoin exclusion is only as good as CoinGecko's categorisation** — `STABLE`
   slipped through. Not hand-corrected, by design.
4. **Class L remains a permanent stub.** No options/DVOL data source exists in this
   project. Recorded in the matrix as permanent rather than pending.

---

## What generalised

**A degraded axis that does not announce its degradation is worse than an absent one.**
The F fix is the direct application, but the same shape recurred three times this cycle:

- A **success ACK is not evidence of a working subscription** (Bybit's 13-arg batch).
- A **row count cannot tell a dead feed from an already-current table** — which is why the
  OI collector's exit code keys on staleness instead.
- An **empty series from an unlisted symbol is indistinguishable from a quiet market** —
  which is why OHLCV symbols are resolved against the live market list and skipped loudly.

In each case the failure mode was the same: a check that returns a plausible value whether
or not the thing it measures actually happened.

The T1 revision added three more, all of the same family:

- **Ask the endpoint a question it is obliged to answer.** The "local middlebox" conclusion
  came from reasoning about what a geo-block *would normally* look like. One
  `LIST_SUBSCRIPTIONS` round-trip settled it in the opposite direction. Signature-matching
  against expectations is a hypothesis; a control message is a measurement.
- **A field can be present, well-typed, in-range and inverted.** Bybit's `S` and Binance's
  `S` are both legal `{BUY, SELL}` values meaning opposite things. No schema constraint, no
  type check and no unit test distinguishes them — only a semantic map, stored in one place
  and re-derived at validation time.
- **Measure an effect against the drift it sits in, not against zero.** The first
  side-convention run showed price rising around *both* liquidation sides, because the whole
  tape was rallying. The raw number was real and the inference from it would have been
  backwards. A random-time baseline from the same tape flipped the sign for one side and
  kept it for the other — and that contrast, not the level, is what carried the conclusion.

---

*Cycle 62A. Companion: `BRIEF_cycle62b_d1_mechanism_prereg.md` (independent).*
