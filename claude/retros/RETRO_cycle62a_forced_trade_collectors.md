# Retro: Cycle 62A -- Forced-Trade Collectors

**Brief:** `BRIEF_cycle62a_forced_trade_collectors.md`
**Date:** 2026-08-19
**Status:** COMPLETE with one blocker (T1 -- no reachable data source, on either path)

---

## Summary

Five of six tasks landed and are running. **T1 is blocked on both of its paths, and the
second one is the decisive finding of the cycle.**

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

## T1 -- Binance forced-order stream: BLOCKED

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

### The Bybit option, not taken

Bybit's `allLiquidation` topic is reachable and carries the same event class (**129 events
in 300s** across BTC/ETH/SOL, against a 31,763-event `publicTrade` control on the same
connection). Not adopted, per instruction. Recorded because it remains the only *live*
liquidation feed reachable from this host, and because the `liquidations` PK already
carries `venue`.

One diagnostic trap worth keeping: an early Bybit probe subscribed to 13 topics in one
request, received `success: true`, and then got nothing. Bybit ACKs and silently drops a
batch that exceeds the per-request arg limit. The first read of that was "Bybit is blocked
too", which was wrong. **A success ACK is not evidence of a working subscription** — only a
control topic that must produce data is.

### The egress finding, worth keeping

Binance futures **REST** works from this host while Binance futures **WS** completes the
upgrade and then silently drops every frame. Geo-blocking would normally take both, or
refuse the handshake outright. Accepting the upgrade and delivering zero bytes points at a
**local middlebox** — something on the path terminating or stalling the WebSocket after
negotiation — rather than Binance refusing the region. Bybit and Hyperliquid WS both work,
so it is not "all WebSockets"; it is specific to `fstream.binance.com`. Worth knowing
before anyone attributes it to Canadian perp restrictions and stops looking.

### What landed for T1

- `liquidations` schema, unchanged from spec, plus a **`source` column** distinguishing
  `stream:*` from `archive:*` — added *before* any row is loaded, because the archive's
  two counting errors make provenance-free counts uninterpretable.
- The stream collector stamps `source` on every row it writes.
- `services/liquidation_collector_service.bat` stays in place, **unscheduled**, carrying
  the evidence inline, for whenever `fstream` becomes reachable.

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

**4. Liquidations — hourly. DO NOT REGISTER YET.** The feed is blocked (see T1); this
would produce an hourly stream of exit-2 runs. Register only after the feed is reachable
or a venue decision is made.
```
schtasks /create /tn "PraxisLiquidationCollector" /tr "cmd.exe /c \"C:\Data\Development\Python\McTheoryApps\praxis\services\liquidation_collector_service.bat\"" /sc hourly /mo 1 /ru jmcphail /f
```

Verify after registering:
```
schtasks /query /tn "PraxisOpenInterestCollector" /fo LIST /v
```

---

## Blockers, stated rather than worked around

1. **T1 has no reachable source of Binance liquidation labels.** Two independent paths,
   both closed:
   - *Live stream:* `fstream.binance.com` accepts the WS upgrade and delivers zero frames.
     Binance futures REST works from the same host, so this is almost certainly a local
     middlebox terminating the WebSocket after negotiation, **not** geo-blocking — worth
     knowing before anyone attributes it to Canadian perp restrictions and stops looking.
   - *Public archive:* no `um` liquidation dataset exists; the `cm` one ends 2024-10-14
     with **zero overlap** against our 2026-04-29-onward `trades`, and is coin-margined
     rather than USDT-margined.

   Binance withdrew the public `allForceOrders` REST endpoint, so there is no third path.
   **Still open:** whether to accept Bybit `allLiquidation` (reachable, same event class,
   different venue) — deliberately not taken this cycle.
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

---

*Cycle 62A. Companion: `BRIEF_cycle62b_d1_mechanism_prereg.md` (independent).*
