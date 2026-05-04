# Schema Notes

> About this file: documentation of every SQLite table across the
> Praxis databases, with column types, semantic meaning, conformance
> status against Rule 35 (temporal data storage standard), and
> read-pattern guidance.
>
> Cycle 17 baseline. Updated each cycle as more tables are migrated.

---

## The standard (Rule 35 summary)

Every table containing temporally indexed data MUST have:

1. A `timestamp` column of INTEGER type, storing Unix epoch
   milliseconds in UTC. Always milliseconds, always UTC, always named
   `timestamp`. Date-only data converts to midnight-UTC ms.
2. `timestamp` is part of the primary key (alone for single-asset
   tables, compound `(asset, timestamp)` / `(slug, timestamp)` /
   `(wallet, timestamp)` for multi-keyed tables).
3. Optional redundant `datetime` (or `date`) TEXT cache, derived from
   `timestamp` in ISO 8601 with explicit `+00:00` offset (or
   `YYYY-MM-DD` for date-only). Canonical column is always
   `timestamp`; the text version is a cache.
4. Collectors MUST convert non-UTC source timestamps to UTC before
   storage.
5. New tables conform from day 1. Existing tables migrate one per
   cycle, tracked in `docs/SCHEMA_MIGRATION_PLAN.md` (forthcoming).
6. Actively-written tables migrate via the dual-write pattern
   (build `_v2`, parallel collect, backfill, cutover, burn-in,
   delete legacy). Batch/daily tables can stop-migrate-start when
   the source API allows full re-fetch.

Full text in `claude/CLAUDE_CODE_RULES.md`, Rule 35.

---

## Read patterns (Rule 34 summary)

Reading a SQLite DB that another process is actively writing to
requires explicit transaction management. Three acceptable patterns:

1. **Fresh connection per logical read pass.** Open, query, close.
   Cheapest and most foolproof. The MCP server's `connect_ro` does
   this.
2. **`isolation_level=None`** for true autocommit -- each statement
   is its own transaction.
3. **`conn.commit()` between SELECTs** -- ends the implicit read
   transaction so the next SELECT begins a fresh transaction with
   current state.

Never keep a single `sqlite3.Connection` open across multiple SELECT
passes spanning more than a few seconds without one of the above.
See `claude/retros/RETRO_sqlite_freshness_diagnostic.md` for the
Cycle 15 investigation.

Full text in `claude/CLAUDE_CODE_RULES.md`, Rule 34.

---

## Database inventory

### crypto_data.db (primary)

Path: `data/crypto_data.db`. Written by `engines/crypto_data_collector.py`
through several scheduled tasks (Praxis*Collector). Read by the MCP
server, all engines, and analysis scripts.

#### Table: fear_greed (CONFORMING -- Cycle 17)

- Columns: `timestamp` (INTEGER PK, ms UTC), `date` (TEXT,
  `YYYY-MM-DD` UTC midnight, derived), `value` (INTEGER 0-100),
  `classification` (TEXT, e.g. `"Fear"`, `"Greed"`).
- Writer: `engines/crypto_data_collector.py:387` `collect_fear_greed`
- Scheduled task: `PraxisFearGreedCollector` (daily 00:30 local).
- Source: alternative.me Fear & Greed Index API (returns seconds;
  collector multiplies by 1000).
- Migration cycle: 17 (this cycle).

#### Table: funding_rates (CONFORMING -- Cycle 21)

- Columns: `asset` (TEXT NOT NULL), `timestamp` (INTEGER NOT NULL,
  **ms** UTC), `datetime` (TEXT NOT NULL, ISO
  `YYYY-MM-DDTHH:MM:SS+00:00`), `funding_rate` (REAL). PK:
  `(asset, timestamp)`.
- Writer: `engines/crypto_data_collector.py` `collect_funding_rates`
- Scheduled task: `PraxisFundingCollector`
  (`00:05`/`08:05`/`16:05` local Toronto, NOT UTC).
- 2,212 rows at migration time (1,106 BTC + 1,106 ETH, 2025-04-30
  onward). Reader at `servers/praxis_mcp/tools/funding.py` uses
  runtime ms/sec autodetect (`ms_mode = ts_sample > 1e12`); migration
  required only a comment-header refresh, no logic change. Reader at
  `engines/lstm_predictor.py:86-90` uses `DATE(datetime)` GROUP BY,
  which SQLite handles for both naive and ISO datetime formats --
  reader-transparent across the format change.
- **Writer alignment (Cycle 21.5 hotfix)**: `collect_funding_rates`
  truncates Binance's `fundingTime` to seconds-aligned ms before
  storage (`ts = (int(r["timestamp"]) // 1000) * 1000`). Sub-second
  jitter from Binance's reporting clock has no information value
  (funding events are aligned to UTC hour boundaries by contract) and
  would otherwise produce duplicate rows for the same hourly event,
  since the compound PK on `(asset, timestamp)` does not collapse
  `.000` and `.NNN` into one row. Cycle 21.5 also deduped the 26
  rows that accumulated between Cycle 21 and the hotfix.

#### Table: market_data (CONFORMING -- Cycle 19)

- Columns: `asset`, `timestamp` (INTEGER ms UTC midnight),
  `date` (TEXT `YYYY-MM-DD` derived), `market_cap`, `total_volume`,
  `circulating_supply`, `total_supply`, `ath`, `ath_change_pct`,
  `btc_dominance`. PK: `(asset, timestamp)`.
- Writer: `engines/crypto_data_collector.py` `collect_market_data`
  (CoinGecko `/coins/{id}` per asset + one `/global` per cycle for
  dominance, threaded through via the CLI handler).
- Scheduled task: `PraxisMarketDataCollector` (daily 00:35 local).
- `btc_dominance` is a single global value (from `/global`); written
  identically across all asset rows for a given collection day. Read
  it from any one row. A single normalized "global state" table would
  be cleaner but isn't worth the join cost given N=3 assets.
- **No historical backfill capability**: CoinGecko's free
  `/coins/{id}` endpoint returns current state only. Table populates
  from Cycle 19 forward. Backfill would require a paid CoinGecko tier
  or an alternative data source (Glassnode, CoinMetrics).

#### Table: ohlcv_1m (CONFORMING -- Cycle 22)

- Columns: `asset`, `timestamp` (INTEGER **ms** UTC),
  `datetime` (TEXT ISO `YYYY-MM-DDTHH:MM:SS+00:00`),
  `open`/`high`/`low`/`close`/`volume`. PK: `(asset, timestamp)`.
- Writer: `engines/crypto_data_collector.py` `collect_ohlcv_1m`
- Scheduled task: `PraxisCrypto1mCollector` (every 6h, --days 180).
- 530,836 rows at migration time (BTC: 265,419, ETH: 265,417;
  2025-10-31 onward). The 2-row asymmetry is a pre-existing data-
  quality footnote: BTC starts 2025-10-31T17:45:00 UTC, ETH starts
  17:47:00, likely a 2-min lag in ETH's first collector backfill
  batch in October. Migration preserved both assets' rows
  independently.
- **Reader fix (Cycle 22)**: `engines/intrabar_predictor.py`
  `load_intrabar_data` line 110 changed from
  `bar_seconds = bar_minutes * 60` to `bar_minutes * 60 * 1000` to
  match the post-migration ms timestamps. Pre-fix the bar-bucketing
  arithmetic produced unique buckets per 1-min row, causing
  `bar_minutes >= 2` to silently return zero aggregated bars. This
  is the first migration cycle in the program requiring a
  non-cosmetic reader change (Cycles 17-21 needed only writer +
  comment-header updates). Verified empirically post-fix:
  `bar_minutes=5` returns aggregated bars at exact 5-min boundaries.
- Reader at `servers/praxis_mcp/tools/ohlcv.py` `get_recent_ohlcv`
  is reader-transparent (sorts and returns the column without
  arithmetic). Docstring updated to specify ms units; pre-Cycle-22
  callers expecting seconds need to adapt.
- Migration script timing: 530,836-row INSERT-SELECT completed in
  0.567s wall-clock (well under the 2-minute concerning threshold).
- **Writer alignment** (per Cycle 21.5 lesson): Binance kline
  `openTime` values from `fetch_ohlcv` are bar-aligned by contract
  (no sub-second jitter). Confirmed empirically post-migration: all
  530,836 rows have `timestamp % 1000 == 0`. The `funding_rates`
  jitter pattern does not apply to kline endpoints. This audit
  result is durable for any future cycle dealing with Binance kline
  data.

#### Table: ohlcv_4h (CONFORMING -- Cycle 20)

- Columns: `asset`, `timestamp` (INTEGER **ms** UTC),
  `datetime` (TEXT ISO `YYYY-MM-DDTHH:MM:SS+00:00`),
  OHLCV columns. PK: `(asset, timestamp)`.
- Writer: `engines/crypto_data_collector.py` `collect_ohlcv_4h`
- Scheduled task: `PraxisOhlcv4hCollector` (daily 00:20 local,
  `--days 7`).
- 10,830 rows (5,415 per asset, 2023-11-11 onward). No external
  reader audited; fully consumed inside the collector pipeline.

#### Table: ohlcv_daily (CONFORMING -- Cycle 18)

- Columns: `asset`, `timestamp` (INTEGER **ms** UTC midnight),
  `date` (TEXT `YYYY-MM-DD`), OHLCV columns. PK: `(asset, timestamp)`.
- Writer: `engines/crypto_data_collector.py` `collect_ohlcv_daily`
- Scheduled task: `PraxisOhlcvDailyCollector` (daily 00:15 local,
  `--days 7`).
- 1,802 rows (901 per asset, 2023-11-12 onward). Reader at
  `engines/lstm_predictor.py:68` queries by `date` (unaffected by the
  migration).

#### Table: onchain_btc (NONCONFORMING -- monitored as of Cycle 17)

- Columns: `id` PK, `date` (TEXT `YYYY-MM-DD`), `active_addresses`,
  `transaction_count`, `hash_rate`, `difficulty`, `block_size`,
  `total_btc`, `market_cap`. **No `timestamp` column.**
- Writer: `engines/crypto_data_collector.py` `collect_onchain` (not
  currently invoked by any registered scheduled task).
- Conformance gaps: missing INTEGER `timestamp` entirely;
  `date`-only.
- Cycle 17 added this table to MCP `get_collector_health`. With no
  scheduled collector running, `is_stale=true` is expected and
  intentional until a collector is registered (see "Currently-stale
  tables" below).
- Migration cycle: TBD.

#### Table: order_book_snapshots (CONFORMING -- Cycle 23, dual-write pilot)

- Columns: `asset`, `timestamp` (INTEGER **ms** UTC, full Binance API
  precision incl. sub-second tail), `datetime` (TEXT ISO with
  microsecond precision + `+00:00` offset), bid/ask top-10 levels +
  spreads + imbalance. PK: `(asset, timestamp)`.
- Writer: `engines/crypto_data_collector.py` `collect_order_book_snapshot`
- Scheduled task: `PraxisOrderBookCollector` (hourly back-to-back,
  3550s windowed, 10s cadence).
- 88,894 rows at cutover (BTC + ETH growing ~12 rows/min).
- **First dual-write cycle in the migration program**. Phase 0-4
  executed in this cycle; Phase 5 (drop `_legacy`, single-write
  collapse) deferred to Cycle 23.5 after 24-48h burn-in.
- **Precision recovery**: pre-Cycle-23 the writer truncated Binance's
  ms `fundingTime` to seconds via `ts_ms // 1000` while the matching
  `datetime` field preserved sub-second precision; the migration
  parses `datetime` to derive ms (via SQLite
  `CAST(ROUND((julianday(dt) - 2440587.5) * 86400000) AS INTEGER)`
  for backfilled rows; native API ms for dual-write rows).
- **MCP tools silently fixed**: `get_order_book_range`'s `WHERE
  timestamp BETWEEN start_ts_ms AND end_ts_ms` clause was
  unit-mismatched pre-migration (table stored sec, clients passed ms)
  and returned `total_in_range = 0` for any sane input. Cycle 23
  migration repairs this without code change. `get_order_book_snapshot`'s
  `ABS(timestamp - at_timestamp_ms)` math also becomes meaningful
  post-migration.
- **Dual-write window writer** (in effect until Cycle 23.5):
  introspects the live table's PK shape on every iteration. If `id`
  column present (pre-cutover state), writes seconds-truncated ts to
  live + ms to `_v2`. If no `id` (post-cutover state), writes ms to
  live + seconds-truncated ts to `_legacy`. The runtime adaptation
  prevents the writer from breaking across the cutover RENAME.

#### Table: trades (NEAR-CONFORMING)

- Columns: `id` PK, `asset`, `trade_id`, `timestamp` (INTEGER **ms**
  UTC -- already conforms), `datetime` (TEXT ISO with `Z` suffix --
  near-conforming, want `+00:00`), `price`, `amount`, `quote_amount`,
  `is_buyer_maker`, `side`.
- Writer: `engines/trades_collector.py`
- Scheduled task: `PraxisTradesCollector` (hourly back-to-back,
  3550s windowed, 30s cadence).
- Conformance gaps: `id` PK instead of `(asset, timestamp, trade_id)`
  PK; `datetime` text uses `Z` instead of `+00:00`.
- Migration cycle: TBD. Cosmetic-ish migration; dual-write still
  recommended given write rate.

### live_collector.db (sidecar)

Path: `data/live_collector.db`. Written by `PraxisLiveCollector`
which polls top Polymarket markets every 60s. Cycle 14 added it to
MCP health monitoring as a sidecar.

#### Table: collection_log (NONCONFORMING)

- Columns: `id` PK, `timestamp` (TEXT ISO with `+00:00`),
  `markets_tracked`, `samples_taken`, `errors`, `duration_ms`.
- TEXT timestamp -- no INTEGER timestamp column.
- Internal log of collector runs.
- Migration cycle: TBD.

#### Table: price_snapshots (NONCONFORMING)

- Columns: `id` PK, `slug` (TEXT NOT NULL), `timestamp` (INTEGER
  **seconds** UTC), `yes_mid`, `yes_bid`, `yes_ask`, `spread`.
- Conformance gaps: timestamp units (sec, want ms); PK shape (want
  `(slug, timestamp)`).
- Migration cycle: TBD. Actively-written; dual-write pattern.

#### Table: spike_alerts (EMPTY)

- Columns: `id` PK, `slug`, `question`, `event_type`, `detected_at`
  (TEXT), price/move/window fields.
- 0 rows. Migration deferred until populated.

#### Table: tracked_markets (state, NOT temporal-row)

- Columns: `slug` (TEXT PK), market metadata, `first_tracked`,
  `last_updated` (TEXT timestamps).
- This is mutable state, not append-only temporal data. Rule 35
  doesn't strictly apply (no row-per-time-point). Out of scope for
  the migration program.

### smart_money.db (sidecar)

Path: `data/smart_money.db`. Written by `PraxisSmartMoney` (every
6h). Wallet discovery + Polymarket position snapshots. Cycle 14
added it to MCP health.

#### Table: convergence_signals (EMPTY)

- Columns: `id` PK, `detected_at` (TEXT NOT NULL), market metadata,
  wallet/size/strength fields.
- 0 rows; convergence detection logic exists but hasn't generated
  signals in the current rebuild window.

#### Table: position_changes (EMPTY)

- Columns: `id` PK, `detected_at` (TEXT NOT NULL), wallet/market
  fields, size delta + price.
- 0 rows; populated only when a wallet's position size changes
  between snapshots.

#### Table: position_snapshots (NONCONFORMING)

- Columns: `id` PK, `snapshot_id`, `timestamp` (TEXT ISO with
  `+00:00`), `wallet`, market info, `size`, `avg_price`,
  `current_price`, `value_usd`, `pnl_usd`.
- Conformance gaps: TEXT timestamp -- no INTEGER timestamp column;
  PK shape (want `(wallet, timestamp, market_slug)`).
- Migration cycle: TBD. Actively-written every 6h; dual-write
  pattern.

#### Table: tracked_wallets (state, NOT temporal-row)

- Columns: `address` (TEXT PK), wallet metadata, `first_tracked`,
  `last_updated`.
- Mutable state per wallet. Out of scope for Rule 35 migration.

---

## Migration status

| Database | Table | Conformance | Cycle | Pattern | Notes |
|---|---|---|---|---|---|
| crypto_data | fear_greed | **CONFORMING** | **17** | stop-migrate-start | Done |
| crypto_data | funding_rates | **CONFORMING** | **21** | stop-migrate-start | Done |
| crypto_data | market_data | **CONFORMING** | **19** | schema-only + collector fix | No backfill (CoinGecko free-tier limit) |
| crypto_data | ohlcv_1m | **CONFORMING** | **22** | stop-migrate-start | Done; 530k rows in 0.567s |
| crypto_data | ohlcv_4h | **CONFORMING** | **20** | stop-migrate-start | Done |
| crypto_data | ohlcv_daily | **CONFORMING** | **18** | stop-migrate-start | Done |
| crypto_data | onchain_btc | NONCONFORMING | TBD | stop-migrate-start | No active collector |
| crypto_data | order_book_snapshots | **CONFORMING (DONE-PARTIAL)** | **23** | dual-write | Phases 0-4 done; Phase 5 cleanup in Cycle 23.5 |
| crypto_data | trades | NEAR-CONFORMING | TBD | dual-write | timestamp already ms |
| live_collector | collection_log | NONCONFORMING | TBD | dual-write | TEXT timestamp |
| live_collector | price_snapshots | NONCONFORMING | TBD | dual-write | Active 60s cadence |
| live_collector | spike_alerts | EMPTY | -- | -- | Defer until populated |
| live_collector | tracked_markets | N/A | -- | -- | State, not temporal |
| smart_money | convergence_signals | EMPTY | -- | -- | Defer until populated |
| smart_money | position_changes | EMPTY | -- | -- | Defer until populated |
| smart_money | position_snapshots | NONCONFORMING | TBD | dual-write | Active 6h cadence |
| smart_money | tracked_wallets | N/A | -- | -- | State, not temporal |

`docs/SCHEMA_MIGRATION_PLAN.md` carries the ordered roadmap and
per-cycle execution log for the remaining migrations.

---

## Notes on currently-stale tables

- `onchain_btc`: monitored as of Cycle 17. No scheduled collector
  registered. Staleness alarms expected (`is_stale=true`) until a
  collector lands. 364 rows, latest `2026-04-28`. Last collected
  during recovery; coverage is roughly 1 year. Registering a
  scheduled task is queued under "Active TODOs" in `claude/TODO.md`.
- `market_data`: CONFORMING as of Cycle 19. Migrated to Rule 35
  schema, writer fixed (added `/global` BTC-dominance call), CLI
  subcommand wired, scheduled task `PraxisMarketDataCollector`
  registered (daily 00:35). 3 rows seeded by manual first-run.
  No historical backfill possible -- table populates forward only.

---

## Read-pattern guidance for analysts

When opening any of these DBs from a script that runs while the
collector is also writing (which for `live_collector.db` and
`crypto_data.db` is essentially always), apply Rule 34. The MCP
server's `connect_ro` is the canonical implementation; analysis
scripts that mimic it (open, query, close) are fine.

When reading `fear_greed` post-Cycle-17, remember the timestamp is
**milliseconds** -- the magnitude check is `> 1e12`. The MCP health
helper auto-detects via this magnitude rule.
