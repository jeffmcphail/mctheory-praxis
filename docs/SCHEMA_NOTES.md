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

#### Table: funding_rates (NONCONFORMING)

- Columns: `id` (INTEGER PK AUTOINCREMENT), `asset` (TEXT NOT NULL),
  `timestamp` (INTEGER NOT NULL, **seconds** UTC), `datetime` (TEXT
  NOT NULL, naive `"YYYY-MM-DD HH:MM:SS"`, no offset suffix),
  `funding_rate` (REAL).
- Writer: `engines/crypto_data_collector.py` `collect_funding_rates`
- Scheduled task: `PraxisFundingCollector`
  (`00:05`/`08:05`/`16:05` local Toronto, NOT UTC).
- Conformance gaps: timestamp units (sec, want ms); `datetime` lacks
  `+00:00`; `id` PK instead of `(asset, timestamp)` PK.
- Migration cycle: TBD.

#### Table: market_data (EMPTY, NONCONFORMING)

- Columns: `id`, `asset`, `date`, `market_cap`, `total_volume`,
  `circulating_supply`, `total_supply`, `ath`, `ath_change_pct`,
  `btc_dominance`. No `timestamp` column.
- 0 rows. Schema exists; collector logic may exist in
  `crypto_data_collector.py` but is not currently invoked.
- Pre-Praxis-recovery artifact. Defer migration until populated
  (or remove if confirmed dead).

#### Table: ohlcv_1m (NONCONFORMING)

- Columns: `id` PK, `asset`, `timestamp` (INTEGER **seconds** UTC),
  `datetime` (TEXT naive `"YYYY-MM-DD HH:MM:SS"`),
  `open`/`high`/`low`/`close`/`volume`.
- Writer: `engines/crypto_data_collector.py` `collect_1m`
- Scheduled task: `PraxisCrypto1mCollector` (every 6h, --days 2).
- Conformance gaps: timestamp units, datetime format, PK shape.
- Migration cycle: TBD. **High row count (~520k); plan dual-write
  pattern.**

#### Table: ohlcv_4h (NONCONFORMING)

- Columns: `id` PK, `asset`, `timestamp` (INTEGER **seconds** UTC),
  `datetime` (TEXT naive), OHLCV columns.
- Writer: `engines/crypto_data_collector.py` `collect_4h`
- Scheduled task: `PraxisOhlcv4hCollector` (daily 00:20 local,
  `--days 7`).
- Conformance gaps: timestamp units, datetime format, PK shape.
- Migration cycle: TBD. Re-fetchable from Binance; stop-migrate-start
  pattern viable.

#### Table: ohlcv_daily (NONCONFORMING)

- Columns: `id` PK, `asset`, `timestamp` (INTEGER **seconds** UTC,
  midnight), `date` (TEXT `YYYY-MM-DD`), OHLCV columns.
- Writer: `engines/crypto_data_collector.py` `collect_daily`
- Scheduled task: `PraxisOhlcvDailyCollector` (daily 00:15 local,
  `--days 7`).
- Conformance gaps: timestamp units, PK shape (date column already
  conforms).
- Migration cycle: TBD. Re-fetchable; stop-migrate-start viable.

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

#### Table: order_book_snapshots (NONCONFORMING)

- Columns: `id` PK, `asset`, `timestamp` (INTEGER **seconds** UTC),
  `datetime` (TEXT ISO with `+00:00` offset -- already conforms),
  bid/ask top-10 levels + spreads + imbalance.
- Writer: `engines/order_book_collector.py`
- Scheduled task: `PraxisOrderBookCollector` (hourly back-to-back,
  3550s windowed, 10s cadence).
- Conformance gaps: timestamp units (sec, want ms); PK shape.
- Migration cycle: TBD. **Actively-written, high frequency; dual-write
  pattern required.**

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
| crypto_data | funding_rates | NONCONFORMING | TBD | stop-migrate-start | Re-fetchable from Binance |
| crypto_data | market_data | EMPTY | -- | -- | Defer until populated |
| crypto_data | ohlcv_1m | NONCONFORMING | TBD | dual-write | ~520k rows; high frequency |
| crypto_data | ohlcv_4h | NONCONFORMING | TBD | stop-migrate-start | Re-fetchable |
| crypto_data | ohlcv_daily | NONCONFORMING | TBD | stop-migrate-start | Re-fetchable |
| crypto_data | onchain_btc | NONCONFORMING | TBD | stop-migrate-start | No active collector |
| crypto_data | order_book_snapshots | NONCONFORMING | TBD | dual-write | High frequency |
| crypto_data | trades | NEAR-CONFORMING | TBD | dual-write | timestamp already ms |
| live_collector | collection_log | NONCONFORMING | TBD | dual-write | TEXT timestamp |
| live_collector | price_snapshots | NONCONFORMING | TBD | dual-write | Active 60s cadence |
| live_collector | spike_alerts | EMPTY | -- | -- | Defer until populated |
| live_collector | tracked_markets | N/A | -- | -- | State, not temporal |
| smart_money | convergence_signals | EMPTY | -- | -- | Defer until populated |
| smart_money | position_changes | EMPTY | -- | -- | Defer until populated |
| smart_money | position_snapshots | NONCONFORMING | TBD | dual-write | Active 6h cadence |
| smart_money | tracked_wallets | N/A | -- | -- | State, not temporal |

Cycle 18 will produce `docs/SCHEMA_MIGRATION_PLAN.md` ordering and
sequencing the remaining migrations.

---

## Notes on currently-stale tables

- `onchain_btc`: monitored as of Cycle 17. No scheduled collector
  registered. Staleness alarms expected (`is_stale=true`) until a
  collector lands. 364 rows, latest `2026-04-28`. Last collected
  during recovery; coverage is roughly 1 year. Registering a
  scheduled task is queued under "Active TODOs" in `claude/TODO.md`.
- `market_data`: empty (0 rows). Schema exists; collector logic may
  be in `crypto_data_collector.py` but is not currently invoked.
  Investigate before populating; pre-Praxis-recovery artifact.

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
