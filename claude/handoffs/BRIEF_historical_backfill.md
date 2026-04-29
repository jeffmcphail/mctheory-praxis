# Implementation Brief: Historical Data Backfill (Tier 1 Recovery)

**Series:** praxis
**Cycle:** 9
**Priority:** P1 — restores LSTM training inputs and funding-strategy ground truth lost in the disk failure
**Mode:** B (multiple one-shot CLI invocations against a working collector; touches data but not collector code logic)

**Estimated Scope:** S (no new code; runs existing `engines.crypto_data_collector` subcommands with explicit `--days` parameters, then validates row counts)
**Estimated Cost:** $0 (all endpoints are public; no auth required)
**Estimated Data Volume:** ~3-4 MB across ~1.4M rows total. Run wall-time ~15-25 min depending on Binance rate-limit responses.
**Kill switch:** No edits to `engines/crypto_data_collector.py`. No new collector code. No edits to `services/*.bat` or `services/*.ps1`. If a planned invocation fails persistently after one retry with backoff, stop, document the failure mode in the retro, and continue to the remaining tables. Partial completion is acceptable; a botched all-or-nothing rerun is not.

Reference: `claude/CLAUDE_CODE_RULES.md` rules 9-15 (progress reporting), rule 16 (validation), rule 19 (ASCII).

---

## Context

Per `claude/handoffs/RECOVERY_PLAN_post_disk_failure.md` §3.3 Tier 1 items 2-4: the disk failure erased all `data/*.db` content, including ~520k OHLCV bars, 2,190 funding rate observations, ~900 fear-greed readings, and 365 days of on-chain BTC metrics. **Schema survived** (Cycle 8 verification confirmed all 10 tables exist in the rebuilt `crypto_data.db`); only the rows are gone.

All five data sources we need are public-API: no Binance authentication, no paid keys. The collector's `ccxt.binance({"enableRateLimit": True})` instantiation passes no API key — these are all public endpoints (OHLCV klines, funding rate history, perpetual contract data) plus `requests.get` to alternative.me (Fear & Greed) and blockchain.info (on-chain BTC).

The OrderBook collector from Cycle 8 is **currently running and accumulating microstructure data** (verified: 1190 snapshots in first 40 min, zero errors). This Brief does NOT touch the OrderBook collector or its scheduled task. The two activities run in parallel against the same `crypto_data.db` — SQLite's WAL mode handles the concurrency.

**Why this matters strategically (recovery plan §3.2):** OHLCV history is no longer the urgent direction-prediction substrate (five lines of evidence say standard TA on time bars doesn't predict 5-min BTC direction), but it remains the LSTM cross-asset feature input and is dirt-cheap to re-pull. Funding rate history is the ground truth for the highest-EV strategy (carry, Sharpe 4.45-10.78 with regime-continuity caveat). Fear & Greed is a memory-flagged LSTM feature. All three are recoverable from API in minutes, so the cost-benefit is overwhelmingly in favor of pulling now even if the immediate use is gated.

---

## Objective

Run six one-shot collector invocations to backfill `crypto_data.db`. Validate row counts after each. Document any failures in the retro.

---

## Detailed Spec

### Phase 0 — Verify preconditions (2 min)

Confirm working state before starting:

```powershell
cd C:\Data\Development\Python\McTheoryApps\praxis
.\.venv\Scripts\activate
```

Verify the OrderBook collector is still running (we don't want to disturb it):

```powershell
Get-ScheduledTask -TaskName PraxisOrderBookCollector
```

State should be `Running` or `Ready`. If `Disabled` or missing, stop and flag — something has gone wrong with Cycle 8's task and that's higher priority than this Brief.

Verify the database is reachable and schema exists:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/crypto_data.db'); print(sorted(r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")))"
```

Expected output should include at minimum: `fear_greed`, `funding_rates`, `ohlcv_1m`, `ohlcv_4h`, `ohlcv_daily`, `onchain_btc`, `order_book_snapshots`, `trades`. If any of these are missing, stop — schema is incomplete.

### Phase 1 — Run the six collector invocations (15-20 min wall-time)

Run each command from the praxis root with the venv activated. **Each command is independent — if one fails, the others can still proceed.** Run them sequentially, not in parallel; multiple concurrent ccxt instances against Binance will trip rate limits.

**1.1 — Daily OHLCV (BTC + ETH, 900 days):**

```powershell
python -m engines.crypto_data_collector collect-ohlcv --asset BTC --days 900
python -m engines.crypto_data_collector collect-ohlcv --asset ETH --days 900
```

Expected: ~900 rows per asset, ~1800 total in `ohlcv_daily`. Should complete in <30s each.

**1.2 — 4-hour OHLCV (BTC + ETH, 900 days):**

```powershell
python -m engines.crypto_data_collector collect-ohlcv-4h --asset BTC --days 900
python -m engines.crypto_data_collector collect-ohlcv-4h --asset ETH --days 900
```

Expected: ~5,400 rows per asset (900 days × 6 bars/day), ~10,800 total in `ohlcv_4h`. NOTE: the collector's hardcoded internal cap may limit this to 180 days unless we override; if so, ~1080 rows per asset is the realistic upper bound. If the collector caps below the requested days, document it in the retro and continue. Do NOT edit the collector to lift the cap.

**1.3 — 1-minute OHLCV (BTC + ETH, 180 days):**

```powershell
python -m engines.crypto_data_collector collect-ohlcv-1m --asset BTC --days 180
python -m engines.crypto_data_collector collect-ohlcv-1m --asset ETH --days 180
```

Expected: ~259,200 rows per asset (180 days × 1440 bars/day), ~518,400 total in `ohlcv_1m`. This is the slowest invocation — Binance returns ~1000 candles per request, so ~260 paginated requests per asset, each with a 0.3s sleep. Estimated wall-time per asset: 90-120 seconds. The collector will print progress every 50 batches.

NOTE on the 180-day cap: per the collector's docstring (`engines/crypto_data_collector.py` line 322-326), Binance actually serves 1m klines back ~730 days, but the collector caps at 180 days to balance fetch time with coverage. **Do not edit this cap as part of this Brief.** If you want more history later, that's a separate scope decision documented in the retro.

**1.4 — Funding rates (BTC + ETH, 365 days):**

```powershell
python -m engines.crypto_data_collector collect-funding --asset BTC --days 365
python -m engines.crypto_data_collector collect-funding --asset ETH --days 365
```

Expected: ~1,095 rows per asset (365 days × 3 funding events/day at 8h cadence), ~2,190 total in `funding_rates`. **This is the most strategically important table** — funding carry strategy ground truth. If this one fails, prioritize debugging it over the rest. Estimated wall-time: 30-60s each.

**1.5 — Fear & Greed Index (900 days):**

```powershell
python -m engines.crypto_data_collector collect-fear-greed --days 900
```

Expected: ~900 rows in `fear_greed`. Single source endpoint (api.alternative.me/fng), should complete in <10s.

**1.6 — On-chain BTC metrics (365 days):**

```powershell
python -m engines.crypto_data_collector collect-onchain --days 365
```

Expected: ~365 rows in `onchain_btc`. Source is blockchain.info charts API. Lower priority than the others (not currently a feature input for any active strategy), but cheap to include while we're here. If it fails, log and continue.

### Phase 2 — Validate row counts (5 min)

After all six invocations complete (or fail), run a single Python check to verify what made it into each table:

```powershell
python -c @"
import sqlite3
c = sqlite3.connect('data/crypto_data.db')
tables = ['ohlcv_daily', 'ohlcv_4h', 'ohlcv_1m', 'funding_rates', 'fear_greed', 'onchain_btc', 'order_book_snapshots']
for t in tables:
    n = c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    if n > 0:
        first, last = c.execute(f'SELECT MIN(timestamp), MAX(timestamp) FROM {t}').fetchone()
        from datetime import datetime, timezone
        first_dt = datetime.fromtimestamp(first, tz=timezone.utc).strftime('%Y-%m-%d') if first else '-'
        last_dt = datetime.fromtimestamp(last, tz=timezone.utc).strftime('%Y-%m-%d') if last else '-'
        print(f'{t:25s} {n:>10,} rows  ({first_dt} -> {last_dt})')
    else:
        print(f'{t:25s} {n:>10,} rows  (EMPTY)')
"@
```

This will print a one-line summary per table, including row count and date range. The retro should include this output verbatim.

NOTE: `order_book_snapshots` is included for visibility into the parallel-running Cycle 8 collector — its count should be growing during this Brief's execution. That's expected; do not be alarmed by a non-zero number that wasn't created by this Brief.

### Phase 3 — Retro

Standard retro at `claude/retros/RETRO_historical_backfill.md`. Include:

1. The validation output from Phase 2 verbatim
2. Wall-time per invocation (use `Measure-Command` if helpful)
3. Any errors or rate-limit hits encountered, with which retry attempts succeeded
4. Whether any of the 1m / 4h pulls hit the internal cap and produced fewer rows than the `--days` parameter requested, with the actual coverage achieved
5. Anything unusual or worth flagging for future cycles (e.g., a particular table that consistently errored, an endpoint that was slow, a schema field that came back null when expected populated)

---

## Acceptance Criteria

1. Each of the six invocation commands has been run (or attempted with documented failure)
2. Phase 2 validation output captured in the retro
3. `funding_rates` table has > 1,800 rows for BTC + ETH combined (this is the strategically critical one; if it's empty or sparse, the cycle is not complete)
4. At least three of the six tables (`ohlcv_daily`, `ohlcv_4h`, `funding_rates` minimum) are populated with at least 80% of the expected row count
5. No edits to `engines/crypto_data_collector.py` or any other source file
6. Retro file exists at `claude/retros/RETRO_historical_backfill.md` with required content
7. `git status` shows `data/crypto_data.db*` as the only modified files (the WAL and shm sidecars will also have changed, that's normal)

---

## Known Pitfalls

- **Binance rate limits.** The collector uses ccxt's `enableRateLimit: True` plus an explicit `time.sleep(0.3)` between batches in the 1m loop. If Binance returns 429 (rate limit) errors, the collector will print them; the cycle should pause for ~60 seconds before retrying. Do not edit the collector to be more aggressive — whatever rate the existing code handles is fine.
- **CCXT ImportError.** If `import ccxt` fails inside the collector, the venv install is broken or you're running outside the venv. Verify `pip show ccxt` shows it's installed before debugging anything else.
- **Duplicate inserts on rerun.** All these collectors use `INSERT OR REPLACE` semantics keyed on `(asset, timestamp)`, so rerunning is idempotent — no risk of doubled rows. Safe to re-execute a single command if it errored partway.
- **Concurrent OrderBook collector.** The Cycle 8 collector is writing to `order_book_snapshots` continuously. SQLite's WAL mode handles the concurrent reads/writes correctly. **Do not stop the OrderBook collector to "make room" for this Brief.** Both run fine in parallel. SQLite is a single-writer database, but the WAL mode serializes writes safely without blocking readers.
- **Funding rate symbol format.** Binance perp contracts use a specific symbol format (e.g., `BTC/USDT:USDT` for USDT-margined perp). The collector's `SUPPORTED_ASSETS[asset]["symbol"]` constant should already be correct, but if you see a "symbol not found" error, that's the place to check before assuming the API is broken.
- **Long fetch times for 1m OHLCV.** The 1m pull for 180 days = ~260K candles per asset, which is ~260 paginated requests per asset. Each request takes 200-400ms plus the 0.3s sleep, so 90-120 seconds per asset. Don't kill the process if it appears to hang — it's working, just slow. The collector prints progress every 50 batches.

---

## What this Brief deliberately does NOT do

- No edits to `engines/crypto_data_collector.py` or any other source file
- No new code (no new collectors, no new utility scripts)
- No new scheduled tasks (those are a separate Brief — see Recovery Plan §1.2 item 2 for funding/fear-greed scheduled collectors going forward)
- No interaction with the `phase3_models.joblib` retrain (separate cycle once `funding_rates` is populated)
- No edits to `services/*.bat` or `services/*.ps1` files (the trades + crypto_1m collectors may share the OrderBook duration race per Cycle 8 retro §6, but auditing them is a separate Brief)
- No commits along the way; the retro is the artifact, the changed `data/crypto_data.db` files are not committed (they're gitignored)

---

## References

- `claude/handoffs/RECOVERY_PLAN_post_disk_failure.md` §3.3 Tier 1 items 2-4 — strategic context for which tables matter most and why
- `engines/crypto_data_collector.py` — already-implemented collector with all six subcommands; no changes needed
- `claude/retros/RETRO_order_book_duration_fix.md` — Cycle 8 retro confirming the parallel OrderBook collector pattern is healthy
- `claude/CLAUDE_CODE_RULES.md` — rules 9-15 (progress), rule 16 (validation), rule 19 (ASCII)
