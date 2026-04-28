# Implementation Brief: Order Book Snapshot Collector (v8.2)

**Series:** praxis
**Priority:** P0 (unlocks microstructure features -- the signal that dominates crypto prediction per literature)
**Mode:** B (live API calls + scheduled task creation + DB writes)

**Estimated Scope:** M (60-120 min: new collector + new table + scheduled task + test runs)
**Estimated Cost:** none (Binance order book API is free)
**Estimated Data Volume:** approximately 1 snapshot every 10 seconds (~8,600/day/asset); at 20 levels each, ~172,000 floats/day/asset; SQLite row growth estimated at 8,600 rows/day/asset
**Kill switch:** N/A (this is a new persistent collector, not a one-shot run)

---

## Context

The crypto LSTM literature (2024-2025) converges on a single dominant finding: **order book microstructure features explain most predictive power in short-horizon crypto price movement**. MDPI Applied Sciences Oct 2025 reports that removing order book depth caused a 68% profit decline while removing technical indicators reduced profit by only 4%. The arXiv Jun 2025 paper titled "Better Inputs Matter More Than Stacking Another Hidden Layer" found the same universal pattern.

Praxis currently collects:
- 1-minute OHLCV (Binance, 180d history now available)
- Fear & Greed Index
- Funding rates
- BTC on-chain metrics
- Polymarket price snapshots

Praxis does NOT currently collect:
- Order book snapshots (bid/ask levels)
- Trade flow (buyer-initiated vs seller-initiated volume)

This Brief starts continuous live collection of order book snapshots so the data accumulates while other work (v8.1, funding strategies, convergence detectors) proceeds. Needs at least 2-4 weeks of data before microstructure-feature modeling becomes meaningful, so the clock needs to start now.

Trade flow (buyer vs seller tagging) is a follow-up Brief; this one focuses on order book depth alone because it's the single highest-value signal per literature.

---

## Objective

Create a scheduled task that captures Binance order book snapshots for BTC/USDT and ETH/USDT every ~10 seconds, stores them in a new SQLite table, and runs 24/7 like existing Praxis collectors.

---

## Detailed Spec

### Step 1: New DB Table

Add to `data/crypto_data.db`:

```sql
CREATE TABLE IF NOT EXISTS order_book_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timestamp INTEGER NOT NULL,       -- Unix seconds
    datetime TEXT NOT NULL,            -- ISO string for human inspection
    mid_price REAL NOT NULL,           -- (best_bid + best_ask) / 2
    best_bid REAL NOT NULL,
    best_ask REAL NOT NULL,
    spread REAL NOT NULL,              -- (best_ask - best_bid)
    spread_bps REAL NOT NULL,          -- spread in basis points relative to mid
    
    -- Top 10 bid levels (price, volume)
    bid_price_1 REAL, bid_vol_1 REAL,
    bid_price_2 REAL, bid_vol_2 REAL,
    bid_price_3 REAL, bid_vol_3 REAL,
    bid_price_4 REAL, bid_vol_4 REAL,
    bid_price_5 REAL, bid_vol_5 REAL,
    bid_price_6 REAL, bid_vol_6 REAL,
    bid_price_7 REAL, bid_vol_7 REAL,
    bid_price_8 REAL, bid_vol_8 REAL,
    bid_price_9 REAL, bid_vol_9 REAL,
    bid_price_10 REAL, bid_vol_10 REAL,
    
    -- Top 10 ask levels
    ask_price_1 REAL, ask_vol_1 REAL,
    ask_price_2 REAL, ask_vol_2 REAL,
    ask_price_3 REAL, ask_vol_3 REAL,
    ask_price_4 REAL, ask_vol_4 REAL,
    ask_price_5 REAL, ask_vol_5 REAL,
    ask_price_6 REAL, ask_vol_6 REAL,
    ask_price_7 REAL, ask_vol_7 REAL,
    ask_price_8 REAL, ask_vol_8 REAL,
    ask_price_9 REAL, ask_vol_9 REAL,
    ask_price_10 REAL, ask_vol_10 REAL,
    
    -- Derived liquidity aggregates (pre-computed for fast query)
    bid_volume_top10 REAL NOT NULL,    -- sum of bid_vol_1 .. bid_vol_10
    ask_volume_top10 REAL NOT NULL,    -- sum of ask_vol_1 .. ask_vol_10
    order_imbalance_top10 REAL NOT NULL,  -- (bid_top10 - ask_top10) / (bid_top10 + ask_top10)
    
    UNIQUE(asset, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ob_asset_timestamp 
    ON order_book_snapshots(asset, timestamp DESC);
```

**Why 10 levels and not more:** 10 levels is the sweet spot in the literature. Binance serves up to 5000, but most papers use 5-20. Depth beyond level 10 is noisy (large limit orders that may cancel). Storing 10 levels keeps the row size manageable (40 floats per snapshot) while capturing signal.

**Why pre-compute aggregates (bid_volume_top10, order_imbalance_top10):** most downstream queries will want these rather than the raw levels. Pre-computing eliminates a hot code path for feature extraction later.

### Step 2: New Collector Function

Add to `engines/crypto_data_collector.py`:

```python
def collect_order_book_snapshot(asset, exchange, conn):
    """Fetch current Binance order book for `asset` and insert one row.
    
    Returns (rows_inserted, error_msg).
    """
    # Symbol mapping
    symbol_map = {"BTC": "BTC/USDT", "ETH": "ETH/USDT"}
    symbol = symbol_map[asset]
    
    ob = exchange.fetch_order_book(symbol, limit=10)
    # ob has keys: 'bids' (list of [price, vol]), 'asks' (same), 'timestamp', 'datetime'
    
    ts = ob.get("timestamp") or int(time.time() * 1000)
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
    
    bids = ob["bids"][:10]
    asks = ob["asks"][:10]
    
    if not bids or not asks:
        return (0, "empty order book")
    
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    spread_bps = (spread / mid) * 10000 if mid else 0
    
    # Pad short sides with zeros
    while len(bids) < 10:
        bids.append([0, 0])
    while len(asks) < 10:
        asks.append([0, 0])
    
    bid_top10 = sum(b[1] for b in bids)
    ask_top10 = sum(a[1] for a in asks)
    imbalance = (bid_top10 - ask_top10) / (bid_top10 + ask_top10) if (bid_top10 + ask_top10) > 0 else 0
    
    # Flatten for INSERT
    bid_flat = [v for b in bids for v in b]  # [price_1, vol_1, price_2, vol_2, ...]
    ask_flat = [v for a in asks for v in a]
    
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO order_book_snapshots (
                asset, timestamp, datetime, mid_price, best_bid, best_ask, spread, spread_bps,
                bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3,
                bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6,
                bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9,
                bid_price_10, bid_vol_10,
                ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3,
                ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6,
                ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9,
                ask_price_10, ask_vol_10,
                bid_volume_top10, ask_volume_top10, order_imbalance_top10
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?)
        """, [
            asset, ts // 1000, dt, mid, best_bid, best_ask, spread, spread_bps,
            *bid_flat, *ask_flat,
            bid_top10, ask_top10, imbalance
        ])
        conn.commit()
        return (cursor.rowcount, None)
    except Exception as e:
        return (0, str(e))
```

### Step 3: New CLI Subcommand `collect-order-book`

```python
p_ob = subs.add_parser("collect-order-book", 
    help="Collect one order book snapshot for each specified asset.")
p_ob.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
```

The implementation loops through assets, calls `collect_order_book_snapshot` for each, prints a compact status line.

### Step 4: Background Polling Mode

Add a separate subcommand for continuous collection without scheduled-task overhead:

```python
p_obc = subs.add_parser("collect-order-book-loop", 
    help="Run continuous order book collection at specified interval.")
p_obc.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
p_obc.add_argument("--interval", type=int, default=10, 
    help="Seconds between snapshots (default 10)")
p_obc.add_argument("--duration", type=int, default=None,
    help="Total seconds to run before exiting (default: forever)")
```

This runs in a simple `while True: sleep(interval)` loop. Handles KeyboardInterrupt and SIGTERM cleanly. The scheduled task will kick off a long-running invocation (duration=3600 or similar) every hour.

### Step 5: Windows Scheduled Task

Create `services/order_book_collector_service.bat`:

```batch
@echo off
cd /d C:\Data\Development\Python\McTheoryApps\praxis
set PYTHONUTF8=1
call .venv\Scripts\activate.bat
python -u -m engines.crypto_data_collector collect-order-book-loop --assets BTC ETH --interval 10 --duration 3600 > logs\order_book_collector.log 2>&1
```

And a PowerShell registration script `services/register_order_book_task.ps1`:

```powershell
# Run every hour. Each invocation runs for duration=3600s, so back-to-back coverage.
$taskName = "PraxisOrderBookCollector"
$action = New-ScheduledTaskAction -Execute "C:\Data\Development\Python\McTheoryApps\praxis\services\order_book_collector_service.bat"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Collect Binance order book snapshots every 10 seconds for BTC/USDT and ETH/USDT"
```

Requires Administrator PowerShell to register.

### Step 6: Verification

After setup, verify data is landing:

```powershell
# Run a 60-second test collection
python -m engines.crypto_data_collector collect-order-book-loop --duration 60

# Check row counts
python -m engines.crypto_data_collector status

# Sample a recent row
python -c "import sqlite3; conn = sqlite3.connect('data/crypto_data.db'); c = conn.execute('SELECT * FROM order_book_snapshots ORDER BY timestamp DESC LIMIT 1'); cols = [d[0] for d in c.description]; row = c.fetchone(); print(dict(zip(cols, row)))"
```

### Step 7: Update `status` Command

If `engines/crypto_data_collector.py` has a `status` command that reports table counts, add `order_book_snapshots` to the list.

---

## Progress Reporting (per CLAUDE_CODE_RULES.md rules 9-15)

This Brief is mostly code additions, not a long-running training run. Progress cadence:

- **T+0:** restate scope; confirm no existing order book collector
- **After each Step completes:** brief status update
- **Before scheduled task registration:** explicit announcement ("about to require Administrator PowerShell")
- **Post-verification:** report row count after 60-second test collection

If any Step takes longer than expected, normal progress-check cadence applies.

---

## Acceptance Criteria

- [ ] `order_book_snapshots` table created in `data/crypto_data.db` with proper schema
- [ ] `collect_order_book_snapshot` function added to collector, handles empty-book edge case
- [ ] `collect-order-book` subcommand runs, inserts one row per specified asset
- [ ] `collect-order-book-loop` subcommand runs continuously, handles interrupts cleanly
- [ ] `services/order_book_collector_service.bat` created, ASCII-only, tested manually
- [ ] `services/register_order_book_task.ps1` created, documented
- [ ] Scheduled task `PraxisOrderBookCollector` registered (or registration-instructions given to Jeff if PowerShell-admin needed)
- [ ] 60-second test run produces >= 10 rows per asset
- [ ] Sample row from DB shows sensible values: mid_price near market, spread_bps single-digit, imbalance in (-1, +1)
- [ ] `status` command reports the new table
- [ ] AST parse + ASCII check pass on all new/modified Python + batch files
- [ ] No impact on existing collectors (PraxisLiveCollector, PraxisSmartMoney, PraxisCrypto1mCollector, etc.)

---

## Known Pitfalls

- **Binance rate limits.** `fetch_order_book(limit=10)` is weight=5 per call. 2 assets every 10 sec = 60 calls/min = weight 300/min. Binance's limit is 1200/min weight for public endpoints. Well under cap. BUT: if in the future we add more assets or shorten the interval, recheck the math.
- **ccxt clock drift.** `ob.get("timestamp")` from Binance may be a few ms off from local clock. For 10-second cadence this doesn't matter. Use Binance's timestamp when available; fall back to local `time.time()` only if missing.
- **Empty order book on low-volume symbols.** Very unlikely for BTC/USDT and ETH/USDT, but the function handles it by returning (0, "empty order book") and NOT raising.
- **Scheduled task auth.** Jeff needs Administrator PowerShell to register. Brief includes that step; if Code runs into permission error, stop and ask Jeff to run the registration script.
- **DB concurrency.** Multiple Python processes writing to `data/crypto_data.db` concurrently (this collector + the 1m collector + smart money) is OK because SQLite with WAL mode serializes writes correctly. Verify `PRAGMA journal_mode` is WAL on the DB (check during Step 1).
- **Cost of inserts at 10 sec intervals.** 8,600 inserts/day/asset = 17,200/day total. Over a year that's 6.3M rows. Manageable for SQLite but monitor size. If size grows uncomfortable, we can partition to a separate DB later.
- **Collector never commits between iterations within a run.** The function above calls `conn.commit()` per insert for safety, which is fine at 10-sec cadence. At higher cadence (e.g., 1 sec) we'd batch commits.
- **Filesystem buffering.** The `--duration 3600` approach with hourly scheduled restarts ensures we don't have a single 30-day Python process accumulating memory or file handles. Fresh process every hour. Small re-launch overhead (~200ms) is negligible.
- **No ASCII compliance needed for the PowerShell file.** (`.ps1` files are not piped through cp1252 the way scheduled task scripts are.) But the `.bat` file IS piped, so no em dashes or emoji there.

---

## What NOT to change

- Existing collectors (1m, funding, sentiment, smart money, live Polymarket)
- Existing DB tables
- Existing scheduled tasks
- Engines that consume existing data
- Any model artifacts

---

## References

- MDPI Applied Sciences Oct 2025 (order book features contribute 73% of profit; removing order book depth caused 68% profit decline)
- arXiv Jun 2025 "Better Inputs Matter More Than Stacking Another Hidden Layer"
- arXiv Jan 2026 "Explainable Patterns in Cryptocurrency Microstructure" (universal patterns across 5 cryptos)
- Binance API docs: https://binance-docs.github.io/apidocs/spot/en/#order-book
- ccxt `fetch_order_book` docs
- Similar existing collector to model after: `engines/crypto_data_collector.py` `collect_ohlcv_1m` function
- Similar existing service: `services/crypto_1m_collector_service.bat`
- Workflow modes: `claude/WORKFLOW_MODES_PRAXIS.md`
- Progress rules: `claude/CLAUDE_CODE_RULES.md` rules 9-15
