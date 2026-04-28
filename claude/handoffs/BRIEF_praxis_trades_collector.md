# Implementation Brief: Trade Flow Collector (v8.2.1)

**Series:** praxis
**Priority:** P0 (parallels v8.2 data accumulation -- extending while order book data is still thin is cheapest now, not later)
**Mode:** B (live API calls + scheduled task creation + DB writes)

**Estimated Scope:** M (60-120 min: similar structure to v8.2, modeled on the already-landed collector)
**Estimated Cost:** none (Binance trades API is free)
**Estimated Data Volume:** variable by asset activity; estimate 50-200 trades/minute for BTC/USDT during active hours, lower overnight. Expect ~100-300k trade rows/day/asset.
**Kill switch:** N/A (new persistent collector, not a one-shot run)

---

## Context

V8.1 triple-barrier probe confirmed with high confidence that the 25-feature OHLCV set cannot predict 5-min directional moves for BTC (Case C, accuracy 41.4% vs 42.0% baseline). V8.2 landed the order book collector -- data is accumulating at ~12 rows/min/asset as scheduled. The retro explicitly flagged extending to trade flow collection as a parallel action:

> "Do we extend the collector to capture trade flow (buyer vs seller tagging)? Brief notes this was deferred; now may be the time."

The literature consistently ranks **order flow imbalance (OFI)** and **buyer-vs-seller initiated volume** as the second-most-predictive microstructure signal after order book depth (arXiv Jun 2025 "Better Inputs Matter More Than Stacking Another Hidden Layer"; arXiv Jan 2026 "Explainable Patterns in Cryptocurrency Microstructure"). Starting trade flow collection NOW means v8.3 (when it comes in 2-4 weeks) has both depth AND flow data accumulated simultaneously, not sequentially.

This Brief adds the trade flow collector using the same structural pattern as the order book collector -- new table, new subcommands, new scheduled task, zero impact on existing collectors.

---

## Objective

Create a persistent collector that records every trade (as reported by Binance) for BTC/USDT and ETH/USDT, with side (buyer vs seller initiated) tagged. Data accumulates 24/7. First-class microstructure feature source for v8.3.

---

## What Trade Flow Gives Us

Binance's public trades API returns every completed trade with:
- price, amount, timestamp
- **isBuyerMaker** flag: `True` means the buyer was the resting limit order (so a seller hit the bid, flagging this as seller-initiated). `False` means the seller was the resting limit order (buyer hit the ask, buyer-initiated).

From this we can derive:
- **Signed volume:** +amount if buyer-initiated, -amount if seller-initiated
- **VPIN-like features:** buyer-initiated volume / total volume over a window
- **Trade intensity:** trades per second
- **Size distribution:** small trades vs whale-size trades
- **Aggressor imbalance:** core OFI signal

These are the features the literature says matter. We store raw trades; feature computation happens downstream.

---

## Detailed Spec

### Step 1: New DB Table

Add to `data/crypto_data.db`:

```sql
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    trade_id INTEGER NOT NULL,          -- Binance's own trade ID, for dedup
    timestamp INTEGER NOT NULL,          -- Unix milliseconds (higher resolution than order book)
    datetime TEXT NOT NULL,              -- ISO string for human inspection
    price REAL NOT NULL,
    amount REAL NOT NULL,                -- base asset quantity (e.g. BTC)
    quote_amount REAL NOT NULL,          -- quote asset (e.g. USDT), = price * amount
    is_buyer_maker INTEGER NOT NULL,     -- 1 if buyer was passive (seller-initiated trade), 0 if buyer was active (buyer-initiated)
    side TEXT NOT NULL,                  -- convenience: 'buy' or 'sell' from taker's perspective
    
    UNIQUE(asset, trade_id)
);

CREATE INDEX IF NOT EXISTS idx_trades_asset_timestamp 
    ON trades(asset, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_asset_tradeid 
    ON trades(asset, trade_id DESC);
```

**Why `side` is stored redundantly:** `is_buyer_maker` is the raw Binance field, but downstream queries want to ask "was this trade buyer-initiated?" Pre-computing `side = 'buy'` if `is_buyer_maker = 0` else `'sell'` makes those queries simpler and faster. Storage cost is trivial.

**Why `trade_id` in UNIQUE:** Binance assigns monotonically increasing trade IDs per symbol. Pagination cursoring works by asking "give me trades with ID > last_seen_id". UNIQUE constraint protects against duplicate inserts on retry or API quirks.

### Step 2: New Collector Functions

Add to `engines/crypto_data_collector.py`:

```python
def collect_recent_trades(asset, exchange, conn, last_trade_id=None):
    """Fetch recent trades for `asset` since `last_trade_id` and insert rows.
    
    If last_trade_id is None, fetches the most recent 1000 trades (Binance max per call).
    Otherwise fetches trades with trade_id > last_trade_id, up to 1000.
    
    Returns (rows_inserted, latest_trade_id, error_msg).
    """
    symbol_map = {"BTC": "BTC/USDT", "ETH": "ETH/USDT"}
    symbol = symbol_map[asset]
    
    # ccxt fetch_trades supports `since` (timestamp ms) and `limit`.
    # For ID-based cursoring, we use Binance-specific params.
    params = {}
    if last_trade_id is not None:
        # Binance-specific: fromId fetches trades starting at this ID (inclusive)
        params["fromId"] = last_trade_id + 1
    
    try:
        trades = exchange.fetch_trades(symbol, limit=1000, params=params)
    except Exception as e:
        return (0, last_trade_id, f"fetch error: {e}")
    
    if not trades:
        return (0, last_trade_id, None)
    
    cursor = conn.cursor()
    inserted = 0
    max_id = last_trade_id or 0
    
    for tr in trades:
        # ccxt normalizes trade structure; for Binance spot we expect:
        # tr['id'] (str) -- Binance trade ID
        # tr['timestamp'] (int ms)
        # tr['datetime'] (ISO str)
        # tr['price'] (float)
        # tr['amount'] (float)
        # tr['side'] ('buy' or 'sell' from taker's perspective)
        # tr['info']['isBuyerMaker'] (bool) in the raw Binance payload
        try:
            trade_id = int(tr["id"])
            ts = int(tr["timestamp"])
            dt = tr["datetime"]
            price = float(tr["price"])
            amount = float(tr["amount"])
            quote_amount = price * amount
            side = tr["side"]  # ccxt gives 'buy' or 'sell' from taker perspective
            # is_buyer_maker = True if the buyer was passive (so the trade was seller-initiated, side='sell')
            is_buyer_maker = 1 if side == "sell" else 0
        except (KeyError, ValueError, TypeError) as e:
            continue  # skip malformed
        
        cursor.execute("""
            INSERT OR IGNORE INTO trades (
                asset, trade_id, timestamp, datetime, price, amount,
                quote_amount, is_buyer_maker, side
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (asset, trade_id, ts, dt, price, amount, quote_amount, is_buyer_maker, side))
        
        if cursor.rowcount > 0:
            inserted += 1
        if trade_id > max_id:
            max_id = trade_id
    
    conn.commit()
    return (inserted, max_id, None)


def get_latest_trade_id(asset, conn):
    """Return the most recent trade_id stored for an asset, or None if none exist."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT MAX(trade_id) FROM trades WHERE asset = ?",
        (asset,)
    )
    result = cursor.fetchone()
    return result[0] if result and result[0] is not None else None
```

### Step 3: New CLI Subcommands

Two new subcommands, modeled on the order book pattern:

**One-shot:**
```python
p_t = subs.add_parser("collect-trades",
    help="Collect recent trades for each specified asset (one-shot, up to 1000 per asset).")
p_t.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
```

**Continuous loop:**
```python
p_tc = subs.add_parser("collect-trades-loop",
    help="Run continuous trade collection at specified interval.")
p_tc.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
p_tc.add_argument("--interval", type=int, default=30,
    help="Seconds between fetch batches (default 30 -- trades accumulate between fetches)")
p_tc.add_argument("--duration", type=int, default=None,
    help="Total seconds to run before exiting (default: forever)")
```

**Why interval default is 30 seconds, not 10:**

Trade flow is different from order book. Order book is a *snapshot* -- the state right now. You want frequent snapshots because state changes continuously.

Trades are *events*. Binance records every trade. You don't need to poll every second; you need to poll often enough that you don't miss trades between calls. At ~200 trades/min during active BTC hours, 30-second intervals mean ~100 trades per fetch -- well under the 1000-per-call limit. During quiet hours (overnight), 30 seconds is plenty.

**Adaptive handling:** if a fetch returns exactly 1000 rows, we know we may have missed trades that happened faster than the polling interval. The loop should IMMEDIATELY refetch (no sleep) to catch up. Only sleep to interval when the fetch returns < 1000 rows. This keeps us current even during volume spikes. Implement this in the loop.

### Step 4: Loop Implementation Notes

The continuous loop pattern:

```python
def cmd_collect_trades_loop(args):
    conn = sqlite3.connect(DB_PATH)
    exchange = ccxt.binance({"enableRateLimit": True})
    
    # Initialize last_trade_id per asset from DB state
    last_ids = {}
    for asset in args.assets:
        last_ids[asset] = get_latest_trade_id(asset, conn)
        print(f"  {asset}: starting from trade_id > {last_ids[asset] or '(latest)'}", flush=True)
    
    start_time = time.time()
    total_inserted = {a: 0 for a in args.assets}
    iteration = 0
    
    try:
        while True:
            iteration += 1
            elapsed = time.time() - start_time
            if args.duration is not None and elapsed >= args.duration:
                break
            
            any_saturated = False  # track if we hit 1000-trade limit for adaptive loop
            for asset in args.assets:
                rows, max_id, err = collect_recent_trades(
                    asset, exchange, conn, last_trade_id=last_ids[asset]
                )
                if err:
                    print(f"  [{asset}] ERROR iter {iteration}: {err}", flush=True)
                    continue
                total_inserted[asset] += rows
                if rows >= 1000:
                    any_saturated = True
                last_ids[asset] = max_id
            
            # Periodic summary (every 6 iterations at 30s = ~3 min)
            if iteration % 6 == 0:
                totals_str = " ".join(f"{a}={total_inserted[a]}" for a in args.assets)
                print(f"  iter {iteration} elapsed={int(elapsed)}s totals: {totals_str}", flush=True)
            
            # Adaptive sleep: if any asset hit the 1000-trade cap, refetch immediately
            if not any_saturated:
                time.sleep(args.interval)
            # else: immediately loop without sleeping to catch up
    except KeyboardInterrupt:
        print("\n  Interrupted; flushing final state...", flush=True)
    finally:
        # Final summary
        elapsed = time.time() - start_time
        print(f"\n  DONE. Ran {elapsed:.1f}s, {iteration} iterations")
        for asset in args.assets:
            print(f"    {asset}: {total_inserted[asset]} trades inserted")
        conn.close()
```

### Step 5: Windows Scheduled Task

Create `services/trades_collector_service.bat`:

```batch
@echo off
cd /d C:\Data\Development\Python\McTheoryApps\praxis
set PYTHONUTF8=1
call .venv\Scripts\activate.bat
python -u -m engines.crypto_data_collector collect-trades-loop --assets BTC ETH --interval 30 --duration 3600 > logs\trades_collector.log 2>&1
```

Create `services/register_trades_task.ps1` modeled on `register_order_book_task.ps1`:

```powershell
$taskName = "PraxisTradesCollector"

# Remove any stale prior task
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction -Execute "C:\Data\Development\Python\McTheoryApps\praxis\services\trades_collector_service.bat"

# Hourly repetition, back-to-back 3600s invocations
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 65)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited

Register-ScheduledTask -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Collect Binance trades (side-tagged) for BTC/USDT and ETH/USDT continuously"

Write-Host "============================================================="
Write-Host "  Task registered: $taskName"
Write-Host "  Trigger: Every 1 hour (back-to-back 3600 s invocations)"
Write-Host "  Collects: BTC + ETH trades with buyer/seller tagging"
Write-Host "  Logs: C:\Data\Development\Python\McTheoryApps\praxis\logs\trades_collector.log"
Write-Host "============================================================="
Write-Host "  Start immediately:"
Write-Host "  Start-ScheduledTask -TaskName $taskName"
```

**Same Admin-PowerShell gotcha as v8.2.** Brief anticipates: if Code hits Access-Denied, fall back to giving Jeff the one-line command to run in elevated PowerShell, exactly as v8.2 resolved.

### Step 6: Extend `status` Command

Add `trades` to the table map. Status output should now include trade count + date range alongside the existing tables.

### Step 7: Verification

After setup:

```powershell
# 60-second test
python -m engines.crypto_data_collector collect-trades-loop --duration 60 --interval 30

# Expected: 2 full iterations (at t=0 and t=30), so up to 2000 trades per asset if volume is high,
# or just recent trades if activity is low. During BTC active hours expect 100-500 trades per asset.

# Verify via DB
python -m engines.crypto_data_collector status

# Sample a recent trade
python -c "import sqlite3; conn = sqlite3.connect('data/crypto_data.db'); c = conn.execute('SELECT * FROM trades WHERE asset=\"BTC\" ORDER BY trade_id DESC LIMIT 5'); cols = [d[0] for d in c.description]; [print(dict(zip(cols, r))) for r in c.fetchall()]"
```

Verify sample rows:
- `side` in {'buy', 'sell'}
- `is_buyer_maker` in {0, 1} and consistent with `side`
- `price` within 0.1% of current market (sanity)
- `amount > 0`
- `quote_amount == price * amount` (within float tolerance)
- `trade_id` monotonic when sorted

---

## Progress Reporting (per CLAUDE_CODE_RULES.md rules 9-15)

This Brief is code additions + scheduled task setup, not long-running training. Progress cadence:

- **T+0:** restate scope; confirm `PraxisOrderBookCollector` is running cleanly (do NOT touch it)
- **After each Step completes:** brief status update
- **Before scheduled task registration:** announce ("about to require Administrator PowerShell")
- **Post-verification:** report row count after 60-second test collection + sample row sanity check

Total Brief should complete inside 90 min. Kill switch N/A since this is just code+setup; the only long-running artifact is the new scheduled task itself (which runs forever).

---

## Acceptance Criteria

- [ ] `trades` table created in `data/crypto_data.db` with proper schema + 2 indexes
- [ ] `collect_recent_trades` function handles errors and ID-based cursoring correctly
- [ ] `get_latest_trade_id` helper returns most recent stored ID per asset
- [ ] `collect-trades` subcommand runs, inserts rows
- [ ] `collect-trades-loop` subcommand runs continuously with adaptive sleep (immediate refetch on 1000-cap hit, sleep to interval otherwise)
- [ ] `services/trades_collector_service.bat` created, ASCII-only
- [ ] `services/register_trades_task.ps1` created, modeled on order book registration
- [ ] Scheduled task `PraxisTradesCollector` registered (or registration-instructions to Jeff if PowerShell-admin needed)
- [ ] 60-second test run produces some trades (expect >= 50 per asset during active hours; may be lower overnight -- just confirm > 0 per asset and no errors)
- [ ] Sample row from DB passes all sanity checks listed in Step 7
- [ ] `status` command reports the new `trades` table
- [ ] AST parse + ASCII check pass on all new/modified Python + batch files
- [ ] Zero impact on existing collectors, especially `PraxisOrderBookCollector` (still running, still writing)

---

## Known Pitfalls

- **`fromId` parameter is Binance-specific.** Other exchanges don't support it. ccxt passes it through `params`. If we expand to other exchanges later, refactor needed.
- **Trade ID gaps.** Binance sometimes has small gaps in trade IDs (internal to their system). Don't treat gaps as data loss; just use whatever `max_id` came back.
- **Binance rate limit budget.** `fetch_trades(limit=1000)` is weight=10. At 30s interval for 2 assets: 4 calls/min = weight 40/min. Binance spot cap is 1200/min weight. Negligible.
- **Timezone handling.** Binance trade timestamps are UTC ms. Store as UTC throughout; no conversion. `datetime` field is just a human-readable mirror of the timestamp.
- **Startup state.** When the loop first runs, there's no `last_trade_id` in DB -- so `fromId` is unset and we get "the last 1000 trades" as of right now. Fine. Subsequent iterations cursor forward by ID.
- **Adaptive refetch safety.** In the rare case of a sustained saturation (every call returns 1000), the loop will poll as fast as `enableRateLimit` permits. This is the desired behavior, and ccxt's rate limiter will prevent us from hammering Binance. If this causes problems (burning CPU during meme-coin crash, say), Code can revisit -- but don't preemptively throttle.
- **DB concurrency.** WAL mode already confirmed working from v8.2. Multiple writers (1m OHLCV + smart money + order book + trades) all serialize cleanly.
- **Storage growth.** At 300k trades/day/asset, 2 assets, 100 bytes/row: ~60 MB/day, ~22 GB/year. Larger than order book. For a 2-4 week window (our v8.3 horizon) this is 850 MB - 1.7 GB. Fine for SQLite but worth noting. If we keep it running indefinitely, consider archiving older data to separate files at some point.
- **ETH trade volume is lower than BTC.** Expect ~30-50% of BTC trade count. Still plenty.

---

## What NOT to change

- Existing `PraxisOrderBookCollector` task or its files
- Existing collectors (1m OHLCV, funding, sentiment, smart money, live Polymarket)
- Existing DB tables
- Existing scheduled tasks (5 pre-existing + PraxisOrderBookCollector = 6 total)
- Engines that consume existing data
- Model artifacts
- `engines/intrabar_predictor.py` (all v8.1+ work stays frozen)

---

## Progress Check Cadence (explicit timestamps per new CLAUDE_CODE_RULES)

Mechanical cadence for this Brief:

- **T+0 (session start):** restate scope, confirm `PraxisOrderBookCollector` task state, announce no-touch policy
- **After Step 1:** table created, confirm via schema dump
- **After Step 2-5:** incremental "N of 7 steps complete" with description
- **Before task registration:** announce Admin requirement
- **At 60-second test:** report trades count per asset + sample row sanity
- **Post-verification:** final summary

If any step unexpectedly takes > 10 min, full out-of-cadence report.

---

## References

- v8.2 order book brief (pattern to follow exactly): `claude/handoffs/BRIEF_praxis_order_book_collector.md`
- v8.2 retro (service file naming, registration workflow): `claude/retros/RETRO_praxis_intrabar_confluence.md` section 2
- Existing order book collector (structural template): `engines/crypto_data_collector.py` `collect_order_book_snapshot` and `cmd_collect_order_book_loop`
- Existing order book service (batch file template): `services/order_book_collector_service.bat`
- Existing order book registration (PowerShell template): `services/register_order_book_task.ps1`
- Binance trades API reference: https://binance-docs.github.io/apidocs/spot/en/#recent-trades-list
- ccxt `fetch_trades` with `fromId`: https://docs.ccxt.com/en/latest/manual.html#trades
- Literature on aggressor imbalance / VPIN as microstructure features: Easley, Lopez de Prado, O'Hara 2012 "Flow Toxicity and Liquidity in a High-Frequency World"
- Workflow modes: `claude/WORKFLOW_MODES_PRAXIS.md`
- Progress rules: `claude/CLAUDE_CODE_RULES.md` rules 9-15
