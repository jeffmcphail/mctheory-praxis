# Implementation Brief: Praxis MCP Server v0.1

**Series:** praxis
**Priority:** P1 (unblocks Chat-side data inspection; dramatically increases signal on what's actually in the DB)
**Mode:** B (new code module + Claude Desktop config changes + local testing)

**Estimated Scope:** M (90-150 min: ~300 lines of server code + install docs + testing)
**Estimated Cost:** none
**Estimated Data Volume:** read-only queries against existing `data/crypto_data.db`; no writes anywhere
**Kill switch:** N/A (one-shot server that starts on demand)

---

## Context

Right now, Chat (the non-Code Claude) can only see what Jeff pastes into conversations. This creates two recurring friction points:

1. Jeff has to manually dump query results for Chat to reason about them. Every "check if the order book collector is writing cleanly" requires Jeff to run a command and paste output.
2. Code retros summarize what Code found, but Chat can't independently verify or explore the data. If a retro says "trade flow looks healthy," Chat has no way to spot-check.

An MCP server exposing `data/crypto_data.db` as read-only tools removes both. Chat calls `get_collector_health()` and sees which collectors are stale. Chat calls `get_recent_trades("BTC", 60)` and sees actual whale activity. Sanity checks that currently require a Code round-trip become instant Chat-side checks.

This Brief builds v0.1 of that MCP server, scoped to `data/crypto_data.db`. `smart_money.db` is explicitly out of scope and gets v0.2.

The MCP server follows a targeted-tools design (not raw-SQL-passthrough). Tools encode the correct joins, filters, and time-range semantics once, so Chat doesn't have to re-derive them. A single `raw_query()` escape hatch exists for genuinely ad-hoc analysis, with a hard row cap.

---

## Objective

Create `servers/praxis_mcp/` containing a Python MCP server that exposes `data/crypto_data.db` read-only to any MCP client (Claude Desktop primarily). Include install docs, a manual CLI smoke test, and a config snippet for `claude_desktop_config.json`.

---

## Detailed Spec

### Step 1: Directory structure

```
servers/
  praxis_mcp/
    __init__.py
    server.py              # the MCP server entry point
    db.py                  # SQLite read-only connection helper
    tools/
      __init__.py
      meta.py              # list_tables, table_stats, get_collector_health
      ohlcv.py             # get_recent_ohlcv
      order_book.py        # get_order_book_snapshot, get_order_book_range
      trades.py            # get_recent_trades, get_trade_flow_summary
      funding.py           # get_funding_rate_history
      raw.py               # raw_query (escape hatch, row-capped)
    README.md              # install + Claude Desktop config
    test_smoke.py          # CLI-invocable smoke test
```

Reason for splitting tools into a submodule: keeps each tool file under ~80 lines, makes adding v0.2 tools (smart_money, Polymarket) a matter of dropping in new files, and isolates failures.

### Step 2: Dependencies

Add to `pyproject.toml` or `requirements.txt` (whichever the repo uses):

```
mcp>=1.2.0
```

The `mcp` package is the official Anthropic SDK. It ships with FastMCP (the decorator-based high-level API) plus the lower-level server primitives. We'll use FastMCP for speed.

### Step 3: Server scaffold (`servers/praxis_mcp/server.py`)

```python
"""
Praxis MCP Server v0.1

Exposes data/crypto_data.db as read-only MCP tools so Claude Desktop (or any
MCP client) can query collected microstructure, OHLCV, funding, and sentiment
data directly without needing Code round-trips.

Transport: stdio (local subprocess; launched by Claude Desktop on demand).
Access mode: read-only via SQLite URI mode=ro (belt) plus no UPDATE/INSERT/
DELETE/DROP tool implementations (suspenders).

IMPORTANT: Never write to stdout. stdout is the MCP JSON-RPC channel; any
print() to stdout will corrupt the protocol. Use sys.stderr or logging
configured to stderr for all diagnostic output.
"""

import sys
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from servers.praxis_mcp.tools import meta, ohlcv, order_book, trades, funding, raw

# Diagnostic logging to stderr only
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[praxis-mcp] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Resolve the DB path relative to the project root (parent of servers/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "crypto_data.db"

if not DB_PATH.exists():
    log.error(f"DB not found at {DB_PATH}")
    # Still start the server; the tools will report the issue per-call rather
    # than failing hard at import time.

mcp = FastMCP("praxis")

# Register tools (each module exposes a `register(mcp, db_path)` function)
meta.register(mcp, DB_PATH)
ohlcv.register(mcp, DB_PATH)
order_book.register(mcp, DB_PATH)
trades.register(mcp, DB_PATH)
funding.register(mcp, DB_PATH)
raw.register(mcp, DB_PATH)


if __name__ == "__main__":
    log.info(f"Starting Praxis MCP server (db={DB_PATH})")
    mcp.run(transport="stdio")
```

### Step 4: Read-only DB helper (`servers/praxis_mcp/db.py`)

```python
"""Read-only SQLite connection helper.

The URI `file:{path}?mode=ro` instructs SQLite to open in read-only mode.
Any attempted write raises sqlite3.OperationalError: attempt to write a
readonly database. This is enforced at the SQLite layer, below anything
Python can accidentally bypass.
"""

import sqlite3
from pathlib import Path


def connect_ro(db_path: Path) -> sqlite3.Connection:
    """Open a read-only connection to the given SQLite DB."""
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row  # dict-like row access
    return conn
```

### Step 5: Meta tools (`servers/praxis_mcp/tools/meta.py`)

```python
from datetime import datetime, timezone
from pathlib import Path

from servers.praxis_mcp.db import connect_ro


def register(mcp, db_path: Path):
    @mcp.tool()
    def list_tables() -> dict:
        """List all tables in the Praxis crypto_data database with their schemas.
        
        Returns a dict with table names as keys. Each value is a list of
        column info dicts: {name, type, notnull, pk}.
        
        Use this first when you need to understand what data is available.
        """
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row["name"] for row in cursor.fetchall()]
            
            result = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                result[table] = [
                    {
                        "name": row["name"],
                        "type": row["type"],
                        "notnull": bool(row["notnull"]),
                        "pk": bool(row["pk"]),
                    }
                    for row in cursor.fetchall()
                ]
            conn.close()
            return {"db_path": str(db_path), "tables": result}
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def table_stats(table_name: str) -> dict:
        """Get row count, date range, and (if applicable) per-asset breakdown
        for a table.
        
        Args:
            table_name: name of the table to analyze. Call list_tables first
                if you're not sure what's available.
        
        Returns:
            Dict with row_count, earliest/latest timestamp if the table has a
            timestamp column, and per-asset row counts if it has an asset
            column.
        """
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            
            # Verify table exists (SQL injection guard -- reject unknown names)
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not cursor.fetchone():
                return {"error": f"Table '{table_name}' not found"}
            
            cursor.execute(f"SELECT COUNT(*) as n FROM {table_name}")
            row_count = cursor.fetchone()["n"]
            
            # Check what columns the table has
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = {row["name"] for row in cursor.fetchall()}
            
            result = {"table": table_name, "row_count": row_count}
            
            if "timestamp" in columns:
                cursor.execute(
                    f"SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts "
                    f"FROM {table_name}"
                )
                row = cursor.fetchone()
                if row["min_ts"] is not None:
                    # Detect timestamp unit: seconds vs milliseconds vs ISO
                    # Heuristic: > 10^12 = ms, > 10^9 = s
                    min_ts = row["min_ts"]
                    max_ts = row["max_ts"]
                    if isinstance(min_ts, (int, float)):
                        if min_ts > 1e12:
                            fmt = "ms"
                            earliest = datetime.fromtimestamp(min_ts / 1000, tz=timezone.utc).isoformat()
                            latest = datetime.fromtimestamp(max_ts / 1000, tz=timezone.utc).isoformat()
                        elif min_ts > 1e9:
                            fmt = "s"
                            earliest = datetime.fromtimestamp(min_ts, tz=timezone.utc).isoformat()
                            latest = datetime.fromtimestamp(max_ts, tz=timezone.utc).isoformat()
                        else:
                            fmt = "unknown"
                            earliest = str(min_ts)
                            latest = str(max_ts)
                    else:
                        fmt = "string"
                        earliest = str(min_ts)
                        latest = str(max_ts)
                    result["timestamp_format"] = fmt
                    result["earliest"] = earliest
                    result["latest"] = latest
            
            if "asset" in columns:
                cursor.execute(
                    f"SELECT asset, COUNT(*) as n FROM {table_name} "
                    f"GROUP BY asset ORDER BY n DESC"
                )
                result["by_asset"] = {row["asset"]: row["n"] for row in cursor.fetchall()}
            
            conn.close()
            return result
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def get_collector_health() -> dict:
        """Snapshot the health of every collector by checking the latest
        timestamp in each live table.
        
        Flags tables whose most recent row is > 1 hour old as potentially
        stale (the collector may be broken).
        
        Returns:
            Dict with per-table status: latest_timestamp, staleness_seconds,
            is_stale (bool), row_count.
        """
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            
            # Tables we expect to have active collectors writing to them
            live_tables = [
                "ohlcv_1m",
                "order_book_snapshots",
                "trades",
                "funding_rates",
                "fear_greed_index",
            ]
            
            now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            result = {"checked_at_utc": datetime.now(tz=timezone.utc).isoformat(), "tables": {}}
            
            for table in live_tables:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,)
                )
                if not cursor.fetchone():
                    result["tables"][table] = {"error": "table not found"}
                    continue
                
                cursor.execute(f"PRAGMA table_info({table})")
                columns = {row["name"] for row in cursor.fetchall()}
                if "timestamp" not in columns:
                    result["tables"][table] = {"error": "no timestamp column"}
                    continue
                
                cursor.execute(
                    f"SELECT COUNT(*) as n, MAX(timestamp) as latest FROM {table}"
                )
                row = cursor.fetchone()
                n = row["n"]
                latest = row["latest"]
                if latest is None:
                    result["tables"][table] = {"row_count": 0, "error": "empty table"}
                    continue
                
                # Normalize to ms
                if latest > 1e12:
                    latest_ms = latest
                elif latest > 1e9:
                    latest_ms = latest * 1000
                else:
                    latest_ms = None
                
                if latest_ms:
                    staleness_s = (now_ms - latest_ms) / 1000
                    latest_iso = datetime.fromtimestamp(latest_ms / 1000, tz=timezone.utc).isoformat()
                else:
                    staleness_s = None
                    latest_iso = str(latest)
                
                result["tables"][table] = {
                    "row_count": n,
                    "latest": latest_iso,
                    "staleness_seconds": staleness_s,
                    "is_stale": staleness_s is not None and staleness_s > 3600,
                }
            
            conn.close()
            return result
        except Exception as e:
            return {"error": str(e)}
```

### Step 6: OHLCV tool (`servers/praxis_mcp/tools/ohlcv.py`)

```python
from pathlib import Path
from servers.praxis_mcp.db import connect_ro


def register(mcp, db_path: Path):
    @mcp.tool()
    def get_recent_ohlcv(asset: str, lookback_bars: int = 60) -> dict:
        """Get the most recent N 1-minute OHLCV bars for an asset.
        
        Args:
            asset: "BTC" or "ETH" (case-insensitive).
            lookback_bars: how many recent bars to return (default 60 = 1 hour).
                Capped at 1440 (1 day) to avoid giant payloads.
        
        Returns:
            Dict with asset, rows (list of bars with timestamp, open, high,
            low, close, volume), and count.
        """
        asset = asset.upper()
        lookback_bars = min(lookback_bars, 1440)
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_1m
                WHERE asset = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (asset, lookback_bars)
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            # Reverse so oldest-first for easier reading
            rows.reverse()
            return {"asset": asset, "count": len(rows), "rows": rows}
        except Exception as e:
            return {"error": str(e)}
```

### Step 7: Order book tool (`servers/praxis_mcp/tools/order_book.py`)

```python
from pathlib import Path
from datetime import datetime, timezone
from servers.praxis_mcp.db import connect_ro


def register(mcp, db_path: Path):
    @mcp.tool()
    def get_order_book_snapshot(asset: str, at_timestamp_ms: int = None) -> dict:
        """Get the order book snapshot nearest to the given timestamp (or
        the most recent if omitted).
        
        Args:
            asset: "BTC" or "ETH".
            at_timestamp_ms: Unix milliseconds. If None, returns the latest.
        
        Returns:
            Dict with all fields from order_book_snapshots for the nearest row,
            including all 10 bid + 10 ask levels, spread, and derived aggregates.
        """
        asset = asset.upper()
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            if at_timestamp_ms is None:
                cursor.execute(
                    """
                    SELECT * FROM order_book_snapshots
                    WHERE asset = ?
                    ORDER BY timestamp DESC LIMIT 1
                    """,
                    (asset,)
                )
            else:
                # Nearest: use absolute difference
                cursor.execute(
                    """
                    SELECT *, ABS(timestamp - ?) as diff
                    FROM order_book_snapshots
                    WHERE asset = ?
                    ORDER BY diff ASC LIMIT 1
                    """,
                    (at_timestamp_ms, asset)
                )
            row = cursor.fetchone()
            conn.close()
            if row is None:
                return {"error": f"no snapshot found for {asset}"}
            return dict(row)
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def get_order_book_range(
        asset: str,
        start_ts_ms: int,
        end_ts_ms: int,
        max_rows: int = 200,
    ) -> dict:
        """Get order book snapshots in a time range.
        
        Args:
            asset: "BTC" or "ETH".
            start_ts_ms, end_ts_ms: Unix ms range.
            max_rows: hard cap (default 200). If the range contains more, rows
                are sampled evenly across the range (not truncated to start).
        
        Returns:
            Dict with asset, count, and rows.
        """
        asset = asset.upper()
        max_rows = min(max_rows, 1000)
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) as n FROM order_book_snapshots
                WHERE asset = ? AND timestamp BETWEEN ? AND ?
                """,
                (asset, start_ts_ms, end_ts_ms)
            )
            total = cursor.fetchone()["n"]
            
            if total <= max_rows:
                cursor.execute(
                    """
                    SELECT * FROM order_book_snapshots
                    WHERE asset = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp ASC
                    """,
                    (asset, start_ts_ms, end_ts_ms)
                )
            else:
                # Even sampling via modulo on row number
                step = total // max_rows
                cursor.execute(
                    """
                    SELECT * FROM (
                        SELECT *, ROW_NUMBER() OVER (ORDER BY timestamp ASC) as rn
                        FROM order_book_snapshots
                        WHERE asset = ? AND timestamp BETWEEN ? AND ?
                    )
                    WHERE rn % ? = 0
                    LIMIT ?
                    """,
                    (asset, start_ts_ms, end_ts_ms, step, max_rows)
                )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return {
                "asset": asset,
                "total_in_range": total,
                "returned": len(rows),
                "sampled": total > max_rows,
                "rows": rows,
            }
        except Exception as e:
            return {"error": str(e)}
```

### Step 8: Trades tool (`servers/praxis_mcp/tools/trades.py`)

```python
from pathlib import Path
from datetime import datetime, timezone
from servers.praxis_mcp.db import connect_ro


def register(mcp, db_path: Path):
    @mcp.tool()
    def get_recent_trades(
        asset: str,
        lookback_minutes: int = 10,
        min_quote_amount: float = 0,
        max_rows: int = 500,
    ) -> dict:
        """Get recent trades for an asset, with optional minimum dollar filter.
        
        Args:
            asset: "BTC" or "ETH".
            lookback_minutes: how far back to look (default 10).
            min_quote_amount: only return trades with quote_amount >= this
                (default 0 = all trades). Use e.g. 100000 to see only $100k+ trades.
            max_rows: hard cap (default 500, max 2000).
        
        Returns:
            Dict with asset, count, rows (timestamp, price, amount,
            quote_amount, side, is_buyer_maker), largest_trade.
        """
        asset = asset.upper()
        max_rows = min(max_rows, 2000)
        cutoff_ms = int(
            (datetime.now(tz=timezone.utc).timestamp() - lookback_minutes * 60) * 1000
        )
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT trade_id, timestamp, datetime, price, amount,
                       quote_amount, side, is_buyer_maker
                FROM trades
                WHERE asset = ?
                  AND timestamp >= ?
                  AND quote_amount >= ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (asset, cutoff_ms, min_quote_amount, max_rows)
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            largest = max(rows, key=lambda r: r["quote_amount"]) if rows else None
            return {
                "asset": asset,
                "lookback_minutes": lookback_minutes,
                "min_quote_amount": min_quote_amount,
                "count": len(rows),
                "rows": rows,
                "largest_trade": largest,
            }
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def get_trade_flow_summary(asset: str, window_minutes: int = 10) -> dict:
        """Summarize trade flow over the last N minutes: total volume,
        buy/sell split, aggressor imbalance.
        
        Args:
            asset: "BTC" or "ETH".
            window_minutes: lookback window (default 10).
        
        Returns:
            Dict with total_trades, total_volume_base, total_volume_quote,
            buy_volume_quote, sell_volume_quote, aggressor_imbalance,
            avg_trade_size_quote, max_trade_size_quote.
        """
        asset = asset.upper()
        cutoff_ms = int(
            (datetime.now(tz=timezone.utc).timestamp() - window_minutes * 60) * 1000
        )
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as n,
                    SUM(amount) as vol_base,
                    SUM(quote_amount) as vol_quote,
                    SUM(CASE WHEN side='buy' THEN quote_amount ELSE 0 END) as buy_quote,
                    SUM(CASE WHEN side='sell' THEN quote_amount ELSE 0 END) as sell_quote,
                    AVG(quote_amount) as avg_quote,
                    MAX(quote_amount) as max_quote
                FROM trades
                WHERE asset = ? AND timestamp >= ?
                """,
                (asset, cutoff_ms)
            )
            r = cursor.fetchone()
            conn.close()
            total_quote = r["vol_quote"] or 0
            buy_quote = r["buy_quote"] or 0
            sell_quote = r["sell_quote"] or 0
            imbalance = (
                (buy_quote - sell_quote) / total_quote if total_quote > 0 else 0
            )
            return {
                "asset": asset,
                "window_minutes": window_minutes,
                "total_trades": r["n"],
                "total_volume_base": r["vol_base"],
                "total_volume_quote": total_quote,
                "buy_volume_quote": buy_quote,
                "sell_volume_quote": sell_quote,
                "aggressor_imbalance": imbalance,
                "avg_trade_size_quote": r["avg_quote"],
                "max_trade_size_quote": r["max_quote"],
            }
        except Exception as e:
            return {"error": str(e)}
```

### Step 9: Funding tool (`servers/praxis_mcp/tools/funding.py`)

```python
from pathlib import Path
from datetime import datetime, timezone
from servers.praxis_mcp.db import connect_ro


def register(mcp, db_path: Path):
    @mcp.tool()
    def get_funding_rate_history(
        asset: str = "BTC",
        lookback_days: int = 30,
        max_rows: int = 500,
    ) -> dict:
        """Get funding rate history for an asset.
        
        Args:
            asset: "BTC", "ETH", or other asset symbol present in funding_rates.
            lookback_days: how far back to pull (default 30).
            max_rows: cap (default 500, max 2000).
        
        Returns:
            Dict with asset, rows (timestamp, funding_rate, ...), latest
            funding_rate, and simple stats (mean, min, max, positive_share).
        """
        asset = asset.upper()
        max_rows = min(max_rows, 2000)
        cutoff_s = int(datetime.now(tz=timezone.utc).timestamp()) - lookback_days * 86400
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            # Verify table exists -- funding_rates may be named differently in
            # different Praxis versions; check first
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='funding_rates'"
            )
            if not cursor.fetchone():
                return {"error": "table funding_rates not found"}
            
            # Funding rates may use seconds or ms timestamps; detect
            cursor.execute("SELECT timestamp FROM funding_rates LIMIT 1")
            sample = cursor.fetchone()
            if sample is None:
                return {"error": "funding_rates table is empty"}
            ts_sample = sample["timestamp"]
            ms_mode = ts_sample > 1e12
            cutoff = cutoff_s * 1000 if ms_mode else cutoff_s
            
            cursor.execute(
                """
                SELECT * FROM funding_rates
                WHERE asset = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (asset, cutoff, max_rows)
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            if not rows:
                return {"asset": asset, "count": 0, "rows": []}
            
            rates = [r.get("funding_rate") for r in rows if r.get("funding_rate") is not None]
            stats = {}
            if rates:
                stats = {
                    "mean": sum(rates) / len(rates),
                    "min": min(rates),
                    "max": max(rates),
                    "positive_share": sum(1 for r in rates if r > 0) / len(rates),
                    "latest": rates[0],  # rows are DESC
                }
            
            return {"asset": asset, "count": len(rows), "stats": stats, "rows": rows}
        except Exception as e:
            return {"error": str(e)}
```

**Note for Code:** the exact schema of `funding_rates` (column names, timestamp unit) depends on how the existing collector writes it. Check the actual schema during implementation and adjust the query if needed. Don't assume.

### Step 10: Raw query escape hatch (`servers/praxis_mcp/tools/raw.py`)

```python
from pathlib import Path
from servers.praxis_mcp.db import connect_ro


# Keywords that must not appear in any query -- extra belt beyond mode=ro
# (SQLite will reject these anyway, but rejecting at the tool layer lets
# the LLM see a cleaner error message and avoids pointless round trips.)
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM",
]


def register(mcp, db_path: Path):
    @mcp.tool()
    def raw_query(sql: str, max_rows: int = 100) -> dict:
        """Execute a read-only SQL query against the Praxis crypto_data database.
        
        Use this ONLY when the targeted tools (get_recent_trades,
        get_trade_flow_summary, etc.) don't cover what you need. Prefer
        targeted tools whenever possible -- they encode the correct filters
        and are less likely to produce bad queries.
        
        Args:
            sql: a SELECT query. Writes and DDL are rejected.
            max_rows: cap on result rows (default 100, max 2000).
        
        Returns:
            Dict with columns (list), rows (list of dicts), and count.
        """
        max_rows = min(max_rows, 2000)
        
        # Tool-layer keyword check (defense in depth; SQLite RO mode also blocks writes)
        upper = sql.upper()
        for kw in FORBIDDEN_KEYWORDS:
            # Match as a whole word to avoid false positives on legit column names
            # (crude but adequate for an escape hatch)
            if f" {kw} " in f" {upper} " or upper.startswith(f"{kw} "):
                return {"error": f"Forbidden keyword: {kw}"}
        
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = []
            for i, row in enumerate(cursor.fetchall()):
                if i >= max_rows:
                    break
                rows.append(dict(row))
            conn.close()
            return {"columns": columns, "count": len(rows), "rows": rows, "truncated": len(rows) >= max_rows}
        except Exception as e:
            return {"error": str(e)}
```

### Step 11: Smoke test (`servers/praxis_mcp/test_smoke.py`)

Purpose: verify the server starts, tools load, and a few representative calls return sensible output -- all without requiring Claude Desktop. Can be invoked directly:

```bash
python -m servers.praxis_mcp.test_smoke
```

```python
"""Smoke test for Praxis MCP server -- invokes tools directly without MCP client.

Doesn't go through stdio; calls the FastMCP tool functions directly for a
quick sanity check. If this passes, Claude Desktop integration should work.
"""

import sys
import json
from pathlib import Path

# Make relative imports work when invoked as `python -m ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from servers.praxis_mcp.db import connect_ro
from servers.praxis_mcp import server  # triggers tool registration


def run():
    print("=" * 60)
    print("Praxis MCP smoke test")
    print("=" * 60)
    
    # FastMCP stores tools internally; we can invoke them via _tool_manager
    # The exact attribute path depends on FastMCP version. Fallback: just
    # verify the server module imported and DB is accessible.
    
    print(f"\nServer name: {server.mcp.name}")
    print(f"DB path: {server.DB_PATH}")
    print(f"DB exists: {server.DB_PATH.exists()}")
    
    # Direct DB sanity check
    conn = connect_ro(server.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row["name"] for row in cursor.fetchall()]
    print(f"\nTables found: {len(tables)}")
    for t in sorted(tables):
        cursor.execute(f"SELECT COUNT(*) as n FROM {t}")
        n = cursor.fetchone()["n"]
        print(f"  {t}: {n} rows")
    conn.close()
    
    # Verify read-only enforcement
    print("\nTesting read-only enforcement...")
    try:
        conn = connect_ro(server.DB_PATH)
        conn.execute("CREATE TABLE mcp_smoke_test_xyz (id INTEGER)")
        print("  FAIL: write succeeded on read-only connection")
        return 1
    except Exception as e:
        print(f"  OK: write rejected: {type(e).__name__}: {e}")
    
    print("\nSmoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
```

### Step 12: Install docs (`servers/praxis_mcp/README.md`)

Must include:

1. **What this is:** one-paragraph summary.
2. **Install:** `pip install mcp>=1.2.0` (or via pyproject if the repo uses that).
3. **Smoke test:** `python -m servers.praxis_mcp.test_smoke`.
4. **Claude Desktop config:** the snippet to add to `claude_desktop_config.json` (Windows path: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "praxis": {
      "command": "python",
      "args": [
        "-m",
        "servers.praxis_mcp.server"
      ],
      "cwd": "C:\\Data\\Development\\Python\\McTheoryApps\\praxis",
      "env": {
        "PYTHONPATH": "C:\\Data\\Development\\Python\\McTheoryApps\\praxis"
      }
    }
  }
}
```

5. **Verification:** restart Claude Desktop; confirm "praxis" appears in the MCP servers indicator; test `list_tables()` in a Chat message.

6. **Troubleshooting:** if the server doesn't appear, check `%APPDATA%\Claude\logs\mcp*.log` on Windows. Common causes: wrong `cwd`, wrong `PYTHONPATH`, Python version < 3.10, `mcp` package not installed in the interpreter Claude Desktop is launching.

7. **Tool list** with one-line descriptions of each.

---

## Progress Reporting (per CLAUDE_CODE_RULES.md rules 9-15)

- **T+0:** confirm `data/crypto_data.db` exists and `mcp` package can be installed
- **After Step 2:** pip install verified
- **After Step 4:** DB helper tested (can open RO connection)
- **After each tool module (Steps 5-10):** brief status
- **After Step 11:** smoke test result
- **After Step 12:** docs complete; note the Claude Desktop config paths used

Total Brief should complete inside 2 hours. No long-running phases.

---

## Acceptance Criteria

- [ ] `servers/praxis_mcp/` directory structure created with all files listed in Step 1
- [ ] `mcp>=1.2.0` added to repo dependencies (pyproject.toml or requirements.txt)
- [ ] Server starts without error: `python -m servers.praxis_mcp.server` runs and waits on stdio (kill with Ctrl-C is fine)
- [ ] `python -m servers.praxis_mcp.test_smoke` passes: lists tables, verifies read-only enforcement, reports row counts
- [ ] Read-only enforcement verified: attempting `CREATE TABLE` on a connect_ro connection raises OperationalError
- [ ] `raw_query` rejects forbidden keywords (test one: `raw_query("DROP TABLE trades")` returns `{"error": "Forbidden keyword: DROP"}`)
- [ ] Every tool returns a dict; no tool raises an uncaught exception on happy-path inputs
- [ ] `README.md` in `servers/praxis_mcp/` covers: install, smoke test, Claude Desktop config JSON with correct paths, verification, troubleshooting, tool list
- [ ] AST parse + ASCII check pass on all new Python files
- [ ] Nothing in `servers/praxis_mcp/` writes to stdout (all diagnostic output via stderr/logging)

---

## Known Pitfalls

- **Never write to stdout.** MCP stdio servers use stdout for JSON-RPC. Any `print()` to stdout corrupts the protocol. Use `print(..., file=sys.stderr)` or `logging` configured to stderr. This is the #1 source of silent MCP failures.
- **Python version.** MCP SDK requires Python >= 3.10. If the repo venv is on 3.9, this won't work. Check first.
- **Claude Desktop absolute paths.** The `cwd` and `PYTHONPATH` in the config snippet are absolute Windows paths. They MUST match Jeff's actual repo location. The Brief includes `C:\Data\Development\Python\McTheoryApps\praxis` because that's the known path; verify before committing to the README.
- **Restart Claude Desktop after config changes.** MCP server configs are read at app startup, not dynamically.
- **SQLite timestamp unit ambiguity.** Different Praxis tables use different timestamp units (seconds vs ms). The `table_stats` and `get_collector_health` tools include detection heuristics; other tools assume the unit of their specific table. Document this per-tool.
- **`sqlite3.Row` is not JSON-serializable by default.** The tools convert via `dict(row)` before returning. MCP serializes the return value to JSON, which requires basic Python types. If any tool returns Row objects directly, MCP will fail on serialization.
- **Concurrent DB access.** The collectors (PraxisTradesCollector, PraxisOrderBookCollector, etc.) are writing to the DB while MCP may be reading. SQLite WAL mode handles this correctly, but in extreme contention queries may block briefly. Set a reasonable SQLite timeout (default 5s) on the RO connection.
- **FastMCP attribute paths.** The smoke test references `server.mcp.name`. FastMCP's internal attribute names have changed across versions. If Code hits AttributeError, fall back to minimal verification (just confirm `server` module imported, DB exists, and RO write is rejected).
- **`raw_query` keyword matching is crude.** It checks for forbidden keywords as whole words. A user could technically inject via column names (`SELECT INSERT_date FROM ...`) but SQLite's RO mode catches actual writes. The tool-layer check is belt; SQLite is suspenders. Don't over-engineer.
- **The `funding_rates` table schema may differ.** The Brief assumes a column named `funding_rate`; the actual schema might use `rate`, `funding`, or something else. Check during Step 9 and adjust. If the column isn't present, the stats dict just won't include them; don't hard-fail.

---

## What NOT to change

- Any collector code in `engines/` or scheduled task config
- Any existing DB tables or indexes
- The `data/crypto_data.db` file itself (read-only access only)
- `smart_money.db` or any other Praxis DB (out of scope for v0.1)
- Existing tests
- `engines/intrabar_predictor.py` (frozen)
- `engines/microstructure_utils.py` or `engines/whale_detector.py` (just shipped; leave alone)

---

## References

- MCP Python SDK docs: https://github.com/modelcontextprotocol/python-sdk
- FastMCP quickstart: `from mcp.server.fastmcp import FastMCP`
- Build an MCP server guide: https://modelcontextprotocol.io/docs/develop/build-server
- Critical: never write to stdout in stdio servers (per official docs)
- Claude Desktop config location (Windows): `%APPDATA%\Claude\claude_desktop_config.json`
- Existing Praxis tables for reference: `order_book_snapshots` (v8.2), `trades` (v8.2.1), `ohlcv_1m` (baseline)
- Repo root: `C:\Data\Development\Python\McTheoryApps\praxis`
- Workflow modes: `claude/WORKFLOW_MODES_PRAXIS.md`
- Progress rules: `claude/CLAUDE_CODE_RULES.md` rules 9-15

---

## Future work (not this Brief)

- **v0.2:** Add `smart_money.db` tools (list leaderboard wallets, get recent positions, track wallet over time). Separate Brief, likely 60 min.
- **v0.2:** Add Polymarket live collector data if it's stored in a Praxis DB.
- **v0.3:** Add computed-feature tools (run whale detector via MCP, compute rolling aggressor imbalance over an arbitrary window, etc.) -- turns MCP into an analysis layer not just a data layer.
- **v0.4:** Consider Alpaca / QuantConnect MCP wrappers (memory #30) -- those are external APIs not local DBs, different architecture.
- **Remote MCP:** if Jeff ever wants Chat-on-mobile to query, switch transport to Streamable HTTP and deploy. Not needed now.
