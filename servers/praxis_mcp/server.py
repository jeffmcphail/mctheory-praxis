"""
Praxis MCP Server v0.1

Exposes data/crypto_data.db as read-only MCP tools so Claude Desktop (or any
MCP client) can query collected microstructure, OHLCV, funding, and sentiment
data directly without needing Code round-trips.

Cycle 14: get_collector_health now also monitors the two sidecar DBs
populated by PraxisLiveCollector and PraxisSmartMoney
(data/live_collector.db, data/smart_money.db). They are read-only here;
the engines themselves write through their own sqlite3 connections.

Transport: stdio (local subprocess; launched by Claude Desktop on demand).
Access mode: read-only via SQLite URI mode=ro (belt) plus no UPDATE/INSERT/
DELETE/DROP tool implementations (suspenders).

IMPORTANT: Never write to stdout. stdout is the MCP JSON-RPC channel; any
print() to stdout will corrupt the protocol. Use sys.stderr or logging
configured to stderr for all diagnostic output.
"""

import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from servers.praxis_mcp.tools import (
    atlas,
    funding,
    meta,
    ohlcv,
    order_book,
    raw,
    trades,
)

# Diagnostic logging to stderr ONLY. Any stdout write corrupts the JSON-RPC
# channel used by stdio MCP clients.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[praxis-mcp] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Resolve the DB path relative to the project root (parent of servers/).
# __file__ -> servers/praxis_mcp/server.py
# parents[0] -> servers/praxis_mcp/
# parents[1] -> servers/
# parents[2] -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "crypto_data.db"
ATLAS_DB_PATH = REPO_ROOT / "data" / "praxis_meta.db"

# Cycle 14: sidecar DBs for collector health monitoring
LIVE_DB_PATH = REPO_ROOT / "data" / "live_collector.db"
SMART_MONEY_DB_PATH = REPO_ROOT / "data" / "smart_money.db"

# Sidecar DB monitoring spec passed into meta.register().
# Each table specifies its threshold + timestamp column + timestamp format
# explicitly, since sidecar schemas don't follow the primary DB's
# conventions.
#
# live_collector / price_snapshots:
#   PraxisLiveCollector samples top 50 Polymarket markets every 60s in a
#   continuous loop (live_collector_service.bat -> python -m
#   engines.live_collector start --top 50 --interval 60). Schema stores
#   `timestamp` as integer Unix seconds. Healthy threshold: 180s (3x
#   cadence). Anything older means the loop crashed or the auto-restart
#   hit the 30s back-off and is mid-cycle.
#
# smart_money / position_snapshots:
#   PraxisSmartMoney runs every 6h via Task Scheduler (smart_money_service
#   .bat -> python -m engines.smart_money discover + snapshot). Schema
#   stores `timestamp` as TEXT in ISO format ("2026-04-29 22:25:24.71",
#   no tz suffix; treat as UTC). Healthy threshold: 28800s (8h = 6h
#   cadence + 2h slack for slow snapshot completion).
SIDECAR_DBS = {
    "live_collector": {
        "path": LIVE_DB_PATH,
        "monitored": {
            "price_snapshots": {
                "threshold_seconds": 180,
                "timestamp_column": "timestamp",
                "timestamp_format": "s",
            },
        },
    },
    "smart_money": {
        "path": SMART_MONEY_DB_PATH,
        "monitored": {
            "position_snapshots": {
                "threshold_seconds": 28800,
                "timestamp_column": "timestamp",
                "timestamp_format": "iso_text",
            },
        },
    },
}

if not DB_PATH.exists():
    log.error(f"DB not found at {DB_PATH}")
    # Still start the server; tools will report per-call rather than
    # failing hard at import.

mcp = FastMCP("praxis")

# Register tools (each module exposes register(mcp, db_path))
meta.register(mcp, DB_PATH, sidecar_dbs=SIDECAR_DBS)
ohlcv.register(mcp, DB_PATH)
order_book.register(mcp, DB_PATH)
trades.register(mcp, DB_PATH)
funding.register(mcp, DB_PATH)
raw.register(mcp, DB_PATH)
atlas.register(mcp, ATLAS_DB_PATH)


if __name__ == "__main__":
    log.info(f"Starting Praxis MCP server (db={DB_PATH})")
    mcp.run(transport="stdio")
