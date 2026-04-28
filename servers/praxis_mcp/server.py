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

import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from servers.praxis_mcp.tools import (
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

if not DB_PATH.exists():
    log.error(f"DB not found at {DB_PATH}")
    # Still start the server; tools will report per-call rather than
    # failing hard at import.

mcp = FastMCP("praxis")

# Register tools (each module exposes register(mcp, db_path))
meta.register(mcp, DB_PATH)
ohlcv.register(mcp, DB_PATH)
order_book.register(mcp, DB_PATH)
trades.register(mcp, DB_PATH)
funding.register(mcp, DB_PATH)
raw.register(mcp, DB_PATH)


if __name__ == "__main__":
    log.info(f"Starting Praxis MCP server (db={DB_PATH})")
    mcp.run(transport="stdio")
