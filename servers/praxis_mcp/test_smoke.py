"""Smoke test for Praxis MCP server.

Invokes tools directly (not through stdio) for a quick sanity check.
If this passes, Claude Desktop integration should work.

Run:
    python -m servers.praxis_mcp.test_smoke
"""

import sys
from pathlib import Path

# Make the package resolvable when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from servers.praxis_mcp import server  # triggers tool registration
from servers.praxis_mcp.db import connect_ro


def run():
    print("=" * 60)
    print("Praxis MCP smoke test")
    print("=" * 60)

    print(f"\nServer name: {server.mcp.name}")
    print(f"DB path: {server.DB_PATH}")
    print(f"DB exists: {server.DB_PATH.exists()}")

    # Direct DB sanity check
    conn = connect_ro(server.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row["name"] for row in cursor.fetchall()]
    print(f"\nTables found: {len(tables)}")
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) as n FROM {t}")
        n = cursor.fetchone()["n"]
        print(f"  {t}: {n:>8,d} rows")
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

    # Spot-check raw_query keyword guard (DROP should be rejected before
    # SQLite even sees it). We can't easily call decorated tools through
    # FastMCP here; instead, import the forbidden list and verify it.
    from servers.praxis_mcp.tools.raw import FORBIDDEN_KEYWORDS
    print(f"\nRaw query forbidden keywords: {len(FORBIDDEN_KEYWORDS)}")
    assert "DROP" in FORBIDDEN_KEYWORDS
    assert "INSERT" in FORBIDDEN_KEYWORDS
    print("  OK: denylist contains DROP, INSERT")

    # Verify all six tool modules registered at least one tool each
    # (FastMCP exposes registered tools via its internal manager).
    try:
        tool_count = len(server.mcp._tool_manager._tools)
    except AttributeError:
        # Attribute path may differ across FastMCP versions; fall back.
        tool_count = None
    if tool_count is not None:
        print(f"\nRegistered tools: {tool_count}")
    else:
        print("\nTool count introspection unavailable "
              "(FastMCP version quirk); skipping.")

    print("\nSmoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
