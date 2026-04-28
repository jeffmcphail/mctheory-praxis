"""Raw query escape hatch: read-only SQL with row cap + keyword denylist."""

from pathlib import Path

from servers.praxis_mcp.db import connect_ro


# Keywords that must not appear in any query -- extra belt beyond mode=ro.
# SQLite will reject writes anyway, but catching at the tool layer lets the
# LLM see a cleaner error and avoids pointless round trips.
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
            Dict with columns (list), rows (list of dicts), count,
            and truncated (bool). On error, a single "error" key.
        """
        max_rows = min(max(1, int(max_rows)), 2000)

        # Whole-word keyword denylist. Crude but adequate; SQLite RO mode
        # catches any actual write that slips through.
        padded = f" {sql.upper()} "
        for kw in FORBIDDEN_KEYWORDS:
            if f" {kw} " in padded:
                return {"error": f"Forbidden keyword: {kw}"}

        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = (
                [d[0] for d in cursor.description]
                if cursor.description else []
            )
            rows = []
            for i, row in enumerate(cursor.fetchall()):
                if i >= max_rows:
                    break
                rows.append(dict(row))
            conn.close()
            return {
                "columns": columns,
                "count": len(rows),
                "rows": rows,
                "truncated": len(rows) >= max_rows,
            }
        except Exception as e:
            return {"error": str(e)}
