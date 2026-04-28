"""Funding rate tool.

The funding_rates table schema (verified during v0.1 implementation):
    asset TEXT, timestamp INTEGER (seconds), datetime TEXT, funding_rate REAL
"""

from datetime import datetime, timezone
from pathlib import Path

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
            asset: "BTC", "ETH", or other asset symbol in funding_rates.
            lookback_days: how far back to pull (default 30).
            max_rows: cap (default 500, max 2000).

        Returns:
            Dict with asset, count, rows (timestamp, datetime, funding_rate),
            and simple stats (mean, min, max, positive_share, latest) when
            data is available.
        """
        asset = asset.upper()
        max_rows = min(max(1, int(max_rows)), 2000)
        cutoff_s = (int(datetime.now(tz=timezone.utc).timestamp())
                    - lookback_days * 86400)
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='funding_rates'"
            )
            if not cursor.fetchone():
                return {"error": "table funding_rates not found"}

            # Detect timestamp unit via a sample row
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
                (asset, cutoff, max_rows),
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()

            if not rows:
                return {"asset": asset, "count": 0, "rows": []}

            rates = [
                r["funding_rate"] for r in rows
                if r.get("funding_rate") is not None
            ]
            stats = {}
            if rates:
                stats = {
                    "mean": sum(rates) / len(rates),
                    "min": min(rates),
                    "max": max(rates),
                    "positive_share": (sum(1 for x in rates if x > 0)
                                       / len(rates)),
                    "latest": rates[0],
                }

            return {
                "asset": asset,
                "count": len(rows),
                "stats": stats,
                "rows": rows,
            }
        except Exception as e:
            return {"error": str(e)}
