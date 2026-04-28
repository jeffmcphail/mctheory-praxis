"""OHLCV tool: recent 1-minute bars."""

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
            low, close, volume, oldest-first), and count.
        """
        asset = asset.upper()
        lookback_bars = min(max(1, int(lookback_bars)), 1440)
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, datetime, open, high, low, close, volume
                FROM ohlcv_1m
                WHERE asset = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (asset, lookback_bars),
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            rows.reverse()  # oldest-first for easier reading
            return {"asset": asset, "count": len(rows), "rows": rows}
        except Exception as e:
            return {"error": str(e)}
