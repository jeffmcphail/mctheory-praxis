"""Order book tools: snapshot near timestamp, range query with sampling."""

from pathlib import Path
from typing import Optional

from servers.praxis_mcp.db import connect_ro


def register(mcp, db_path: Path):
    @mcp.tool()
    def get_order_book_snapshot(
        asset: str,
        at_timestamp_ms: Optional[int] = None,
    ) -> dict:
        """Get the order book snapshot nearest to the given timestamp (or
        the most recent if omitted).

        Args:
            asset: "BTC" or "ETH".
            at_timestamp_ms: Unix milliseconds. If None, returns the latest.

        Returns:
            Dict with all fields from order_book_snapshots for the nearest row,
            including all 10 bid + 10 ask levels, spread, and derived
            aggregates (bid_volume_top10, ask_volume_top10,
            order_imbalance_top10).
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
                    (asset,),
                )
            else:
                cursor.execute(
                    """
                    SELECT *, ABS(timestamp - ?) as diff
                    FROM order_book_snapshots
                    WHERE asset = ?
                    ORDER BY diff ASC LIMIT 1
                    """,
                    (int(at_timestamp_ms), asset),
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
        """Get order book snapshots in a time range, evenly sampled if needed.

        Args:
            asset: "BTC" or "ETH".
            start_ts_ms, end_ts_ms: Unix ms range.
            max_rows: hard cap (default 200, max 1000). If the range contains
                more, rows are sampled evenly across the range.

        Returns:
            Dict with asset, total_in_range, returned, sampled (bool), rows.
        """
        asset = asset.upper()
        max_rows = min(max(1, int(max_rows)), 1000)
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) as n FROM order_book_snapshots
                WHERE asset = ? AND timestamp BETWEEN ? AND ?
                """,
                (asset, int(start_ts_ms), int(end_ts_ms)),
            )
            total = cursor.fetchone()["n"]

            if total <= max_rows:
                cursor.execute(
                    """
                    SELECT * FROM order_book_snapshots
                    WHERE asset = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp ASC
                    """,
                    (asset, int(start_ts_ms), int(end_ts_ms)),
                )
            else:
                step = max(1, total // max_rows)
                cursor.execute(
                    """
                    SELECT * FROM (
                        SELECT *, ROW_NUMBER() OVER (
                            ORDER BY timestamp ASC) as rn
                        FROM order_book_snapshots
                        WHERE asset = ? AND timestamp BETWEEN ? AND ?
                    )
                    WHERE rn % ? = 0
                    LIMIT ?
                    """,
                    (asset, int(start_ts_ms), int(end_ts_ms), step, max_rows),
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
