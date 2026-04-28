"""Trade flow tools: recent trades + windowed flow summary."""

from datetime import datetime, timezone
from pathlib import Path

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
                (default 0 = all trades). E.g. 100000 to see $100k+ trades.
            max_rows: hard cap (default 500, max 2000).

        Returns:
            Dict with asset, count, rows (trade_id, timestamp, datetime,
            price, amount, quote_amount, side, is_buyer_maker),
            largest_trade. Rows are ordered newest-first.
        """
        asset = asset.upper()
        max_rows = min(max(1, int(max_rows)), 2000)
        cutoff_ms = int(
            (datetime.now(tz=timezone.utc).timestamp()
             - lookback_minutes * 60) * 1000
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
                (asset, cutoff_ms, float(min_quote_amount), max_rows),
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            largest = (max(rows, key=lambda r: r["quote_amount"])
                       if rows else None)
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
    def get_trade_flow_summary(
        asset: str,
        window_minutes: int = 10,
    ) -> dict:
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
            (datetime.now(tz=timezone.utc).timestamp()
             - window_minutes * 60) * 1000
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
                    SUM(CASE WHEN side='buy' THEN quote_amount ELSE 0 END)
                        as buy_quote,
                    SUM(CASE WHEN side='sell' THEN quote_amount ELSE 0 END)
                        as sell_quote,
                    AVG(quote_amount) as avg_quote,
                    MAX(quote_amount) as max_quote
                FROM trades
                WHERE asset = ? AND timestamp >= ?
                """,
                (asset, cutoff_ms),
            )
            r = cursor.fetchone()
            conn.close()
            total_quote = r["vol_quote"] or 0
            buy_quote = r["buy_quote"] or 0
            sell_quote = r["sell_quote"] or 0
            imbalance = (
                (buy_quote - sell_quote) / total_quote
                if total_quote > 0 else 0
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
