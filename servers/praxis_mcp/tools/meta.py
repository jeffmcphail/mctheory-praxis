"""Meta tools: schema introspection and collector health."""

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
                "SELECT name FROM sqlite_master WHERE type='table' "
                "ORDER BY name"
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
                (table_name,),
            )
            if not cursor.fetchone():
                return {"error": f"Table '{table_name}' not found"}

            cursor.execute(f"SELECT COUNT(*) as n FROM {table_name}")
            row_count = cursor.fetchone()["n"]

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
                    min_ts = row["min_ts"]
                    max_ts = row["max_ts"]
                    if isinstance(min_ts, (int, float)):
                        if min_ts > 1e12:
                            fmt = "ms"
                            earliest = datetime.fromtimestamp(
                                min_ts / 1000, tz=timezone.utc).isoformat()
                            latest = datetime.fromtimestamp(
                                max_ts / 1000, tz=timezone.utc).isoformat()
                        elif min_ts > 1e9:
                            fmt = "s"
                            earliest = datetime.fromtimestamp(
                                min_ts, tz=timezone.utc).isoformat()
                            latest = datetime.fromtimestamp(
                                max_ts, tz=timezone.utc).isoformat()
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
                result["by_asset"] = {
                    row["asset"]: row["n"] for row in cursor.fetchall()
                }

            conn.close()
            return result
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def get_collector_health() -> dict:
        """Snapshot the health of every collector by checking the latest
        timestamp in each live table.

        Each monitored table has its own staleness threshold matching the
        underlying collector's natural cadence (continuous vs. hourly
        windowed vs. multi-hour batch). Tables with no scheduled collector
        are intentionally excluded from the health check and listed under
        `unmonitored`.

        See `claude/retros/RETRO_praxis_collector_outage_triage.md` for
        the investigation that established the monitored-vs-unmonitored
        split and the per-table thresholds.

        Returns:
            Dict with:
              checked_at_utc: ISO timestamp
              tables: per-monitored-table status -- latest (ISO), staleness_seconds,
                      threshold_seconds, is_stale (bool), row_count
              unmonitored: list of tables that exist in the DB but have no
                           scheduled collector (populated only by manual CLI
                           runs; not alarmed on)
        """
        try:
            conn = connect_ro(db_path)
            cursor = conn.cursor()

            # Tables with active scheduled collectors. Threshold = natural
            # cadence + slack for scheduler jitter / windowed-gap behavior.
            #
            # trades                -- PraxisTradesCollector, continuous 10s.
            # order_book_snapshots  -- PraxisOrderBookCollector, 10s on-hour
            #                          but currently 1h-on/1h-off pattern
            #                          (see retro). 65 min tolerance covers
            #                          the worst-case sampling moment.
            # ohlcv_1m              -- PraxisCrypto1mCollector, 6h batch.
            #                          7h tolerance covers batch + slack.
            #
            # funding_rates and fear_greed are NOT monitored: no scheduled
            # task writes to them. Populated only by manual one-shot CLI
            # runs (python -m engines.crypto_data_collector collect-funding /
            # collect-fear-greed). Listed under `unmonitored` below.
            monitored_tables = {
                "trades": 120,
                "order_book_snapshots": 3900,
                "ohlcv_1m": 25200,
            }
            unmonitored_tables = ["funding_rates", "fear_greed"]

            now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            result = {
                "checked_at_utc": datetime.now(
                    tz=timezone.utc).isoformat(),
                "tables": {},
                "unmonitored": unmonitored_tables,
            }

            for table, threshold_s in monitored_tables.items():
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name=?",
                    (table,),
                )
                if not cursor.fetchone():
                    result["tables"][table] = {"error": "table not found"}
                    continue

                cursor.execute(f"PRAGMA table_info({table})")
                columns = {row["name"] for row in cursor.fetchall()}
                if "timestamp" not in columns:
                    # Fall back to date column if present (e.g. fear_greed)
                    if "date" in columns:
                        cursor.execute(
                            f"SELECT COUNT(*) as n, MAX(date) as latest "
                            f"FROM {table}"
                        )
                        row = cursor.fetchone()
                        result["tables"][table] = {
                            "row_count": row["n"],
                            "latest": row["latest"],
                            "note": "date-only table; staleness not computed",
                        }
                    else:
                        result["tables"][table] = {
                            "error": "no timestamp or date column"}
                    continue

                cursor.execute(
                    f"SELECT COUNT(*) as n, MAX(timestamp) as latest "
                    f"FROM {table}"
                )
                row = cursor.fetchone()
                n = row["n"]
                latest = row["latest"]
                if latest is None:
                    result["tables"][table] = {
                        "row_count": 0, "error": "empty table"}
                    continue

                if latest > 1e12:
                    latest_ms = latest
                elif latest > 1e9:
                    latest_ms = latest * 1000
                else:
                    latest_ms = None

                if latest_ms:
                    staleness_s = (now_ms - latest_ms) / 1000
                    latest_iso = datetime.fromtimestamp(
                        latest_ms / 1000, tz=timezone.utc).isoformat()
                else:
                    staleness_s = None
                    latest_iso = str(latest)

                result["tables"][table] = {
                    "row_count": n,
                    "latest": latest_iso,
                    "staleness_seconds": staleness_s,
                    "threshold_seconds": threshold_s,
                    "is_stale": (staleness_s is not None
                                 and staleness_s > threshold_s),
                }

            conn.close()
            return result
        except Exception as e:
            return {"error": str(e)}
