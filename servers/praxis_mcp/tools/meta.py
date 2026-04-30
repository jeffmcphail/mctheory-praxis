"""Meta tools: schema introspection and collector health."""

from datetime import datetime, timezone
from pathlib import Path

from servers.praxis_mcp.db import connect_ro


def register(mcp, db_path: Path, sidecar_dbs: dict = None):
    """Register meta tools.

    Args:
        mcp: FastMCP server instance
        db_path: primary DB (data/crypto_data.db)
        sidecar_dbs: optional dict mapping logical name -> Path for additional
            DBs to include in get_collector_health. Each sidecar contributes
            its monitored tables and unmonitored tables to the health
            response, scoped under the sidecar's logical name.

            Schema:
              {
                "live_collector": {
                  "path": Path("data/live_collector.db"),
                  "monitored": {
                    "price_snapshots": {
                      "threshold_seconds": 180,
                      "timestamp_column": "timestamp",
                      "timestamp_format": "s",  # or "ms" or "iso_text"
                    },
                  },
                },
                ...
              }
    """
    sidecar_dbs = sidecar_dbs or {}

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
        """Snapshot the health of every collector across primary + sidecar DBs.

        Each monitored table has its own staleness threshold matching the
        underlying collector's natural cadence (continuous vs. hourly
        windowed vs. multi-hour batch). Tables with no scheduled collector
        are intentionally excluded from the health check and listed under
        `unmonitored` (per-DB).

        Sidecar DBs (live_collector.db, smart_money.db) are monitored
        through the same response under per-DB scoping in `databases`.
        The top-level `tables` and `unmonitored` keys preserve the
        pre-Cycle-14 response shape for the primary `crypto_data.db`.

        See `claude/retros/RETRO_praxis_collector_outage_triage.md` for
        the original monitored-vs-unmonitored split, and
        `claude/retros/RETRO_health_expansion.md` for the Cycle 11
        expansion that added Cycle 10's scheduled tables. Cycle 14
        added the sidecar-DB monitoring + ISO TEXT timestamp support
        + funding_rates threshold widening.

        Returns:
            Dict with:
              checked_at_utc: ISO timestamp
              tables: per-monitored-table status for the PRIMARY DB
                (preserves pre-Cycle-14 shape for backward compat).
                Each entry has: row_count, latest (ISO), staleness_seconds,
                threshold_seconds, is_stale (bool).
              unmonitored: list of tables in the PRIMARY DB without
                scheduled collectors.
              databases: dict scoped per-DB (including primary). Same
                shape as top-level tables/unmonitored. Use this when
                you want full visibility across all sidecars in one
                response, including the new live_collector and
                smart_money DBs (Cycle 14).
        """
        # Primary DB monitoring config (Cycle 11 + Cycle 14 funding fix)
        #
        # trades                -- PraxisTradesCollector, continuous 30s
        #                          (now 3550s windowed per Cycle 10
        #                          patch matching OrderBook).
        # order_book_snapshots  -- PraxisOrderBookCollector, 10s on-hour,
        #                          3550s windowed (Cycle 8 fix). 65 min
        #                          tolerance covers the worst-case
        #                          sampling moment plus inter-window gap.
        # ohlcv_1m              -- PraxisCrypto1mCollector, 6h batch.
        #                          7h tolerance covers batch + slack.
        # funding_rates         -- PraxisFundingCollector. Cadence in DB
        #                          is 8h between funding events (Binance
        #                          schedule). Scheduled collector runs at
        #                          00:05/08:05/16:05 local Toronto time.
        #                          Worst case gap between (now) and
        #                          (latest data point in DB) is roughly
        #                          one full funding cycle plus collector
        #                          gap, which can approach 16h depending
        #                          on which run last touched the table.
        #                          Cycle 14 widened from 9h to 17h to
        #                          stop legitimate cadence-aligned data
        #                          from being flagged stale.
        # fear_greed            -- PraxisFearGreedCollector, daily at
        #                          00:30 local (Cycle 10). 26h tolerance.
        # ohlcv_daily           -- PraxisOhlcvDailyCollector, daily at
        #                          00:15 local (Cycle 10). 26h tolerance.
        # ohlcv_4h              -- PraxisOhlcv4hCollector, daily at
        #                          00:20 local (Cycle 10). 26h tolerance.
        primary_monitored = {
            "trades": 120,
            "order_book_snapshots": 3900,
            "ohlcv_1m": 25200,
            "funding_rates": 61200,    # Cycle 14: 17h (was 32400 / 9h)
            "fear_greed": 93600,       # 24h + 2h slack
            "ohlcv_daily": 93600,      # 24h + 2h slack
            "ohlcv_4h": 93600,         # 24h + 2h slack
        }

        result = {
            "checked_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "tables": {},
            "unmonitored": [],
            "databases": {},
        }

        # Primary DB
        primary_status = _collect_db_health(
            db_path=db_path,
            monitored_tables=primary_monitored,
            timestamp_format="auto",   # autodetect ms vs s for primary
        )
        result["tables"] = primary_status["tables"]
        result["unmonitored"] = primary_status["unmonitored"]
        result["databases"]["crypto_data"] = {
            "path": str(db_path),
            "tables": primary_status["tables"],
            "unmonitored": primary_status["unmonitored"],
        }

        # Sidecar DBs (Cycle 14)
        for name, cfg in sidecar_dbs.items():
            scfg_path = cfg["path"]
            scfg_monitored = cfg.get("monitored", {})
            sidecar_status = _collect_db_health_sidecar(
                db_path=scfg_path,
                monitored_spec=scfg_monitored,
            )
            result["databases"][name] = {
                "path": str(scfg_path),
                "tables": sidecar_status["tables"],
                "unmonitored": sidecar_status["unmonitored"],
            }

        return result


# ---- helpers below the register() closure ----

def _collect_db_health(*, db_path: Path, monitored_tables: dict,
                       timestamp_format: str) -> dict:
    """Inspect the primary DB. Preserves the pre-Cycle-14 shape.

    Args:
        db_path: SQLite DB path
        monitored_tables: {table_name: threshold_seconds}
        timestamp_format: "auto" (autodetect ms vs s by magnitude),
            "ms", "s", or "iso_text"

    Returns dict with `tables` (per-table status) and `unmonitored` (list).
    """
    out_tables = {}
    unmonitored = []
    try:
        conn = connect_ro(db_path)
        cursor = conn.cursor()

        # Compute unmonitored set
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        all_tables = {row["name"] for row in cursor.fetchall()}
        sqlite_internal = {"sqlite_sequence"}
        unmonitored = sorted(
            all_tables - set(monitored_tables.keys()) - sqlite_internal
        )

        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        for table, threshold_s in monitored_tables.items():
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name=?",
                (table,),
            )
            if not cursor.fetchone():
                out_tables[table] = {"error": "table not found"}
                continue

            cursor.execute(f"PRAGMA table_info({table})")
            columns = {row["name"] for row in cursor.fetchall()}
            if "timestamp" not in columns:
                if "date" in columns:
                    cursor.execute(
                        f"SELECT COUNT(*) as n, MAX(date) as latest "
                        f"FROM {table}"
                    )
                    row = cursor.fetchone()
                    out_tables[table] = {
                        "row_count": row["n"],
                        "latest": row["latest"],
                        "note": "date-only table; staleness not computed",
                    }
                else:
                    out_tables[table] = {
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
                out_tables[table] = {"row_count": 0, "error": "empty table"}
                continue

            latest_ms = _to_latest_ms(latest, timestamp_format)
            if latest_ms is None:
                out_tables[table] = {
                    "row_count": n,
                    "latest": str(latest),
                    "error": "could not parse timestamp",
                }
                continue

            staleness_s = (now_ms - latest_ms) / 1000
            latest_iso = datetime.fromtimestamp(
                latest_ms / 1000, tz=timezone.utc).isoformat()

            out_tables[table] = {
                "row_count": n,
                "latest": latest_iso,
                "staleness_seconds": staleness_s,
                "threshold_seconds": threshold_s,
                "is_stale": staleness_s > threshold_s,
            }

        conn.close()
    except Exception as e:
        out_tables["__error__"] = str(e)
    return {"tables": out_tables, "unmonitored": unmonitored}


def _collect_db_health_sidecar(*, db_path: Path, monitored_spec: dict) -> dict:
    """Inspect a sidecar DB. Each table has explicit timestamp_column +
    timestamp_format spec rather than autodetect, since sidecars use
    schemas that don't always follow the primary DB's conventions.

    monitored_spec format:
      {
        "table_name": {
          "threshold_seconds": int,
          "timestamp_column": str,    # column to MAX() over
          "timestamp_format": "ms" | "s" | "iso_text",
        },
        ...
      }
    """
    out_tables = {}
    unmonitored = []
    try:
        if not db_path.exists():
            return {
                "tables": {"__error__": f"DB not found at {db_path}"},
                "unmonitored": [],
            }
        conn = connect_ro(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        all_tables = {row["name"] for row in cursor.fetchall()}
        sqlite_internal = {"sqlite_sequence"}
        monitored_names = set(monitored_spec.keys())
        unmonitored = sorted(all_tables - monitored_names - sqlite_internal)

        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        for table, spec in monitored_spec.items():
            ts_col = spec["timestamp_column"]
            threshold_s = spec["threshold_seconds"]
            ts_fmt = spec.get("timestamp_format", "auto")

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name=?",
                (table,),
            )
            if not cursor.fetchone():
                out_tables[table] = {"error": "table not found"}
                continue

            cursor.execute(f"PRAGMA table_info({table})")
            columns = {row["name"] for row in cursor.fetchall()}
            if ts_col not in columns:
                out_tables[table] = {
                    "error": f"timestamp column '{ts_col}' not in table"}
                continue

            cursor.execute(
                f"SELECT COUNT(*) as n, MAX({ts_col}) as latest FROM {table}"
            )
            row = cursor.fetchone()
            n = row["n"]
            latest = row["latest"]
            if latest is None:
                out_tables[table] = {"row_count": 0, "error": "empty table"}
                continue

            latest_ms = _to_latest_ms(latest, ts_fmt)
            if latest_ms is None:
                out_tables[table] = {
                    "row_count": n,
                    "latest": str(latest),
                    "error": (
                        f"could not parse timestamp "
                        f"(format={ts_fmt}, sample={latest!r})"),
                }
                continue

            staleness_s = (now_ms - latest_ms) / 1000
            latest_iso = datetime.fromtimestamp(
                latest_ms / 1000, tz=timezone.utc).isoformat()

            out_tables[table] = {
                "row_count": n,
                "latest": latest_iso,
                "staleness_seconds": staleness_s,
                "threshold_seconds": threshold_s,
                "is_stale": staleness_s > threshold_s,
            }

        conn.close()
    except Exception as e:
        out_tables["__error__"] = str(e)
    return {"tables": out_tables, "unmonitored": unmonitored}


def _to_latest_ms(latest, fmt: str):
    """Convert a `latest` timestamp value to Unix milliseconds.

    fmt:
      "auto"     -- numeric: detect ms vs s by magnitude (>1e12 -> ms,
                    >1e9 -> s); else None.
      "ms"       -- numeric milliseconds since epoch.
      "s"        -- numeric seconds since epoch.
      "iso_text" -- ISO 8601 string (e.g. "2026-04-29 22:25:24.71" or
                    "2026-04-29T22:25:24+00:00"). datetime.fromisoformat
                    handles both space and T separators in Python 3.11+.

    Returns int (ms) or None if unparseable.
    """
    if latest is None:
        return None

    if fmt == "iso_text" or isinstance(latest, str):
        # Try ISO parsing. Smart_money's timestamps look like
        # "2026-04-29 22:25:24.71" without a tz suffix; treat as UTC.
        s = latest.strip()
        # Replace space-separator with T to widen fromisoformat compat
        s_iso = s.replace(" ", "T", 1)
        try:
            dt = datetime.fromisoformat(s_iso)
        except ValueError:
            # Try stripping fractional seconds beyond microseconds
            try:
                # Trim everything after the decimal point in the seconds
                # field if it has more digits than microseconds allows.
                # Fall back: try without fractional seconds at all.
                if "." in s_iso:
                    head, _frac = s_iso.split(".", 1)
                    dt = datetime.fromisoformat(head)
                else:
                    return None
            except ValueError:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    if not isinstance(latest, (int, float)):
        return None

    if fmt == "ms":
        return int(latest)
    if fmt == "s":
        return int(latest * 1000)
    # "auto"
    if latest > 1e12:
        return int(latest)
    if latest > 1e9:
        return int(latest * 1000)
    return None
