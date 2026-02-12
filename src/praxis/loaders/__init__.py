"""
YFinance Data Loader (Phase 2.3).

Loader lifecycle: fetch → ldr_yfinance (working) → security match →
ldr_yfinance_hist (archive) → fact_price_daily.

Safety: working table never wiped until all records are terminal.

Usage:
    loader = YFinanceLoader(conn, security_master)
    result = loader.load(["AAPL", "MSFT"], start="2023-01-01", end="2024-01-01")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import duckdb
import numpy as np

from praxis.datastore.keys import EntityKeys
from praxis.logger.core import PraxisLogger
from praxis.security import SecurityMaster

# Optional dependency - module-level for mockability
try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class LoadResult:
    """Result of a loader batch."""
    batch_id: str
    symbols_requested: list[str]
    symbols_loaded: int = 0
    rows_fetched: int = 0
    rows_matched: int = 0
    rows_created: int = 0
    rows_rejected: int = 0
    rows_stored: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class YFinanceLoader:
    """
    §2.6: YFinance data loader with full lifecycle.

    Pipeline:
    1. Fetch OHLCV from yfinance → pandas DataFrame
    2. Insert into ldr_yfinance (working table) with batch_id
    3. For each symbol: match_or_create in SecurityMaster
    4. Update process_status in working table
    5. Insert matched rows into fact_price_daily
    6. Archive: copy working → ldr_yfinance_hist
    7. Safety check: wipe working only if all records terminal
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        security_master: SecurityMaster,
    ):
        self._conn = conn
        self._master = security_master
        self._log = PraxisLogger.instance()

    def load(
        self,
        symbols: str | list[str],
        start: str | None = None,
        end: str | None = None,
        sec_type: str = "EQUITY",
    ) -> LoadResult:
        """
        Load price data for given symbols.

        Args:
            symbols: Ticker(s) to load.
            start: Start date (ISO string).
            end: End date (ISO string).
            sec_type: Security type for all symbols.

        Returns:
            LoadResult with batch statistics.
        """
        start_time = time.monotonic()

        if isinstance(symbols, str):
            symbols = [symbols]

        batch_id = f"yf_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        self._log.info(
            f"YFinanceLoader: starting batch {batch_id} "
            f"({len(symbols)} symbols, {start}→{end})",
            tags={"data_pipeline", "trade_cycle"},
            batch_id=batch_id,
            symbols=symbols,
        )

        result = LoadResult(
            batch_id=batch_id,
            symbols_requested=symbols,
        )

        # ── Step 1: Fetch from yfinance ───────────────────────
        raw_data = self._fetch(symbols, start, end)
        if not raw_data:
            result.errors.append("No data fetched from yfinance")
            return result

        # ── Step 2: Insert into working table ─────────────────
        total_rows = self._insert_working(raw_data, batch_id)
        result.rows_fetched = total_rows

        self._log.info(
            f"Loaded {total_rows} rows into ldr_yfinance",
            tags={"data_pipeline"},
            batch_id=batch_id,
        )

        # ── Step 3-4: Security matching ───────────────────────
        matched, created, rejected = self._match_securities(batch_id, sec_type)
        result.rows_matched = matched
        result.rows_created = created
        result.rows_rejected = rejected
        result.symbols_loaded = len(set(
            r[0] for r in self._conn.execute(
                "SELECT DISTINCT symbol FROM ldr_yfinance WHERE batch_id = $1 AND process_status != 'rejected'",
                [batch_id]
            ).fetchall()
        ))

        # ── Step 5: Insert into fact_price_daily ──────────────
        stored = self._store_prices(batch_id)
        result.rows_stored = stored

        # ── Step 6: Archive working → hist ────────────────────
        self._archive(batch_id)

        # ── Step 7: Safety wipe ───────────────────────────────
        self._safe_wipe(batch_id)

        result.duration_seconds = time.monotonic() - start_time

        self._log.info(
            f"YFinanceLoader: batch {batch_id} complete — "
            f"{result.rows_stored} prices stored, "
            f"{result.symbols_loaded} symbols",
            tags={"data_pipeline", "trade_cycle"},
            batch_id=batch_id,
            rows_stored=result.rows_stored,
        )

        return result

    # ── Step 1: Fetch ─────────────────────────────────────────

    def _fetch(
        self,
        symbols: list[str],
        start: str | None,
        end: str | None,
    ) -> dict[str, Any]:
        """Fetch OHLCV from yfinance. Returns {symbol: pandas_df}."""
        if yf is None:
            raise ImportError("yfinance required. Install: pip install yfinance")

        data = {}
        for symbol in symbols:
            try:
                self._log.debug(
                    f"Fetching {symbol} from yfinance",
                    tags={"datastore.loader", "data_pipeline"},
                )
                kwargs = {"progress": False, "auto_adjust": False}
                if start:
                    kwargs["start"] = start
                if end:
                    kwargs["end"] = end

                df = yf.download(symbol, **kwargs)

                if df is not None and not df.empty:
                    # Flatten MultiIndex columns
                    if hasattr(df.columns, 'levels'):
                        df.columns = [
                            col[0] if isinstance(col, tuple) else col
                            for col in df.columns
                        ]
                    data[symbol] = df
                    self._log.debug(
                        f"{symbol}: {len(df)} bars",
                        tags={"datastore.loader"},
                    )
                else:
                    self._log.warning(
                        f"No data for {symbol}",
                        tags={"datastore.loader"},
                    )
            except Exception as e:
                self._log.error(
                    f"Failed to fetch {symbol}: {e}",
                    tags={"datastore.loader"},
                )

        return data

    # ── Step 2: Working table ─────────────────────────────────

    def _insert_working(
        self,
        data: dict[str, Any],
        batch_id: str,
    ) -> int:
        """Insert fetched data into ldr_yfinance working table."""
        now = datetime.now(timezone.utc)
        total = 0

        for symbol, df in data.items():
            df = df.reset_index()

            # Normalize column names
            col_map = {}
            for col in df.columns:
                lower = col.lower().strip()
                if lower in ("date", "datetime", "index"):
                    col_map[col] = "date"
                elif lower == "open":
                    col_map[col] = "open"
                elif lower == "high":
                    col_map[col] = "high"
                elif lower == "low":
                    col_map[col] = "low"
                elif lower == "close":
                    col_map[col] = "close"
                elif lower in ("adj close", "adj_close"):
                    col_map[col] = "adj_close"
                elif lower == "volume":
                    col_map[col] = "volume"
            df = df.rename(columns=col_map)

            for _, row in df.iterrows():
                load_id = int(time.time_ns())
                time.sleep(0.000001)  # Ensure unique IDs

                self._conn.execute("""
                    INSERT INTO ldr_yfinance (
                        load_id, load_timestamp, batch_id,
                        symbol, date, open, high, low, close, adj_close, volume,
                        process_status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'pending')
                """, [
                    load_id, now, batch_id,
                    symbol,
                    row.get("date"),
                    float(row["open"]) if "open" in row and not _is_nan(row["open"]) else None,
                    float(row["high"]) if "high" in row and not _is_nan(row["high"]) else None,
                    float(row["low"]) if "low" in row and not _is_nan(row["low"]) else None,
                    float(row["close"]) if "close" in row and not _is_nan(row["close"]) else None,
                    float(row["adj_close"]) if "adj_close" in row and not _is_nan(row.get("adj_close")) else None,
                    int(row["volume"]) if "volume" in row and not _is_nan(row["volume"]) else None,
                ])
                total += 1

        return total

    # ── Step 3-4: Security matching ───────────────────────────

    def _match_securities(
        self,
        batch_id: str,
        sec_type: str,
    ) -> tuple[int, int, int]:
        """Match each symbol to SecurityMaster. Returns (matched, created, rejected)."""
        symbols = self._conn.execute("""
            SELECT DISTINCT symbol FROM ldr_yfinance
            WHERE batch_id = $1 AND process_status = 'pending'
        """, [batch_id]).fetchall()

        matched = 0
        created = 0
        rejected = 0

        for (symbol,) in symbols:
            try:
                # Build identifiers based on sec_type
                # CRYPTO/FX use SYMBOL, everything else uses TICKER
                if sec_type in ("CRYPTO", "FX"):
                    identifiers = {"SYMBOL": symbol}
                elif sec_type == "INDEX":
                    identifiers = {"INDEX_CODE": symbol}
                else:
                    identifiers = {"TICKER": symbol}

                # Check if already exists
                existing = self._master.lookup(sec_type, identifiers)

                if existing is not None:
                    security_base_id = existing
                    status = "matched"
                    matched_count = self._conn.execute("""
                        SELECT COUNT(*) FROM ldr_yfinance
                        WHERE batch_id = $1 AND symbol = $2
                    """, [batch_id, symbol]).fetchone()[0]
                    matched += matched_count
                else:
                    security_base_id = self._master.match_or_create(
                        sec_type,
                        identifiers,
                        source="yfinance",
                        batch_id=batch_id,
                    )
                    status = "created"
                    created_count = self._conn.execute("""
                        SELECT COUNT(*) FROM ldr_yfinance
                        WHERE batch_id = $1 AND symbol = $2
                    """, [batch_id, symbol]).fetchone()[0]
                    created += created_count

                # Update all rows for this symbol
                self._conn.execute("""
                    UPDATE ldr_yfinance
                    SET security_base_id = $1, process_status = $2
                    WHERE batch_id = $3 AND symbol = $4
                """, [security_base_id, status, batch_id, symbol])

            except Exception as e:
                self._log.error(
                    f"Security matching failed for {symbol}: {e}",
                    tags={"datastore.loader", "security_resolve"},
                )
                rej_count = self._conn.execute("""
                    SELECT COUNT(*) FROM ldr_yfinance
                    WHERE batch_id = $1 AND symbol = $2
                """, [batch_id, symbol]).fetchone()[0]
                rejected += rej_count

                self._conn.execute("""
                    UPDATE ldr_yfinance
                    SET process_status = 'rejected', reject_reason = $1
                    WHERE batch_id = $2 AND symbol = $3
                """, [str(e), batch_id, symbol])

        return matched, created, rejected

    # ── Step 5: Store prices ──────────────────────────────────

    def _store_prices(self, batch_id: str) -> int:
        """Insert matched rows into fact_price_daily."""
        rows = self._conn.execute("""
            SELECT symbol, date, open, high, low, close, adj_close, volume,
                   security_base_id
            FROM ldr_yfinance
            WHERE batch_id = $1 AND process_status IN ('matched', 'created')
              AND close IS NOT NULL
        """, [batch_id]).fetchall()

        stored = 0
        for row in rows:
            symbol, trade_date, open_, high, low, close, adj_close, volume, sec_base_id = row
            price_bpk = f"{sec_base_id}|{trade_date}"
            keys = EntityKeys.create(price_bpk)

            try:
                self._conn.execute("""
                    INSERT INTO fact_price_daily (
                        price_hist_id, price_base_id, price_bpk,
                        security_base_id, trade_date,
                        open, high, low, close, adj_close, volume,
                        source, batch_id, created_by
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'yfinance', $12, 'yfinance_loader')
                """, [
                    keys.hist_id, keys.base_id, keys.bpk,
                    sec_base_id, trade_date,
                    open_, high, low, close, adj_close, volume,
                    batch_id,
                ])
                stored += 1
            except Exception as e:
                # Duplicate price (same security+date) — skip
                self._log.debug(
                    f"Price insert skipped for {symbol} {trade_date}: {e}",
                    tags={"datastore.loader"},
                )

        self._log.debug(
            f"Stored {stored} prices in fact_price_daily",
            tags={"data_pipeline"},
            batch_id=batch_id,
        )

        return stored

    # ── Step 6: Archive ───────────────────────────────────────

    def _archive(self, batch_id: str) -> int:
        """Copy working table rows to ldr_yfinance_hist."""
        result = self._conn.execute("""
            INSERT INTO ldr_yfinance_hist
            SELECT load_id, load_timestamp, batch_id,
                   symbol, date, open, high, low, close, adj_close, volume,
                   security_base_id, process_status, reject_reason,
                   CURRENT_TIMESTAMP
            FROM ldr_yfinance
            WHERE batch_id = $1
        """, [batch_id])

        count = self._conn.execute("""
            SELECT COUNT(*) FROM ldr_yfinance_hist WHERE batch_id = $1
        """, [batch_id]).fetchone()[0]

        self._log.debug(
            f"Archived {count} rows to ldr_yfinance_hist",
            tags={"data_pipeline"},
        )
        return count

    # ── Step 7: Safe wipe ─────────────────────────────────────

    def _safe_wipe(self, batch_id: str) -> bool:
        """
        Wipe working table ONLY if all records have terminal status.
        Safety check per §2.6.
        """
        pending = self._conn.execute("""
            SELECT COUNT(*) FROM ldr_yfinance
            WHERE batch_id = $1 AND process_status = 'pending'
        """, [batch_id]).fetchone()[0]

        if pending > 0:
            self._log.warning(
                f"Cannot wipe ldr_yfinance: {pending} pending records in batch {batch_id}",
                tags={"data_pipeline"},
            )
            return False

        self._conn.execute(
            "DELETE FROM ldr_yfinance WHERE batch_id = $1",
            [batch_id],
        )

        self._log.debug(
            f"Wiped ldr_yfinance for batch {batch_id}",
            tags={"data_pipeline"},
        )
        return True

    # ── Query prices ──────────────────────────────────────────

    def get_prices(
        self,
        security_base_id: int,
        start: str | None = None,
        end: str | None = None,
    ):
        """Query stored prices from fact_price_daily as Polars DataFrame."""
        import polars as pl

        query = "SELECT trade_date, open, high, low, close, adj_close, volume FROM fact_price_daily WHERE security_base_id = $1"
        params = [security_base_id]

        if start:
            query += " AND trade_date >= $2::DATE"
            params.append(start)
        if end:
            idx = len(params) + 1
            query += f" AND trade_date <= ${idx}::DATE"
            params.append(end)

        query += " ORDER BY trade_date"

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return pl.DataFrame()

        cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        return pl.DataFrame([dict(zip(cols, r)) for r in rows])


def _is_nan(val) -> bool:
    """Check if value is NaN."""
    if val is None:
        return True
    try:
        return np.isnan(float(val))
    except (TypeError, ValueError):
        return False
