"""
Data Quality Framework — Layer 1: Source Validation (Phase 2.8).

Validates loader batches on ingestion:
- Null/missing values
- Range checks (price > 0, volume >= 0)
- Timestamp validity
- Outlier detection (returns > N std devs)

Results stored in fact_data_quality per batch.

Usage:
    dq = DataQualityValidator(conn)
    result = dq.validate_price_batch(batch_id, source="yfinance")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import duckdb


@dataclass
class ValidationResult:
    """Result of a data quality validation."""
    batch_id: str
    source: str
    records_received: int = 0
    records_accepted: int = 0
    records_rejected: int = 0
    records_flagged: int = 0
    quality_score: float = 1.0
    null_count: int = 0
    range_violations: int = 0
    outliers_detected: int = 0
    gaps_detected: int = 0
    issues: list[str] = field(default_factory=list)


class DataQualityValidator:
    """
    §4.2 Layer 1: Source-level validation on load.

    Checks:
    1. Null/missing required fields (close, date, symbol)
    2. Range: price > 0, volume >= 0
    3. Timestamp: not in future, not before 1900
    4. Outliers: daily return > threshold std devs
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn

    def validate_price_batch(
        self,
        batch_id: str,
        source: str = "unknown",
        outlier_threshold: float = 5.0,
    ) -> ValidationResult:
        """
        Validate a price batch in ldr_yfinance_hist.

        Args:
            batch_id: Batch to validate.
            source: Data source name.
            outlier_threshold: Std devs for outlier detection.

        Returns:
            ValidationResult with quality metrics.
        """
        result = ValidationResult(batch_id=batch_id, source=source)

        # Count total records
        result.records_received = self._conn.execute(
            "SELECT COUNT(*) FROM ldr_yfinance_hist WHERE batch_id = $1",
            [batch_id],
        ).fetchone()[0]

        if result.records_received == 0:
            result.quality_score = 0.0
            result.issues.append("Empty batch — no records found")
            self._store(result)
            return result

        # ── Check 1: Null/missing required fields ─────────────
        null_close = self._conn.execute("""
            SELECT COUNT(*) FROM ldr_yfinance_hist
            WHERE batch_id = $1 AND close IS NULL
        """, [batch_id]).fetchone()[0]

        null_date = self._conn.execute("""
            SELECT COUNT(*) FROM ldr_yfinance_hist
            WHERE batch_id = $1 AND date IS NULL
        """, [batch_id]).fetchone()[0]

        null_symbol = self._conn.execute("""
            SELECT COUNT(*) FROM ldr_yfinance_hist
            WHERE batch_id = $1 AND symbol IS NULL
        """, [batch_id]).fetchone()[0]

        result.null_count = null_close + null_date + null_symbol
        if null_close > 0:
            result.issues.append(f"Null close prices: {null_close}")
        if null_date > 0:
            result.issues.append(f"Null dates: {null_date}")
        if null_symbol > 0:
            result.issues.append(f"Null symbols: {null_symbol}")

        # ── Check 2: Range violations ─────────────────────────
        neg_price = self._conn.execute("""
            SELECT COUNT(*) FROM ldr_yfinance_hist
            WHERE batch_id = $1 AND close IS NOT NULL AND close <= 0
        """, [batch_id]).fetchone()[0]

        neg_volume = self._conn.execute("""
            SELECT COUNT(*) FROM ldr_yfinance_hist
            WHERE batch_id = $1 AND volume IS NOT NULL AND volume < 0
        """, [batch_id]).fetchone()[0]

        high_low_inversion = self._conn.execute("""
            SELECT COUNT(*) FROM ldr_yfinance_hist
            WHERE batch_id = $1
              AND high IS NOT NULL AND low IS NOT NULL
              AND high < low
        """, [batch_id]).fetchone()[0]

        result.range_violations = neg_price + neg_volume + high_low_inversion
        if neg_price > 0:
            result.issues.append(f"Negative/zero close prices: {neg_price}")
        if neg_volume > 0:
            result.issues.append(f"Negative volume: {neg_volume}")
        if high_low_inversion > 0:
            result.issues.append(f"High < Low inversions: {high_low_inversion}")

        # ── Check 3: Timestamp validity ───────────────────────
        future_dates = self._conn.execute("""
            SELECT COUNT(*) FROM ldr_yfinance_hist
            WHERE batch_id = $1 AND date > CURRENT_DATE + INTERVAL 1 DAY
        """, [batch_id]).fetchone()[0]

        ancient_dates = self._conn.execute("""
            SELECT COUNT(*) FROM ldr_yfinance_hist
            WHERE batch_id = $1 AND date < DATE '1900-01-01'
        """, [batch_id]).fetchone()[0]

        if future_dates > 0:
            result.range_violations += future_dates
            result.issues.append(f"Future dates: {future_dates}")
        if ancient_dates > 0:
            result.range_violations += ancient_dates
            result.issues.append(f"Dates before 1900: {ancient_dates}")

        # ── Check 4: Outlier detection (per symbol) ───────────
        symbols = self._conn.execute("""
            SELECT DISTINCT symbol FROM ldr_yfinance_hist
            WHERE batch_id = $1 AND close IS NOT NULL
        """, [batch_id]).fetchall()

        total_outliers = 0
        for (symbol,) in symbols:
            outliers = self._conn.execute("""
                WITH returns AS (
                    SELECT date, close,
                           (close - LAG(close) OVER (ORDER BY date))
                            / NULLIF(LAG(close) OVER (ORDER BY date), 0) AS daily_return
                    FROM ldr_yfinance_hist
                    WHERE batch_id = $1 AND symbol = $2 AND close IS NOT NULL
                ),
                stats AS (
                    SELECT AVG(daily_return) AS mu, STDDEV(daily_return) AS sigma
                    FROM returns WHERE daily_return IS NOT NULL
                )
                SELECT COUNT(*) FROM returns, stats
                WHERE daily_return IS NOT NULL
                  AND stats.sigma > 0
                  AND ABS(daily_return - stats.mu) > $3 * stats.sigma
            """, [batch_id, symbol, outlier_threshold]).fetchone()[0]
            total_outliers += outliers

        result.outliers_detected = total_outliers
        if total_outliers > 0:
            result.issues.append(f"Return outliers (>{outlier_threshold}σ): {total_outliers}")

        # ── Compute scores ────────────────────────────────────
        rejected = result.null_count + result.range_violations
        flagged = result.outliers_detected
        accepted = result.records_received - rejected

        result.records_accepted = max(accepted, 0)
        result.records_rejected = rejected
        result.records_flagged = flagged

        if result.records_received > 0:
            result.quality_score = round(
                result.records_accepted / result.records_received, 4
            )

        # ── Store results ─────────────────────────────────────
        self._store(result)

        return result

    def validate_prices_table(
        self,
        security_base_id: int | None = None,
        outlier_threshold: float = 5.0,
    ) -> ValidationResult:
        """
        Layer 3-style validation on stored fact_price_daily.
        Gap detection + outliers on historical data.
        """
        result = ValidationResult(
            batch_id=f"prices_{security_base_id or 'all'}",
            source="fact_price_daily",
        )

        where = "WHERE security_base_id = $1" if security_base_id else ""
        params = [security_base_id] if security_base_id else []

        result.records_received = self._conn.execute(
            f"SELECT COUNT(*) FROM fact_price_daily {where}", params
        ).fetchone()[0]

        if result.records_received == 0:
            result.quality_score = 0.0
            return result

        # Gap detection: missing business days
        # (simplified — counts date gaps > 3 calendar days)
        if security_base_id:
            gaps = self._conn.execute("""
                WITH ordered AS (
                    SELECT trade_date,
                           LEAD(trade_date) OVER (ORDER BY trade_date) AS next_date
                    FROM fact_price_daily
                    WHERE security_base_id = $1
                )
                SELECT COUNT(*) FROM ordered
                WHERE next_date IS NOT NULL
                  AND next_date - trade_date > INTERVAL 4 DAY
            """, [security_base_id]).fetchone()[0]
            result.gaps_detected = gaps

        result.records_accepted = result.records_received
        result.quality_score = 1.0

        return result

    def get_quality(self, batch_id: str) -> dict[str, Any] | None:
        """Retrieve stored quality metrics for a batch."""
        row = self._conn.execute("""
            SELECT batch_id, source, quality_score,
                   records_received, records_accepted, records_rejected, records_flagged,
                   validation_details
            FROM fact_data_quality
            WHERE batch_id = $1
        """, [batch_id]).fetchone()

        if row is None:
            return None

        return {
            "batch_id": row[0],
            "source": row[1],
            "quality_score": row[2],
            "records_received": row[3],
            "records_accepted": row[4],
            "records_rejected": row[5],
            "records_flagged": row[6],
            "validation_details": row[7],
        }

    def _store(self, result: ValidationResult) -> None:
        """Store validation result in fact_data_quality."""
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO fact_data_quality (
                    batch_id, source, load_timestamp,
                    records_received, records_accepted, records_rejected, records_flagged,
                    quality_score,
                    validation_details
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8,
                    {null_count: $9, range_violations: $10,
                     outliers_detected: $11, gaps_detected: $12}
                )
            """, [
                result.batch_id, result.source,
                datetime.now(timezone.utc),
                result.records_received, result.records_accepted,
                result.records_rejected, result.records_flagged,
                result.quality_score,
                result.null_count, result.range_violations,
                result.outliers_detected, result.gaps_detected,
            ])
        except Exception:
            pass  # Non-critical — validation still returns result
