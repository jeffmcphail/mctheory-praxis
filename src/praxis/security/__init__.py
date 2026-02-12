"""
Security Master (§3.6).

Match-or-create protocol: walk identifier hierarchy,
match existing security or create new. Conflicts go to conflict_queue.

Usage:
    master = SecurityMaster(conn)
    base_id = master.match_or_create(
        sec_type="EQUITY",
        identifiers={"TICKER": "AAPL", "ISIN": "US0378331005"},
        source="yfinance",
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

import duckdb

from praxis.datastore.keys import EntityKeys
from praxis.logger.core import PraxisLogger
from praxis.security.hierarchy import (
    SECID_HIERARCHY,
    SECID_TO_COLUMN,
    SecIdType,
    SecType,
    get_preferred_bpk,
)


class SecurityMaster:
    """
    §3.6: Simplified matching protocol.

    match_or_create() is the single entry point:
    1. Try to match any identifier to existing security (walk hierarchy)
    2. If match found, populate any missing identifiers (new hist_id version)
    3. If no match, create new security
    4. Conflicts go to conflict_queue
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn
        self._log = PraxisLogger.instance()

    def match_or_create(
        self,
        sec_type: str,
        identifiers: dict[str, str],
        *,
        name: str | None = None,
        currency: str | None = None,
        exchange: str | None = None,
        sector: str | None = None,
        industry: str | None = None,
        country: str | None = None,
        source: str = "unknown",
        batch_id: str | None = None,
    ) -> int:
        """
        Match incoming identifiers to existing security, or create new.

        Args:
            sec_type: Security type (EQUITY, BOND, CRYPTO, etc.).
            identifiers: {SecIdType: value} e.g., {"TICKER": "AAPL"}.
            name: Security name.
            currency: Trading currency.
            exchange: Primary exchange.
            source: Data source that provided these identifiers.
            batch_id: Batch identifier for conflict tracking.

        Returns:
            security_base_id (int).
        """
        sec_type_enum = SecType(sec_type)
        hierarchy = SECID_HIERARCHY[sec_type_enum]
        norm_ids = {k.upper(): v for k, v in identifiers.items()}

        self._log.info(
            f"SecurityMaster: match_or_create {sec_type} "
            f"identifiers={list(norm_ids.keys())}",
            tags={"security_resolve"},
            sec_type=sec_type,
            source=source,
        )

        # ── Step 1: Try to match any identifier ──────────────
        for secid_type in hierarchy:
            value = norm_ids.get(secid_type.value)
            if not value:
                continue

            column = SECID_TO_COLUMN[secid_type]
            existing = self._lookup_by_identifier(column, value)

            if existing is not None:
                existing_base_id = existing["security_base_id"]

                # Check for conflicts: same identifier, different sec_type
                if existing["sec_type"] != sec_type:
                    self._log.warning(
                        f"SecurityMaster: sec_type mismatch for {secid_type.value}={value}: "
                        f"existing={existing['sec_type']}, incoming={sec_type}",
                        tags={"security_resolve"},
                    )
                    self._queue_conflict(
                        security_base_id=existing_base_id,
                        source=source,
                        batch_id=batch_id or "unknown",
                        conflict_type="sectype_mismatch",
                        detail={
                            "secid_type": secid_type.value,
                            "secid_value": value,
                            "existing_sec_type": existing["sec_type"],
                            "incoming_sec_type": sec_type,
                        },
                    )
                    # Still return the existing — conflict is flagged for review
                    return existing_base_id

                # Match found — populate missing identifiers
                updated = self._update_missing_identifiers(
                    existing_base_id, norm_ids, name, currency,
                    exchange, sector, industry, country, source,
                )

                # Audit all identifiers from this source
                self._audit_identifiers(existing_base_id, norm_ids, source)

                self._log.info(
                    f"SecurityMaster: matched {secid_type.value}={value} → "
                    f"base_id={existing_base_id}"
                    + (" (updated)" if updated else ""),
                    tags={"security_resolve"},
                )
                return existing_base_id

        # ── Step 2: No match — create new ────────────────────
        bpk, preferred = get_preferred_bpk(sec_type, norm_ids)
        keys = EntityKeys.create(bpk)

        self._insert_new_security(
            keys, sec_type, norm_ids, name, currency,
            exchange, sector, industry, country, source,
        )

        # Audit all identifiers
        self._audit_identifiers(keys.base_id, norm_ids, source)

        self._log.info(
            f"SecurityMaster: created new {sec_type} → "
            f"bpk={bpk}, base_id={keys.base_id}",
            tags={"security_resolve"},
        )

        return keys.base_id

    # ── Lookup ────────────────────────────────────────────────

    def _lookup_by_identifier(
        self, column: str, value: str
    ) -> dict[str, Any] | None:
        """Look up security by identifier column in current view."""
        # Use vew_security (latest version) for matching
        try:
            row = self._conn.execute(f"""
                SELECT security_base_id, security_bpk, sec_type
                FROM vew_security
                WHERE {column} = $1
                LIMIT 1
            """, [value]).fetchone()
        except duckdb.CatalogException:
            # vew_security might not exist in ephemeral mode — fall back to raw table
            row = self._conn.execute(f"""
                SELECT security_base_id, security_bpk, sec_type
                FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY security_base_id
                        ORDER BY security_hist_id DESC
                    ) AS rn FROM dim_security
                ) WHERE rn = 1 AND {column} = $1
                LIMIT 1
            """, [value]).fetchone()

        if row is None:
            return None

        return {
            "security_base_id": row[0],
            "security_bpk": row[1],
            "sec_type": row[2],
        }

    def lookup(self, sec_type: str, identifiers: dict[str, str]) -> int | None:
        """
        Look up a security without creating.

        Returns security_base_id or None if not found.
        """
        sec_type_enum = SecType(sec_type)
        hierarchy = SECID_HIERARCHY[sec_type_enum]
        norm_ids = {k.upper(): v for k, v in identifiers.items()}

        for secid_type in hierarchy:
            value = norm_ids.get(secid_type.value)
            if not value:
                continue
            column = SECID_TO_COLUMN[secid_type]
            existing = self._lookup_by_identifier(column, value)
            if existing is not None:
                return existing["security_base_id"]

        return None

    def get(self, security_base_id: int) -> dict[str, Any] | None:
        """Get full security record by base_id."""
        try:
            row = self._conn.execute("""
                SELECT * FROM vew_security
                WHERE security_base_id = $1
            """, [security_base_id]).fetchone()
        except duckdb.CatalogException:
            row = self._conn.execute("""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY security_base_id
                        ORDER BY security_hist_id DESC
                    ) AS rn FROM dim_security
                ) WHERE rn = 1 AND security_base_id = $1
            """, [security_base_id]).fetchone()

        if row is None:
            return None

        cols = [desc[0] for desc in self._conn.description]
        return dict(zip(cols, row))

    # ── Insert / Update ───────────────────────────────────────

    def _insert_new_security(
        self,
        keys: EntityKeys,
        sec_type: str,
        identifiers: dict[str, str],
        name: str | None,
        currency: str | None,
        exchange: str | None,
        sector: str | None,
        industry: str | None,
        country: str | None,
        source: str,
    ) -> None:
        """Insert a brand new security into dim_security."""
        # Build column values from identifiers
        id_cols = {}
        for secid_name, value in identifiers.items():
            try:
                secid_type = SecIdType(secid_name)
                col = SECID_TO_COLUMN[secid_type]
                id_cols[col] = value
            except (ValueError, KeyError):
                continue

        self._conn.execute("""
            INSERT INTO dim_security (
                security_hist_id, security_base_id, security_bpk,
                sec_type,
                isin, cusip, figi, sedol, ticker, ric,
                bloomberg_id, permid, cik, lei,
                contract_address, symbol, occ_symbol, exchange_code,
                name, currency, exchange, sector, industry, country,
                status, created_by
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14,
                $15, $16, $17, $18,
                $19, $20, $21, $22, $23, $24,
                'ACTIVE', $25
            )
        """, [
            keys.hist_id, keys.base_id, keys.bpk,
            sec_type,
            id_cols.get("isin"), id_cols.get("cusip"),
            id_cols.get("figi"), id_cols.get("sedol"),
            id_cols.get("ticker"), id_cols.get("ric"),
            id_cols.get("bloomberg_id"), id_cols.get("permid"),
            id_cols.get("cik"), id_cols.get("lei"),
            id_cols.get("contract_address"), id_cols.get("symbol"),
            id_cols.get("occ_symbol"), id_cols.get("exchange_code"),
            name, currency, exchange, sector, industry, country,
            source,
        ])

    def _update_missing_identifiers(
        self,
        security_base_id: int,
        identifiers: dict[str, str],
        name: str | None,
        currency: str | None,
        exchange: str | None,
        sector: str | None,
        industry: str | None,
        country: str | None,
        source: str,
    ) -> bool:
        """
        If any identifiers or descriptive fields are new, insert a new
        version (new hist_id) with the merged data.

        Returns True if an update was made.
        """
        current = self.get(security_base_id)
        if current is None:
            return False

        # Check if any identifier columns are missing and we have them
        needs_update = False
        merged = {}

        for secid_name, value in identifiers.items():
            try:
                secid_type = SecIdType(secid_name)
                col = SECID_TO_COLUMN[secid_type]
            except (ValueError, KeyError):
                continue

            current_val = current.get(col)
            if current_val is None and value:
                merged[col] = value
                needs_update = True

        # Check descriptive fields
        for field, new_val in [
            ("name", name), ("currency", currency), ("exchange", exchange),
            ("sector", sector), ("industry", industry), ("country", country),
        ]:
            if new_val and current.get(field) is None:
                merged[field] = new_val
                needs_update = True

        if not needs_update:
            return False

        # Insert new version with merged data
        new_keys = EntityKeys.new_version(
            current["security_bpk"], security_base_id
        )

        # Start with current values, overlay new ones
        id_cols = {}
        for secid_type in SecIdType:
            col = SECID_TO_COLUMN[secid_type]
            id_cols[col] = merged.get(col, current.get(col))

        desc = {
            "name": merged.get("name", current.get("name")),
            "currency": merged.get("currency", current.get("currency")),
            "exchange": merged.get("exchange", current.get("exchange")),
            "sector": merged.get("sector", current.get("sector")),
            "industry": merged.get("industry", current.get("industry")),
            "country": merged.get("country", current.get("country")),
        }

        self._conn.execute("""
            INSERT INTO dim_security (
                security_hist_id, security_base_id, security_bpk,
                sec_type,
                isin, cusip, figi, sedol, ticker, ric,
                bloomberg_id, permid, cik, lei,
                contract_address, symbol, occ_symbol, exchange_code,
                name, currency, exchange, sector, industry, country,
                status, created_by
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14,
                $15, $16, $17, $18,
                $19, $20, $21, $22, $23, $24,
                $25, $26
            )
        """, [
            new_keys.hist_id, new_keys.base_id, new_keys.bpk,
            current["sec_type"],
            id_cols.get("isin"), id_cols.get("cusip"),
            id_cols.get("figi"), id_cols.get("sedol"),
            id_cols.get("ticker"), id_cols.get("ric"),
            id_cols.get("bloomberg_id"), id_cols.get("permid"),
            id_cols.get("cik"), id_cols.get("lei"),
            id_cols.get("contract_address"), id_cols.get("symbol"),
            id_cols.get("occ_symbol"), id_cols.get("exchange_code"),
            desc["name"], desc["currency"], desc["exchange"],
            desc["sector"], desc["industry"], desc["country"],
            current.get("status", "ACTIVE"), source,
        ])

        self._log.debug(
            f"Updated security {security_base_id}: "
            f"added {list(merged.keys())}",
            tags={"security_resolve"},
        )

        return True

    # ── Audit ─────────────────────────────────────────────────

    def _audit_identifiers(
        self,
        security_base_id: int,
        identifiers: dict[str, str],
        source: str,
    ) -> None:
        """Record which identifiers came from which source."""
        for secid_name, value in identifiers.items():
            try:
                SecIdType(secid_name)  # Validate
            except ValueError:
                continue

            audit_bpk = f"{security_base_id}|{secid_name}|{source}"
            keys = EntityKeys.create(audit_bpk)

            self._conn.execute("""
                INSERT INTO dim_security_identifier_audit (
                    audit_hist_id, audit_base_id, audit_bpk,
                    security_base_id, secid_type, secid_value,
                    source, confidence
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'HIGH')
            """, [
                keys.hist_id, keys.base_id, keys.bpk,
                security_base_id, secid_name, value, source,
            ])

    # ── Conflict Queue ────────────────────────────────────────

    def _queue_conflict(
        self,
        security_base_id: int,
        source: str,
        batch_id: str,
        conflict_type: str,
        detail: dict,
    ) -> None:
        """Add a conflict to the queue for resolution."""
        import time
        conflict_id = int(time.time_ns())

        self._conn.execute("""
            INSERT INTO conflict_queue (
                conflict_id, security_base_id, source, batch_id,
                conflict_type, conflict_detail,
                resolution_status
            ) VALUES ($1, $2, $3, $4, $5, $6::JSON, 'OPEN')
        """, [
            conflict_id, security_base_id, source, batch_id,
            conflict_type, json.dumps(detail),
        ])

        self._log.warning(
            f"Conflict queued: {conflict_type} for base_id={security_base_id}",
            tags={"security_resolve"},
        )

    def get_conflicts(self, status: str = "OPEN") -> list[dict]:
        """Get conflicts from the queue."""
        rows = self._conn.execute("""
            SELECT conflict_id, security_base_id, source, batch_id,
                   conflict_type, conflict_detail, resolution_status
            FROM conflict_queue
            WHERE resolution_status = $1
            ORDER BY created_timestamp DESC
        """, [status]).fetchall()

        cols = [
            "conflict_id", "security_base_id", "source", "batch_id",
            "conflict_type", "conflict_detail", "resolution_status",
        ]
        return [dict(zip(cols, row)) for row in rows]

    # ── Stats ─────────────────────────────────────────────────

    def count(self) -> int:
        """Number of unique securities."""
        try:
            return self._conn.execute(
                "SELECT COUNT(DISTINCT security_base_id) FROM dim_security"
            ).fetchone()[0]
        except Exception:
            return 0
