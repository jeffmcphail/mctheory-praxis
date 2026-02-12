"""
Security Classifications (Phase 2.10, §15).

Core: Platform's operational taxonomy (asset_class, instrument_type, settlement_type)
External: Vendor taxonomies (GICS, ICB, ratings)
Mapping: Bridge from external codes → core classifications

Usage:
    cls = ClassificationManager(conn)
    cls.bootstrap_gics_mappings()
    cls.assign_core(security_base_id, "EQUITY", "COMMON_STOCK")
    cls.assign_external(security_base_id, "GICS", raw_code="45102010", ...)
    cls.auto_classify(security_base_id, "GICS", "45102010")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import duckdb

from praxis.datastore.keys import EntityKeys
from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  GICS Bootstrap Data — §15.7
#  Maps GICS sectors to core asset_class + instrument_type
# ═══════════════════════════════════════════════════════════════════

GICS_SECTOR_MAPPINGS = [
    # (raw_code, level_1, level_2, core_asset_class, core_instrument_type)
    ("10", "Energy", None, "EQUITY", "COMMON_STOCK"),
    ("15", "Materials", None, "EQUITY", "COMMON_STOCK"),
    ("20", "Industrials", None, "EQUITY", "COMMON_STOCK"),
    ("25", "Consumer Discretionary", None, "EQUITY", "COMMON_STOCK"),
    ("30", "Consumer Staples", None, "EQUITY", "COMMON_STOCK"),
    ("35", "Health Care", None, "EQUITY", "COMMON_STOCK"),
    ("40", "Financials", None, "EQUITY", "COMMON_STOCK"),
    ("45", "Information Technology", None, "EQUITY", "COMMON_STOCK"),
    ("50", "Communication Services", None, "EQUITY", "COMMON_STOCK"),
    ("55", "Utilities", None, "EQUITY", "COMMON_STOCK"),
    ("60", "Real Estate", None, "EQUITY", "COMMON_STOCK"),
]

# Industry group level mappings for common sub-types
GICS_INDUSTRY_MAPPINGS = [
    ("4510", "Information Technology", "Software & Services", "EQUITY", "COMMON_STOCK"),
    ("4520", "Information Technology", "Technology Hardware & Equipment", "EQUITY", "COMMON_STOCK"),
    ("4530", "Information Technology", "Semiconductors", "EQUITY", "COMMON_STOCK"),
    ("2550", "Consumer Discretionary", "Retailing", "EQUITY", "COMMON_STOCK"),
    ("3510", "Health Care", "Health Care Equipment & Services", "EQUITY", "COMMON_STOCK"),
    ("3520", "Health Care", "Pharmaceuticals & Biotechnology", "EQUITY", "COMMON_STOCK"),
    ("4010", "Financials", "Banks", "EQUITY", "COMMON_STOCK"),
    ("4020", "Financials", "Diversified Financials", "EQUITY", "COMMON_STOCK"),
    ("4030", "Financials", "Insurance", "EQUITY", "COMMON_STOCK"),
]


class ClassificationManager:
    """§15: Security classification management."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn
        self._log = PraxisLogger.instance()

    # ── Core Classification ───────────────────────────────────

    def assign_core(
        self,
        security_base_id: int,
        asset_class: str,
        instrument_type: str,
        *,
        settlement_type: str | None = None,
        derivative_underlier_type: str | None = None,
        is_otc: bool = False,
        created_by: str = "manual",
    ) -> int:
        """
        Assign core classification to a security.

        Returns class_base_id.
        """
        bpk = f"{security_base_id}|core"
        keys = EntityKeys.create(bpk)

        self._conn.execute("""
            INSERT INTO dim_classification_core (
                class_hist_id, class_base_id, class_bpk,
                security_base_id, asset_class, instrument_type,
                settlement_type, derivative_underlier_type, is_otc,
                created_by
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """, [
            keys.hist_id, keys.base_id, keys.bpk,
            security_base_id, asset_class, instrument_type,
            settlement_type, derivative_underlier_type, is_otc,
            created_by,
        ])

        self._log.debug(
            f"Core classification: {asset_class}/{instrument_type} → sec {security_base_id}",
            tags={"classification"},
        )
        return keys.base_id

    def get_core(self, security_base_id: int) -> dict[str, Any] | None:
        """Get current core classification for a security."""
        row = self._conn.execute("""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY class_base_id
                           ORDER BY class_hist_id DESC
                       ) AS rn
                FROM dim_classification_core
                WHERE security_base_id = $1
            ) WHERE rn = 1
        """, [security_base_id]).fetchone()

        if row is None:
            return None
        cols = [desc[0] for desc in self._conn.description]
        return dict(zip(cols, row))

    # ── External Classification ───────────────────────────────

    def assign_external(
        self,
        security_base_id: int,
        classification_system: str,
        *,
        version: str | None = None,
        level_1: str | None = None,
        level_2: str | None = None,
        level_3: str | None = None,
        level_4: str | None = None,
        level_5: str | None = None,
        raw_code: str | None = None,
        source: str = "manual",
        confidence: str = "HIGH",
        created_by: str = "manual",
    ) -> int:
        """
        Assign external vendor classification.

        Returns ext_class_base_id.
        """
        bpk = f"{security_base_id}|{classification_system}|{version or 'latest'}"
        keys = EntityKeys.create(bpk)

        self._conn.execute("""
            INSERT INTO dim_classification_external (
                ext_class_hist_id, ext_class_base_id, ext_class_bpk,
                security_base_id, classification_system, classification_version,
                level_1, level_2, level_3, level_4, level_5,
                raw_code, source, confidence, created_by
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        """, [
            keys.hist_id, keys.base_id, keys.bpk,
            security_base_id, classification_system, version,
            level_1, level_2, level_3, level_4, level_5,
            raw_code, source, confidence, created_by,
        ])

        return keys.base_id

    def get_external(
        self,
        security_base_id: int,
        classification_system: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get external classifications for a security."""
        if classification_system:
            rows = self._conn.execute("""
                SELECT * EXCLUDE (rn) FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY ext_class_base_id
                               ORDER BY ext_class_hist_id DESC
                           ) AS rn
                    FROM dim_classification_external
                    WHERE security_base_id = $1 AND classification_system = $2
                ) WHERE rn = 1
            """, [security_base_id, classification_system]).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT * EXCLUDE (rn) FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY ext_class_base_id
                               ORDER BY ext_class_hist_id DESC
                           ) AS rn
                    FROM dim_classification_external
                    WHERE security_base_id = $1
                ) WHERE rn = 1
            """, [security_base_id]).fetchall()

        cols = [desc[0] for desc in self._conn.description]
        return [dict(zip(cols, r)) for r in rows]

    # ── Mapping ───────────────────────────────────────────────

    def add_mapping(
        self,
        classification_system: str,
        core_asset_class: str,
        core_instrument_type: str,
        *,
        external_level_1: str | None = None,
        external_level_2: str | None = None,
        external_level_3: str | None = None,
        external_raw_code: str | None = None,
        mapping_method: str = "manual",
        confirmed: bool = True,
    ) -> int:
        """Add a mapping from external → core classification."""
        bpk = f"{classification_system}|{external_raw_code or external_level_1}|{core_asset_class}|{core_instrument_type}"
        keys = EntityKeys.create(bpk)

        self._conn.execute("""
            INSERT INTO dim_classification_mapping (
                mapping_hist_id, mapping_base_id, mapping_bpk,
                classification_system,
                external_level_1, external_level_2, external_level_3,
                external_raw_code,
                core_asset_class, core_instrument_type,
                mapping_method, confirmed, confirmed_by, confirmed_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """, [
            keys.hist_id, keys.base_id, keys.bpk,
            classification_system,
            external_level_1, external_level_2, external_level_3,
            external_raw_code,
            core_asset_class, core_instrument_type,
            mapping_method, confirmed,
            "bootstrap" if confirmed else None,
            datetime.now(timezone.utc) if confirmed else None,
        ])

        return keys.base_id

    def lookup_mapping(
        self,
        classification_system: str,
        raw_code: str,
    ) -> dict[str, Any] | None:
        """Look up core classification from an external code."""
        # Try exact raw_code match first, then prefix match
        row = self._conn.execute("""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY mapping_base_id
                           ORDER BY mapping_hist_id DESC
                       ) AS rn
                FROM dim_classification_mapping
                WHERE classification_system = $1 AND external_raw_code = $2
            ) WHERE rn = 1
            LIMIT 1
        """, [classification_system, raw_code]).fetchone()

        if row is None:
            # Try prefix match (e.g., code "45102010" matches mapping "45")
            for prefix_len in range(len(raw_code) - 1, 0, -1):
                prefix = raw_code[:prefix_len]
                row = self._conn.execute("""
                    SELECT * EXCLUDE (rn) FROM (
                        SELECT *,
                               ROW_NUMBER() OVER (
                                   PARTITION BY mapping_base_id
                                   ORDER BY mapping_hist_id DESC
                               ) AS rn
                        FROM dim_classification_mapping
                        WHERE classification_system = $1 AND external_raw_code = $2
                    ) WHERE rn = 1
                    LIMIT 1
                """, [classification_system, prefix]).fetchone()
                if row:
                    break

        if row is None:
            return None

        cols = [desc[0] for desc in self._conn.description]
        return dict(zip(cols, row))

    # ── Auto-classify ─────────────────────────────────────────

    def auto_classify(
        self,
        security_base_id: int,
        classification_system: str,
        raw_code: str,
        *,
        source: str = "auto",
    ) -> bool:
        """
        §15.6: Auto-classify security from external code via mapping table.

        Returns True if classification was assigned, False if no mapping found.
        """
        mapping = self.lookup_mapping(classification_system, raw_code)
        if mapping is None:
            self._log.warning(
                f"No mapping for {classification_system}/{raw_code}",
                tags={"classification"},
            )
            return False

        # Assign core classification
        self.assign_core(
            security_base_id,
            mapping["core_asset_class"],
            mapping["core_instrument_type"],
            created_by=f"auto:{classification_system}",
        )

        self._log.info(
            f"Auto-classified sec {security_base_id}: "
            f"{mapping['core_asset_class']}/{mapping['core_instrument_type']} "
            f"from {classification_system}/{raw_code}",
            tags={"classification"},
        )
        return True

    # ── Bootstrap ─────────────────────────────────────────────

    def bootstrap_gics_mappings(self) -> int:
        """
        §15.7: Bootstrap GICS sector + industry group mappings.

        Returns number of mappings created.
        """
        count = 0

        for raw_code, level_1, level_2, asset_class, instrument_type in GICS_SECTOR_MAPPINGS:
            self.add_mapping(
                "GICS", asset_class, instrument_type,
                external_level_1=level_1,
                external_raw_code=raw_code,
                mapping_method="bootstrap",
            )
            count += 1

        for raw_code, level_1, level_2, asset_class, instrument_type in GICS_INDUSTRY_MAPPINGS:
            self.add_mapping(
                "GICS", asset_class, instrument_type,
                external_level_1=level_1,
                external_level_2=level_2,
                external_raw_code=raw_code,
                mapping_method="bootstrap",
            )
            count += 1

        self._log.info(
            f"Bootstrapped {count} GICS mappings",
            tags={"classification"},
        )
        return count

    def count_mappings(self) -> int:
        """Total mapping entries."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM dim_classification_mapping"
        ).fetchone()[0]
