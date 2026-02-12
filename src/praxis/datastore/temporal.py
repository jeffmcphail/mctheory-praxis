"""
Bi-temporal query helpers (Phase 2.9).

AS-IS: "What's the current state?" → uses today as system-state date
AS-WAS: "What did we know at time T?" → uses historical run_timestamp

Both use the same vt2_ views — the only difference is which date
goes into the BETWEEN start_date AND end_date filter.

Usage:
    from praxis.datastore.temporal import TemporalQuery

    tq = TemporalQuery(conn)

    # AS-IS: current knowledge of AAPL
    sec = tq.security_as_is(security_base_id)

    # AS-WAS: what we knew about AAPL when the backtest ran
    sec = tq.security_as_was(security_base_id, as_of="2024-01-10")

    # Compare: did a correction change anything?
    diff = tq.compare(security_base_id, as_was_date="2024-01-10")
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

import duckdb

from praxis.logger.core import PraxisLogger


class TemporalQuery:
    """
    §2.3-2.4: AS-IS / AS-WAS query patterns.

    Uses vt2_security and vt2_model_definition views with
    start_date/end_date derived from hist_id sequences.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn
        self._log = PraxisLogger.instance()

    # ── Security queries ──────────────────────────────────────

    def security_as_is(
        self, security_base_id: int
    ) -> dict[str, Any] | None:
        """
        AS-IS: Current version of a security.

        "What do we know NOW about this security?"
        Uses today as system-state date → picks up all corrections.
        """
        row = self._conn.execute("""
            SELECT *
            FROM vt2_security
            WHERE security_base_id = $1
              AND CURRENT_DATE BETWEEN start_date AND end_date
        """, [security_base_id]).fetchone()

        if row is None:
            return None
        cols = [desc[0] for desc in self._conn.description]
        return dict(zip(cols, row))

    def security_as_was(
        self,
        security_base_id: int,
        as_of: str | date | datetime,
    ) -> dict[str, Any] | None:
        """
        AS-WAS: Historical version of a security at a specific system-state date.

        "What did we know about this security when the backtest ran on {as_of}?"
        Uses the original run date → reproduces exactly what was seen then.
        """
        if isinstance(as_of, (date, datetime)):
            as_of_str = as_of.isoformat()[:10]
        else:
            as_of_str = as_of

        row = self._conn.execute("""
            SELECT *
            FROM vt2_security
            WHERE security_base_id = $1
              AND $2::DATE BETWEEN start_date AND end_date
        """, [security_base_id, as_of_str]).fetchone()

        if row is None:
            return None
        cols = [desc[0] for desc in self._conn.description]
        return dict(zip(cols, row))

    def security_all_versions(
        self, security_base_id: int
    ) -> list[dict[str, Any]]:
        """Get all temporal versions of a security."""
        rows = self._conn.execute("""
            SELECT *
            FROM vt2_security
            WHERE security_base_id = $1
            ORDER BY start_date ASC
        """, [security_base_id]).fetchall()

        cols = [desc[0] for desc in self._conn.description]
        return [dict(zip(cols, row)) for row in rows]

    # ── Model definition queries ──────────────────────────────

    def model_as_is(
        self, model_def_base_id: int
    ) -> dict[str, Any] | None:
        """AS-IS: Current version of a model definition."""
        row = self._conn.execute("""
            SELECT *
            FROM vt2_model_definition
            WHERE model_def_base_id = $1
              AND CURRENT_DATE BETWEEN start_date AND end_date
        """, [model_def_base_id]).fetchone()

        if row is None:
            return None
        cols = [desc[0] for desc in self._conn.description]
        return dict(zip(cols, row))

    def model_as_was(
        self,
        model_def_base_id: int,
        as_of: str | date | datetime,
    ) -> dict[str, Any] | None:
        """AS-WAS: Historical model definition at a system-state date."""
        if isinstance(as_of, (date, datetime)):
            as_of_str = as_of.isoformat()[:10]
        else:
            as_of_str = as_of

        row = self._conn.execute("""
            SELECT *
            FROM vt2_model_definition
            WHERE model_def_base_id = $1
              AND $2::DATE BETWEEN start_date AND end_date
        """, [model_def_base_id, as_of_str]).fetchone()

        if row is None:
            return None
        cols = [desc[0] for desc in self._conn.description]
        return dict(zip(cols, row))

    # ── Compare: AS-IS vs AS-WAS ──────────────────────────────

    def compare_security(
        self,
        security_base_id: int,
        as_was_date: str | date,
    ) -> dict[str, Any]:
        """
        Compare AS-IS vs AS-WAS for a security.

        Returns:
            {
                "changed": bool,
                "as_is": dict | None,
                "as_was": dict | None,
                "differences": {field: {"old": val, "new": val}},
            }
        """
        as_is = self.security_as_is(security_base_id)
        as_was = self.security_as_was(security_base_id, as_was_date)

        differences = {}
        if as_is and as_was:
            # Compare relevant fields (exclude temporal metadata)
            skip = {"security_hist_id", "start_date", "end_date", "rn_within_day"}
            for key in as_is:
                if key in skip:
                    continue
                if as_is.get(key) != as_was.get(key):
                    differences[key] = {
                        "old": as_was.get(key),
                        "new": as_is.get(key),
                    }

        return {
            "changed": len(differences) > 0,
            "as_is": as_is,
            "as_was": as_was,
            "differences": differences,
        }
