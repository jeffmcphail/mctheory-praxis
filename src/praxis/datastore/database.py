"""
DuckDB initialization and connection management.

§2.4: Applications NEVER query dim_*/fact_* tables directly. Always use views.

Supports two modes:
- persistent: DuckDB file on disk, full temporal plumbing
- ephemeral: In-memory DuckDB, no views (Phase 1.9)

Phase 1.3 scope: Create tables, views, wire logger DatabaseAdapter.
"""

from pathlib import Path
from typing import Optional

import duckdb

from praxis.datastore.schema import ALL_TABLES, ALL_INDEXES, ALL_VIEWS
from praxis.logger.core import PraxisLogger
from praxis.logger.adapters import DatabaseAdapter
from praxis.logger.records import LogLevel


class PraxisDatabase:
    """
    DuckDB connection manager with schema initialization.

    Usage:
        db = PraxisDatabase("data/praxis.duckdb")
        db.initialize()                  # Creates tables + views
        db.wire_logger()                 # Connects DatabaseAdapter to fact_log
        conn = db.connection             # Use for queries

        # Ephemeral mode
        db = PraxisDatabase.ephemeral()  # In-memory, no file
        db.initialize()
    """

    def __init__(self, path: str | Path | None = None):
        """
        Create database connection.

        Args:
            path: Path to DuckDB file. None = in-memory (ephemeral mode).
        """
        self._path = Path(path) if path else None
        self._mode = "persistent" if path else "ephemeral"

        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self._path))
        else:
            self._conn = duckdb.connect(":memory:")

        self._initialized = False
        self._log = PraxisLogger.instance()

    @classmethod
    def ephemeral(cls) -> "PraxisDatabase":
        """Create an in-memory database (no file, no temporal views)."""
        return cls(path=None)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Raw DuckDB connection for queries."""
        return self._conn

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ── Schema Initialization ─────────────────────────────────────

    def initialize(self) -> dict:
        """
        Create all tables, indexes, and views.

        Returns dict with counts: {tables: N, indexes: N, views: N}
        """
        self._log.info(
            f"Initializing DuckDB ({self._mode})",
            tags={"datastore.init"},
            mode=self._mode,
            path=str(self._path) if self._path else ":memory:",
        )

        results = {"tables": 0, "indexes": 0, "views": 0}

        # Tables
        for name, ddl in ALL_TABLES:
            self._conn.execute(ddl)
            results["tables"] += 1
            self._log.debug(f"Table created: {name}", tags={"datastore.init"})

        # Indexes
        for name, ddl in ALL_INDEXES:
            self._conn.execute(ddl)
            results["indexes"] += 1
            self._log.debug(f"Index created: {name}", tags={"datastore.init"})

        # Views (skip in ephemeral mode — no temporal plumbing)
        if self._mode == "persistent":
            for name, ddl in ALL_VIEWS:
                self._conn.execute(ddl)
                results["views"] += 1
                self._log.debug(f"View created: {name}", tags={"datastore.init"})
        else:
            self._log.info(
                "Ephemeral mode: views skipped",
                tags={"datastore.init"},
            )

        self._initialized = True
        self._log.info(
            f"DuckDB initialized: {results['tables']} tables, "
            f"{results['indexes']} indexes, {results['views']} views",
            tags={"datastore.init"},
        )
        return results

    # ── Logger Integration ────────────────────────────────────────

    def wire_logger(self) -> bool:
        """
        Connect the logger's DatabaseAdapter to this database.

        Finds the DatabaseAdapter in the logger's adapters and
        injects this connection, enabling log writes to fact_log.

        Returns True if wired successfully, False if no DatabaseAdapter found.
        """
        log = PraxisLogger.instance()
        db_adapter = log.get_adapter("database")

        if db_adapter is None or not isinstance(db_adapter, DatabaseAdapter):
            self._log.debug(
                "No database adapter found in logger — skipping wire",
                tags={"datastore.init"},
            )
            return False

        db_adapter.connect(self._conn)
        self._log.info(
            "Logger DatabaseAdapter wired to DuckDB",
            tags={"datastore.init"},
        )
        return True

    # ── Schema Introspection ──────────────────────────────────────

    def tables(self) -> list[str]:
        """List all user tables."""
        result = self._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        ).fetchall()
        return [row[0] for row in result]

    def views(self) -> list[str]:
        """List all views."""
        result = self._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'VIEW' "
            "ORDER BY table_name"
        ).fetchall()
        return [row[0] for row in result]

    def table_columns(self, table_name: str) -> list[dict]:
        """Get column info for a table."""
        result = self._conn.execute(
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table_name],
        ).fetchall()
        return [
            {"name": row[0], "type": row[1], "nullable": row[2]}
            for row in result
        ]

    def row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        result = self._conn.execute(
            f'SELECT COUNT(*) FROM "{table_name}"'
        ).fetchone()
        return result[0] if result else 0

    def status(self) -> dict:
        """Database status for diagnostics."""
        return {
            "mode": self._mode,
            "path": str(self._path) if self._path else ":memory:",
            "initialized": self._initialized,
            "tables": {t: self.row_count(t) for t in self.tables()},
            "views": self.views(),
        }

    # ── Cleanup ───────────────────────────────────────────────────

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
