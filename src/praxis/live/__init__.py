"""
Live Trading Data Layer (Phase 4.8).

Abstract store protocol for live trading data (orders, fills, positions)
with two backends:
- DuckDB (development / testing)
- PostgreSQL + TimescaleDB (production live trading)

The spec (§2.1) mandates PostgreSQL for live trading but DuckDB-first
for development. This module provides the abstraction so the rest of
the platform doesn't care which backend is active.

Usage:
    store = LiveStore.duckdb()          # Dev
    store = LiveStore.postgres(dsn=...) # Prod
    store.initialize()
    store.insert_order(order)
    store.insert_fill(fill)
    fills = store.get_fills(model_id="burgess_v1")
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

import duckdb

from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class LiveOrder:
    """An order in the live trading system."""
    order_id: str = ""
    model_id: str = ""
    asset: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"ord_{uuid4().hex[:12]}"


@dataclass
class LiveFill:
    """A fill (execution) in the live trading system."""
    fill_id: str = ""
    order_id: str = ""
    model_id: str = ""
    asset: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    filled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    venue: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.fill_id:
            self.fill_id = f"fill_{uuid4().hex[:12]}"


@dataclass
class LivePosition:
    """A position in the live trading system."""
    model_id: str = ""
    asset: str = ""
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def side(self) -> str:
        if self.quantity > 0:
            return "long"
        elif self.quantity < 0:
            return "short"
        return "flat"


# ═══════════════════════════════════════════════════════════════════
#  Abstract Store Protocol
# ═══════════════════════════════════════════════════════════════════

class LiveStoreBackend(ABC):
    """Abstract interface for live trading data storage."""

    @abstractmethod
    def initialize(self) -> None:
        """Create tables/schema."""
        ...

    @abstractmethod
    def insert_order(self, order: LiveOrder) -> str:
        """Insert an order. Returns order_id."""
        ...

    @abstractmethod
    def update_order_status(self, order_id: str, status: OrderStatus) -> None:
        """Update an order's status."""
        ...

    @abstractmethod
    def insert_fill(self, fill: LiveFill) -> str:
        """Insert a fill. Returns fill_id."""
        ...

    @abstractmethod
    def upsert_position(self, position: LivePosition) -> None:
        """Insert or update a position."""
        ...

    @abstractmethod
    def get_orders(
        self, model_id: str | None = None, status: OrderStatus | None = None,
    ) -> list[LiveOrder]:
        """Query orders with optional filters."""
        ...

    @abstractmethod
    def get_fills(
        self, model_id: str | None = None, order_id: str | None = None,
    ) -> list[LiveFill]:
        """Query fills with optional filters."""
        ...

    @abstractmethod
    def get_positions(
        self, model_id: str | None = None,
    ) -> list[LivePosition]:
        """Query current positions."""
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> LiveOrder | None:
        """Get a single order by ID."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close connections."""
        ...


# ═══════════════════════════════════════════════════════════════════
#  DuckDB Backend (Development)
# ═══════════════════════════════════════════════════════════════════

_DUCKDB_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_orders (
    order_id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    asset VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    order_type VARCHAR NOT NULL,
    quantity DOUBLE NOT NULL,
    limit_price DOUBLE,
    stop_price DOUBLE,
    status VARCHAR NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    metadata JSON
);

CREATE TABLE IF NOT EXISTS live_fills (
    fill_id VARCHAR PRIMARY KEY,
    order_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    asset VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    quantity DOUBLE NOT NULL,
    price DOUBLE NOT NULL,
    commission DOUBLE DEFAULT 0.0,
    filled_at TIMESTAMP NOT NULL,
    venue VARCHAR,
    metadata JSON
);

CREATE TABLE IF NOT EXISTS live_positions (
    model_id VARCHAR NOT NULL,
    asset VARCHAR NOT NULL,
    quantity DOUBLE NOT NULL,
    avg_entry_price DOUBLE NOT NULL,
    current_price DOUBLE DEFAULT 0.0,
    unrealized_pnl DOUBLE DEFAULT 0.0,
    realized_pnl DOUBLE DEFAULT 0.0,
    updated_at TIMESTAMP NOT NULL,
    PRIMARY KEY (model_id, asset)
);
"""


class DuckDBLiveStore(LiveStoreBackend):
    """DuckDB implementation for development/testing."""

    def __init__(self, conn: duckdb.DuckDBPyConnection | None = None):
        self._conn = conn or duckdb.connect(":memory:")
        self._initialized = False

    def initialize(self) -> None:
        for stmt in _DUCKDB_SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)
        self._initialized = True

    def insert_order(self, order: LiveOrder) -> str:
        self._conn.execute(
            """INSERT INTO live_orders
               (order_id, model_id, asset, side, order_type, quantity,
                limit_price, stop_price, status, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                order.order_id, order.model_id, order.asset,
                order.side.value, order.order_type.value, order.quantity,
                order.limit_price, order.stop_price, order.status.value,
                order.created_at, order.updated_at, str(order.metadata),
            ],
        )
        return order.order_id

    def update_order_status(self, order_id: str, status: OrderStatus) -> None:
        now = datetime.now(timezone.utc)
        self._conn.execute(
            "UPDATE live_orders SET status = ?, updated_at = ? WHERE order_id = ?",
            [status.value, now, order_id],
        )

    def insert_fill(self, fill: LiveFill) -> str:
        self._conn.execute(
            """INSERT INTO live_fills
               (fill_id, order_id, model_id, asset, side, quantity,
                price, commission, filled_at, venue, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                fill.fill_id, fill.order_id, fill.model_id, fill.asset,
                fill.side.value, fill.quantity, fill.price, fill.commission,
                fill.filled_at, fill.venue, str(fill.metadata),
            ],
        )
        return fill.fill_id

    def upsert_position(self, position: LivePosition) -> None:
        # Delete then insert (DuckDB doesn't have ON CONFLICT in all versions)
        self._conn.execute(
            "DELETE FROM live_positions WHERE model_id = ? AND asset = ?",
            [position.model_id, position.asset],
        )
        self._conn.execute(
            """INSERT INTO live_positions
               (model_id, asset, quantity, avg_entry_price, current_price,
                unrealized_pnl, realized_pnl, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                position.model_id, position.asset, position.quantity,
                position.avg_entry_price, position.current_price,
                position.unrealized_pnl, position.realized_pnl,
                position.updated_at,
            ],
        )

    def get_orders(
        self, model_id: str | None = None, status: OrderStatus | None = None,
    ) -> list[LiveOrder]:
        query = "SELECT * FROM live_orders WHERE 1=1"
        params: list = []
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC"

        rows = self._conn.execute(query, params).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [self._row_to_order(dict(zip(cols, row))) for row in rows]

    def get_fills(
        self, model_id: str | None = None, order_id: str | None = None,
    ) -> list[LiveFill]:
        query = "SELECT * FROM live_fills WHERE 1=1"
        params: list = []
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        if order_id:
            query += " AND order_id = ?"
            params.append(order_id)
        query += " ORDER BY filled_at DESC"

        rows = self._conn.execute(query, params).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [self._row_to_fill(dict(zip(cols, row))) for row in rows]

    def get_positions(
        self, model_id: str | None = None,
    ) -> list[LivePosition]:
        query = "SELECT * FROM live_positions WHERE 1=1"
        params: list = []
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        rows = self._conn.execute(query, params).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [self._row_to_position(dict(zip(cols, row))) for row in rows]

    def get_order(self, order_id: str) -> LiveOrder | None:
        row = self._conn.execute(
            "SELECT * FROM live_orders WHERE order_id = ?", [order_id]
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.description]
        return self._row_to_order(dict(zip(cols, row)))

    def close(self) -> None:
        pass  # DuckDB in-memory, nothing to close

    @staticmethod
    def _row_to_order(d: dict) -> LiveOrder:
        return LiveOrder(
            order_id=d["order_id"], model_id=d["model_id"],
            asset=d["asset"], side=OrderSide(d["side"]),
            order_type=OrderType(d["order_type"]),
            quantity=d["quantity"], limit_price=d.get("limit_price"),
            stop_price=d.get("stop_price"),
            status=OrderStatus(d["status"]),
            created_at=d["created_at"], updated_at=d["updated_at"],
        )

    @staticmethod
    def _row_to_fill(d: dict) -> LiveFill:
        return LiveFill(
            fill_id=d["fill_id"], order_id=d["order_id"],
            model_id=d["model_id"], asset=d["asset"],
            side=OrderSide(d["side"]), quantity=d["quantity"],
            price=d["price"], commission=d.get("commission", 0),
            filled_at=d["filled_at"], venue=d.get("venue", ""),
        )

    @staticmethod
    def _row_to_position(d: dict) -> LivePosition:
        return LivePosition(
            model_id=d["model_id"], asset=d["asset"],
            quantity=d["quantity"], avg_entry_price=d["avg_entry_price"],
            current_price=d.get("current_price", 0),
            unrealized_pnl=d.get("unrealized_pnl", 0),
            realized_pnl=d.get("realized_pnl", 0),
            updated_at=d["updated_at"],
        )


# ═══════════════════════════════════════════════════════════════════
#  PostgreSQL Backend Stub (Production)
# ═══════════════════════════════════════════════════════════════════

_POSTGRES_SCHEMA = """
-- Orders table
CREATE TABLE IF NOT EXISTS live_orders (
    order_id VARCHAR(64) PRIMARY KEY,
    model_id VARCHAR(128) NOT NULL,
    asset VARCHAR(64) NOT NULL,
    side VARCHAR(8) NOT NULL,
    order_type VARCHAR(16) NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    limit_price DOUBLE PRECISION,
    stop_price DOUBLE PRECISION,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);
CREATE INDEX IF NOT EXISTS idx_orders_model ON live_orders(model_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON live_orders(status);

-- Fills table (TimescaleDB hypertable candidate)
CREATE TABLE IF NOT EXISTS live_fills (
    fill_id VARCHAR(64) PRIMARY KEY,
    order_id VARCHAR(64) NOT NULL REFERENCES live_orders(order_id),
    model_id VARCHAR(128) NOT NULL,
    asset VARCHAR(64) NOT NULL,
    side VARCHAR(8) NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION DEFAULT 0.0,
    filled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    venue VARCHAR(64),
    metadata JSONB
);
CREATE INDEX IF NOT EXISTS idx_fills_order ON live_fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_model ON live_fills(model_id);

-- Positions table
CREATE TABLE IF NOT EXISTS live_positions (
    model_id VARCHAR(128) NOT NULL,
    asset VARCHAR(64) NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    avg_entry_price DOUBLE PRECISION NOT NULL,
    current_price DOUBLE PRECISION DEFAULT 0.0,
    unrealized_pnl DOUBLE PRECISION DEFAULT 0.0,
    realized_pnl DOUBLE PRECISION DEFAULT 0.0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (model_id, asset)
);
"""


class PostgresLiveStore(LiveStoreBackend):
    """
    PostgreSQL + TimescaleDB backend for production.

    Requires psycopg2 or asyncpg. Not usable without a running
    PostgreSQL instance. Falls back to DuckDB in tests.
    """

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            try:
                import psycopg2
                self._conn = psycopg2.connect(self._dsn)
            except ImportError:
                raise RuntimeError(
                    "psycopg2 not installed. Install with: "
                    "pip install psycopg2-binary"
                )
        return self._conn

    def initialize(self) -> None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(_POSTGRES_SCHEMA)
        conn.commit()

    def insert_order(self, order: LiveOrder) -> str:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO live_orders
                   (order_id, model_id, asset, side, order_type, quantity,
                    limit_price, stop_price, status, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    order.order_id, order.model_id, order.asset,
                    order.side.value, order.order_type.value, order.quantity,
                    order.limit_price, order.stop_price, order.status.value,
                    order.created_at, order.updated_at,
                ),
            )
        conn.commit()
        return order.order_id

    def update_order_status(self, order_id: str, status: OrderStatus) -> None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE live_orders SET status = %s, updated_at = NOW() WHERE order_id = %s",
                (status.value, order_id),
            )
        conn.commit()

    def insert_fill(self, fill: LiveFill) -> str:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO live_fills
                   (fill_id, order_id, model_id, asset, side, quantity,
                    price, commission, filled_at, venue)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    fill.fill_id, fill.order_id, fill.model_id, fill.asset,
                    fill.side.value, fill.quantity, fill.price, fill.commission,
                    fill.filled_at, fill.venue,
                ),
            )
        conn.commit()
        return fill.fill_id

    def upsert_position(self, position: LivePosition) -> None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO live_positions
                   (model_id, asset, quantity, avg_entry_price, current_price,
                    unrealized_pnl, realized_pnl, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (model_id, asset) DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    avg_entry_price = EXCLUDED.avg_entry_price,
                    current_price = EXCLUDED.current_price,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    realized_pnl = EXCLUDED.realized_pnl,
                    updated_at = EXCLUDED.updated_at""",
                (
                    position.model_id, position.asset, position.quantity,
                    position.avg_entry_price, position.current_price,
                    position.unrealized_pnl, position.realized_pnl,
                    position.updated_at,
                ),
            )
        conn.commit()

    def get_orders(self, model_id=None, status=None) -> list[LiveOrder]:
        raise NotImplementedError("PostgreSQL queries require psycopg2 at runtime")

    def get_fills(self, model_id=None, order_id=None) -> list[LiveFill]:
        raise NotImplementedError("PostgreSQL queries require psycopg2 at runtime")

    def get_positions(self, model_id=None) -> list[LivePosition]:
        raise NotImplementedError("PostgreSQL queries require psycopg2 at runtime")

    def get_order(self, order_id: str) -> LiveOrder | None:
        raise NotImplementedError("PostgreSQL queries require psycopg2 at runtime")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ═══════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════

class LiveStore:
    """Factory for live trading stores."""

    @staticmethod
    def duckdb(conn: duckdb.DuckDBPyConnection | None = None) -> DuckDBLiveStore:
        store = DuckDBLiveStore(conn)
        store.initialize()
        return store

    @staticmethod
    def postgres(dsn: str) -> PostgresLiveStore:
        return PostgresLiveStore(dsn)
