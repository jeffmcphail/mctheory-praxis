"""
engines/forced_trade/common.py

Shared plumbing for the Cycle 61 forced-trade data audit.

READ-ONLY CONTRACT
------------------
Every connection here is opened with SQLite URI `mode=ro`. That is not a
convention, it is enforced by the driver: a write against these handles raises
`sqlite3.OperationalError: attempt to write a readonly database`. The brief
requires no writes to crypto_data.db and this is how that is guaranteed rather
than promised.

RULE 34 (stale-read quirk)
--------------------------
crypto_data.db is written continuously by the live collectors. Long-lived
sqlite3 connections can pin a snapshot from an implicit BEGIN and silently
serve hours-old data. Every connection opened here uses `isolation_level=None`
(true autocommit, no implicit BEGIN sticks around), which is one of the three
patterns Rule 34 permits, and is scoped to a single logical read pass.

CACHING
-------
The `trades` table is the bulk of a 24 GB database. Re-scanning it once per
threshold setting would make the sensitivity sweep unaffordable, so the scan is
done ONCE into a coarse time-bucket table and cached to parquet. Every
threshold setting is then evaluated in memory against that cache. The cache is
a pure aggregation of source rows -- no thresholds are applied while building
it, so caching cannot influence any reported count.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "data" / "crypto_data.db"
DEFAULT_CACHE_DIR = REPO / "data" / "external" / "forced_trade"
DEFAULT_OUT_DIR = REPO / "outputs" / "forced_trade"

MS_PER_SEC = 1000
MS_PER_MIN = 60 * MS_PER_SEC
MS_PER_HOUR = 60 * MS_PER_MIN
MS_PER_DAY = 24 * MS_PER_HOUR

logger = logging.getLogger("forced_trade")


# ---------------------------------------------------------------- time -----

def ms_to_utc(ms):
    """Epoch milliseconds (Rule 35 canonical form) -> aware UTC datetime."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def ms_to_str(ms, date_only: bool = False) -> str:
    """Render epoch ms as an ISO-ish UTC string for reports."""
    dt = ms_to_utc(ms)
    if dt is None:
        return "None"
    return dt.strftime("%Y-%m-%d") if date_only else dt.strftime("%Y-%m-%d %H:%M:%S")


def utc_to_ms(dt: datetime) -> int:
    """Aware (or naive-assumed-UTC) datetime -> epoch milliseconds."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def days_between(start_ms: int, end_ms: int) -> float:
    return (end_ms - start_ms) / MS_PER_DAY


def per_year(count: int, span_days: float) -> float:
    """Annualise a raw count observed over `span_days`.

    Reported everywhere ALONGSIDE the raw count, never instead of it: with a
    ~4-month span the annualisation is a 3.3x extrapolation and the raw count
    is the honest number.
    """
    if span_days <= 0:
        return float("nan")
    return count * 365.25 / span_days


# ------------------------------------------------------------------ db -----

@contextmanager
def read_only_db(db_path=DEFAULT_DB):
    """Open a fresh READ-ONLY connection for one logical read pass.

    Fresh-per-pass plus isolation_level=None: both Rule 34 remedies at once.
    """
    p = Path(db_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"database not found: {p}")
    conn = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True,
                           isolation_level=None)
    try:
        yield conn
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,))
    return cur.fetchone() is not None


def column_names(conn: sqlite3.Connection, table: str) -> list:
    return [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')")]


@dataclass(frozen=True)
class Span:
    """Observed coverage of one (table, asset) pair."""
    asset: str
    first_ms: int
    last_ms: int
    rows: Optional[int] = None

    @property
    def days(self) -> float:
        return days_between(self.first_ms, self.last_ms)

    def describe(self) -> str:
        n = f"{self.rows:,}" if self.rows is not None else "?"
        return (f"{self.asset}: {n} rows  {ms_to_str(self.first_ms)} -> "
                f"{ms_to_str(self.last_ms)}  ({self.days:.1f} days)")


def asset_spans(conn: sqlite3.Connection, table: str,
                with_counts: bool = False) -> list:
    """Per-asset first/last timestamp for `table`.

    `with_counts=False` by default: on `trades` a COUNT(*) is a full index scan
    that takes minutes, while MIN/MAX are index seeks that take microseconds.
    """
    assets = [r[0] for r in conn.execute(
        f"SELECT DISTINCT asset FROM {table} ORDER BY asset")]
    out = []
    for a in assets:
        mn = conn.execute(
            f"SELECT MIN(timestamp) FROM {table} WHERE asset=?", (a,)).fetchone()[0]
        mx = conn.execute(
            f"SELECT MAX(timestamp) FROM {table} WHERE asset=?", (a,)).fetchone()[0]
        n = None
        if with_counts:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE asset=?", (a,)).fetchone()[0]
        out.append(Span(asset=a, first_ms=mn, last_ms=mx, rows=n))
    return out


# -------------------------------------------------------------- output -----

# Third-party loggers that emit full HTTP request/response bodies at DEBUG.
# Rule 25 asks for maximum verbosity from OUR code; inheriting ccxt DEBUG
# turns a readable audit into 470 KB of option-chain JSON and buries the
# findings, so these are pinned at WARNING independently of -vv.
NOISY_LOGGERS = ("ccxt", "ccxt.base.exchange", "urllib3", "requests",
                 "asyncio", "matplotlib")


def setup_logging(verbose: int = 2) -> None:
    """-v=INFO, -vv=DEBUG. Rule 25: default is maximum verbosity."""
    level = logging.DEBUG if verbose >= 2 else (
        logging.INFO if verbose == 1 else logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(levelname)-7s %(name)s: %(message)s",
        force=True,
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def banner(title: str, width: int = 78) -> str:
    return "\n".join(["", "=" * width, title, "=" * width])


def fmt_table(rows, headers, aligns=None) -> str:
    """Minimal fixed-width ASCII table (Rule 20: no box-drawing characters)."""
    cells = [[("" if c is None else str(c)) for c in r] for r in rows]
    ncol = len(headers)
    widths = [len(h) for h in headers]
    for r in cells:
        for i in range(ncol):
            if i < len(r):
                widths[i] = max(widths[i], len(r[i]))
    if aligns is None:
        aligns = ["<"] + [">"] * (ncol - 1)
    head = "  ".join(f"{headers[i]:{aligns[i]}{widths[i]}}" for i in range(ncol))
    sep = "  ".join("-" * widths[i] for i in range(ncol))
    body = ["  ".join(f"{(r[i] if i < len(r) else ''):{aligns[i]}{widths[i]}}"
                      for i in range(ncol)) for r in cells]
    return "\n".join([head, sep] + body)
