"""
engines/collector_common.py

Shared plumbing for the Cycle 62A collectors (T1 liquidations, T2 open
interest). Kept separate from engines/forced_trade/common.py, which carries a
hard read-only contract -- these are writers.

WHAT LIVES HERE
---------------
1. Rule 35 time rendering. One implementation, so `datetime` can never drift
   from `timestamp` between collectors.
2. Gap recording. The Cycle 62A brief's standing requirement: a gap that
   leaves no trace is indistinguishable from a quiet market. Gaps land in the
   `collector_gaps` table, not just the log.
3. CWD-independent DB path. Cycle 46's DB_PATH lesson: scheduled .bat files
   and stray `cd`s made a relative path resolve against process CWD and write
   a phantom database. Anchor to __file__, always.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "data" / "crypto_data.db"

logger = logging.getLogger("collector")

# Sanity window for any timestamp claiming to be epoch-ms. 2020-01-01 to
# 2100-01-01. A seconds-since-epoch value fed in by mistake lands far below
# the lower bound, which is exactly the Rule 35.4 foot-gun this catches.
MS_MIN = 1_577_836_800_000
MS_MAX = 4_102_444_800_000


def now_ms() -> int:
    """Current time as epoch milliseconds UTC (Rule 35 canonical form)."""
    return int(time.time() * 1000)


def ms_to_iso(ms: int) -> str:
    """Epoch ms -> ISO 8601 with explicit +00:00 offset (Rule 35.3 cache)."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def assert_ms(ms, field: str = "timestamp") -> int:
    """Verify a value really is epoch-MILLISECONDS before it reaches storage.

    Rule 35.4 names the foot-gun: a feed that returns seconds, or local time,
    silently corrupts the series. Trusting the vendor is how that happens, so
    every timestamp is range-checked on the way in.
    """
    try:
        v = int(ms)
    except (TypeError, ValueError):
        raise ValueError(f"{field}: not an integer ms value: {ms!r}")
    if not (MS_MIN <= v <= MS_MAX):
        raise ValueError(
            f"{field}: {v} is outside the plausible epoch-ms range "
            f"[{MS_MIN}, {MS_MAX}] -- seconds instead of ms?"
        )
    return v


def open_db(db_path: Path | None = None, timeout: float = 30.0) -> sqlite3.Connection:
    """Open the collector DB in WAL with autocommit (Rule 34 stale-read guard)."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


# ------------------------------------------------------------------ gaps ---

def record_gap(conn, collector: str, venue: str, start_ms: int,
               end_ms: int | None, reason: str) -> None:
    """Persist a collection gap as a row.

    Idempotent on (collector, venue, start_ms): a gap that is first recorded
    open (end_ms=None) and later closed will be UPDATEd in place rather than
    duplicated. That is the one sanctioned write-after-insert in these
    collectors, and it exists so a crash still leaves a trace.
    """
    gap_seconds = None if end_ms is None else (end_ms - start_ms) / 1000.0
    conn.execute(
        """
        INSERT INTO collector_gaps
            (collector, venue, timestamp, datetime, gap_end, gap_seconds,
             reason, detected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(collector, venue, timestamp) DO UPDATE SET
            gap_end     = excluded.gap_end,
            gap_seconds = excluded.gap_seconds,
            reason      = excluded.reason
        """,
        (collector, venue, start_ms, ms_to_iso(start_ms), end_ms, gap_seconds,
         reason, now_ms()),
    )
    if end_ms is None:
        logger.warning("GAP OPENED  %s/%s at %s (%s)",
                       collector, venue, ms_to_iso(start_ms), reason)
    else:
        logger.warning("GAP CLOSED  %s/%s %s -> %s (%.1fs, %s)",
                       collector, venue, ms_to_iso(start_ms),
                       ms_to_iso(end_ms), gap_seconds, reason)


def setup_logging(verbose: int) -> None:
    """Verbosity levels; the brief asks for maximum by default.

    3 = DEBUG (every event), 2 = INFO (default, per-batch), 1 = WARNING
    (gaps and failures only), 0 = ERROR.
    """
    level = {0: logging.ERROR, 1: logging.WARNING,
             2: logging.INFO, 3: logging.DEBUG}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
