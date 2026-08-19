"""
Cycle 62A T1 (revised) -- add `source` to liquidations.

Provenance has to travel with the row. The two possible origins of a
liquidation record are not equivalent measurements:

  stream:<venue>   the live !forceOrder WebSocket feed -- every event Binance
                   chose to push, subject to its documented throttle.
  archive:<path>   data.binance.vision daily liquidationSnapshot files.

The archive was MEASURED (Cycle 62A) to be a sampled, duplicated view:
  - never more than ONE distinct liquidation per symbol per second, matching
    the documented "largest order per 1000ms" throttle -- so counts and
    volumes from it are a LOWER BOUND, never a complete record;
  - every row is written exactly TWICE (duplication factor 2.00 across every
    file checked), so a naive row count doubles the true event count.

Mixing the two provenances in one column without a discriminator would make
any count silently wrong in two different directions at once. Hence this
column, added before any archive row is ever loaded rather than after.

`source` is nullable so the existing (currently empty) table migrates without
a rewrite; the stream collector fills it going forward.

Idempotent: re-running on an already-migrated table prints "already present"
and exits 0.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"


def has_column(conn, table: str, col: str) -> bool:
    return any(r[1] == col for r in conn.execute("PRAGMA table_info(%s)" % table))


def main() -> int:
    print("[migrate] Opening %s" % DB_PATH)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        if conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='liquidations'"
        ).fetchone() is None:
            print("[migrate] ERROR: liquidations table does not exist. Run "
                  "cycle62a_liquidations_schema.py first.", file=sys.stderr)
            return 3

        if has_column(conn, "liquidations", "source"):
            print("[migrate] liquidations.source already present. Exiting cleanly.")
            return 0

        n = conn.execute("SELECT COUNT(*) FROM liquidations").fetchone()[0]
        print("[migrate] adding source column (table currently holds %d rows)" % n)
        conn.execute("ALTER TABLE liquidations ADD COLUMN source TEXT")
        conn.commit()

        if not has_column(conn, "liquidations", "source"):
            print("[migrate] ERROR: column not present after ALTER",
                  file=sys.stderr)
            return 4

        print("[migrate] liquidations.source added OK.")
        for r in conn.execute("PRAGMA table_info(liquidations)"):
            print("  %-13s %-8s notnull=%d pk=%d" % (r[1], r[2], r[3], r[5]))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
