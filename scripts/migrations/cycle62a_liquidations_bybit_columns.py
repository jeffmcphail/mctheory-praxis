"""
Cycle 62A T1 (Bybit adoption) -- add `side_raw` and `price_basis`.

Bybit `allLiquidation` becomes the T1 venue because the Binance forced-order
stream is unreachable and NO backfill exists on any path (the public archive
has no USD-margined liquidation dataset at all, and the coin-margined one ends
2024-10-14 against trades coverage starting 2026-04-29 -- zero overlap).
Liquidations are therefore forward-only forever: a day not recorded is gone.

That makes Bybit the source of record, and these two columns exist because
Bybit's payload is NOT a drop-in for Binance's in two ways that would
otherwise corrupt rows silently.

1. `side_raw` -- THE SIDE CONVENTIONS ARE INVERTED
   -----------------------------------------------
   Binance `forceOrder.S` is the ORDER side. Closing a long sends a market
   SELL, so on Binance  SELL => a LONG was liquidated.

   Bybit `allLiquidation.S` is documented as the POSITION side: "Position
   side. Buy,Sell. When you receive a `Buy` update, this means that a long
   position has been liquidated". So on Bybit  BUY => a LONG was liquidated.

   Same letter, opposite meaning. Writing Bybit's raw `S` into the same
   `side` column as Binance's would invert every Bybit row against every
   Binance row, and nothing downstream would raise. `side` therefore holds ONE
   normalised convention for all venues -- the Binance ORDER-side convention,
   because that is what the existing column already means and what the Cycle
   61 A2 detector compares against -- and `side_raw` preserves what the venue
   actually sent so the normalisation stays auditable and reversible.

   The mapping is asserted against live data, not taken from the docs alone;
   see engines/liquidation_common.py and the Cycle 62A retro.

2. `price_basis` -- THE PRICES ARE NOT THE SAME KIND OF PRICE
   ----------------------------------------------------------
   Binance sends an order price `p` and an average FILL price `ap`; the
   collector prefers `ap x z`, so its quote_qty is close to true executed
   notional.

   Bybit's `p` is documented as the BANKRUPTCY price -- the price at which the
   position's margin reaches zero, not the price anything traded at. It is
   systematically worse than the actual fill. `quote_qty` for a Bybit row is
   therefore an approximation with a known directional bias, not a measured
   notional, and summing notional across venues without accounting for that
   compares two different quantities.

   `price_basis` records which kind each row carries: 'executed', 'order' or
   'bankruptcy'.

Both columns are nullable so the existing table migrates without a rewrite.

Idempotent: re-running prints "already present" and exits 0.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "crypto_data.db"

NEW_COLUMNS = [
    ("side_raw", "TEXT"),
    ("price_basis", "TEXT"),
]


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

        n = conn.execute("SELECT COUNT(*) FROM liquidations").fetchone()[0]
        added = []
        for col, typ in NEW_COLUMNS:
            if has_column(conn, "liquidations", col):
                print("[migrate] liquidations.%s already present." % col)
                continue
            print("[migrate] adding %s (%s); table holds %d rows" % (col, typ, n))
            conn.execute("ALTER TABLE liquidations ADD COLUMN %s %s" % (col, typ))
            added.append(col)
        conn.commit()

        for col, _ in NEW_COLUMNS:
            if not has_column(conn, "liquidations", col):
                print("[migrate] ERROR: %s not present after ALTER" % col,
                      file=sys.stderr)
                return 4

        # A venue index matters now that the table is genuinely multi-venue:
        # every honest query filters on it, because counts are not poolable
        # across venues (different perp share, different liquidation engine,
        # different stream throttle).
        conn.execute("CREATE INDEX IF NOT EXISTS idx_liquidations_venue_ts "
                     "ON liquidations(venue, timestamp)")
        conn.commit()

        print("[migrate] OK (%s)." % (", ".join(added) if added else "no change"))
        for r in conn.execute("PRAGMA table_info(liquidations)"):
            print("  %-13s %-8s notnull=%d pk=%d" % (r[1], r[2], r[3], r[5]))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
