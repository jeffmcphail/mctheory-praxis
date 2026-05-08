"""
Cycle 33 -- Atlas DB schema extension.

Adds four new columns to atlas_experiments:
  - test_conditions        TEXT  (JSON-encoded dict)
  - revival_hypotheses     TEXT  (JSON-encoded list of dicts)
  - regime_state_at_test   TEXT  (JSON-encoded dict; often "not_measured")
  - computational_engine   INTEGER  (1-7, or NULL if multi/ambiguous)

These columns are populated by `engines/atlas_sync.py` from
new structured sections in TRADING_ATLAS.md. Existing
columns are unchanged. The `regime_classes`,
`strategy_regime_relevance`, `atlas_embeddings`, and `sync_log`
tables are unchanged.

Idempotent: detects already-applied state (columns already
present) and exits cleanly without re-running.

Pre-condition: existing DB must be at the post-Cycle-12 schema.
Run from the repo root:

    python scripts/migrations/cycle33_atlas_schema_extension.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "praxis_meta.db"

NEW_COLUMNS = [
    ("test_conditions", "TEXT"),
    ("revival_hypotheses", "TEXT"),
    ("regime_state_at_test", "TEXT"),
    ("computational_engine", "INTEGER"),
]


def main() -> int:
    print(f"[cycle33] Opening {DB_PATH}")
    if not DB_PATH.exists():
        print(f"[cycle33] ABORT: {DB_PATH} doesn't exist. Run "
              f"`python -m engines.atlas_sync` first to bootstrap "
              f"the DB.", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Pre-state inventory
        existing_cols = {
            row[1] for row in
            conn.execute("PRAGMA table_info(atlas_experiments)").fetchall()
        }
        print(f"[cycle33] Pre-state: atlas_experiments has "
              f"{len(existing_cols)} columns")

        already_added = [name for name, _ in NEW_COLUMNS
                         if name in existing_cols]
        to_add = [(name, typ) for name, typ in NEW_COLUMNS
                  if name not in existing_cols]

        if already_added and not to_add:
            print(f"[cycle33] Already migrated -- all 4 new columns "
                  f"present: {already_added}. Exiting cleanly.")
            return 0

        if already_added:
            print(f"[cycle33] WARN: partial prior migration detected. "
                  f"Already-added columns: {already_added}. "
                  f"Will add the missing ones: "
                  f"{[n for n, _ in to_add]}.")

        # Schema migration
        print(f"[cycle33] Adding {len(to_add)} new columns...")
        conn.execute("BEGIN")
        try:
            for col_name, col_type in to_add:
                stmt = (f"ALTER TABLE atlas_experiments "
                        f"ADD COLUMN {col_name} {col_type}")
                print(f"  + {stmt}")
                conn.execute(stmt)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Post-state verification
        post_cols = {
            row[1] for row in
            conn.execute("PRAGMA table_info(atlas_experiments)").fetchall()
        }
        for col_name, _ in NEW_COLUMNS:
            if col_name not in post_cols:
                print(f"[cycle33] FAIL: column {col_name} missing "
                      f"after ALTER TABLE.", file=sys.stderr)
                return 3

        # All existing rows have NULL in the new columns -- expected
        # state. Atlas_sync will populate them from markdown.
        existing_rows = conn.execute(
            "SELECT COUNT(*) FROM atlas_experiments"
        ).fetchone()[0]
        print(f"[cycle33] Post-state: {len(post_cols)} columns, "
              f"{existing_rows} existing rows. New columns are NULL "
              f"for all rows; atlas_sync will populate them after "
              f"markdown is updated with structured sections.")

        print()
        print("=" * 60)
        print("[cycle33] SCHEMA EXTENSION COMPLETE")
        print(f"  Added: {[name for name, _ in to_add]}")
        print(f"  Total columns now: {len(post_cols)}")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Update markdown templates in TRADING_ATLAS.md")
        print("     experiments to include the new structured sections.")
        print("  2. Update engines/atlas_sync.py parser to extract "
              "those sections into the new columns.")
        print("  3. Re-run atlas_sync; populated rows should show in "
              "atlas_get(N).")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
