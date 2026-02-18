#!/usr/bin/env python3
"""
Merge surface data from a remote DuckDB into a local DuckDB.

Uses DuckDB's ATTACH to copy tables directly between databases.
Handles both scalar surface tables and artifact tables.

Designed for the workflow:
  1. Desktop runs Phase 1 → data/surfaces.duckdb
  2. Laptop runs Phase 2-4 → laptop_surfaces.duckdb
  3. Copy laptop file to desktop
  4. Run this script to merge laptop data into desktop DB

Usage:
    python scripts/merge_surfaces.py --source laptop_surfaces.duckdb --target data/surfaces.duckdb

    # Dry run (show what would be merged)
    python scripts/merge_surfaces.py --source laptop_surfaces.duckdb --target data/surfaces.duckdb --dry-run

    # Docker:
    docker compose run --rm -v /path/to/remote:/remote merge --source /remote/surfaces.duckdb --target /app/data/surfaces.duckdb
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import duckdb


def get_surface_tables(conn: duckdb.DuckDBPyConnection, prefix: str = "") -> list[str]:
    """Get all cv_surf_* and cv_artifact_* table names."""
    tables = conn.execute("SHOW TABLES").fetchall()
    surface_tables = []
    for (name,) in tables:
        if name.startswith("cv_surf_") or name.startswith("cv_artifact_") or name == "cv_surface_meta":
            surface_tables.append(f"{prefix}{name}" if prefix else name)
    return surface_tables


def table_info(conn: duckdb.DuckDBPyConnection, table: str) -> dict:
    """Get row count and column names for a table."""
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    cols = [row[0] for row in conn.execute(f"DESCRIBE {table}").fetchall()]
    return {"count": count, "columns": cols}


def merge_surfaces(
    source_path: str,
    target_path: str,
    dry_run: bool = False,
) -> dict:
    """
    Merge surface tables from source DB into target DB.

    For each table in the source:
    - If table doesn't exist in target: copy entire table
    - If table exists: INSERT rows that don't already exist (by grid point key)

    Returns summary dict with counts.
    """
    source_path = str(Path(source_path).resolve())
    target_path = str(Path(target_path).resolve())

    if not Path(source_path).exists():
        print(f"ERROR: Source not found: {source_path}")
        sys.exit(1)

    # Create target if it doesn't exist
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(target_path)
    conn.execute(f"ATTACH '{source_path}' AS remote (READ_ONLY)")

    # Get tables from both
    target_tables = get_surface_tables(conn)
    remote_tables = conn.execute(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = 'remote'"
    ).fetchall()
    remote_surface = [name for (name,) in remote_tables
                      if name.startswith("cv_surf_") or name.startswith("cv_artifact_") or name == "cv_surface_meta"]

    summary = {
        "source": source_path,
        "target": target_path,
        "tables_merged": 0,
        "rows_added": 0,
        "rows_skipped": 0,
        "details": [],
    }

    print(f"\n  Source: {source_path}")
    print(f"  Target: {target_path}")
    print(f"  Remote tables: {len(remote_surface)}")
    print(f"  Local tables:  {len(target_tables)}")
    print()

    for table_name in remote_surface:
        remote_info = conn.execute(f"SELECT COUNT(*) FROM remote.{table_name}").fetchone()[0]

        if table_name not in target_tables:
            # Table doesn't exist locally — full copy
            print(f"  + {table_name}: {remote_info} rows (new table)")
            if not dry_run:
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM remote.{table_name}")
            summary["rows_added"] += remote_info
            summary["tables_merged"] += 1
            summary["details"].append({"table": table_name, "action": "created", "rows": remote_info})

        else:
            # Table exists — merge missing rows
            local_info = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            # Get column names to build merge query
            cols = [row[0] for row in conn.execute(f"DESCRIBE {table_name}").fetchall()]

            # Identify key columns (everything except 'values', 'pct_*', 'artifact')
            # Key columns are the grid point identifiers
            key_cols = [c for c in cols if c not in ("values", "artifact")
                        and not c.startswith("pct_")]

            if not key_cols:
                # Fallback: use all columns
                key_cols = cols

            # Build anti-join: insert rows from remote that don't exist locally
            key_match = " AND ".join(f"r.{c} = l.{c}" for c in key_cols)

            count_sql = f"""
                SELECT COUNT(*) FROM remote.{table_name} r
                WHERE NOT EXISTS (
                    SELECT 1 FROM {table_name} l WHERE {key_match}
                )
            """
            new_rows = conn.execute(count_sql).fetchone()[0]

            if new_rows > 0:
                print(f"  ~ {table_name}: {new_rows} new rows (local has {local_info}, remote has {remote_info})")
                if not dry_run:
                    insert_sql = f"""
                        INSERT INTO {table_name}
                        SELECT r.* FROM remote.{table_name} r
                        WHERE NOT EXISTS (
                            SELECT 1 FROM {table_name} l WHERE {key_match}
                        )
                    """
                    conn.execute(insert_sql)
                summary["rows_added"] += new_rows
                summary["tables_merged"] += 1
                summary["details"].append({"table": table_name, "action": "merged", "rows": new_rows})
            else:
                skipped = remote_info
                print(f"  = {table_name}: all {remote_info} rows already present")
                summary["rows_skipped"] += skipped
                summary["details"].append({"table": table_name, "action": "skipped", "rows": 0})

    conn.execute("DETACH remote")
    conn.close()

    # Final report
    print()
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"  {prefix}Tables merged: {summary['tables_merged']}")
    print(f"  {prefix}Rows added:    {summary['rows_added']}")
    print(f"  {prefix}Rows skipped:  {summary['rows_skipped']}")

    if dry_run:
        print(f"\n  Re-run without --dry-run to apply.")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Merge surface data from a remote DuckDB into local.",
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to source (remote) DuckDB file",
    )
    parser.add_argument(
        "--target", default="data/surfaces.duckdb",
        help="Path to target (local) DuckDB file (default: data/surfaces.duckdb)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be merged without making changes",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  McTheory Praxis -- Surface Merge")
    print("=" * 60)

    merge_surfaces(args.source, args.target, args.dry_run)

    print("=" * 60)


if __name__ == "__main__":
    main()
