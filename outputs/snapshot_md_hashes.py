"""Capture md_hash for all atlas experiments for pre/post comparison."""
import sqlite3
import sys

conn = sqlite3.connect("data/praxis_meta.db")
cur = conn.cursor()
rows = cur.execute(
    "SELECT id, source_section, md_hash FROM atlas_experiments "
    "WHERE source_file='TRADING_ATLAS.md' ORDER BY id"
).fetchall()
print(f"label={sys.argv[1] if len(sys.argv) > 1 else 'snapshot'}  count={len(rows)}")
for r in rows:
    print(f"  id={r[0]:2d}  hash={r[2][:16]}...  {r[1][:60]}")
