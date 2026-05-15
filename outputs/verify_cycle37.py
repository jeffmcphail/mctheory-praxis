"""Cycle 37 post-correction spot-check on entries nearest the edit."""
import sqlite3

conn = sqlite3.connect("data/praxis_meta.db")
cur = conn.cursor()

# id=10 is Exp 12 (TA × FX G10 TB), the entry just before the edited block.
# id=11 is Exp 13 (FUNDING), the next parsed entry after the edited block.
for entry_id in [8, 10, 11]:
    row = cur.execute(
        "SELECT id, source_section, result_class, md_hash, length(full_markdown), date_run "
        "FROM atlas_experiments WHERE id=?", (entry_id,)
    ).fetchone()
    print(f"\nid={row[0]} {row[1]!r}")
    print(f"  result_class: {row[2]}")
    print(f"  md_hash: {row[3]}")
    print(f"  full_markdown_len: {row[4]}")
    print(f"  date_run: {row[5]}")

    # Check that the edited "CRITICAL FINDING [SUPERSEDED]" content did NOT leak
    # into this entry's full_markdown.
    fm = cur.execute(
        "SELECT full_markdown FROM atlas_experiments WHERE id=?", (entry_id,)
    ).fetchone()[0]
    leaked_phrases = [
        "CRITICAL FINDING [SUPERSEDED]",
        "free aspirational guess rather than derived arithmetic",
        "completed differently: Cycle 36b uses",
    ]
    for p in leaked_phrases:
        if p in fm:
            print(f"  LEAK DETECTED: '{p}' appears in this entry's full_markdown")
        else:
            print(f"  no leak: '{p[:50]}...'")

print("\nSpot-check complete.")
