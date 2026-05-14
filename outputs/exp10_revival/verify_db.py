"""Post-sync verification of atlas_experiments id=8."""
import json
import sqlite3

conn = sqlite3.connect("data/praxis_meta.db")
cur = conn.cursor()
row = cur.execute(
    "SELECT id, result_class, md_hash, length(full_markdown), date_run, "
    "result_summary, revival_hypotheses, test_conditions "
    "FROM atlas_experiments WHERE id=8"
).fetchone()

print("Final post-sync atlas_experiments id=8:")
print("  result_class:", row[1])
print("  md_hash (pre):  71e2ca00d8e1ee8ceb29be0c8d08b09b56b973597b74824f09b881a4b41c92df")
print("  md_hash (post):", row[2])
print("  md_hash changed:", row[2] != "71e2ca00d8e1ee8ceb29be0c8d08b09b56b973597b74824f09b881a4b41c92df")
print("  full_markdown_len (was 4302):", row[3])
print("  date_run:", row[4])
print()
print("  result_summary (full):")
print("   ", row[5])
print()
rh = json.loads(row[6]) if row[6] else None
print(f"  revival_hypotheses: {len(rh) if rh else None} items")
if rh:
    for i, h in enumerate(rh, 1):
        title = h.get("title", "?")
        like = h.get("likelihood", "?")
        desc = (h.get("description") or "")[:140]
        print(f"    {i}. title={title!r}")
        print(f"       likelihood={like!r}")
        print(f"       desc={desc!r}")
print()
tc = json.loads(row[7]) if row[7] else None
keys = list(tc.keys()) if tc else None
print(f"  test_conditions keys: {keys}")
if tc:
    fs = tc.get("feature_set")
    rm = tc.get("risk_management")
    print(f"    feature_set: {fs}")
    print(f"    risk_management: {rm}")

# Also check that the search query "leverage cap revival" returns id=8
print()
print("Sanity: does the new verdict text appear in DB full_markdown?")
fm = cur.execute("SELECT full_markdown FROM atlas_experiments WHERE id=8").fetchone()[0]
phrases = [
    "Sharpe was invariant to 4 decimals",
    "decisively refuting the leverage-cap revival hypothesis",
    "110 signal configs x 72 barrier configs = 7,920",
    "TESTED, REFUTED",
    "a2202a7 fabrication pattern",
]
for p in phrases:
    found = p in fm
    print(f"    {'YES' if found else 'NO ':3s}  '{p}'")
