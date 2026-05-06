# Cycle 25.5 -- writer collapse (hybrid mini-brief)

**Predecessor:** Cycle 25 (`36fb44a`, `874bf81`, `4e47659`)
**Mode:** hybrid (Claude drafts, Code applies code change to file)
**Process pattern:** PraxisSmartMoney is a SCHEDULED task (every
6h), NOT a long-lived process. Per Cycle 23.5/24.5 lessons: the
kill-and-relaunch dance does NOT apply here. Next scheduled fire
spawns a fresh process with new code automatically.

## What

Collapse the dual-write `position_snapshots` writer in
`engines/smart_money.py` to single-write. Remove the runtime
PK-introspection branch added during Cycle 25 (commit `36fb44a`);
remove the `position_snapshots_v2` `CREATE TABLE` from `init_db()`.

There are TWO writer sites per the Cycle 25 retro:
1. **`cmd_snapshot()` around line 371** (or whatever the
   post-Cycle-25 line is) -- the ONE-SHOT path; this is what the
   scheduled .bat invokes every 6h. Production path.
2. **`cmd_loop()` around line 703** -- the CONTINUOUS-mode CLI
   for ad-hoc monitoring. Off the production path but still
   needs the collapse for consistency.

Both writers had identical dual-INSERT + introspection blocks
added in Cycle 25. Both must be collapsed identically.

## Specifics for Code

In `engines/smart_money.py`:

1. **`init_db()`**: Remove the `CREATE TABLE IF NOT EXISTS
   position_snapshots_v2 (...)` block + any associated index
   that was added in Cycle 25 Phase 0 (commit `36fb44a`). The
   `position_snapshots` CREATE block itself stays. ~15-20
   lines removed.

2. **`cmd_snapshot()` writer site (line ~371)**: replace the
   runtime-introspection block + dual-INSERT logic with a
   single-write path against the live `position_snapshots`
   (post-cutover schema: ms timestamp + datetime + compound
   PK on `(snapshot_id, wallet, market_slug, outcome)`).

   The post-Cycle-25 single-write should look roughly like:

   ```python
   # Single-write to the post-Cycle-25-cutover position_snapshots
   # table (Rule 35: ms timestamp + datetime + compound PK on
   # (snapshot_id, wallet, market_slug, outcome)).
   try:
       conn.execute("""
           INSERT OR REPLACE INTO position_snapshots
           (snapshot_id, timestamp, datetime, wallet,
            market_slug, market_title, outcome, size,
            avg_price, current_price, value_usd, pnl_usd)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       """, (snapshot_id, now_ms, now_iso, address, slug,
             str(title)[:100], outcome, size, avg_price,
             cur_price, value, pnl))
   except Exception as e:
       # ... existing error handling ...
   ```

   Match the existing style. Note `INSERT OR REPLACE` (Cycle 25
   established this; preserves existing semantics where same
   snapshot_id+wallet+market_slug+outcome overwrites prior).

3. **`cmd_loop()` writer site (line ~703)**: same collapse as
   site #2.

   DRY note: if Code wants to extract a shared helper
   `_insert_position_row(conn, snapshot_id, now_ms, now_iso,
   address, slug, title, outcome, size, avg_price, cur_price,
   value, pnl)` that both sites call, that's a clean refactor.
   Optional; small duplication is also fine.

4. **Remove imports** that are no longer used after the collapse,
   if any (likely none).

5. **py_compile clean check**: `python -m py_compile
   engines/smart_money.py`.

## Verification

py_compile only. Don't run the writer in-process -- the
verification step is the cleanup script's pre-flight #4 (detects
whether legacy was written within 60s). Since PraxisSmartMoney
fires every 6h and the next fire is many hours from now, the
legacy-age check will pass trivially.

## Why a brief instead of a full delta zip

Claude's local checkout pre-dates Cycle 25, so the post-Cycle-25
file content (specifically the runtime-introspection writers
added in `36fb44a` and the cutover changes in `874bf81`) is not
on disk for safe diffing. Code reads the actual file, which
avoids a memory-reconstruction error.

The cleanup script (separate file in `scripts/migrations/`) is
fully self-contained because it operates on the DB schema, not
on the writer code.

## Commit message (use this verbatim)

```
Cycle 25.5 step 1: position_snapshots writer collapse

Removes runtime PK introspection + dual-INSERT branches from
both cmd_snapshot() and cmd_loop() writer sites; removes
position_snapshots_v2 CREATE from init_db().

Cleanup script (drop _legacy + _v2) runs as a separate step
after this commits. PraxisSmartMoney is scheduled-not-long-lived
so the kill-and-relaunch dance from Cycle 24.5 doesn't apply
here -- the next 6h scheduled fire automatically spawns a fresh
process with this code.

Brief: claude/handoffs/BRIEF_position_snapshots_phase5_cleanup.md
```
