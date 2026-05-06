# Cycle 24.5 -- writer collapse (hybrid mini-brief)

**Predecessor:** Cycle 24 (`b8fa847`, `6ca1796`, `dbecb23`)
**Mode:** hybrid (Claude drafts, Code applies code change to file)
**Process pattern:** PraxisLiveCollector is a LONG-LIVED process
(`python -u -m engines.live_collector start --top 50 --interval 60`).
Per Cycle 23.5 lesson: writer collapse FIRST, then KILL the process,
then run the cleanup script. Do NOT run cleanup before the kill.

## What

Collapse the dual-write `price_snapshots` writer in
`engines/live_collector.py` to single-write. Remove the runtime
PK-introspection branch added during Cycle 24's mid-cycle dual-
write; remove the `price_snapshots_v2` `CREATE TABLE` from
`init_db()`.

## Specifics for Code

In `engines/live_collector.py`:

1. **`init_db()`**: Remove the `CREATE TABLE IF NOT EXISTS
   price_snapshots_v2 (...)` block (and any associated index)
   that was added in Cycle 24 Phase 0 (commit `b8fa847`). The
   `price_snapshots` CREATE block itself stays. Total: ~20-30
   lines removed depending on schema width.

2. **The price-write path** (the function/method that inserts
   into `price_snapshots` per market iteration): replace the
   runtime-introspection block + dual-INSERT logic with a
   single-write path against the live `price_snapshots`
   (post-cutover schema: ms timestamp + compound PK on whatever
   Cycle 24 chose -- check the on-disk schema via PRAGMA
   table_info to confirm before writing).

   The post-Cycle-24 single-write should look roughly like:

   ```python
   # Single-write to the post-Cycle-24-cutover price_snapshots
   # table (Rule 35: ms timestamp, compound PK on
   # (market_id_or_token, timestamp)).
   try:
       cursor = conn.cursor()
       cursor.execute(
           "INSERT OR IGNORE INTO price_snapshots "
           "(... cols ...) VALUES (...)",
           values,
       )
       conn.commit()
       return (cursor.rowcount, None)
   except Exception as e:
       return (0, f"insert: {type(e).__name__}: {e}")
   ```

   Match the existing style (`INSERT OR IGNORE`, error-tuple
   return, etc.).

3. **Remove imports** that are no longer used after the collapse,
   if any.

4. **py_compile clean check**: `python -m py_compile
   engines/live_collector.py`.

## Verification

py_compile only. Don't run the writer in-process here -- the
verification step is the cleanup script's pre-flight #4 (detects
whether legacy is still being written to). After the kill +
fresh-spawn, the cleanup script's `legacy last write: Ns ago`
output will confirm the new code is live.

## Why a brief instead of a full delta zip

Claude's local checkout pre-dates Cycle 24, so the post-Cycle-24
file content (specifically the runtime-introspection writer added
in `b8fa847` and the cutover changes in `6ca1796`) is not on
disk for safe diffing. Code reads the actual file, which avoids
a memory-reconstruction error.

The cleanup script (separate file in `scripts/migrations/`) is
fully self-contained because it operates on the DB schema, not
on the writer code.
