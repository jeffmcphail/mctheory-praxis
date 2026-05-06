# Cycle 23.5 -- writer collapse (hybrid mini-brief)

**Predecessor:** Cycle 23 (`ca5c719`, `10724bc`, `5cf1c03`)
**Mode:** hybrid (Claude drafts, Code applies code change to file)

## What

Collapse the dual-write `order_book_snapshots` writer in
`engines/crypto_data_collector.py` to single-write. Remove the
runtime PK-introspection branch added during Cycle 23's mid-cycle
recovery; remove the `order_book_snapshots_v2` `CREATE TABLE` from
`init_db()`.

## Specifics for Code

In `engines/crypto_data_collector.py`:

1. **`init_db()`**: remove the `CREATE TABLE IF NOT EXISTS
   order_book_snapshots_v2 (...)` block AND the
   `CREATE INDEX ... idx_ob_v2_asset_timestamp` that follows it
   (added in commit `ca5c719`). The `order_book_snapshots` CREATE
   block itself stays as-is. Total: ~47 lines removed.

2. **`collect_order_book_snapshot`** (the writer added in `ca5c719`,
   modified in `10724bc`): replace the runtime-introspection block
   + dual-INSERT logic with a single-write path against the live
   `order_book_snapshots` (post-cutover schema: ms timestamp +
   datetime + compound PK on `(asset, timestamp)`):

   ```python
   # Single-write to the post-Cycle-23-cutover order_book_snapshots
   # table (Rule 35: ms timestamp, datetime ISO with +00:00 microsecond
   # precision, compound PK on (asset, timestamp), no `id`).
   cols_sql = (
       "asset, timestamp, datetime, mid_price, best_bid, best_ask, "
       "spread, spread_bps, "
       "bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, "
       "bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6, "
       "bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9, "
       "bid_price_10, bid_vol_10, "
       "ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, "
       "ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6, "
       "ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9, "
       "ask_price_10, ask_vol_10, "
       "bid_volume_top10, ask_volume_top10, order_imbalance_top10"
   )
   placeholders = ", ".join(["?"] * 51)
   values = [
       asset, ts_ms, dt, mid, best_bid, best_ask, spread, spread_bps,
       *bid_flat, *ask_flat,
       bid_top10, ask_top10, imbalance,
   ]
   try:
       cursor = conn.cursor()
       cursor.execute(
           f"INSERT OR IGNORE INTO order_book_snapshots ({cols_sql}) "
           f"VALUES ({placeholders})",
           values,
       )
       conn.commit()
       return (cursor.rowcount, None)
   except Exception as e:
       return (0, f"insert: {type(e).__name__}: {e}")
   ```

   The `ts_ms` and `dt` variables already exist earlier in the
   function (per `ca5c719`). The `bid_flat`, `ask_flat`, etc.
   exist as well. No other helpers need changing.

3. **Remove imports** that are no longer used after the collapse,
   if any (likely none -- the introspection used `conn.execute`
   which is unchanged).

4. **py_compile clean check**: `python -m py_compile
   engines/crypto_data_collector.py`.

## Verification

Just `py_compile`. Don't run the writer in-process; the next
hourly invocation of `PraxisOrderBookCollector` (fires at :00 of
each hour) will exercise it. Chat will verify post-deploy via
`get_collector_health` after the cleanup migration script runs +
the writer collapse commits.

## Why a brief instead of a full delta zip

Claude doesn't have the post-Cycle-23 file contents in local checkout
to safely diff against. Rather than reconstruct the writer from
memory and risk a typo, this brief asks Code to apply the targeted
change against the actual on-disk file. Any uncertainty about the
exact prior shape is resolved by reading the file directly.

If Code prefers, the equivalent unified diff against `ca5c719`
(reverse the writer change to single-write) is also acceptable --
the goal is the writer ends up doing one INSERT into
`order_book_snapshots` with ms timestamp, no introspection, no `_v2`
or `_legacy` references anywhere.
