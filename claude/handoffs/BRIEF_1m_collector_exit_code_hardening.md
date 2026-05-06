# Cycle 28 -- 1m collector exit-code hardening (hybrid mini-brief)

**Predecessor:** Cycle 22 (ohlcv_1m migration)
**Mode:** hybrid (Claude drafts, Code applies)
**Trigger:** Cycle 23.5 outage investigation 2026-05-05.
The `engines/crypto_data_collector.py` 1m collector silently swallowed
CCXT errors AND a SQLite `OperationalError: database is locked`,
logged "Collection complete", and exited 0 on multiple consecutive
fires. Task Scheduler reported `LastTaskResult=0` for all of them,
and the resulting data gaps (07:24 UTC, 19:24 UTC May 5; 01:24 UTC
May 6) accumulated unnoticed until ohlcv_1m showed in
`get_collector_health` as 10+ hours stale.

## The bug

`collect_ohlcv_1m(asset, days, conn)` in
`engines/crypto_data_collector.py` (around line 319 pre-Cycle-22;
verify line numbers post-Cycle-22 if shifted):

1. Returns `None` implicitly. No success/failure signal.
2. Catches CCXT exceptions in the fetch loop, prints them, and
   `break`s -- so a CCXT failure on the very first batch yields
   `len(all_candles) == 0` and the function proceeds to the store
   loop with nothing to store.
3. Catches SQLite exceptions per-row in the store loop, prints
   `pass`, and continues -- so a `database is locked` on every
   row yields `stored == 0` with no error surfaced.
4. Always prints "Stored N 1m candles (X.X days)" even when N=0.
5. The dispatch site at `main()` calls
   `dispatch[args.command](args)` without checking the return
   value, so `main()` returns `None`, and the script exits 0.

Result: Task Scheduler can't distinguish "ran cleanly with 2880
candles fetched" from "ran cleanly with 0 candles fetched after
total CCXT failure" -- both report `LastTaskResult=0`.

## The fix (3 changes)

### Change 1: `collect_ohlcv_1m` returns a status dict

Replace the function's return path at the bottom (currently
implicit `return None` after `print(f"    Stored ...")`) with:

```python
    conn.commit()
    days_covered = len(all_candles) / 1440 if all_candles else 0
    print(f"    Stored {stored} 1m candles ({days_covered:.1f} days)")

    # Cycle 28: explicit status return so main() can exit non-zero
    # when a transient error caused us to write 0 rows. Task Scheduler
    # uses LastTaskResult to surface failures; without an explicit
    # non-zero exit on real failures, silent gaps accumulate.
    fetched = len(all_candles)
    if fetched == 0:
        return {
            "status": "error",
            "reason": "CCXT fetch returned 0 candles (network/API failure)",
            "fetched": 0,
            "stored": 0,
        }
    if stored == 0:
        return {
            "status": "error",
            "reason": f"Fetched {fetched} candles but stored 0 (DB write failure -- check for lock contention)",
            "fetched": fetched,
            "stored": 0,
        }
    return {
        "status": "ok",
        "fetched": fetched,
        "stored": stored,
    }
```

Note: this fix treats partial success (fetched > 0, stored > 0) as
ok even if `stored < fetched`. The store loop's per-row try/except
swallows individual row failures (e.g. unique-constraint violations
on idempotent re-fetches), which is fine. The error case is when
we wrote NOTHING.

### Change 2: dispatch wrapper at `main()` checks the return value

The current dispatch site:

```python
    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()
```

becomes:

```python
    if args.command in dispatch:
        result = dispatch[args.command](args)
        # Cycle 28: commands MAY return a status dict; if they do
        # and report error, exit non-zero so Task Scheduler surfaces
        # the failure. Commands that return None (most of them, for
        # now) are treated as success (preserves pre-Cycle-28
        # behavior for unmodified commands).
        if isinstance(result, dict) and result.get("status") == "error":
            print(
                f"\n[FAIL] {args.command}: {result.get('reason', 'unknown')}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        parser.print_help()
```

This is backwards-compatible: only commands that return the new
dict shape get the new exit-code behavior. All other commands
keep their existing implicit-None return and exit 0.

### Change 3: import `sys` if not already imported

Verify `sys` is imported at the top of the file. It probably
already is (used elsewhere). If not, add `import sys`.

## What's NOT in scope

- Other CCXT-dependent collectors (`collect_ohlcv_daily`,
  `collect_ohlcv_4h`, `collect_fear_greed`, `collect_funding_rates`,
  `collect_onchain_btc`) have the same silent-exit-on-CCXT-error
  pattern. They should get the same treatment, but in a follow-up
  cycle (Cycle 29 or rolled into a single audit pass). This cycle
  only fixes the one we know is causing real outages.
- The order-book / trades loop collectors (`cmd_collect_order_book`,
  `cmd_collect_order_book_loop`, `cmd_collect_trades`,
  `cmd_collect_trades_loop`) are long-lived and have their own
  retry/restart machinery. Different failure model. Out of scope.

## Verification

py_compile only:
```
python -m py_compile engines/crypto_data_collector.py
```

End-to-end behavior verification will happen at the next 6h
scheduled fire of `PraxisCrypto1mCollector` (next runs at the :23
of every 6th hour). If the fire succeeds, `LastTaskResult=0` and
data flows. If a CCXT/lock failure recurs, `LastTaskResult=1`
will be visible in `Get-ScheduledTask -TaskName
"PraxisCrypto1mCollector" | Get-ScheduledTaskInfo`, and the next
fire's log will end with the new `[FAIL] collect-ohlcv-1m: ...`
line on stderr.

## Commit message (use this verbatim)

```
Cycle 28: 1m collector exit-code hardening

collect_ohlcv_1m now returns a status dict; main() exits non-zero
when a command reports error. Catches the silent-exit-on-CCXT-error
pattern that masked three real outages on 2026-05-05/06 (07:24,
19:24, 01:24 UTC) where Task Scheduler reported success despite
0 candles written.

Backwards-compatible: other commands still return None and exit 0
implicitly. Same pattern can be extended to ohlcv_daily, ohlcv_4h,
fear_greed, funding_rates, onchain_btc in a follow-up audit cycle.

Triggered by Cycle 23.5 outage investigation. Per memory entry on
collector exit-code honesty (2026-05-06).
```
