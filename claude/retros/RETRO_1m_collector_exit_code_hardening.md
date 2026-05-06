# Retro: Cycle 28 -- 1m collector exit-code hardening

**Brief:** `claude/handoffs/BRIEF_1m_collector_exit_code_hardening.md`
**Date:** 2026-05-06
**Mode:** Hybrid (Claude drafted, Code applied)
**Status:** DONE
**Predecessor:** Cycle 23.5 outage investigation 2026-05-05.

---

## Summary

Cycle 28 fixes the silent-exit-on-CCXT-error pattern in
`engines/crypto_data_collector.py`'s `collect_ohlcv_1m` function.
Pre-Cycle-28 the function caught CCXT exceptions, logged them,
and exited 0 -- producing `LastTaskResult=0` in Windows Task
Scheduler regardless of whether 0 candles or 2880 candles were
actually written. This masked three real outages on 2026-05-05/06
(07:24, 19:24, 01:24 UTC) where the collector "ran cleanly" but
wrote nothing.

The fix:

1. `collect_ohlcv_1m` now returns a status dict
   (`{"status": "ok"|"error", "reason": ..., "fetched": N,
   "stored": M}`).
2. `main()`'s dispatch site now checks the return value and
   `sys.exit(1)` if the command reports error.
3. Other commands' return-None pattern preserved -- new behavior
   is opt-in per command.

Net change: 37 insertions / 1 deletion (+36 net) in
`engines/crypto_data_collector.py`. py_compile clean.

---

## Why this matters

Memory entry from 2026-05-06: "Scheduled collectors must
non-zero-exit when transient errors caused 0 rows written.
Otherwise Task Scheduler reports LastTaskResult=0 and silent
gaps accumulate."

The Cycle 23.5 outage investigation surfaced four distinct 1m
collector failures masked by exit code 0:

| UTC fire | Cause | Rows written |
|---|---|---|
| 2026-05-05 07:24 | CCXT GET exchangeInfo failed | 0 |
| 2026-05-05 19:24 | CCXT GET klines failed at batch 1 | 1000 (BTC partial), 2880 (ETH OK) |
| 2026-05-06 01:24 | SQLite OperationalError: database is locked (x2) -- traced to OrderBook long-lived process holding lock after Cycle 23.5 cleanup script dropped `_legacy` | 0 |
| 2026-05-06 07:24 | Self-recovered after OrderBook PIDs killed | normal |

All four reported `LastTaskResult=0`. Without external
verification via `get_collector_health`, this would have
continued silently for an unknown duration.

---

## Execution log

### Step 1: Apply brief

Code applied the three changes per the brief:

- `collect_ohlcv_1m` (line 313) gained a status-dict return
  at function exit. Distinguishes:
  - `fetched == 0` -> error (CCXT/network failure)
  - `fetched > 0 AND stored == 0` -> error (DB write failure)
  - `fetched > 0 AND stored > 0` -> ok (partial idempotent
    re-fetches still count as ok)
- `main()` (line 1115) gained the dispatch-result check:
  `sys.exit(1)` if `result["status"] == "error"`. The existing
  dispatch lambda for `collect-ohlcv-1m` already returns the
  function's value, so no lambda change needed.
- `import sys` already present at the top of the file (line 31);
  no import change required.

### Step 2: py_compile

Clean.

### Step 3: Commit + push

Commit `<CYCLE_28_HASH>` on origin/master.

### Step 4: Verification

The next `PraxisCrypto1mCollector` fire (next at :23 of every
6th hour) will exercise the new path. Expected outcomes:

- **Normal success**: `LastTaskResult = 0`, data flows. Same as
  before.
- **CCXT failure**: `LastTaskResult = 1`. Stderr ends with
  `[FAIL] collect-ohlcv-1m: CCXT fetch returned 0 candles...`.
  Visible to `Get-ScheduledTask -TaskName
  "PraxisCrypto1mCollector" | Get-ScheduledTaskInfo`.
- **DB lock**: `LastTaskResult = 1`. Stderr ends with `[FAIL]
  collect-ohlcv-1m: Fetched N candles but stored 0 (DB write
  failure...)`.

Verification of the ok path will be observable at the next 6h
fire (no special action needed). Verification of the error path
is harder to force -- next time CCXT genuinely fails (likely
within a week given the recent outage frequency), we'll see
the new behavior in action.

---

## Notes

### Scope kept narrow on purpose

Other CCXT-dependent collectors (daily, 4h, fear_greed, funding,
onchain) have the same silent-exit pattern. Could have been
fixed in this cycle. Two reasons not to:

1. The 1m collector is the one that has CONCRETE OUTAGE EVIDENCE
   from the last 24h. Others might also fail this way but we
   haven't observed it. Fix what's known broken; audit the rest
   in a follow-up.
2. Each of those functions has slightly different shape (some
   use `requests`, some use `ccxt`; some have inner loops, some
   don't). Auditing them needs read-the-actual-file work that
   this cycle isn't doing.

Cycle 29 (or whenever) should be a full audit pass: same status-
dict pattern applied to all CCXT-dependent collectors with
similar exit-code semantics. Worth a half-hour.

### Hybrid-workflow speedup

Cycle 28 is the third hybrid cycle (after 23.5 and 24.5). Active
drafting time: ~<DURATION>. The brief format is settling into a
predictable shape: predecessor, bug description, exact code
changes, scope-out, verification, commit message. ~150 lines
total -- much shorter than the old-workflow 600+-line briefs.

### Cycle 23.5 ordering lesson reinforced

Cycle 28 is downstream of the Cycle 23.5 incident. The chain:

Cycle 23.5 (Phase 5 cleanup, wrong order)
  -> OrderBook hits dropped `_legacy` for 5h
  -> SQLite write-lock contention on crypto_data.db
  -> 1m collector fails with `database is locked`
  -> 1m collector swallows error, exits 0 silently
  -> data gap invisible until external observability

Memory entry #11 captures the ordering lesson; Cycle 28 closes
the observability gap. Next time something similar happens
(any cause), Task Scheduler will surface it.

---

## Open items / next cycle inputs

- **Cycle 29 (audit pass)**: same status-dict pattern for
  `collect_ohlcv_daily`, `collect_ohlcv_4h`, `collect_fear_greed`,
  `collect_funding_rates`, `collect_onchain_btc`. Half-hour cycle.
- **Cycle 25.5** (position_snapshots Phase 5 cleanup): still
  queued. PraxisSmartMoney is scheduled (not long-lived), so the
  natural ordering trick from Cycle 23.5/24.5 doesn't apply --
  next scheduled invocation just picks up new code.
- **Cycle 26** (trades): still queued. Largest table; near-
  conforming already (timestamp already INTEGER ms). Need to
  check process pattern (long-lived or scheduled) before
  drafting.
