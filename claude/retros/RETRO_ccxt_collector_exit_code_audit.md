# Retro: Cycle 29 -- CCXT collector exit-code audit

**Brief:** `claude/handoffs/BRIEF_ccxt_collector_exit_code_audit.md`
**Date:** 2026-05-06
**Mode:** Hybrid (Claude drafted, Code applied)
**Status:** DONE
**Predecessor:** Cycle 28 (`4fa6941`) -- 1m collector exit-code hardening

---

## Summary

Applied Cycle 28's status-dict return pattern to the other five
CCXT/HTTP-dependent collectors:

- `collect_ohlcv_daily`
- `collect_ohlcv_4h`
- `collect_fear_greed`
- `collect_funding_rates`
- `collect_onchain_btc`

Each now returns a status dict identical in shape to the one
Cycle 28 added to `collect_ohlcv_1m`. The dispatch wrapper in
`main()` from Cycle 28 already handles this shape; no
dispatch-side changes were needed in this cycle.

Net change: 120 insertions / 1 deletion (+119 net) across 5
functions in `engines/crypto_data_collector.py`. The single
deletion is the `stored = 0` line inside `collect_fear_greed`'s
outer try/except, which was hoisted out of the try so the
function-tail status return can read it on the network-failure
path. py_compile clean.

---

## Why this matters

Per memory entry on collector exit-code honesty (2026-05-06):
"Scheduled collectors must non-zero-exit when transient errors
caused 0 rows written. Otherwise Task Scheduler reports
LastTaskResult=0 and silent gaps accumulate."

Cycle 28 fixed the one collector with concrete outage evidence
(1m). The other five had the same pattern but no observed
outages. With the recent network instability (router replaced,
intermittent HTTP failures observed in 1m collector logs), it
was only a matter of time before one of the others experienced
the same silent failure. Cycle 29 closes the gap pre-emptively.

The five fixed functions all run on scheduled (not long-lived)
tasks: ohlcv_daily and ohlcv_4h fire daily, fear_greed fires
daily, funding_rates fires every 8h, and onchain_btc was
intended for daily fires (currently no scheduled task -- see
"Open items" below).

---

## Execution log

### Step 1: Apply brief to all 5 functions

Code edited `engines/crypto_data_collector.py`:

| Function | Line | Counter pattern | Lines added |
|---|---|---|---|
| `collect_ohlcv_daily` | 211 | `len(all_candles)` / `stored` | +23 |
| `collect_ohlcv_4h` | 285 | `len(all_candles)` / `stored` | +23 |
| `collect_fear_greed` | 452 | `len(data)` / `stored` | +24 (incl. hoist of `stored=0` and `data=[]` init out of the outer try/except so the tail return can read them on the network-failure path) |
| `collect_funding_rates` | 511 | `len(all_rates)` / `stored` | +23 |
| `collect_onchain_btc` | 590 | `len(all_data)` / `stored` | +27 (incl. extra comment explaining that `len(all_data)` is "distinct dates with at least one of six metrics", since this collector aggregates across six blockchain.info endpoints) |

Total: +120 / -1 = +119 net.

Per-function notes:
- All four CCXT/HTTP collectors with a single `len(<accumulator>)`
  source (daily, 4h, funding_rates) got the standard 23-line
  template applied unchanged.
- `collect_fear_greed` needed a small restructure because its
  whole body was wrapped in one outer `try/except` -- moved
  `data = []` and `stored = 0` to function scope so the
  post-try status return reads them as zero on the network-
  failure path, exactly like `collect_ohlcv_1m`'s post-CCXT-error
  branch in Cycle 28.
- `collect_onchain_btc`'s "fetched" counter is necessarily
  fuzzier than the others -- it makes six independent API calls
  and aggregates per-date. Picked `len(all_data)` (distinct dates
  with at least one metric) because that maps 1:1 to what the
  store loop iterates over. If every API call fails, all_data is
  empty and the error branch fires.

The dispatch wrapper in `main()` (added in Cycle 28) was not
touched -- it already handles any returned `{"status": "error"}`
shape correctly.

### Step 2: py_compile

Clean.

### Step 3: Commit + push

Commit `<CYCLE_29_HASH>` on origin/master.

### Step 4: Verification

End-to-end verification is observational: each function will be
exercised at its next scheduled fire. Future CCXT/HTTP failures
will now surface as `LastTaskResult=1` in
`Get-ScheduledTask | Get-ScheduledTaskInfo` instead of silent
exit 0.

---

## Notes

### Cycle 28 dispatch wrapper proved out

Cycle 28's dispatch-side change was designed to be backwards-
compatible: any command returning a dict with `status: "error"`
triggers `sys.exit(1)`; any command returning anything else
(including `None`) exits 0 implicitly. Cycle 29 confirms this
design works -- adding the status-dict return to additional
functions required zero dispatch-side changes.

### Out-of-scope items (unchanged)

- `collect_market_data`: different shape (not CCXT/HTTP-fetcher
  in the same template). Audit separately if needed.
- `collect_order_book_snapshot`, `collect_recent_trades`: part
  of long-lived loop collectors with their own retry/restart
  machinery. Different failure model.

### Hybrid workflow: sixth cycle

Cycle 29 is the sixth hybrid cycle (after 23.5, 24.5, 28, 25.5,
27). The "apply same pattern to N functions" shape worked well
as a hybrid brief because the pattern is mechanical -- Code reads
each function, identifies the counter variables, applies the
same template, py_compiles. ~5-10 minutes of Code time for 5
functions.

---

## Open items / next cycle inputs

- **`collect_onchain_btc` underlying staleness**: this collector
  has been stale since 2026-04-28 (per `get_collector_health`).
  Cycle 29's fix will make any FUTURE failure visible, but does
  not address why the collector hasn't run in 9 days. Likely
  causes: no scheduled task registered, API endpoint changed,
  credential expired, or the source script was never wired up.
  Filed as a separate TODO; not part of Cycle 29's scope.

- **Cycle 26**: now the only remaining schema migration. trades
  table (~8.5M rows). Already near-conforming.
