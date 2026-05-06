# Cycle 29 -- CCXT collector exit-code audit (hybrid mini-brief)

**Predecessor:** Cycle 28 (`4fa6941`) -- 1m collector exit-code hardening
**Mode:** hybrid (Claude drafts, Code applies)
**Risk:** low. Same structural change as Cycle 28, applied to 5 more
functions. Backwards-compatible: each function still returns the same
shape OR a status dict; `main()`'s dispatch wrapper from Cycle 28
already knows how to handle either.

## What

Apply Cycle 28's status-dict return + dispatch-side `sys.exit(1)` pattern
to the other CCXT/HTTP-dependent collectors in
`engines/crypto_data_collector.py`:

- `collect_ohlcv_daily` (line ~217)
- `collect_ohlcv_4h` (line ~268)
- `collect_fear_greed` (line ~387)
- `collect_funding_rates` (line ~422)
- `collect_onchain_btc` (line ~473)

Cycle 28 fixed only `collect_ohlcv_1m` because that was the one with
concrete outage evidence. The other five have the same silent-exit-on-
CCXT-error pattern but no observed outages yet. Cycle 29 closes the
gap pre-emptively before any of them experience a transient outage that
goes silent.

The dispatch wrapper in `main()` from Cycle 28 already does:

```python
result = dispatch[args.command](args)
if isinstance(result, dict) and result.get("status") == "error":
    print(f"\n[FAIL] {args.command}: {result.get('reason', 'unknown')}",
          file=sys.stderr)
    sys.exit(1)
```

So the only change in this cycle is per-function: each of the five
gains the same status-dict return at the bottom.

## NOT in scope

- `collect_market_data` (line 538): not a CCXT/HTTP-dependent fetcher
  in the same shape -- review separately if needed.
- `collect_order_book_snapshot` (line 585): part of the long-lived
  loop collector; different failure model (loop-level retry).
  Out of scope.
- `collect_recent_trades` (line 757): part of long-lived trades loop.
  Same reason.

## Specifics for Code

For each of the five target functions, apply the same pattern as
Cycle 28's `collect_ohlcv_1m` fix. The exact "fetched" / "stored"
counter names will vary per function -- read the actual code to find
the right variables. The pattern:

```python
# At the function tail, replace the implicit `return None` (or the
# end of the function body) with:

# Cycle 29: explicit status return so main() can exit non-zero
# when a transient error caused us to write 0 rows.
fetched = <count of items returned by the API; e.g. len(data) or
          a per-asset accumulator>
if fetched == 0:
    return {
        "status": "error",
        "reason": "<API> fetch returned 0 records (network/API failure)",
        "fetched": 0,
        "stored": 0,
    }
if stored == 0:
    return {
        "status": "error",
        "reason": f"Fetched {fetched} records but stored 0 (DB write failure)",
        "fetched": fetched,
        "stored": 0,
    }
return {
    "status": "ok",
    "fetched": fetched,
    "stored": stored,
}
```

Per-function notes (use the on-disk function body to confirm the right
counter variable names; my local checkout pre-dates Cycle 22 so my
references may not match exactly):

### collect_ohlcv_daily(asset, days, conn) (~line 217)

- API: `ccxt.binance().fetch_ohlcv(symbol, '1d', ...)`
- Fetched count: typically `len(ohlcv)` or accumulated `total_fetched`
- Stored count: in the per-row loop, count successful executes
- Reason text: "CCXT daily OHLCV fetch returned 0 candles (network/API failure)"

### collect_ohlcv_4h(asset, days, conn) (~line 268)

- Same shape as daily. Substitute "4h" in the reason text.

### collect_fear_greed(days, conn) (~line 387)

- API: `requests.get("https://api.alternative.me/fng/")`
- Fetched count: `len(json_data["data"])` or similar
- Reason text: "Fear & Greed API returned 0 records (network/API failure)"

### collect_funding_rates(asset, days, conn) (~line 422)

- API: `ccxt.binance().fetch_funding_rate_history(symbol, ...)`
- Fetched count: typically `len(rates)` or accumulated total
- Reason text: "CCXT funding rate fetch returned 0 records"

### collect_onchain_btc(days, conn) (~line 473)

- API: typically a third-party requests call (mempool.space or similar)
- Fetched count: `len(data)` or similar
- Reason text: "Onchain BTC API returned 0 records"

NOTE: `collect_onchain_btc` has been stale since 2026-04-28 per the
current `get_collector_health` output (`is_stale=true`,
`staleness_seconds=750917`). This cycle does NOT fix the underlying
root cause of that staleness -- the collector likely has no scheduled
task, OR the API endpoint changed, OR a credential expired. Adding
the status-dict return won't make it succeed; it will just make
future failures (once it has a scheduled task again) visible to Task
Scheduler. The underlying onchain_btc fix is a separate TODO.

## Verification

py_compile only:

```
python -m py_compile engines/crypto_data_collector.py
```

End-to-end behavior verification: each function will be exercised
at its next scheduled fire. If a CCXT/HTTP failure occurs at any of
them, `LastTaskResult=1` will surface in
`Get-ScheduledTask | Get-ScheduledTaskInfo` instead of silent
exit-0 success.

## Why a brief instead of a delta zip

Each function lives in `engines/crypto_data_collector.py` and the
on-disk shape may differ from my local checkout (which pre-dates
Cycle 22). Code reads the actual function body and adapts the
counter-variable names per function.

## Commit message (use this verbatim)

```
Cycle 29: CCXT collector exit-code audit

Applies Cycle 28's status-dict return pattern to the other five
CCXT/HTTP-dependent collectors: ohlcv_daily, ohlcv_4h, fear_greed,
funding_rates, onchain_btc. Each now returns a status dict; the
dispatch wrapper in main() (added in Cycle 28) exits non-zero on
{"status": "error"}.

Closes the same observability gap as Cycle 28 across the rest of
the scheduled-fire collectors, before any of them experience a
transient outage that goes silent.

Backwards-compatible. Loop-collectors (order_book, trades) and
collect_market_data are out of scope (different failure models).
```
