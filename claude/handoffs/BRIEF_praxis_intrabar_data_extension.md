# Implementation Brief: Intrabar Data Extension Test (v4)

**Series:** praxis
**Priority:** P1 (unblocks strategy decision)
**Mode:** B (live API calls, DB writes, live environment)
**Estimated Scope:** S (Small -- 30-45 min, diagnostic test + conditional path)
**Date:** 2026-04-22
**Follow-up to:** RETRO_praxis_intrabar_confluence.md (v3)

---

## Context

V3 retro's Case 4 (direction concentration: 163 SHORT / 0 LONG) fires and invalidates Case 3's "strategy is dead" verdict. The test window was pure downtrend -- we need balanced regime data (uptrend + sideways + downtrend) to conclude anything.

Initial check found `ohlcv_1m` spans only 34.8 days (2026-03-18 -> 2026-04-22). The collector code at line 271 of `engines/crypto_data_collector.py` hard-caps lookback at 30 days with a comment asserting "Binance retains ~30 days of 1-min data." But that comment was assumed, not verified. Before building a full historical loader for Binance ZIP archives (1-3 hours of work), we should test whether the live API actually enforces a 30-day cap or whether we can simply request more.

This brief tests that assumption in ~30 minutes. The result determines whether we get free dataset extension or whether we need to commit to the ZIP loader path.

---

## Objective

Determine experimentally how far back Binance's `fetch_ohlcv` endpoint will serve 1-minute candles for BTC and ETH. Write the results to the retro so Chat can decide the next step.

---

## Detailed Spec

### Step 1: Bypass the collector's 30-day cap

Create a one-off test script at `scripts/test_binance_1m_history.py` (new file, DO NOT modify the existing collector). The script should:

1. Load environment with `load_dotenv()` (API keys not strictly needed for public klines, but keep the pattern).
2. Use ccxt's `binance` exchange.
3. Attempt to fetch 1-minute OHLCV starting from progressively earlier timestamps:
   - 60 days ago
   - 90 days ago
   - 180 days ago
   - 365 days ago
4. For each attempt, fetch in batches (ccxt default limit is 500-1000 per call) and stop after the first successful batch OR the first empty/error response.
5. Report for each start date:
   - Whether the initial batch returned data
   - The actual oldest timestamp returned (may differ from requested)
   - The number of candles returned in the first batch
   - Any error messages verbatim

### Step 2: Conditional action based on Step 1

**If the API serves >=90 days:** extend the `ohlcv_1m` table directly by modifying the existing `collect_ohlcv_1m` function's `min(days, 30)` cap to `min(days, 180)` and running a backfill for BTC and ETH. Verify the row counts increase and the oldest timestamps move back.

**If the API caps at ~30 days:** do NOT modify the collector. Write the result to the retro and stop. Chat will decide between building the ZIP loader or pivoting.

### Step 3: Update retro

Whichever branch of Step 2 runs, document the outcome clearly in `claude/retros/RETRO_praxis_intrabar_data_extension.md` (new retro file, NOT overwriting v3). Include:

- Exact oldest timestamp returned for each probe date
- Whether collector was modified (and how much new data landed if so)
- Recommendation for Chat: can we proceed with Option C (extend), or do we need to pivot/build-loader?

---

## First-Pass Code Sketch

```python
# scripts/test_binance_1m_history.py
"""One-off test: how far back does Binance serve 1-min OHLCV?

This does NOT modify the live data collector. It only probes the API.
"""
import os
import sys
from datetime import datetime, timezone, timedelta

import ccxt
from dotenv import load_dotenv

load_dotenv()

PROBE_DAYS = [60, 90, 180, 365]
SYMBOLS = ["BTC/USDT", "ETH/USDT"]

def probe(exchange, symbol, days_back):
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)
    print(f"\n  Probing {symbol} starting {days_back} days back "
          f"(since={datetime.fromtimestamp(since_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')})")
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe="1m", since=since_ms, limit=1000)
        if not candles:
            print(f"    EMPTY response (API rejected or no data)")
            return None
        oldest_ts = candles[0][0] / 1000
        oldest_dt = datetime.fromtimestamp(oldest_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"    Returned {len(candles)} candles")
        print(f"    Oldest in batch: {oldest_dt}")
        print(f"    Requested was:   {datetime.fromtimestamp(since_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        return oldest_dt
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        return None

def main():
    exchange = ccxt.binance({"enableRateLimit": True})
    for symbol in SYMBOLS:
        print(f"\n{'='*60}\n  {symbol}\n{'='*60}")
        for days in PROBE_DAYS:
            probe(exchange, symbol, days)

if __name__ == "__main__":
    main()
```

### If Step 1 Succeeds (>=90 days available)

Modify `engines/crypto_data_collector.py` line 256-271 area:

```python
def collect_ohlcv_1m(asset, days, conn):
    """Collect 1-minute OHLCV.

    Binance historical availability: determined experimentally to be ~X days
    (see scripts/test_binance_1m_history.py and RETRO_praxis_intrabar_data_extension.md).
    """
    # ... existing code ...
    since = int((datetime.now(timezone.utc) - timedelta(days=min(days, 180))).timestamp() * 1000)
    # ... rest unchanged ...
```

Then run:
```powershell
python -m engines.crypto_data_collector collect-ohlcv-1m --asset BTC --days 180
python -m engines.crypto_data_collector collect-ohlcv-1m --asset ETH --days 180
python -m engines.crypto_data_collector status
```

Report the new row counts and date ranges in the retro.

---

## Acceptance Criteria

- [ ] `scripts/test_binance_1m_history.py` runs without errors and prints probe results for BTC and ETH at 60/90/180/365 days
- [ ] AST parse and ASCII check pass on any new/modified files
- [ ] Retro clearly states whether Binance serves >=90 days of 1-min data
- [ ] IF YES: collector modified, BTC+ETH backfilled, new row counts reported (should be ~130K+ rows per asset if 90 days landed cleanly)
- [ ] IF NO: collector NOT modified, retro flags that ZIP loader is needed for Option C

---

## Known Pitfalls

- **Rate limiting.** ccxt's `enableRateLimit=True` handles Binance's weight-per-minute cap automatically. Do NOT disable it.
- **`since` parameter vs actual returned data.** Binance may silently clamp `since` to whatever its oldest retained data is, returning recent data without error. The probe logic checks for this (compares requested vs oldest returned).
- **Empty response handling.** If Binance returns `[]` for a `since` value beyond retention, that's a successful API call with an empty payload -- not an error. The probe treats this as "not available."
- **DB duplicates.** The existing collector uses `INSERT OR REPLACE` so a backfill won't create duplicates, but the new rows will still count as new in row-count diffs (since they didn't exist before).
- **DO NOT modify the scheduled task** `PraxisCrypto1mCollector`. The one-off backfill is a manual run via the CLI. The scheduled task stays on its 30-day cap -- we don't need it doing 180-day runs every 6 hours.

---

## Do NOT

- Do NOT retrain any models in this brief. Training will come AFTER Chat reviews the retro and decides next steps.
- Do NOT modify `engines/intrabar_predictor.py`.
- Do NOT modify the scheduled task.
- Do NOT commit code (still uncommitted from v3).

---

## Open Questions (OK to decide during implementation, document in retro)

1. If Binance serves more than 365 days at 1-min, should we try even further (2+ years)? (Recommendation: YES, test 730 days too -- "too much history" is never a problem)
2. Should we also probe ohlcv_5m availability via the API? Some exchanges retain 5-minute data longer than 1-minute. (Recommendation: YES, as a bonus probe -- could give us a direct 5-min history source bypassing the resample step)

---

## References

- Retro v3: `claude/retros/RETRO_praxis_intrabar_confluence.md`
- Existing collector: `engines/crypto_data_collector.py` line 256-271
- Workflow mode doc: `claude/WORKFLOW_MODES_PRAXIS.md` (this is Mode B -- live API + DB writes)
