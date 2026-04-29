# Implementation Brief: Expand MCP get_collector_health to Cover Cycle 10 Scheduled Tables

**Series:** praxis
**Cycle:** 11
**Priority:** P2 -- functional bug, but no incorrect alerts; bookkeeping/observability completeness fix
**Mode:** A (single-file edit to `servers/praxis_mcp/tools/meta.py`; no behavior change to other tools, no DB changes, no schema changes, no new infrastructure)

**Estimated Scope:** XS (~30 lines of code change inside one function)
**Estimated Cost:** $0
**Kill switch:** No edits to any file other than `servers/praxis_mcp/tools/meta.py`. No modifications to the other MCP tool modules (`funding.py`, `ohlcv.py`, `order_book.py`, `raw.py`, `trades.py`). No DB migrations. Existing acceptance criteria from Cycle 7's MCP work must continue to pass (`monitored_tables` dict structure, per-table thresholds, stale/not-stale flagging behavior).

Reference: `claude/CLAUDE_CODE_RULES.md` rules 9-15 (progress reporting), rule 16 (validation), rule 19 (ASCII).

---

## Context

Cycle 7 introduced the `monitored_tables` / `unmonitored` split in `get_collector_health()` to fix false-positive staleness alarms on tables that had no scheduled collector. The fix used hardcoded lists at the time:

- `monitored_tables`: trades / order_book_snapshots / ohlcv_1m
- `unmonitored_tables`: ["funding_rates", "fear_greed"]

**Cycle 10 changed that landscape.** Four new scheduled tasks now write to:

- `funding_rates` (PraxisFundingCollector, 8h cadence)
- `fear_greed` (PraxisFearGreedCollector, 24h cadence)
- `ohlcv_daily` (PraxisOhlcvDailyCollector, 24h cadence)
- `ohlcv_4h` (PraxisOhlcv4hCollector, 24h cadence)

`get_collector_health()` was not updated; it still hardcodes the pre-Cycle-10 split. Today's verification call returned:

- `funding_rates` and `fear_greed` listed in `unmonitored` despite now having scheduled collectors (incorrect classification but harmless -- no false-positive staleness alarms)
- `ohlcv_4h`, `ohlcv_daily`, `onchain_btc`, `market_data` missing from `unmonitored` entirely (incomplete bookkeeping; the user can't see at a glance which tables exist in the DB but aren't being checked)

Both issues stem from the same root cause: the `unmonitored_tables` list is hardcoded. The fix replaces it with dynamic computation derived from the DB schema.

The reframe: **anything in the DB that isn't in `monitored_tables` and isn't a SQLite internal table is unmonitored.** This is correct, future-proof, and self-maintaining as new tables are added.

There is also a small comment-block update needed in the source: the existing comment ("funding_rates and fear_greed are NOT monitored: no scheduled task writes to them. Populated only by manual one-shot CLI runs") is no longer true after Cycle 10. The fix updates the comment to reflect Cycle 10's scheduled tasks.

---

## Objective

Expand `get_collector_health()` in `servers/praxis_mcp/tools/meta.py` to:

1. Add four entries to `monitored_tables` covering the four scheduled collectors Cycle 10 introduced
2. Compute `unmonitored` dynamically from the DB schema (not hardcoded)
3. Update the inline comment block to match Cycle 10 reality
4. Preserve all existing behavior for the three pre-Cycle-10 monitored tables (trades / order_book_snapshots / ohlcv_1m)

---

## Detailed Spec

### Phase 0 -- Verify preconditions (1 min)

```powershell
cd C:\Data\Development\Python\McTheoryApps\praxis
.\.venv\Scripts\activate
```

Confirm the MCP server's smoke test passes BEFORE editing (so we have a clean baseline):

```powershell
python -m servers.praxis_mcp.test_smoke
```

Expected: "Smoke test PASSED." If it doesn't pass, stop and flag -- something has regressed since the last cycle.

### Phase 1 -- Edit `servers/praxis_mcp/tools/meta.py` (10 min)

Open `servers/praxis_mcp/tools/meta.py` and locate the `get_collector_health()` function. The change is confined to the body of that function plus the immediately preceding comment block.

**Find the existing block** (currently around line 145-160, identified by content -- exact line numbers may shift):

```python
            # Tables with active scheduled collectors. Threshold = natural
            # cadence + slack for scheduler jitter / windowed-gap behavior.
            #
            # trades                -- PraxisTradesCollector, continuous 10s.
            # order_book_snapshots  -- PraxisOrderBookCollector, 10s on-hour
            #                          but currently 1h-on/1h-off pattern
            #                          (see retro). 65 min tolerance covers
            #                          the worst-case sampling moment.
            # ohlcv_1m              -- PraxisCrypto1mCollector, 6h batch.
            #                          7h tolerance covers batch + slack.
            #
            # funding_rates and fear_greed are NOT monitored: no scheduled
            # task writes to them. Populated only by manual one-shot CLI
            # runs (python -m engines.crypto_data_collector collect-funding /
            # collect-fear-greed). Listed under `unmonitored` below.
            monitored_tables = {
                "trades": 120,
                "order_book_snapshots": 3900,
                "ohlcv_1m": 25200,
            }
            unmonitored_tables = ["funding_rates", "fear_greed"]
```

**Replace it with:**

```python
            # Tables with active scheduled collectors. Threshold = natural
            # cadence + slack for scheduler jitter / windowed-gap behavior.
            #
            # trades                -- PraxisTradesCollector, continuous 30s
            #                          (now 3550s windowed per Cycle 10
            #                          patch matching OrderBook).
            # order_book_snapshots  -- PraxisOrderBookCollector, 10s on-hour,
            #                          3550s windowed (Cycle 8 fix). 65 min
            #                          tolerance covers the worst-case
            #                          sampling moment plus inter-window gap.
            # ohlcv_1m              -- PraxisCrypto1mCollector, 6h batch.
            #                          7h tolerance covers batch + slack.
            # funding_rates         -- PraxisFundingCollector, 8h cadence
            #                          aligned approximately to Binance
            #                          funding events (Cycle 10). 9h
            #                          tolerance covers cadence + slack.
            # fear_greed            -- PraxisFearGreedCollector, daily at
            #                          00:30 local (Cycle 10). 26h tolerance.
            # ohlcv_daily           -- PraxisOhlcvDailyCollector, daily at
            #                          00:15 local (Cycle 10). 26h tolerance.
            # ohlcv_4h              -- PraxisOhlcv4hCollector, daily at
            #                          00:20 local (Cycle 10). 26h tolerance.
            #                          (Daily refresh of 4h bars; the cadence
            #                          is the refresh frequency, not the bar
            #                          frequency.)
            #
            # Tables NOT in this dict are computed dynamically below as
            # `unmonitored` -- the set of tables in the DB minus monitored
            # minus SQLite internals. Currently this captures onchain_btc
            # (no scheduled collector) and market_data (legacy/empty).
            monitored_tables = {
                "trades": 120,
                "order_book_snapshots": 3900,
                "ohlcv_1m": 25200,
                "funding_rates": 32400,    # 8h + 1h slack
                "fear_greed": 93600,       # 24h + 2h slack
                "ohlcv_daily": 93600,      # 24h + 2h slack
                "ohlcv_4h": 93600,         # 24h + 2h slack
            }

            # Dynamically compute unmonitored: every table in the DB that
            # isn't monitored and isn't a SQLite internal table.
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "ORDER BY name"
            )
            all_tables = {row["name"] for row in cursor.fetchall()}
            sqlite_internal = {"sqlite_sequence"}
            unmonitored_tables = sorted(
                all_tables - set(monitored_tables.keys()) - sqlite_internal
            )
```

The threshold constants are derived as follows:
- `funding_rates: 32400` = 8h * 3600 + 1h * 3600 = 28800 + 3600 = 32400 (cadence + 1h slack)
- `fear_greed: 93600` = 24h * 3600 + 2h * 3600 = 86400 + 7200 = 93600 (cadence + 2h slack to absorb F&G publishing delays from alternative.me)
- `ohlcv_daily: 93600` = same shape as fear_greed (24h cadence with 2h slack)
- `ohlcv_4h: 93600` = same shape (24h refresh cadence; 2h slack)

The 2h slack on the daily-cadence collectors is intentional: PowerShell Task Scheduler runs at "local" time (Toronto = UTC-4/UTC-5), so when `latest` is read by `get_collector_health()`, the bar cadence + the Task Scheduler local-time offset can stack. 2h covers normal slack; anything beyond that is a real problem worth flagging.

**Note on the staleness check semantics for daily-cadence tables:** the comment block now distinguishes between "cadence" (how often the collector runs) and "bar frequency" (the granularity of the data being collected). For `ohlcv_4h`, the collector runs daily but each run pulls multiple 4-hour bars; the staleness threshold reflects how often new data is *added*, which is determined by the collector's run cadence, not the bar's natural frequency.

There is no need to modify any other function. The dynamic-computation logic uses an SQL query that's already structurally identical to the one in `list_tables()`, so it follows established patterns.

### Phase 2 -- Verify (5 min)

After saving the file, **re-run the smoke test** to confirm nothing regressed:

```powershell
python -m servers.praxis_mcp.test_smoke
```

Expected: "Smoke test PASSED." Same as Phase 0.

Then verify the actual fix using the real DB. Since the MCP server is currently running as a subprocess inside Claude Desktop, **the running instance still has the old code in memory**; testing through the live MCP requires a Claude Desktop restart (Jeff's manual step). **In this Brief, verification happens directly via the Python module rather than through Claude Desktop.**

Write a small `claude/scratch/verify_health_expansion.py` helper:

```python
"""Verify the meta.get_collector_health expansion against the real DB."""
from pathlib import Path
import sys

# Ensure imports resolve from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.fastmcp import FastMCP
from servers.praxis_mcp.tools import meta

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "crypto_data.db"

# Spin up a one-shot FastMCP harness to register the tools and grab them
mcp = FastMCP("test")
meta.register(mcp, DB_PATH)

# FastMCP doesn't expose tools by name in a stable public API; fall back
# to importing the inner function via a re-implementation strategy.
# Instead, call the underlying logic by re-running the function definition:
# the cleanest path is to invoke meta.register again and capture the wrapped
# function. For verification simplicity, we'll re-import and call the body
# logic by running list_tools through the MCP interface.

# Simplest path: call get_collector_health via FastMCP's tool registry
import asyncio

async def main():
    tools = await mcp.list_tools()
    health_tool = next(
        (t for t in tools if t.name == "get_collector_health"), None
    )
    if health_tool is None:
        print("FAIL: get_collector_health not registered")
        return 1

    # FastMCP tool invocation
    result = await mcp.call_tool("get_collector_health", arguments={})

    # FastMCP returns a list of TextContent; parse the JSON payload
    import json
    payload_text = result[0].text if hasattr(result[0], "text") else str(result[0])
    payload = json.loads(payload_text)

    print("=== Verification: get_collector_health expansion ===")
    print()

    expected_monitored = {
        "trades", "order_book_snapshots", "ohlcv_1m",
        "funding_rates", "fear_greed", "ohlcv_daily", "ohlcv_4h",
    }
    actual_monitored = set(payload.get("tables", {}).keys())

    if actual_monitored == expected_monitored:
        print(f"PASS: monitored set is exactly {sorted(expected_monitored)}")
    else:
        missing = expected_monitored - actual_monitored
        extra = actual_monitored - expected_monitored
        print(f"FAIL: monitored mismatch")
        print(f"  missing: {sorted(missing)}")
        print(f"  extra:   {sorted(extra)}")
        return 1

    expected_unmonitored = {"onchain_btc", "market_data"}
    actual_unmonitored = set(payload.get("unmonitored", []))

    if actual_unmonitored == expected_unmonitored:
        print(f"PASS: unmonitored set is exactly {sorted(expected_unmonitored)}")
    else:
        missing = expected_unmonitored - actual_unmonitored
        extra = actual_unmonitored - expected_unmonitored
        print(f"FAIL: unmonitored mismatch")
        print(f"  missing: {sorted(missing)}")
        print(f"  extra:   {sorted(extra)}")
        return 1

    # Verify each new monitored table reports a sane state (not stale unless
    # actually stale; staleness_seconds present and non-negative)
    for tbl in ("funding_rates", "fear_greed", "ohlcv_daily", "ohlcv_4h"):
        info = payload["tables"].get(tbl, {})
        if "error" in info:
            print(f"WARN: {tbl} reported error: {info['error']}")
            continue
        ss = info.get("staleness_seconds")
        threshold = info.get("threshold_seconds")
        is_stale = info.get("is_stale")
        print(f"  {tbl}: staleness={ss}s threshold={threshold}s stale={is_stale}")

    print()
    print("=== Verification complete ===")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

Run it:

```powershell
python claude\scratch\verify_health_expansion.py
```

Expected output: PASS lines for both monitored and unmonitored sets, plus per-table staleness status for the four new tables. None of the four should be `is_stale=True` since Cycle 10's collectors all populated their tables earlier today.

If any FAIL prints, do not commit -- diagnose first. The most likely failure modes are typos in the table names in `monitored_tables` keys or wrong threshold constants.

### Phase 3 -- Retro

Standard retro at `claude/retros/RETRO_health_expansion.md`. Include:

1. The diff against `servers/praxis_mcp/tools/meta.py` (use `git diff` output)
2. Smoke test result before edit (Phase 0)
3. Smoke test result after edit (Phase 2)
4. Output of `verify_health_expansion.py` showing the monitored/unmonitored sets and per-table status
5. Confirmation the file is ASCII-only and Python syntax is valid (verifiable via the smoke test passing)
6. Note for Jeff: he must restart Claude Desktop to pick up the change in the running MCP server subprocess (the fix doesn't auto-reload; Claude Desktop loads the server once at launch)

---

## Acceptance Criteria

1. `servers/praxis_mcp/tools/meta.py` modified with the expanded `monitored_tables` dict (7 entries instead of 3) and dynamic `unmonitored_tables` computation
2. Comment block in `get_collector_health()` updated to reflect Cycle 10's scheduled collectors
3. Phase 0 smoke test passed (baseline)
4. Phase 2 smoke test passed (post-edit)
5. `claude/scratch/verify_health_expansion.py` reports PASS for both monitored set and unmonitored set
6. No edits to any file other than `servers/praxis_mcp/tools/meta.py` (the helper script in `claude/scratch/` is gitignored per Cycle 9's `.gitignore` update)
7. Retro at `claude/retros/RETRO_health_expansion.md` with diff + verification output + Phase 0/2 smoke test status

---

## Known Pitfalls

- **Don't restart the running MCP server during this Brief.** The MCP server is currently running as a subprocess of Claude Desktop. Editing the source on disk doesn't affect the running instance -- it picks up the change only on next Claude Desktop launch. This is fine: verification happens via the standalone Python script, not through the live MCP. Jeff handles the Claude Desktop restart manually after this Brief lands.
- **Threshold constants are seconds, not milliseconds.** All existing entries use seconds; preserve that convention. Don't accidentally provide ms values (you'll get instant false-positive staleness on every check).
- **The `cursor` variable is already in scope.** The dynamic-computation block uses the existing `cursor` from earlier in the function; don't open a new connection.
- **`sqlite_internal` is a set, not a list.** This matters because the set difference operation (`all_tables - monitored_keys - sqlite_internal`) requires set semantics on both sides.
- **Helper script in scratch is fine to leave around.** Per Cycle 9, `claude/scratch/` is gitignored. Code can write `verify_health_expansion.py` there and leave it.
- **PowerShell here-strings + python -c (Cycle 9 reprise).** Don't fight the harness. Write the verification helper as a real `.py` file in `claude/scratch/` and invoke it normally.
- **CRLF for .py files is fine.** Unlike `.bat` and `.ps1`, Python doesn't care about line endings on Windows. No `unix2dos` step needed for the verification helper.

---

## What this Brief deliberately does NOT do

- Does not modify any other tool module (`funding.py`, `ohlcv.py`, `order_book.py`, `raw.py`, `trades.py`)
- Does not modify `server.py`, `db.py`, `test_smoke.py`, or `__init__.py`
- Does not change the `monitored_tables` thresholds for the three pre-Cycle-10 tables (trades / order_book_snapshots / ohlcv_1m). Their original Cycle 7 thresholds remain.
- Does not address the `onchain_btc.date` schema heterogeneity flagged in Cycle 9 retro 6.4 (separate cycle)
- Does not fix the trades.timestamp millisecond-vs-seconds heterogeneity (separate cycle, also flagged)
- Does not modify the `claude_desktop_config.json` Claude Desktop config (lives in MSIX virtualized path; out of repo)
- Does not address the `engines/burgess.py` legacy / `src/praxis/models/burgess.py` migration -- separate cycle
- Does not write `docs/SCHEMA_NOTES.md` -- separate cycle, queued

---

## References

- `servers/praxis_mcp/tools/meta.py` -- the only file modified
- `claude/retros/RETRO_praxis_collector_outage_triage.md` -- Cycle 7 retro that established the monitored/unmonitored pattern this Brief expands
- `claude/handoffs/BRIEF_scheduled_collectors.md` -- Cycle 10 Brief that added the four new scheduled collectors
- `claude/retros/RETRO_scheduled_collectors.md` -- Cycle 10 retro confirming all four scheduled tasks are operational
- `services/register_funding_task.ps1`, `register_fear_greed_task.ps1`,
  `register_ohlcv_daily_task.ps1`, `register_ohlcv_4h_task.ps1` -- the
  registration scripts whose collectors this Brief teaches `get_collector_health()` about
- `claude/CLAUDE_CODE_RULES.md` -- standing rules; rule 19 (ASCII), rule 16 (validation)
