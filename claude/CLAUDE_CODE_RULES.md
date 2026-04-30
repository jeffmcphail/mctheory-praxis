# Dual-Claude Workflow Protocol

## McTheory Development -- Praxis (Prediction Markets & Systematic Trading)

**Version:** 1.4
**Author:** Jeff McPhail / Claude Chat
**Date:** April 30, 2026
**Adapted from:** AI Agent Factory protocol (validated, first retro: 30 min vs 2+ hours)

**Changelog:**
- v1.0 (2026-04-20): Initial Praxis protocol.
- v1.1 (2026-04-22): Added Progress Reporting Rules (9-15) operationalizing the ETA/Progress Reporting section of WORKFLOW_MODES_PRAXIS.md. Added Brief/Retro retention rules (36-37). Added required reading of WORKFLOW_MODES_PRAXIS.md on session start (rule 3). Renumbered subsequent rules accordingly.
- v1.2 (2026-04-29): Four new/extended rules from cycles 8-12: extended Rule 20 (ASCII-only) with the Unicode-runtime-recognition pattern using module-level constants; new Rule 21 (CRLF preservation for .bat/.ps1); new Rule 32 (MCP server changes require Claude Desktop full-quit + relaunch including kill of any orphaned Python processes); new Rule 33 (prefer .py file over inline `python -c` for non-trivial diagnostics). Refreshed the Running Services table to reflect Cycle 10's registered tasks plus PraxisLiveCollector and PraxisSmartMoney reactivation. Refreshed the Key Directories tree to include `servers/`, `tests/`, `src/`, `contracts/`, `dashboards/`, `gui/`, `k8s/`, `spikes/`, `examples/`, `battle_results/`, `market_data/`, and `claude/scratch/` -- long-present directories that were never added. Renamed "Testing Rules" to "Testing and Diagnostics Rules". Renumbered subsequent rules accordingly. Total rule count: 40 (was 37).
- v1.3 (2026-04-30): One new rule from Cycle 15 diagnostic investigation: new Rule 34 (explicit transaction management on SQLite reads against actively-written DBs). Python's sqlite3 module has documented quirks around implicit BEGIN that can cause long-lived connections to see snapshot views from past states. The defensive practice is fresh connections per logical read pass OR `isolation_level=None` OR explicit `conn.commit()` between SELECTs to release the implicit read transaction. Rule lands in the Testing and Diagnostics Rules subsection. Renumbered Retro Rules from 34-40 to 35-41. Total rule count: 41 (was 40).
- v1.4 (2026-04-30): One new rule from Cycle 17: new Rule 35 (temporal data storage standard). Establishes the canonical schema for any table holding temporally indexed data: INTEGER `timestamp` column in ms-since-epoch UTC, part of the primary key, optionally with a derived `datetime`/`date` TEXT cache in ISO 8601 with `+00:00` offset. Codifies the dual-write migration pattern for high-frequency tables. Lands in a new "Data Storage Rules" subsection between "Testing and Diagnostics Rules" and "Retro Rules". Renumbered Retro Rules from 35-41 to 36-42. Total rule count: 42 (was 41).

---

## Overview

This protocol defines a two-Claude development workflow where **Claude Chat** handles strategy, architecture, research, and cross-session memory, while **Claude Code** handles implementation, debugging, testing, and file operations. The two communicate via structured handoff documents and retro logs, minimizing context bloat in Chat and eliminating the delta-zip cycle.

---

## The Two Roles

### Claude Chat (Strategist)

- Architecture decisions and design discussions
- Research deep dives (web search, competitive analysis, article reviews)
- Cross-session memory (remembers project state, conventions, decisions)
- Implementation planning with detailed specs
- Review of retro logs from Claude Code sessions
- Maintains the project narrative across all sessions
- Trading strategy research and validation analysis

### Claude Code (Implementer)

- Direct file editing in the live repo
- Running and testing code (scripts, engines, data collectors)
- Debugging with real-time log inspection
- Git operations (commits, branches, diffs)
- Multi-file refactors
- Script execution and verification
- Maintains a structured session log of everything done

---

## Workflow Cycle

```
+-----------------------------------------------------------------+
|  CLAUDE CHAT SESSION                                            |
|                                                                 |
|  1. Review retro log from last Code session                     |
|  2. Discuss strategy, architecture, next steps                  |
|  3. Produce IMPLEMENTATION BRIEF (.md)                          |
|     - Detailed spec of what to build                            |
|     - First-pass code (if helpful)                              |
|     - Acceptance criteria                                       |
|     - Known pitfalls / warnings                                 |
|                                                                 |
|  Output: claude/handoffs/BRIEF_<slug>.md                        |
+-----------------------------------------------------------------+
                       |
                       v
+-----------------------------------------------------------------+
|  CLAUDE CODE SESSION                                            |
|                                                                 |
|  1. Read the Implementation Brief                               |
|  2. Read CLAUDE_CODE_RULES.md (this protocol)                   |
|  3. Implement, test, debug, iterate                             |
|  4. Maintain SESSION_LOG in real-time                            |
|  5. On completion: write RETRO_<slug>.md                        |
|                                                                 |
|  Output: claude/retros/RETRO_<slug>.md                          |
+-----------------------------------------------------------------+
                       |
                       v
+-----------------------------------------------------------------+
|  CLAUDE CHAT SESSION (next)                                     |
|                                                                 |
|  1. Jeff uploads RETRO_<slug>.md (or it's in project docs)      |
|  2. Chat reviews what was done, what failed, what changed       |
|  3. Updates memory, project state                               |
|  4. Plans next cycle                                            |
|                                                                 |
+-----------------------------------------------------------------+
```

---

## File Structure

All handoff documents live in the repo under `claude/`:

```
praxis/
+-- claude/
|   +-- handoffs/                    # Chat -> Code briefs
|   |   +-- BRIEF_lstm_lightgbm.md
|   |   +-- BRIEF_convergence_detector.md
|   |   +-- ...
|   +-- retros/                      # Code -> Chat session logs
|   |   +-- RETRO_lstm_lightgbm.md
|   |   +-- RETRO_convergence_detector.md
|   |   +-- ...
|   +-- scratch/                     # Diagnostic helpers, throwaway scripts (gitignored)
|   +-- CLAUDE_CODE_RULES.md         # This protocol (for Code to read)
+-- .claude/
    +-- commands/                    # Slash commands for Claude Code
        +-- brief.md
        +-- retro.md
        +-- commit.md
        +-- syntax-check.md
```

---

## Implementation Brief Format

Claude Chat produces this. It tells Claude Code exactly what to do.

```markdown
# Implementation Brief: <Title>

**Series:** praxis
**Priority:** P0 (blocking) / P1 (important) / P2 (nice-to-have)
**Estimated Scope:** S/M/L (Small = < 1 hour, Medium = 1-3 hours, Large = 3+ hours)
**Date:** YYYY-MM-DD

## Context
Why this work is needed. What happened that led to this task.
Reference to specific Chat session or test results if applicable.

## Objective
One clear sentence: what does "done" look like?

## Detailed Spec

### What to change
- File: `engines/lstm_predictor.py`
  - Function `train_multi_horizon_lstm()`: add LightGBM as third model
  - Add quantile regression with `objective="quantile"`

### First-Pass Code (optional)
If Chat has already drafted the implementation, include it here.

### What NOT to change
Explicit list of files/functions to leave alone.

## Acceptance Criteria
- [ ] LightGBM trains alongside XGBoost and LSTM
- [ ] Walk-forward backtest shows results for all three models
- [ ] `python -m engines.lstm_predictor train --asset BTC` completes without errors

## Known Pitfalls
- All Python files used in scheduled tasks must be ASCII-only (no em dashes, emoji)
- Always use `load_dotenv()` before accessing API keys
- Windows Task Scheduler pipes through cp1252 encoding

## References
- Chat session: praxis_main_current
- Related files: `engines/lstm_predictor.py`, `engines/crypto_data_collector.py`
```

---

## Retro Log Format

Claude Code produces this. It tells Claude Chat exactly what happened.

```markdown
# Retro: <Title>

**Brief:** BRIEF_<slug>.md
**Date:** YYYY-MM-DD
**Duration:** ~2 hours
**Status:** COMPLETE / PARTIAL / BLOCKED

## Summary
One paragraph: what was accomplished.

## Changes Made

### Files Modified
| File | Change | Lines |
|------|--------|-------|
| `engines/lstm_predictor.py` | Added LightGBM ensemble | 700-750 |
| `engines/crypto_data_collector.py` | Added liquidation data collection | 280-320 |

### Key Decisions
- Chose LightGBM over CatBoost because lighter dependency
- Used `quantile` objective instead of custom loss

## Test Results

### Passed
- `python -m engines.lstm_predictor train --asset BTC` completes
- LightGBM 7d directional accuracy: 54.2%
- Walk-forward backtest runs without errors

### Failed / Known Issues
- 30d horizon still overfits (96% train, 37% test)
- Liquidation data only available from 2025 onward

## Failures & Debugging Trail
Document what went wrong during implementation, even if fixed.

### Attempt 1: LightGBM quantile
- `objective="quantile"` requires `alpha` param, not `quantile_alpha`
- Fixed by checking LightGBM docs

## Commits
- `abc1234` -- feat: add LightGBM to multi-horizon ensemble
- `def5678` -- feat: liquidation data collection from Binance

## Open Items for Chat
- Should we add Transformer model too? (Corvino found it didn't help)
- Liquidation data gaps before 2025 -- use synthetic data or skip?
- Consider scheduled task for daily model retrain

## Artifacts
- `models/lstm/BTC_multi_horizon.joblib` -- updated with 3-model ensemble
```

---

## Rules for Claude Code

### General Rules
1. **Read the Brief first.** Always start by reading the Implementation Brief in `claude/handoffs/`.
2. **Read this rules file.** Every session.
3. **Read `claude/WORKFLOW_MODES_PRAXIS.md`** on session start. It defines Mode A / Mode B, the ETA and Progress Reporting protocol, Brief/Retro retention rules, and periodic check-in cadence.
4. **Maintain the session log in real-time.** Write to the retro as you go.
5. **Test before declaring done.** Run the code. Check the output.
6. **Document failures.** The debugging trail is as valuable as the fix.
7. **Don't make architectural decisions.** If something requires a design choice not covered in the Brief, note it in the retro's "Open Items for Chat" and either ask Jeff or make the minimal/safe choice.
8. **Commit incrementally.** Small commits with clear messages.

### Progress Reporting Rules (MANDATORY for long-running tasks)

These rules operationalize the "ETA and Progress Reporting" section of `WORKFLOW_MODES_PRAXIS.md`. Follow them as mechanical instructions, not guidelines.

9. **State the estimate at session start.** Before launching any task expected to take >5 min, restate the Brief's scope and expected runtime. If initial investigation changes the picture, announce the revised estimate before proceeding.

10. **Proactive self-polling for background tasks.** When you launch any command with `run_in_background: true` that is expected to run >5 min (training, backtesting, historical data pulls, long simulations), you MUST schedule your own status checks. Do NOT wait for the user to prompt you. Specifically:
    - At T+5 min after launch: tail the log, report current state in format `"X min of ~Y min estimated . <current phase>"`
    - At T+10 min: another check. Report per-iteration timing if relevant (epoch time, rows processed, etc.).
    - Every ~5 min thereafter until completion or intervention.
    - Use a Monitor tool or a bash poll loop or manual checks between other work -- mechanism doesn't matter, but the cadence is not optional.

11. **Flag state changes immediately.** Any of these require an out-of-cadence report, not waiting for the next 5-min window:
    - Per-iteration time changes by more than 50% vs. previous iteration (e.g., epoch 4 takes 1.5x epoch 3)
    - Loss/metric trajectory becomes flat when it should be descending
    - Any warning or error appears in the log
    - Projected completion time exceeds the Brief's estimate by >50%

12. **Kill switch compliance.** Every Brief for long-running work specifies a kill switch (maximum acceptable runtime). If actual runtime approaches 90% of the kill switch without completion:
    - Pause and report: elapsed, projected completion, reason for overrun, recommendation (continue / abort / change parameters)
    - Do NOT let the task run past the kill switch without explicit approval from the user.

13. **Real-money-adjacent work.** If the task touches any trade execution path (Polymarket orders, Binance trades, WCOL movements, fund transfers), report every 2 minutes regardless of duration, and require explicit user approval before any real-money action executes.

14. **Silent progress is a bug.** If a long-running task is producing no log output for >10 min, investigate before assuming it's "just working slowly." Check:
    - Is the process alive? (`tasklist | findstr python`)
    - Is the log file being written? (mtime check)
    - Is CPU accumulating? (wmic or equivalent)
    - Is memory stable or growing? (working set check)
    Report what you find, then decide continue/kill.

15. **The user should never need to ask for a progress update.** If they do, you've already missed the cadence. When this happens, acknowledge the miss briefly, give the update, and restore cadence going forward.

### File Rules (CRITICAL)
16. **NEVER modify `.env` files.** Reference secrets via `os.getenv()` with `load_dotenv()`.
17. **NEVER modify `*.db` database files directly.** Use migration scripts or JSON exports.
18. **NEVER delete or overwrite `data/` directory contents** without explicit confirmation from Jeff.
19. **Always run `python -c "import ast; ast.parse(open('file.py').read())"` after editing Python files.**
20. **All scripts must be ASCII-only** -- no em dashes, box-drawing chars, emoji. Windows Task Scheduler pipes through cp1252. **When a parser must RECOGNIZE Unicode characters at runtime** (e.g., `engines/atlas_sync.py` parsing markdown that contains em dashes, multiplication signs, etc.), define module-level constants like `EM_DASH = "\u2014"`, `EN_DASH = "\u2013"`, `MULT_SIGN = "\u00d7"`, and concatenate them into regex patterns:
    ```python
    EM_DASH = "\u2014"
    MULT_SIGN = "\u00d7"
    parts = re.split(r"\s+[xX" + MULT_SIGN + r"]\s+", body, maxsplit=1)
    ```
    This keeps source ASCII while runtime correctly handles non-ASCII input. Cycle 12 retro section 8 has the canonical example.
21. **Preserve CRLF line endings on `.bat` and `.ps1` files.** The Edit tool silently rewrites these files with LF-only line endings, which can break Windows Task Scheduler execution. After every edit to a `.bat` or `.ps1` file:
    - Run `unix2dos <path>` to restore CRLF
    - Verify with `file <path>` showing "with CRLF line terminators"
    - Verify with `tr -cd '\r' < <path> | wc -c` showing a non-zero count matching the line count
    Caught early in Cycle 8 (RETRO_order_book_duration_fix.md section 4); a recurring trap. Don't skip the verification step -- the failure mode is silent at edit time and only surfaces when Task Scheduler tries to invoke the file.

### Praxis-Specific Rules
22. **Always use `load_dotenv()` before accessing any API keys.** Never assume raw environment variables.
23. **Always check the resolution oracle/source** for any Polymarket bet before trading.
24. **Every script that moves tokens or money** MUST have `input("Type YES to confirm: ")` gate before execution.
25. **Default to maximum validation and verbose output.** Add `--validate` and `--verbose` args. Relax as confidence increases.
26. **Polymarket API field names:** `proxyWallet` (not `userAddress`), `vol` (not `volume`), `userName` (not `username`).

### Testing and Diagnostics Rules
27. **Run the script manually** with `python -m engines.<module> <command>` and paste the output.
28. **For scheduled task scripts:** verify no non-ASCII characters with `grep -P "[^\x00-\x7F]" file.py`.
29. **For data collection:** run `python -m engines.crypto_data_collector status` to verify data landed.
30. **For trading scripts:** ALWAYS dry-run first. Never `--execute` without Jeff's explicit approval.
31. **For Polymarket redemption/trading:** check on-chain balances before and after.
32. **For MCP server changes:** the running MCP subprocess in Claude Desktop won't pick up source changes until Desktop is fully relaunched (right-click tray icon -> Quit, kill any lingering Claude.exe via `Get-Process -Name "Claude*" | Stop-Process -Force`, kill any orphaned Python from the praxis venv via `Get-Process -Name "python*" | Where-Object { $_.Path -like '*praxis\.venv*' } | Stop-Process -Force`, then relaunch). Verification of MCP changes happens via `python -m servers.praxis_mcp.test_smoke` (standalone) or via a `claude/scratch/` helper script that loads tools through `FastMCP` directly. The smoke test gives the canonical "Registered tools: N" count.
33. **Prefer a `.py` file over inline `python -c` for any non-trivial diagnostic.** PowerShell's quote-mangling on Windows breaks inline Python with embedded single+double quotes, format strings, or multi-line content. Write the helper to `claude/scratch/<name>.py` (gitignored per the convention established in Cycle 9) and invoke it normally:
    ```powershell
    @'
    import sqlite3
    c = sqlite3.connect('data/crypto_data.db')
    print(c.execute('SELECT COUNT(*) FROM trades').fetchone()[0])
    '@ | Out-File -Encoding utf8 claude\scratch\check_trades.py
    python claude\scratch\check_trades.py
    ```
    The single-quoted here-string `@'...'@` preserves quotes literally without PowerShell variable interpolation. Trivial one-liners (no embedded quotes, no format strings) are still fine inline. Cycle 9 retro 6.7 and Cycle 11 implementation notes both surfaced this.
34. **Always explicitly manage transactions when reading a SQLite DB that another process is actively writing to.** Python's `sqlite3` module has documented quirks around implicit `BEGIN` that can cause a long-lived connection to see a snapshot view from a past state, missing all writes that committed since. The exact reproduction is version- and pattern-dependent and can manifest as "I'm reading the same DB the live MCP server is reading and getting hours-old data." Three acceptable patterns:
    - **Fresh connection per logical read pass.** Open, query, close. The MCP server's `connect_ro` does this and never sees stale data. Cheapest and most foolproof:
        ```python
        def fresh_read():
            conn = sqlite3.connect("data/live_collector.db")
            try:
                cur = conn.cursor()
                cur.execute("SELECT MAX(timestamp) FROM price_snapshots")
                return cur.fetchone()
            finally:
                conn.close()
        ```
    - **`isolation_level=None` for true autocommit.** Each statement is its own transaction; no implicit BEGIN sticks around:
        ```python
        conn = sqlite3.connect("data/live_collector.db", isolation_level=None)
        ```
    - **Explicit `conn.commit()` between SELECTs.** On a read connection, `commit()` ends any open implicit transaction without changing data; the next SELECT begins a fresh transaction with current state.

    Do NOT keep a single `sqlite3.Connection` open across multiple SELECT passes that span more than a few seconds without one of the above. Cycle 15 diagnostic confirmed this is real and the cause of mysterious "stale data" reads. See `claude/retros/RETRO_sqlite_freshness_diagnostic.md` for the full investigation.

### Data Storage Rules
35. **Temporal data storage standard.** Every table that contains
    temporally indexed data MUST have:

    1. A `timestamp` column of INTEGER type, storing Unix epoch
       milliseconds in UTC. Always the same units (milliseconds, not
       seconds). Always UTC. The column name is always `timestamp`.
       Date-only data converts to midnight-UTC milliseconds before
       storage.

    2. `timestamp` is part of the table's primary key. Alone for
       single-asset tables, or compound (`(asset, timestamp)`,
       `(slug, timestamp)`, etc.) for multi-keyed tables. Whatever
       uniquely identifies a row, the temporal component IS the
       `timestamp` column.

    3. Optionally, a redundant `datetime` (or `date`) TEXT column if
       read-side speed matters. When present, it's strictly derived
       from `timestamp` and stored in ISO 8601 with explicit UTC offset
       (`+00:00`) for datetime, or `YYYY-MM-DD` (interpreted as UTC
       midnight) for date-only. The `timestamp` column remains
       canonical; `datetime`/`date` is a cache.

    4. For data feeds that return non-UTC timestamps, the collector MUST
       convert to UTC before storage. Common foot-guns: APIs that return
       local time without offset, APIs that return seconds-since-epoch
       but anchor to local-midnight, file-based feeds (CSV) where the
       timezone is documented in the source's API docs but invisible in
       the data itself. When in doubt, audit a known-time sample (e.g.
       fetch a Binance kline that should close at 04:00:00 UTC and
       confirm the stored timestamp matches).

    5. For new tables / new collectors, conformance is mandatory.
       For existing tables, migration is tracked in
       `docs/SCHEMA_MIGRATION_PLAN.md` and executes one table per cycle.

    6. For migrations of actively-written tables, use the dual-write
       pattern:
       - Phase 0: Build new schema as `<table>_v2` and a parallel
         collector path
       - Phase 1: Register a new scheduled task targeting `_v2`; old
         task keeps running
       - Phase 2: Backfill historical rows from old -> `_v2` (timestamp
         converted, UTC text rendered with `+00:00`)
       - Phase 3: Verify the overlap window (rows collected by both
         tasks) is identical at the source-API level, and all readers
         work against `_v2`
       - Phase 4: Stop old task, rename old table to `<table>_legacy`,
         rename `_v2` -> `<table>`, point readers at the renamed table
       - Phase 5: Burn-in observation (24-48h) before deleting `_legacy`

       For batch/daily collectors where the source API allows full
       re-fetch (Binance OHLCV, funding rates, etc.), the simpler
       stop-migrate-start pattern is acceptable as long as the gap
       window can be backfilled from the API after the new collector
       starts. Document which pattern is used for each table in
       `docs/SCHEMA_MIGRATION_PLAN.md`.

    Migration recipe (SQLite, simple pattern): create new table with
    target schema, INSERT-SELECT from old table converting units and
    rendering UTC text, DROP old, RENAME new. Verify all readers
    (engines, MCP tools, analysis scripts) still work before
    committing.

### Retro Rules (cross-reference `WORKFLOW_MODES_PRAXIS.md` for Brief/Retro retention)
36. **Write the retro before ending the session.** Save to `claude/retros/RETRO_<slug>.md`.
37. **Include ALL files modified** with line ranges.
38. **Include the debugging trail** -- what was tried, what failed, why.
39. **Include test results** -- both passes and failures.
40. **Include open items** -- anything that needs Chat's attention for strategy.
41. **Retros are permanent.** Never delete a retro, even for failed/partial attempts. Failed experiments are data.
42. **Briefs are permanent once a matching retro exists.** Do not delete Briefs to "clean up" `claude/handoffs/`. The only removable case: a Brief never executed and superseded -- rename to `ARCHIVED_<slug>.md`, do not delete.

---

## Rules for Claude Chat

### Planning Rules
1. **Always review the retro first** when starting a new session.
2. **Update memory** with key decisions, state changes, and new conventions from the retro.
3. **Write briefs with acceptance criteria.** Claude Code needs clear "done" definitions.
4. **Include first-pass code when possible.** It dramatically speeds up Code's implementation.
5. **Include "Known Pitfalls" from experience.** (e.g., cp1252 encoding, WCOL collateral, phantom NegRisk opps)

### Context Management
6. **Don't ask for full file dumps.** Read project knowledge and retros instead.
7. **Don't do implementation over many iterations.** Design it, write the Brief, hand it off.
8. **Track the project state in memory**, not in the conversation.

### Brief Naming Convention
- `BRIEF_praxis_<feature>.md`
- Examples: `BRIEF_praxis_lightgbm_ensemble.md`, `BRIEF_praxis_convergence_detector.md`, `BRIEF_praxis_atr_position_mgmt.md`

### Retro Naming Convention (must match brief)
- `RETRO_praxis_<feature>.md`

---

## Quick Reference: When to Use Which Claude

| Task | Claude Chat | Claude Code |
|------|:-----------:|:-----------:|
| "What should we build next?" | X | |
| "Build it" | | X |
| "Why isn't this working?" (needs log analysis) | | X |
| "Why isn't this working?" (needs architecture rethink) | X | |
| Research (articles, strategies, competitive analysis) | X | |
| Web search for API docs | X | |
| Multi-file refactor | | X |
| Writing a Brief | X | |
| Writing a Retro | | X |
| Debugging a script | | X |
| Strategy research / article deep dives | X | |
| Git operations | | X |
| Cross-session context ("what did we decide about X?") | X | |
| Running scheduled tasks / checking logs | | X |
| Trading strategy validation / backtest analysis | X | |

---

## Praxis Project Context

### Repo Path
`C:\Data\Development\Python\McTheoryApps\praxis`

### Key Directories

```
praxis/
+-- engines/           # Trading engines, data collectors, models, analysis
+-- scripts/           # One-off scripts (redeem, batch_sell, debug tools)
+-- services/          # Windows scheduled task scripts (.bat + register_*.ps1)
+-- servers/
|   +-- praxis_mcp/    # MCP server exposing read-only data access (Cycle 7)
|       +-- tools/     # MCP tool modules (meta, ohlcv, order_book, trades, funding, raw, atlas)
+-- src/               # Modern post-migration src layout (praxis package)
+-- tests/             # pytest suite
+-- data/              # SQLite databases (crypto_data.db, smart_money.db, praxis_meta.db, etc.); gitignored
+-- models/            # Trained models; gitignored
+-- logs/              # Service logs; gitignored
+-- docs/              # TRADING_ATLAS.md, REGIME_MATRIX.md, ATLAS_DB.md, etc.
+-- contracts/         # Solidity contracts for on-chain execution
+-- dashboards/        # Streamlit / web dashboards
+-- gui/               # Desktop GUI components (mcb_studio)
+-- k8s/               # Kubernetes manifests (forward-looking)
+-- spikes/            # Throwaway exploration code
+-- examples/          # Reference implementations
+-- battle_results/    # Backtest output archives
+-- market_data/       # Vendor data captures
+-- claude/
|   +-- handoffs/      # Chat -> Code briefs (permanent per WORKFLOW_MODES_PRAXIS.md)
|   +-- retros/        # Code -> Chat session logs (permanent)
|   +-- scratch/       # Diagnostic helpers, throwaway scripts (gitignored, Cycle 9)
+-- .claude/
    +-- commands/      # Slash commands for Claude Code (brief.md, retro.md, etc.)
```

### Running Services (Windows Scheduled Tasks)

After Cycle 10's recovery and Cycle 13's reactivation of `PraxisLiveCollector` and `PraxisSmartMoney`, the canonical set of registered tasks is below. All registration scripts live in `services/`. The meta-registrar at `services/register_all_tasks.ps1` discovers them by glob and supports `-Only` to target a subset.

| Task | Schedule | Source bat | Notes |
|------|----------|------------|-------|
| PraxisOrderBookCollector | Hourly back-to-back | order_book_collector_service.bat | 3550s windowed (Cycle 8 race fix); 10s cadence; BTC+ETH order book snapshots |
| PraxisTradesCollector | Hourly back-to-back | trades_collector_service.bat | 3550s windowed (Cycle 10 audit fix); 30s cadence; BTC+ETH trades with buyer/seller tagging |
| PraxisCrypto1mCollector | Every 6 hours | crypto_1m_collector_service.bat | BTC+ETH 1m candles, --days 2 overlap |
| PraxisOhlcvDailyCollector | Daily 00:15 | ohlcv_daily_collector_service.bat | Cycle 10; BTC+ETH daily candles, --days 7 overlap |
| PraxisOhlcv4hCollector | Daily 00:20 | ohlcv_4h_collector_service.bat | Cycle 10; BTC+ETH 4h candles, --days 7 overlap |
| PraxisFundingCollector | 00:05/08:05/16:05 | funding_collector_service.bat | Cycle 10; local time, not UTC; --days 7 absorbs offset |
| PraxisFearGreedCollector | Daily 00:30 | fear_greed_collector_service.bat | Cycle 10; alternative.me F&G index |
| PraxisLiveCollector | At system startup, restart-on-failure loop | live_collector_service.bat | Pre-recovery + Cycle 13 reactivation; samples top 50 Polymarket markets every 60s; auto-restart with 30s delay; writes to data/live.db (or similar) |
| PraxisSmartMoney | Every 6 hours | smart_money_service.bat | Pre-recovery + Cycle 13 reactivation; wallet discovery + position snapshots; writes to data/smart_money.db |

The MCP server (`servers/praxis_mcp/`) provides `get_collector_health()` for live staleness monitoring of the registered set. Per-table thresholds defined in `servers/praxis_mcp/tools/meta.py` (Cycle 11 expansion). Note: `PraxisLiveCollector` and `PraxisSmartMoney` write to separate SQLite DBs (`live.db` / `smart_money.db`) which are NOT currently monitored by the MCP health check; expanding `get_collector_health()` to cover those is queued as a future cycle.

### Key Principles
- "Everything is a parameter" -- never restrict at design time
- Maximum validation and verbose output by default
- Always use `load_dotenv()` before API keys
- ASCII-only for scheduled task scripts (cp1252 encoding); use module-level Unicode constants when runtime parsing needs non-ASCII recognition (Rule 20)
- CRLF line endings on `.bat` and `.ps1` files; verify with `file <path>` after every edit (Rule 21)
- Every token/money movement needs confirmation gate (Rule 24)
- Diagnostics in `claude/scratch/` as `.py` files; avoid quote-heavy inline `python -c` (Rule 33)
- Reads against actively-written SQLite DBs need explicit transaction management: fresh connections per pass, `isolation_level=None`, or `conn.commit()` between SELECTs (Rule 34)
- Temporal data: store as INTEGER ms-since-epoch UTC `timestamp` column, primary key. Optional `datetime`/`date` TEXT cache renders ISO 8601 with `+00:00` (Rule 35).
