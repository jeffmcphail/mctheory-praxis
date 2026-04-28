# Dual-Claude Workflow Protocol

## McTheory Development — Praxis (Prediction Markets & Systematic Trading)

**Version:** 1.1
**Author:** Jeff McPhail / Claude Chat
**Date:** April 22, 2026
**Adapted from:** AI Agent Factory protocol (validated, first retro: 30 min vs 2+ hours)

**Changelog:**
- v1.0 (2026-04-20): Initial Praxis protocol.
- v1.1 (2026-04-22): Added Progress Reporting Rules (9-15) operationalizing the ETA/Progress Reporting section of WORKFLOW_MODES_PRAXIS.md. Added Brief/Retro retention rules (36-37). Added required reading of WORKFLOW_MODES_PRAXIS.md on session start (rule 3). Renumbered subsequent rules accordingly.

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
20. **All scripts must be ASCII-only** -- no em dashes, box-drawing chars, emoji. Windows Task Scheduler pipes through cp1252.

### Praxis-Specific Rules
21. **Always use `load_dotenv()` before accessing any API keys.** Never assume raw environment variables.
22. **Always check the resolution oracle/source** for any Polymarket bet before trading.
23. **Every script that moves tokens or money** MUST have `input("Type YES to confirm: ")` gate before execution.
24. **Default to maximum validation and verbose output.** Add `--validate` and `--verbose` args. Relax as confidence increases.
25. **Polymarket API field names:** `proxyWallet` (not `userAddress`), `vol` (not `volume`), `userName` (not `username`).

### Testing Rules
26. **Run the script manually** with `python -m engines.<module> <command>` and paste the output.
27. **For scheduled task scripts:** verify no non-ASCII characters with `grep -P "[^\x00-\x7F]" file.py`.
28. **For data collection:** run `python -m engines.crypto_data_collector status` to verify data landed.
29. **For trading scripts:** ALWAYS dry-run first. Never `--execute` without Jeff's explicit approval.
30. **For Polymarket redemption/trading:** check on-chain balances before and after.

### Retro Rules (cross-reference `WORKFLOW_MODES_PRAXIS.md` for Brief/Retro retention)
31. **Write the retro before ending the session.** Save to `claude/retros/RETRO_<slug>.md`.
32. **Include ALL files modified** with line ranges.
33. **Include the debugging trail** -- what was tried, what failed, why.
34. **Include test results** -- both passes and failures.
35. **Include open items** -- anything that needs Chat's attention for strategy.
36. **Retros are permanent.** Never delete a retro, even for failed/partial attempts. Failed experiments are data.
37. **Briefs are permanent once a matching retro exists.** Do not delete Briefs to "clean up" `claude/handoffs/`. The only removable case: a Brief never executed and superseded -- rename to `ARCHIVED_<slug>.md`, do not delete.

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
+-- engines/           # Trading engines (lstm_predictor, ai_ensemble, smart_money, etc.)
+-- scripts/           # One-off scripts (redeem, batch_sell, debug tools)
+-- services/          # Windows scheduled task scripts (.bat, .ps1)
+-- data/              # SQLite databases (crypto_data.db, smart_money.db, etc.)
+-- models/            # Trained models (lstm/, crypto/)
+-- logs/              # Service logs
+-- docs/              # Documentation (TRADING_ATLAS.md, REGIME_MATRIX.md)
+-- claude/            # Dual-Claude handoffs and retros
+-- .claude/commands/  # Slash commands for Claude Code
```

### Running Services (Windows Scheduled Tasks)
| Task | Schedule | What |
|------|----------|------|
| PraxisLiveCollector | 60 seconds | Polymarket price snapshots |
| PraxisSmartMoney | 6 hours | Wallet discovery + position snapshots |
| PraxisCrypto1mCollector | 6 hours | BTC+ETH 1-minute candles |
| Praxis Funding Monitor | Periodic | Funding rate alerts |
| Praxis Sentiment Collector | Periodic | Sentiment data collection |

### Key Principles
- "Everything is a parameter" -- never restrict at design time
- Maximum validation and verbose output by default
- Always use `load_dotenv()` before API keys
- ASCII-only for scheduled task scripts (cp1252 encoding)
- Every token/money movement needs confirmation gate
