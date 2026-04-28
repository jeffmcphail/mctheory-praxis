# Workflow Mode Arbitration — Praxis

Rules for how work gets done between Claude Chat, Claude Code, and Jeff on the Praxis project. **Chat is the arbiter** — it picks the mode and announces the choice. Jeff just follows the resulting instructions.

---

## The Two Working Modes

### Mode A — Chat edits directly

**Chat makes the file changes itself and delivers a delta zip.** Jeff extracts the zip into the repo. Done.

**When Chat uses Mode A:**
- Small, contained changes (1-3 files, well-defined scope)
- Chat's sandbox can verify the change is correct (syntax check, test run, or the change is simple enough not to need testing)
- No live environment needed (no running service, no real crypto data DB, no real API calls, no scheduled task modifications)
- **No impact on any code path that could execute a real trade or move real funds**

**Jeff's instructions for Mode A:**
1. Download the delta zip Chat provides
2. Extract into `C:\Data\Development\Python\McTheoryApps\praxis\`
3. Reply "done" — or paste any error

---

### Mode B — Claude Code implements and tests

**Chat writes a Brief. Jeff points Claude Code at it. Claude Code implements, tests, writes a Retro. Jeff uploads the Retro back.**

Claude Code does **everything** — implementation, testing, debugging, git operations. Jeff's only jobs are: pass the brief to Code, wait, upload the retro.

**When Chat uses Mode B:**
- Live environment interaction needed (Binance/CCXT, Polymarket CTF/NegRisk, Deribit, yfinance, real API calls)
- **Anything that could execute a real trade or move real funds** (automatic Mode B, no exceptions)
- Touches Windows scheduled tasks (PraxisLiveCollector, PraxisSmartMoney, Funding Monitor, Sentiment Collector)
- Reads/writes live crypto data DB, model files (`phase3_models.joblib`, etc.), or collected market snapshots
- Multi-file investigation — grep across engines/scripts/dashboards to figure out what's there
- Unknown scope — might uncover issues that change the plan mid-implementation
- Large refactors (4+ files) or long-running work (>30 min of Code session)
- Backtesting runs that need real historical data
- Model training (LSTM, quantamental, XGBoost) or retraining
- Anything requiring `load_dotenv()` + real API keys

**Jeff's instructions for Mode B:**
1. Download the delta zip with the brief(s)
2. Extract into the repo root
3. Open Claude Code Desktop
4. Run: `Read claude/handoffs/BRIEF_<slug>.md and implement it.`
5. Wait for completion
6. Upload `claude/retros/RETRO_<slug>.md` back to Chat

---

## Brief and Retro Retention (Mode B artifacts)

**Briefs and Retros are both permanent project artifacts once created.** They form the audit trail the Dual-Claude protocol is designed to produce.

- **Briefs** document what was asked for and how Chat scoped it. Useful for reviewing Chat's scoping accuracy (wrong paths, stale field names, missed context) and for future chats understanding design intent.
- **Retros** document what actually happened during implementation — decisions Code made, pitfalls hit, open items flagged, test results, spend, backtest outputs, model training runs.

The Brief+Retro **pair** is the unit of record. Reading `claude/handoffs/` alongside `claude/retros/` should let a future chat reconstruct the project's evolution — *especially* critical for trading-strategy work where the rationale for a strategy change is as important as the change itself.

**Retention rules:**
- Once a Brief has a matching Retro, both are permanent.
- Do NOT delete Briefs to "clean up" `claude/handoffs/`. The directory is not a todo list — it's a historical record.
- The only time a Brief should be removed is if it was never executed AND is being superseded by a revised version. Even then, prefer renaming to `ARCHIVED_<slug>.md` over hard deletion.
- Retros are never deleted. Full stop. A retro documenting a failed backtest or a model that didn't generalize is as valuable as one documenting a success — it prevents re-exploring dead ends.

---

## ETA and Progress Reporting (both modes)

**Every task must include an explicit time estimate before starting, and a running progress indicator while executing. Extra weight on runtime-heavy work like backtests, model training, and live data collection.**

### Chat's obligations (Mode A)

When announcing a Mode A task, Chat states the expected time to deliver the delta zip (e.g., "~2 minutes for the edit + verification"). Short tasks are fine to just note briefly.

### Brief obligations (Mode B)

Every Brief that Chat writes must include at the top:

```
**Estimated Scope:** <XS/S/M/L/XL> (<minutes> min)
**Estimated Cost:** $<X> in LLM/API spend (or "none" if no external calls)
**Estimated Data Volume:** <rows/snapshots/calls> touched (for live data work)
```

Scope guide: XS = <5 min, S = 5-30 min, M = 30 min-2 hr, L = 2-6 hr, XL = 6+ hr. Be generous — include typical overrun. If scope is genuinely unknown (e.g., "train an LSTM and see how it performs"), the brief should say so explicitly and start with a scoping investigation phase whose output is a revised estimate.

For backtesting / model training / long historical pulls, the Brief must also estimate:
- Expected runtime (wall-clock)
- Expected data scan (rows, snapshots, candles processed)
- Kill switch: maximum acceptable runtime before pause-for-approval

### Claude Code's obligations

For any task expected to take >5 minutes, Code should:

1. **State the estimate at start**, restating what the Brief said (or updating it if initial investigation changes the picture).
2. **Report progress every ~5 min** in the status window in the format:
   ```
   "X min of ~Y min estimated · <current phase>"
   ```
   Examples:
   - `"18 min of ~60 min estimated · backtesting funding rate carry on 2024 data (8 of 12 months)"`
   - `"5 min of ~30 min estimated · training XGBoost fold 3 of 5"`
   - `"2 min of ~10 min estimated · collecting Polymarket NegRisk order book snapshots"`
3. **Flag overruns early.** If actual progress suggests the task will exceed the estimate by >50%, pause and report: actual elapsed, new projection, reason for overrun, recommendation (continue / pause for approval / abort).
4. **For long-running background processes** (backtests, model training, data pulls), tail the log at each check-in and report: current phase, elapsed, last log line, projected completion, any error/warning spikes.
5. **For anything touching trade execution paths**, report at minimum every 2 minutes regardless of duration, and require Jeff approval before any real-money action.

### Enforcement

If Code starts a long-running task without an estimate, or runs more than 10 minutes without progress reports, Chat flags it in the next retro review as a protocol miss and updates the brief template if the root cause is estimator-side (bad initial estimate from Chat).

---

## Periodic Manual Check-ins (orthogonal to mode)

Every **3–5 completed cycles** (Mode A or Mode B, mixed), Chat designs a short hands-on exercise for Jeff — 5–10 minutes of running a script, checking a dashboard, or eyeballing a signal so he stays connected to the system's behavior.

This is **not** per-task testing — Code tests its own work in Mode B, and Mode A is low-risk by definition. The check-in is about Jeff maintaining human intuition for the system (market regimes, signal behavior, strategy P&L patterns), not QA.

**Cadence:**
- Chat tracks cycle count in conversation
- When count hits 3, Chat flags: "Check-in due after this cycle — want me to prep one?"
- Jeff can defer (push to 5) or accept
- Counter resets after the check-in is completed

**Check-in content (Chat produces):**
- Short list of what changed since last check-in
- 5–10 min numbered checklist (run X, check Y dashboard, eyeball Z signal)
- Specific things to notice (new funding rate divergence, new smart money positioning, etc.)
- Known-risk areas worth poking at

**What's NOT a check-in:**
- Full regression testing (that's what proper test scripts are for)
- >15 minutes of Jeff's time
- Anything spending real money beyond trivial test calls

---

## Arbitration Decision Tree

For each task, Chat picks A or B. First match wins.

**Pick Mode B if ANY of:**
- Task could affect any code path that executes a real trade or moves real funds → **always Mode B, always**
- Task needs live API calls (Binance, Polymarket, Deribit, yfinance)
- Task reads/writes live DB state (crypto data, collected snapshots, funding history, model artifacts)
- Task touches Windows scheduled tasks (create/modify/delete/inspect)
- Task is >3 files or needs grep-discovery
- Scope is unknown
- Task takes >30 min of implementation time
- Task involves model training, backtesting, or simulation runs
- Task requires `.env` credentials

**Otherwise Mode A.**

**Default if unclear:** Mode A. But when in doubt about trade execution impact → Mode B.

---

## Validation Principle (Praxis-specific)

For any new scripts or significant logic changes, always err on **maximal validation and verbose output**. Especially critical for anything touching trade execution or real money. This applies in both Mode A (Chat codes defensively) and Mode B (Brief specifies `--validate` + `--verbose` flags, level-based verbosity, default maximum).

Relax validation only as confidence grows — never at design time.

---

## Chat's Announcement Format

At the start of any task, Chat states the mode and estimated time:

> **Mode A.** [one-sentence rationale] (~X min)
>
> [edits, delta zip, Mode-A instructions]

> **Mode B.** [one-sentence rationale — especially if real-money-adjacent] (~Y min Code time, ~$Z spend, data volume if applicable)
>
> [brief, delta zip, Mode-B instructions]

---

## Escalation Rule

If a Mode A task hits >2 rounds of "that broke," Chat pauses and escalates to Mode B. Cheap to correct early, expensive to brute-force.

---

## Examples

| Task | Mode | Why |
|------|------|-----|
| Adjust a threshold constant in one engine script | A | 1 file, 1 line, no live state |
| Fix typo or formatting in TRADING_ATLAS.md | A | Markdown only |
| Update REGIME_MATRIX.md section | A | Docs only |
| Rename a function argument in a standalone utility | A | Contained, no live state |
| Tweak Streamlit dashboard color/label | A | Cosmetic, no live data risk |
| Add a new CLI flag to an existing script (no live call path) | A | 1-2 files, chat can test syntax |
| Add a new engine that places real Polymarket orders | B | Real money path |
| Modify `negrisk_arb.py` or `ai_ensemble.py` trade logic | B | Real money adjacent |
| Build crypto prediction system (LSTM + quantamental) | B | Multi-file, model training, live data |
| Backtest a new strategy over historical data | B | DB-dependent, long runtime |
| Modify a Windows scheduled task or its script | B | Live scheduled infra |
| Retrain `phase3_models.joblib` | B | Model artifact + data + long runtime |
| Build convergence speed detector using live collector data | B | Live DB + grep across collector code |
| Refactor shared utilities across 5 engines | B | Multi-file, regression risk |
| Add a new metric to a monitoring dashboard (read-only) | A | Contained, read-only, cosmetic |
| Investigate why funding monitor is logging errors | B | Needs live log inspection |

---

## Cycle Counter

Chat keeps running count in-conversation. Hits 3 → flag a check-in. Resets after check-in completed.

---

*Last updated: 2026-04-22 (Chat: ai_factory_main_current, written for praxis_main_current)*
*Changes: Added Brief/Retro Retention section. Briefs and Retros are both permanent once paired; `claude/handoffs/` is a historical record, not a todo list. Only exception: un-executed, superseded Briefs may be renamed to ARCHIVED_*. Retros are never deleted — failed backtests / non-generalizing models are as valuable as successes for avoiding re-exploration of dead ends.*
