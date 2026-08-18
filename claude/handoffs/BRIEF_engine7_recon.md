# Cycle 39 RECON -- Engine 7 (Funding Carry) live-vs-paper reconciliation scoping

**Predecessor:** Cycle 38 RECON (commits `b0ace4b` + `a5fd142`). Atlas
Exp 13 is the target entry (CONFIRMED POSITIVE; Cycle 37 suspect-LOW
on the headline Sharpe figures pending operational verification).

**Mode:** Read-only investigation. No code changes, no commits, no
experimental runs. Output is a structured findings report that
becomes the input to a follow-up implementation brief.

**Risk:** very low. Cycle scope is documentation + code reading +
DB queries on existing live data.

**Scope cap:** ~2 hours Code time. Four investigation streams:
- L1-L3 live deployment archaeology (~30 min)
- L4-L6 historical-period reconciliation feasibility (~30 min)
- L7-L9 current-regime sit-out verification (~30 min)
- L10-L11 design synthesis (~30 min)

## Why this cycle is RECON not implementation

Claude pulled atlas Exp 13 + funding_rates table state before drafting
the brief and found three constraints that materially reshape what
"reconciliation" means as a cycle:

**Constraint 1: Live collector covers 2/6 of validated universe.**
Atlas Exp 13's universe is BTC, ETH, SOL, XRP, ADA, AVAX (BNB excluded).
funding_rates table contains BTC + ETH only (1,143 events each from
2025-04-30 to 2026-05-15). The four missing assets (SOL, XRP, ADA, AVAX)
contributed materially to atlas's validation Sharpe -- e.g. AVAX +63%
cum / Sharpe +16.5; ADA was the primary's top model at +10.5% cum /
Sharpe +7.21. A full reconciliation cycle requires either (a) extending
the live collector to the full universe before reconciling, or (b)
scoping reconciliation to BTC + ETH only (matches what's live but
covers ~33% of validated universe).

**Constraint 2: Current regime is bear-side for BTC, mild-bull for
ETH.** Last 30d BTC funding mean: -2.07% annualized; positive_share
0.348. Last 10 ETH events: +3.0% annualized; positive_share 0.90. Per
atlas Exp 13's confirmed properties, the strategy should sit-out
completely on BTC and trade selectively on ETH. The validation Sharpe
numbers (+4.65 primary / +10.78 validation) came from sustained-bull
funding regimes (Q1-Q3 2024 in validation; 2025 H1 in primary). We
**cannot reproduce those numbers in current regime conditions** --
only the sit-out behavior is testable real-time.

**Constraint 3: The "live system" status is unknown to Claude.**
Memory references `funding_monitor.py` as a scheduled task that's
been running for cycles. But Claude doesn't know: is it paper-trading
(generating hypothetical fills)? signal-only (logging "would trade"
events)? Or just monitoring (logging funding rates without strategy
output)? The reconciliation cycle's scope depends on what the live
system actually does.

These three constraints mean Cycle 39 should investigate first, then
the user picks an implementation track. The candidate tracks are
sketched in L10-L11 below; the RECON cycle's job is to surface the
information needed to choose between them.

## Investigation tasks

### L1: Live system inventory

What's deployed and running. Investigate:

- Find `funding_monitor.py` in repo. What does it do? (Signal-only
  logging? Paper-trading? Both? Neither?)
- Does it run inside Windows Task Scheduler? If so, with what cadence?
  (Per `get-scheduledtask` output equivalent or by reading the
  scheduled_tasks docs/scripts.)
- Does it write to a database table or file? If a table, which one?
  If a file, where?
- Are there any other Engine-7-related scripts running?
  (`carry_monitor.py`, `funding_collector.py`, `funding_strategy.py`,
  etc. - grep the repo.)

Output: a clear "Engine 7 live deployment consists of [N] processes:
[list]. Each one [does X], writes to [Y], runs on [cadence Z]."

### L2: Live signal output schema

If L1 finds a paper-trading or signal-logging process:

- What's the output schema? (timestamp, asset, signal_strength,
  predicted_prob, would_enter, would_exit, etc.)
- How many records exist? (`SELECT COUNT(*)` on the relevant table
  or `wc -l` on the file.)
- Date range covered.
- Per-asset breakdown if applicable.

If L1 finds no paper-trading process exists, document explicitly:
"No live paper-trading process found. Engine 7 live deployment is
currently signal-monitoring-only / data-collection-only."

### L3: Live phase3 model artifacts

Does a trained model exist on disk that's currently being used for
live signal generation? Per atlas Exp 13's training: `2024, 7 assets x
36 configs`. Per memory: "Per project memory the main machine SSD was
lost; recovery underway." So:

- `find . -name "phase3_models*.joblib" -o -name "phase3_models*.pkl"`
  - Does ANY funding-rate phase3 artifact survive on disk?
- If yes: what's its training period, model count, AUC range?
  (Load it and inspect; this is a 5-minute Python script.)
- If no: confirm explicitly.

This determines whether "live system" means "trained model making
predictions" or "rule-based monitoring using funding-rate-percentile
heuristics."

### L4: Historical reconciliation feasibility -- input data

Atlas Exp 13's training and validation periods are 2024-01-01 to
2024-12-31 (training) and 2025-01-01 to 2026-03-26 (primary OOS).
Could the reconciliation be done against the OOS period using
current funding_rates data?

- Does funding_rates table cover 2025-01-01 onward for BTC + ETH?
  (We know it starts 2025-04-30, so partial coverage.)
- Are there gaps? (`SELECT date(datetime), COUNT(*) FROM funding_rates
  WHERE asset='BTC' GROUP BY date(datetime) HAVING COUNT(*) != 3
  ORDER BY date(datetime) LIMIT 50` -- checks for incomplete days.)
- What's the latest paper-validated cum return / Sharpe for the
  BTC + ETH subset of the OOS period? (Without re-running phase4
  on the full universe; if atlas only reports portfolio-level
  numbers, we can't isolate the BTC + ETH contribution easily.)

### L5: Historical reconciliation feasibility -- atlas paper trades

Atlas Exp 13 says "Individual models: BTC_FUNDING 77 days, Sharpe
+5.86, cum +3.4%. ETH_FUNDING 126 days, Sharpe +6.58, cum +5.0%"
for the primary OOS period (2025-01-01 to 2026-03-26).

To reconcile, we'd want to know on which specific days each model
fired (entered/exited) and what its hypothetical P&L was. Is that
data anywhere in the repo?

- `output/funding_rate/` -- exists?
- `output/funding_rate/cpo/phase4_portfolio.parquet` -- exists?
  (Probably not; project memory says phase4 outputs were deleted in
  Cycle 36 timeframe.)
- `output/funding_rate/cpo/phase3_models.joblib` -- covered by L3.
- Any per-trade log or signal-history file from a prior live or
  paper run?

If atlas's paper trades aren't reconstructable from disk, the
"historical reconciliation" track requires re-running phase2/3/4
against the BTC + ETH subset first -- which is a ~3-5h cycle, not
RECON.

### L6: Phase4 reproducibility cost estimate

If atlas's headline numbers (Sharpe +4.65 / +10.78) require re-running
phase4 to verify, what's the compute budget?

- Engine 7's `run_cpo.py` invocation path: what's the existing
  command-line shape? (Look at `scripts/run_cpo.py` or
  `scripts/run_funding_carry.py` or equivalent.)
- For the BTC + ETH subset on the training window 2024-01-01 to
  2024-12-31, how many configs? (Per atlas: "7 assets x 36 configs"
  full universe; BTC + ETH subset = 2 x 36 = 72 configs.)
- Phase2 estimated runtime for 72 configs on 365 days: from atlas's
  Exp 10 cycle 36c experience, ~1-2 minutes per config-day on the
  triple-barrier path; funding-rate-carry path should be faster
  (no triple barriers, just position lifecycle). Estimate 10-30 min
  phase2 total.
- Phase3 RF: per atlas, 6 models, 0.98-0.99 AUC, fast to train.
  ~5-15 min.
- Phase4 portfolio backtest: ~5-15 min.

Total estimated cost: ~30-60 min compute for BTC + ETH subset
reproduction. Cheap enough that "re-run + reconcile" could be one
cycle rather than a separate prerequisite.

### L7: Current-regime sit-out verification -- expected behavior

Per atlas Exp 13 confirmed properties:

- Gate P > 0.70 (current recommended); models sit out below this
- Bear-side funding (negative average, low positive_share):
  RF should output P < 0.70 systematically -- no trades
- Mild-bull funding (positive average, high positive_share):
  RF may output some P > 0.70 events -- selective trades

Given current regime (BTC bear, ETH mild bull), expected live system
behavior:
- BTC_FUNDING model: 0 entries in last 30 days; ~all P scores < 0.70
- ETH_FUNDING model: maybe 0-3 entries in last 10-15 days; some P scores
  > 0.70 if RF correctly identifies positive-funding windows

If the live system has signal output (per L1-L2), we can compare these
expected behaviors against actual log output. This is the
**bear-market validation by happenstance** opportunity.

### L8: Current-regime sit-out verification -- actual behavior

If L1 finds the live system has signal output, dump the last 30 days
of recorded signals and tabulate:

- Entries triggered (P > 0.70 in atlas calibration; or whatever
  threshold the live system uses)
- Exits triggered
- Per-asset breakdown
- Compare against the expected behavior in L7

If actuals match expected: this is real-time validation of the
"sit-out in bear regimes" claim. If actuals don't match: surface
the discrepancy explicitly -- could indicate a calibration drift,
data freshness issue, or a different model than atlas documents.

### L9: Bear-market validation context

Atlas Exp 13 explicitly flags "2022 bear validation (sustained
negative funding) not yet run. -0.03% max DD across two tested
periods strongly suggests the strategy flat-lines rather than loses
in bear markets, but formal confirmation pending."

Current April-May 2026 BTC funding is sustained negative (mean
-2.07% annualized, positive_share 0.348). This is a bear-side
funding regime, mechanically equivalent to the unverified 2022 case
even though the calendar date is different.

If L7 + L8 show clean sit-out behavior over the current window,
we can update the atlas with a real-time confirmation: "2026-04 to
2026-05 sustained-bear regime confirmed sit-out behavior; max DD
on BTC_FUNDING: [actual value, expected ~0]." This is a real-time
addition to atlas's confirmed-properties list, even without running
any experimental code.

### L10: Implementation track candidates

Based on L1-L9 findings, surface 2-4 concrete options for the
implementation cycle:

**Track A: BTC + ETH subset reproduction.** Re-run phase2/3/4 against
BTC + ETH on training window 2024 / OOS 2025-01-01 onward. Confirms
or refutes atlas's headline Sharpe figures for the subset that's
currently collectable. ~30-60 min compute + reconciliation analysis.
Outcome: confirmed-or-refuted Sharpe on covered universe.

**Track B: Live system audit only.** If L1 finds a paper-trading
process, verify it's writing signal output correctly, the
calibration matches atlas, and the sit-out behavior matches L7
expectations. Add an atlas note recording the 2026-04 to 2026-05
bear-regime sit-out as a real-time confirmation of the unverified
2022 bear validation hypothesis. No phase4 re-run. ~2-3h analysis
+ docs.

**Track C: Universe extension first.** Extend live collector from
BTC + ETH to BTC + ETH + SOL + XRP + ADA + AVAX. This is a
real engineering cycle (new CCXT calls, new schema/migration if
needed, new Task Scheduler entries). After deployment, reconciliation
can cover the full validated universe but waits for live data to
accumulate. ~3-5h engineering + 1-3 month wait.

**Track D: Combined A + B.** Run Track A's reproduction in parallel
with Track B's live audit. The reproduction confirms paper-numbers
on subset; the audit adds the bear-regime confirmation as a
real-time data point. ~4-6h total.

**Track E: Punt / pivot.** If L1 finds no paper-trading process and
phase3 artifacts are gone, "live-vs-paper reconciliation" doesn't
actually have a "paper" to reconcile against without first running
Track A. In that case Track A becomes prerequisite to anything
called "reconciliation"; the cycle reframes as "paper reproduction
+ optional live audit" rather than reconciliation.

### L11: Design recommendation

Per investigation findings L1-L9, recommend 1-2 tracks for the
implementation cycle, with rationale. Explicitly note any track
that's blocked by findings (e.g. if Track B requires a paper-trading
process that doesn't exist, mark it dependent on Track A).

## Output: structured findings report

Reply with a single structured report:

```
# Cycle 39 RECON findings -- Engine 7 reconciliation scoping

## L1-L3: Live deployment inventory
- Live processes: [list with cadences + I/O]
- Signal output schema: [table/file, columns, row count, date range]
- Phase3 artifact status: [exists @path / missing]

## L4-L6: Historical reconciliation feasibility
- funding_rates coverage gaps: [if any]
- Paper trades reconstructable from disk: [yes / no / partial]
- Phase4 reproducibility cost: [estimated compute time]

## L7-L9: Current-regime sit-out verification
- Expected behavior (BTC, ETH): [from atlas]
- Actual behavior (BTC, ETH): [from L8 query / "no signal output to query"]
- Match status: [aligned / discrepancy / N/A]
- Bear-regime confirmation opportunity: [yes / no, with rationale]

## L10-L11: Implementation track candidates + recommendation
[Per L10 template; 2-4 tracks with cost/outcome characterization]
[Recommendation: track + rationale]
```

The recommendation in L11 is the load-bearing output. L1-L9 are
the means; L11 is what feeds the next cycle.

## Pause point

After Code's report lands, Claude reviews + either approves an
implementation track (Cycle 40 brief) or asks for additional
investigation (Cycle 39b). The recon explicitly leaves room for
"Track E: re-scope" if findings surface that the original
reconciliation framing doesn't fit reality.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | All four investigation streams (L1-L3, L4-L6, L7-L9, L10-L11) report findings |
| 2 | L1-L3 surface concrete inventory of what's deployed (or explicit "nothing deployed" finding) |
| 3 | L4-L6 estimate compute cost for phase4 BTC + ETH reproduction (whether or not we do it this cycle) |
| 4 | L7-L9 either run the comparison (if live signals exist) or document explicitly why it's not possible |
| 5 | L10-L11 surface 2-4 concrete tracks with cost / outcome / dependency characterization |
| 6 | No code changes, no commits, no experimental runs |

## Commit at end of cycle

The investigation itself doesn't produce a commit (same pattern as
Cycle 38 RECON before Option D-prime). If findings surface that
warrant atlas documentation (e.g. real-time bear-regime confirmation),
hold the documentation for the next cycle's commit so the substantive
finding ships alongside the implementation work.

## Out of scope

- Running anything (no phase2/3/4, no data acquisition, no live
  system changes)
- Touching TRADING_ATLAS.md
- Other atlas entries
- Other commits
- Universe extension (Track C) is an option for next cycle, NOT
  for this cycle's investigation

## Notes for Code

- The four streams are largely independent. Run in parallel where
  possible.
- For L1, the canonical entry points are: project memory mentions
  `funding_monitor.py`; check `scripts/` and `engines/` directories;
  grep for any "funding" or "carry" reference in Task Scheduler
  exports.
- For L3, the phase3 joblib file may be on the lost SSD per memory
  notes about the 2026-04-24 disk failure. If it's not in the
  current backup tree, that's a key finding (Track A becomes the
  default since there's no live model to reconcile).
- For L8, if you find a signal log, just dump the last 30 days into
  a small markdown table. Don't build any complex visualization.
- For L11, the recommendation should be calibrated to findings, not
  a generic "I think Track X is best." Specifically: if L1 finds no
  live paper-trading, Track B is blocked and Track A becomes the
  default. If L1 finds a healthy live system, Track B is the
  cheapest informative cycle.
- Any data finding that contradicts the brief's framing (e.g.
  funding_rates has SOL/XRP/ADA/AVAX after all, or phase3 artifacts
  are intact) should be surfaced explicitly -- don't paper over the
  contradiction.

