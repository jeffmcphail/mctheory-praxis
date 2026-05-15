# Cycle 38 RECON -- Chan/Burgess pairs trading code archaeology + feature/info-bar design memo

**Predecessor:** Cycle 37 (commits `f732c3e` + `ebfb438`). Atlas Exp 1
is the canonical entry for this thread; result_class=NEGATIVE per
the 2026-04-02 rebuild documented in atlas.

**Mode:** Read-only investigation. No code changes, no commits, no
experimental runs. Output is a structured findings report that
becomes the input to a follow-up implementation brief.

**Risk:** very low. Cycle scope is documentation + code reading +
arithmetic on existing data.

**Scope cap:** ~2 hours Code time. The investigation has four
streams that should be roughly parallel:
- I1-I3 code archaeology (~30 min)
- I4-I5 Chan paper spec (~30 min)
- I6-I8 info bars feasibility (~30 min)
- I9-I11 TC arithmetic stress test (~30 min)

## Why this cycle exists

Atlas Exp 1 documents a 2026-04-02 minute-bar rebuild of the
Chan/Burgess pairs trading strategy:

- 112 minute features (7 indicators x 16 lookback combinations)
- AUC 0.855-0.896 (improved from daily-bar 0.77-0.87)
- Gross win rate 57.6% (real conditional lift from ~28% base rate)
- Gross alpha 5.3 bps/model-day
- TC 11 bps/model-day (2.7 trades x 4 bps RT)
- Net -5.6 bps/model-day, Sharpe -3.18
- Verdict: NEGATIVE after TC

Project memory describes a planned rebuild with 8 lookback windows
(30/60/90/120/180/240/360/720 min) but doesn't explain how that
differs from the 16-lookback Chan spec the atlas already documents.
Cycle 37's audit marked the 2026-04-02 result suspect-LOW (artifacts
deleted in a2202a7) but the result is internally consistent and
the atlas's analysis (gross alpha < TC) is mechanical.

The user's working theory for revival is info bars / dollar bars
(unlike Exp 10 where Sharpe-invariance refuted the cap-revival
hypothesis, info bars genuinely change which bars are sampled, so
they can change both gross alpha and trade count -- and thus the
gross-vs-TC arithmetic).

This cycle's purpose: lay the design groundwork before any code or
data work. We need to know:

1. What feature set was actually used in the 2026-04-02 run, and
   does it match Chan's paper? (Code archaeology)
2. What does Chan's paper actually specify? Is the 8-lookback set
   in project memory the same as Chan's spec, a subset, or
   different? (Paper review)
3. Are info bars feasible for SP500 equities given current data,
   and what trade-count reduction is plausible at various dollar
   thresholds? (Feasibility)
4. What's the actual trade-count reduction needed to flip net
   positive at the existing gross alpha? (TC arithmetic)

The output of this cycle is a design memo that synthesizes all
four streams into a concrete proposal for the next cycle.

## Investigation tasks

### I1: Code archaeology -- what survives in current tree

`engines/chan_cpo.py` was deleted in commit a2202a7 per Cycle 37's
S1 inventory. `scripts/run_chan_cpo.py` was modified in the same
commit. Investigate:

- Does `engines/chan_cpo.py` exist in current HEAD? Confirm absent.
- What does `scripts/run_chan_cpo.py` currently call into for
  feature generation? (Grep for imports.)
- Is there a successor module (`engines/pairs_cpo.py`, or merged
  into `engines/cpo_core.py`, or...)? Trace the import chain.
- If chan_cpo logic was inlined or merged into another module,
  identify where the feature-generation function lives now.

Report: a clear "the current chan_cpo code path goes:
scripts/run_chan_cpo.py -> [module] -> [function]" trace, or
"chan_cpo logic is no longer in the tree" if it was removed
without successor.

### I2: Code archaeology -- recover the 2026-04-02 feature set

The atlas result was produced by code that's not in HEAD. To
understand what 16 lookbacks were used, check the parent commit
of a2202a7:

```powershell
# Read the deleted engines/chan_cpo.py from a2202a7's parent
git show a2202a7^:engines/chan_cpo.py > outputs/chan_cpo_pre_a2202a7.py
# Or trace through earlier commits if a2202a7 wasn't the introducing commit
git log --all --oneline -- engines/chan_cpo.py | head -10
git log --all --oneline -- scripts/run_chan_cpo.py | head -10
```

Find the feature-generation function (probably named something
like `compute_pair_features` or `generate_features`). Report:

- File + line range where features are computed
- The list of indicators (the "7 indicators" from atlas)
- The list of lookback windows (the "16 lookbacks" from atlas)
- Any specific filter or aggregation logic

If chan_cpo went through several iterations, capture the version
that was current at 2026-04-02 (i.e. the commit immediately before
or contemporary with the atlas's "Corrected results 2026-04-02"
claim).

### I3: Code archaeology -- understand the trade-count source

The atlas claims "2.7 trades/day" average. Find where that comes
from in the code:

- Trading logic (entry/exit thresholds, hold period, etc.)
- Whether trade count is a derived statistic from a backtest or
  a configured parameter
- Whether changing bar type (minute -> dollar bars) would mechanically
  change the trade count, or whether other logic (e.g. fixed daily
  cycles) caps it independent of bar type

This is critical for the TC arithmetic in I11.

### I4: Chan paper review -- canonical lookback / indicator spec

Read the Chan paper(s) in project root. Two PDFs:
- `Conditional_Portfolio_Optimization_Using_Machine__Learning_to_Adapt_Capital_Allocations_to_Market_Regimes_Ernest_Chan.pdf`
- `CPOPaper__Chan.pdf`

(One may be a draft / preprint of the other; if so, read both and
note differences.)

Extract Chan's specification for:
- Indicator list (the "7 indicators" or whatever Chan actually lists)
- Lookback windows (the "16 lookbacks" or Chan's actual set)
- Feature construction methodology (raw values? z-scores? something
  derived?)

Report Chan's canonical spec in a clean enumerated form.

### I5: Spec comparison

Cross-reference three feature designs:

A. Chan paper canonical spec (from I4)
B. 2026-04-02 actually-implemented spec (from I2)
C. Project memory 8-lookback set: `30, 60, 90, 120, 180, 240, 360, 720` min

For each, capture:
- Are the lookbacks in min, hours, days?
- Is C a subset of A, a different overlapping set, or genuinely
  different?
- Is B a faithful implementation of A, or does it deviate?
- What's the mechanical implication of using one set vs another?
  (E.g. C's max lookback is 720 min = 12 hours; A may go out to
  multi-day windows. The choice shapes whether the strategy can
  capture cross-day mean-reversion patterns.)

### I6: Info bars feasibility -- data availability

Per atlas, Exp 1 used Polygon.io minute bars for 38 SP500 pairs.
Investigate:

- Does the praxis repo have any SP500 minute-bar data still
  on disk? (Probably yes if the run was done locally.)
- Where is it stored? (Look for `data/polygon_minute_bars/`,
  `data/sp500/`, etc.)
- What does the data schema look like? (timestamp, open, high,
  low, close, volume -- and is volume populated?)
- If data is absent, what would re-acquiring cost? (Polygon.io
  has a paid tier; the user may have credentials.)

Report: data state, schema (if present), and a rough estimate
of what re-acquisition would entail if needed.

### I7: Info bars feasibility -- bar construction approach

If minute-bar data is available, what's the implementation cost
of constructing dollar bars from it?

The standard approach (per Lopez de Prado, "Advances in Financial
Machine Learning"):
- Compute dollar_volume per minute bar = volume * close (or VWAP if
  available)
- Accumulate cumulative_dollar_volume forward in time
- Bar boundaries fall when cumulative_dollar_volume crosses fixed
  threshold (e.g. every $10M of traded volume)
- Each bar has OHLC over its dollar-volume window + total volume

Estimate:
- Lines of code: this is a 30-50 line function in pandas/polars
- Per-pair compute cost: small (one pass over a year of minutes)
- Storage: similar to or smaller than minute bars

Report: a rough "X lines, Y minutes per pair" estimate.

### I8: Info bars feasibility -- threshold tuning

The dollar threshold is the key parameter. Too low = same as minute
bars (no compression); too high = too few bars to support intraday
trading (one bar per day defeats the point).

A useful target: a threshold that gives ~10-30 bars per pair per
day during normal hours (vs ~390 minute bars per day for 6.5
trading hours).

To estimate: pick a representative SP500 pair (e.g. AAPL/MSFT or
similar), look at typical daily dollar volume, divide by target
bar count, get threshold.

Example arithmetic: AAPL trades ~$10B/day. To get ~20 bars/day,
threshold = $500M per bar. That's per-side; for a pair, total
dollar volume depends on both legs.

Report: a rough threshold estimate for SP500 pairs, with arithmetic
shown.

### I9: TC arithmetic -- pin down the atlas's 5.3 bps / 11 bps / -5.6 bps

These three numbers anchor the atlas's "TC is the bottleneck"
conclusion. Verify their provenance:

- Where in the 2026-04-02 code/output do these come from?
- Is "5.3 bps gross" a per-model-day average over OOS days?
- Is "11 bps TC" derived as "2.7 trades * 4 bps RT"? (4 bps RT =
  2 bps/leg per the atlas's "TC: 4 bps round-trip per pair" line.)
- Are these numbers reproducible from any surviving artifact, or
  do we have to trust them based on the suspect-LOW Cycle 37
  classification?

If reproducible: confirm them.
If not reproducible: flag explicitly.

### I10: TC arithmetic -- break-even trade count

At 4 bps RT per trade, what trade count would flip net positive
holding gross alpha at the atlas's claimed 5.3 bps/day?

Simple: break-even when 4 * trades_per_day <= 5.3, so
trades_per_day <= 1.325. Currently 2.7. Halving trade count would
exactly break even at the gross alpha number; getting to 1
trade/day at 5.3 bps gross would produce 1.3 bps net per day,
small but positive.

Report: a quick arithmetic table showing net P&L at
trades/day in [2.7, 2.0, 1.5, 1.0, 0.5] with gross_alpha held at
5.3 bps, plus columns for "what if gross alpha rises to 8 bps?"
and "what if 4 bps RT becomes 2 bps RT?" sensitivity.

### I11: TC arithmetic -- info-bar trade-count plausibility

The key empirical question: would dollar bars at the I8 threshold
actually halve trade count?

Two things govern this:
- Bar count: dollar bars at threshold X produce N bars/day vs
  390 minute bars; ratio is roughly N/390.
- Signal-firing rate: signals fire on bar events. If the gate
  fires at z-score threshold and the spread crosses threshold at
  the same rate regardless of sampling, trade count drops
  proportionally to bar count.

But: dollar bars sample at *event-times*, not clock-times, so
signal-firing can concentrate around active periods (open, close,
news events) where the gross alpha may be higher. This is the
mechanism the user is intuiting -- info bars don't just reduce
trades, they reduce *low-quality* trades preferentially.

This is hard to verify without running. But we can estimate:
- If trade-count drops with bar count (proportional), the
  arithmetic in I10 dominates: any plausible threshold halves
  trade count.
- If trade-count drops AND gross-alpha-per-trade rises (event-
  concentration), both numerator and denominator move favorably.

Report: a qualitative breakdown of the two mechanisms and a
reasoned guess at expected trade-count reduction range at a
"reasonable" threshold from I8. Don't run anything; just reason.

## Output: structured findings report

Reply with a single structured report:

```
# Cycle 38 RECON findings -- Chan/Burgess pairs trading

## I1-I3: Code archaeology
- Current chan_cpo path: [trace]
- 2026-04-02 feature set:
  - Indicators: [list of 7]
  - Lookbacks: [list of 16]
  - Code path: [file:line]
- Trade-count source: [where 2.7 trades/day comes from]

## I4-I5: Chan paper spec comparison
- Chan canonical spec:
  - Indicators: [list]
  - Lookbacks: [list]
- 2026-04-02 implementation vs Chan canonical:
  - Match / Deviation / Reason
- Project memory 8-lookback set vs Chan + actual:
  - Same as Chan? Same as actual? Different?
  - Mechanical implication

## I6-I8: Info bars feasibility
- Data availability: [yes / no / partial]
- Bar construction cost: [lines, minutes]
- Threshold tuning: [estimate, arithmetic]

## I9-I11: TC arithmetic
- Atlas numbers provenance: [reproducible / trust-only / refuted]
- Break-even table: [trades/day -> net bps]
- Info-bar trade-count plausibility: [qualitative]

## Design recommendation (the synthesis)
Given findings I1-I11, the next cycle should:
- [Option A: re-verify 2026-04-02 result as standalone reproduction cycle]
- [Option B: dollar-bar rebuild with threshold X, expected trade-count reduction Y]
- [Option C: something else surfaced by the investigation]
- [Option D: punt entirely; atlas's existing analysis already gives the answer]

For each option, capture:
- What it tests
- Expected compute budget
- What outcome would change the atlas's NEGATIVE verdict
```

The design recommendation section is the load-bearing output. The
investigation tasks are means; the recommendation is what feeds
the next cycle.

## Pause point

After Code's report lands, Claude reviews and either:
- Approves a specific recommendation -> Cycle 39 implementation brief
- Asks for additional investigation -> Cycle 38b
- Decides the atlas's existing answer is sufficient -> close thread

The brief explicitly leaves room for "Option D: the atlas's existing
analysis is sufficient." If I9-I11 confirm the gross-vs-TC arithmetic
is rock-solid and I6-I8 don't surface a credible path to material
trade-count reduction, the right move may be to accept the atlas's
NEGATIVE verdict and pick a different research direction.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | All four investigation streams (I1-I3, I4-I5, I6-I8, I9-I11) report findings |
| 2 | Code archaeology surfaces a concrete enumeration of the 2026-04-02 feature set (indicators + lookbacks) |
| 3 | Chan paper spec is captured in enumerated form for direct comparison |
| 4 | Info bars feasibility includes a concrete threshold estimate with arithmetic shown |
| 5 | TC arithmetic includes a break-even trade-count table |
| 6 | Design recommendation surfaces 2-4 concrete options with compute/outcome characterizations |
| 7 | No code changes, no commits, no experimental runs |

## Commit at end of cycle

The investigation itself doesn't produce a commit. Findings live in
the conversation. If you want a persistent record:

- Brief + findings could be committed together as a small read-only
  recon artifact (similar to how Cycle 36 RECON's findings fed
  Cycles 36a/b/c without a recon-specific commit).
- My preference: skip the commit until we have a Cycle 39 to commit
  alongside it.

## Out of scope

- Running anything (no data acquisition, no feature generation, no
  backtests)
- Restoring deleted code (engines/chan_cpo.py recovery is
  read-only via `git show` only)
- Touching TRADING_ATLAS.md
- Other atlas entries
- Other commits

## Notes for Code

- The four streams are roughly independent. Run in parallel where
  possible.
- For I2 (recovering deleted code), use
  `git show a2202a7^:engines/chan_cpo.py` as the starting point
  if a2202a7 is where deletion happened; trace back further if
  earlier commits had different versions.
- For I4 (Chan paper review), the PDFs are large. Skim for the
  feature-spec sections; don't try to read the whole paper.
  Section titles like "Features", "Indicators", "Variables", "Inputs"
  are the targets.
- For I8 (threshold tuning), if minute-bar data isn't available,
  estimate from public sources (e.g. SP500 average daily volume
  data). Don't run anything; reason from public knowledge.
- For I11 (info-bar trade-count plausibility), be careful not to
  overclaim. The two-mechanism argument (bar count down + per-bar
  quality up) is intuition, not measurement. The recommendation
  should be calibrated to that uncertainty.
- If at any point a finding contradicts the brief's premise (e.g.
  Chan paper doesn't describe 16 lookbacks, or the 2026-04-02 code
  used some other feature set entirely), surface it explicitly --
  don't paper over the contradiction.
