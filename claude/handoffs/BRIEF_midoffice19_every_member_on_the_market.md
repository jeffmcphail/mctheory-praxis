# Brief: Mid-office 19 -- midoffice18's rulings, then every member on the market and the market's own multiplier

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-13 (EST and UTC)
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-13T21:32Z (16:32 EST), the first clock reading after it; transcribed verbatim below, the
paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice18 (caca97e), checked in `git log` before any
file was touched; then come his rulings on midoffice18's decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice19_every_member_on_the_market.md`
**Builds on:** caca97e (the country factor as a covariance basis, measured).

---

## The brief, verbatim

```text
committed -- midoffice18. If this is a new session, read
claude/MIDOFFICE_STATE.md and the midoffice18 retro before starting.

On its decisions:

1. The factor covariance: D16's, by the rule, for now.
2. The two-block pass: yes, after ruling 3's fix, with a higher bar.
   The design came from reading C18's halves, so choosing on the first
   and checking on the second guards it less well: it must beat the
   reference's criterion on both halves, after the optimiser's test.
   Its half-life is the regime adjustment's 21, untuned.
3. No member goes without the market. A member in no industry factor
   has no market exposure in this model, and the optimiser has used
   it: on 2026-09-30 both minimum-variance portfolios held 9.4% of
   CTVA (midoffice13's retro, diagnose_day.txt in its appendix), 2.4
   times their next-largest holding, though its forecast was 2.00% a
   day, as high as any of their top names -- what a minimum-variance
   optimiser does with a stock the model says has no market risk. Give
   CTVA, FRCB and SBNY each the industry its business makes it: by its
   SIC code where the data carry one, by an override with its evidence
   where they do not. If the scheme's floor is what left them out, a
   group below the floor joins its nearest industry instead of going
   without one. This becomes the default, the old scheme or panel
   kept and rebuildable.
4. The intermittent failure is named in your final_checks.txt:
   tests/test_battle_round3.py::TestSurfaceIntegration::
   test_surface_interpolated_query. Record it in the state file.
5. Next, after this piece: the largest members' specific risk -- or,
   if the two-block pass passes, its build first (your retro's item
   4, with the constraint's spurious direction removed and the exact
   diagonal in it), then specific risk.

Next piece, midoffice19: every member on the market, then the market's
own multiplier. The rulings first, then:

1. Before the fix, from D16's stored arrays: CTVA's weight in each
   minimum-variance portfolio by half, and its share of their realised
   variance. After it, the new defaults' bias run (D19) against D16:
   the same two readings, the minimum-variance portfolios' realised
   volatility and statistics by half, and wherever else anything
   moves.
2. The country basis in one pass beside D19's model, on two
   multipliers as two settings: C18's, one over the basis's 26
   factors; and the two blocks -- lambda_C from the country's own
   squared statistic, lambda_R from the other 25 factors', each from
   its own forecasts' history on 21 days and one until 21 days, the
   basis's covariance scaled by blocks before it is mapped back (the
   country's variance by lambda_C squared, the other 25 by lambda_R
   squared, their cross terms by lambda_C lambda_R). Judged by the
   rule against D19, with ruling 2's bar for the two blocks. Beside
   them: rank 1, the country factor and the index by half; lambda_C's
   path by half; how much the index's forecast moves from day to day;
   the optimiser by eigen-direction; the members by quintile.
3. The failing test: run it alone fifty times and say what it depends
   on -- an unseeded draw, the clock, memory. Fix it only if the cause
   is a missing seed, and say so; otherwise leave it named.

Runs one at a time through the runner. Change no default beyond ruling
3. Report, and say which you would choose and why. Same discipline:
retro with its git add and the state file in the same commit, then
stop.
```
