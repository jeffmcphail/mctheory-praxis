# Brief: Mid-office 03 -- the forecast agrees with the estimation, and two definitions measured

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-06
**Source:** written in Chat and pasted into the Claude Code session; transcribed verbatim below.
Its first line is Jeff's "committed" for midoffice02 (956331e) and the Rule 43 correction
(6e29b2c), both checked in `git log` before any file was touched.
**Retro:** `claude/retros/RETRO_midoffice03_forecast_agrees_with_estimation.md`
**Builds on:** 956331e (each regression fixed the day before its return; an empty window gives
no value), 6e29b2c (Rule 43 v1.6: mid-office briefs are written in Chat and pasted in).

---

## The brief, verbatim

```text
committed -- midoffice02, then the Rule 43 correction on its own.

Next piece, midoffice03: make the forecast agree with the estimation,
and measure two descriptor definitions before anyone chooses them. Same
discipline as before: in this order, each change decomposed against the
one before it, retro and git add line at the end, then stop.

1. Coverage: set min_coverage's default to 0.5. A twelve-month
   descriptor computed from under six months of its window is not that
   descriptor, and the measured cost is R-squared +0.000014 and two
   phantom identities, ECHO and VMRK, that the rebuild will fix. 0
   stays available as a value.

2. SERDP: drop it from the forecast blend, so every forecast exposure
   is built from the descriptors the last estimated period's exposures
   were built from. Turn the run's flag into a guard that stops the
   run: no descriptor may contribute to a forecast exposure if it
   contributed to no estimated period's. Computing SERDP historically
   waits for a reason to; it has never had a usable period in the
   screen. Add a standing line giving each risk index's blend
   composition over the estimated periods, with the date each
   descriptor enters, so the 2025-04-23 change is visible in every run.

3. fit_index_weights: lag it now -- descriptor on t, return on t+1 --
   though the run does not call it, so it cannot bring back what the lag
   removed. Leave it whole-sample, say so in its docstring, and keep
   test_lookahead's third test as the record of it.

4. RSTR: add a skip parameter, the trading days excluded from the end
   of its window, default 0, which is today's definition. Report
   Momentum's vol, mean and t at 0 and at 21, USE4's skip. Do not change
   the default.

5. The market model: BTSG, HALPHA and VLVR first reach the estimated
   exposures on 2025-04-23, so for 552 of 938 periods Volatility has no
   beta, the descriptor the screen ranks first. Report the market
   model's window and half-life, and rerun the screen and the model
   with the market model on 252 trading days and half-life 63, as USE4
   estimates beta: when each of the three first appears, the screen,
   R-squared and the style vols. Do not change the default.

6. As a separate one-line change with its own git add: correct the
   Source lines in midoffice01's brief and retro to say the brief was
   written in Chat and pasted in.

Not now, but record it for the rebuild: share counts must change on a
split's execution date, using the split events rather than the price
adjustment factor -- DD's 2.39 and 0.33 suggest that factor also
carries actions that are not splits. Check each applied split against
the next share sample, flag any that disagree, and guard on zero
member-days pairing a post-split price with a pre-split count.
```
