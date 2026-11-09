# Brief: Mid-office 04 -- settle the definitions midoffice03 measured

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-09
**Source:** written in Chat and pasted into the Claude Code session; transcribed verbatim below.
Its first line is Jeff's "committed" for the midoffice01 provenance fix (08087bf) and midoffice03
(992f11a), both checked in `git log` before any file was touched.
**Retro:** `claude/retros/RETRO_midoffice04_settle_measured_definitions.md`
**Builds on:** 992f11a (every forecast exposure built the way its factor was estimated; RSTR's
skip and USE4's beta measured, defaults unchanged), 08087bf (midoffice01's brief provenance).

---

## The brief, verbatim

```text
committed -- the midoffice01 provenance fix, then midoffice03.

Next piece, midoffice04: settle the definitions midoffice03 measured.
Same discipline: in this order, each change decomposed against the
one before it, retro and git add line at the end, then stop.

The policy behind items 1 to 3: USE3's definitions by default, and
USE4's wherever USE3's choice produces a measured problem on this
panel, each kept switchable.

1. The market model: default to 252 trading days with half-life 63,
   USE4's beta, in compute_descriptors itself and in the run, so the
   run no longer overrides the library. 828 equal-weighted days stays
   runnable. Today's choice keeps beta, the screen's strongest
   descriptor, out of 552 of 938 periods; USE4's puts it in all of
   them.

2. One definition per factor across the sample: a period is estimated
   only if every risk index's blend in it has exactly the descriptors
   the forecast's has. Extend midoffice03's guard from the last
   estimated period to every estimated period, as an assertion. Under
   item 1 this should start the estimation on 2023-02-08, the same 938
   periods, and drop the 24 earlier ones in which Momentum is HALPHA
   alone. Report what it drops.

3. The size pair: build Size Nonlinearity USE4's way, as a parameter --
   size_nonlinearity="use4" by default, "use3" keeping today's raw cube
   runnable. Cube the Size exposure; treat the cube like any descriptor
   except for the raw MAD clip, which on a cube binds at |z| about 1.2
   -- so standardise it and clip at +/-4; then orthogonalise it against
   Size, per period, with the regression weights, over the estimation
   universe; then the final exact standardisation.
   Guard: on every period, the regression-weighted covariance of the
   Size and Size Nonlinearity exposures over the estimation universe is
   zero to machine precision.
   Report against the raw cube: the two exposures' correlation (0.93
   now), their factor returns' correlation (-0.87), both vols and
   means, the pair's risk for the smallest 85 members, the largest 50
   and the index, R-squared, how many members the clip moves, and
   Size's factor returns with and without Size Nonlinearity in the
   model. Under USE4 those two series should be close; under USE3 they
   are not.

4. The residual spread in EQ 3-14's V fit: make its scale a parameter,
   defaulting to a MAD-based one (1.4826 x the MAD), with the standard
   deviation runnable. FRC is 71% of the squared spread today, and
   moving the sample start by 24 days moved the spread from 0.368 to
   0.279. Report the spread on the 938- and 962-period samples under
   both scales -- the MAD one should barely move -- and what it does
   to the blend weights and the delivered forecasts.

5. RSTR's skip stays 0: at 21 it changed nothing material and cost a
   little R-squared. Say in the parameter's docstring that it cuts the
   window's last k days, the academic twelve-minus-one reading, and
   that USE4's lag is a full window ending k days back, measured in
   midoffice03 and not built.
```
