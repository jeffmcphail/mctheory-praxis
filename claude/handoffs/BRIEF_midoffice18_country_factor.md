# Brief: Mid-office 18 -- midoffice17's rulings, then the country factor

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-13 (EST and UTC)
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-13T19:41Z (14:41 EST), the first clock reading after it; transcribed verbatim below, the
paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice17 (c80a112), checked in `git log` before any
file was touched; then come his rulings on midoffice17's decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice18_country_factor.md`
**Builds on:** c80a112 (the plain run's regime history from the full fit; the calm half's two
levers measured).

---

## The brief, verbatim

```text
committed -- midoffice17. If this is a new session, read
claude/MIDOFFICE_STATE.md and the midoffice17 retro before starting.

On its decisions:

1. The factor covariance as it stands: correlations 504, volatilities
   90, the factor side on 21, no eigenvalue adjustment. The eigenvalue
   adjustment is closed: three tests have fixed the statistics it
   targets and not the optimiser's realised risk.
2. The calm half's market direction: neither left nor given a
   multiplier of its own. Test USE4's country factor first (below).
   Item 3 found the market direction's variance carried by the
   industries' correlations on the slow window; a country factor
   carries it on the volatilities' window instead.
3. Next: the country factor, then the largest members' specific risk.
   If the country factor fails the rule, the factor side is settled
   and specific risk is the piece after it.

Next piece, midoffice18: the country factor. The rulings first, then:

1. The country factor as a covariance basis. With every member exposed
   to it and the cap-weighted industry factor returns constrained to sum
   to zero, as USE4 has it, the regression is the same fit -- the same
   fitted values and specific returns -- and its factor returns are a
   change of basis: the country's return is the cap-weighted mean of the
   industries' returns, each industry's return becomes its old return
   less the country's, and the styles are unchanged. What changes is
   the covariance, because the two-half-life estimator is not
   basis-invariant: in this basis the market's variance rides on the
   volatilities' 90 days, and the industries' correlations, net of the
   market, stay on 504. Check it on a few dates before the pass: a fit
   with the country factor and the constraint gives the transformed
   factor returns and the same residuals, to rounding. USE4's own
   account, section 3.1:
   https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf
2. One pass beside the model, as C16 and V17 were: each refit's factor
   returns taken to the country basis, the covariance estimated there
   (Newey-West as the horizon wants it) and mapped back to the model's
   factors, so the rest of the test runs unchanged. Its regime history
   from its own forecasts, B_F over the basis's 26 factors, the country
   among them. Judged by the same rule against D16, with C16's tables
   and item 3's readings beside it: the forecast correlations between
   the industries by quarter against the realised, and rank 1 by half.
3. The expectation, for the test to check: the index's forecast becomes
   nearly the country factor's own variance, so most of its calm-half
   over-forecast should go, and some of its first-half under-forecast
   with it. What the optimiser does is the test.
4. If it passes, say what building it into the model would take -- the
   country factor in the reports, the plain run, the defaults -- and
   don't build it.

Runs one at a time through the runner. Change no default. Report, and
say which you would choose and why. Same discipline: retro with its
git add and the state file in the same commit, then stop.
```
