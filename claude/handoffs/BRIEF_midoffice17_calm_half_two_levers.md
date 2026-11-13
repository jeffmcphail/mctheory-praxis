# Brief: Mid-office 17 -- midoffice16's rulings, then the calm half's shape: two levers

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-13 (EST and UTC)
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-13T17:45Z (12:45 EST), the first clock reading after it; transcribed verbatim below, the
paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice16 (eb2b7f9), checked in `git log` before any
file was touched; then come his rulings on midoffice16's decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice17_calm_half_two_levers.md`
**Builds on:** eb2b7f9 (the regime adjustment on, the released panel, correlation half-lives
measured).

---

## The brief, verbatim

```text
committed -- midoffice16. If this is a new session, read
claude/MIDOFFICE_STATE.md and the midoffice16 retro before starting.

On its decisions:

1. The correlations stay on 504 days, as the rule chose. The trade is
   narrower than the halves make it look: at 63 days the broad
   portfolios improve in both halves (their MRAD 0.068 to 0.056 on the
   first, 0.159 to 0.113 on the second), and the whole cost is the
   optimiser's, in the small eigen-directions -- the eigenvalue
   adjustment's job, on a matrix with more for it to correct than the
   504-day one had.
2. Item 4's test: run it (item 1 below).
3. The estimator's diagonal: left as it is. Fold the exact diagonal
   into the first default change that moves every covariance anyway,
   so the stored runs' baseline breaks once.
4. The plain run's regime history from the full fit: build it, with a
   guard -- a few as-of refits spread over the history, each checked
   against the full fit's prefix to the bit, the run refusing on any
   difference -- so the out-of-sample property stays checked, not
   assumed. Report the plain run's time. The bias test keeps its
   refits.
5. Next: the calm half's shape once more (below), then the largest
   members' specific risk. This reverses midoffice15's order: the
   factor covariance is settled first, so the specific side is
   measured on the factor side it will keep.

Next piece, midoffice17: the calm half's shape, two levers. The rulings
first, then:

1. Item 4's test as you proposed it: the one-sided eigenvalue
   adjustment, a = 1, 2, 3, on the 63-day correlations, in one pass,
   its model test C16's 63-day setting to the bit. Judged by the rule
   against D16: both minimum-variance portfolios realising no more over
   the whole test than 10.13% and 10.19% a year, to the hundredth; then
   the criterion on the first half against 0.098, checked on the
   second. Beside it: ranks 21-25, the optimiser's share and statistic
   in ranks 16-25, the broad portfolios' MRAD by half, and rank 1 by
   half.
2. The volatilities' half-life: 21, 42 and 63 days against 90 (the
   default), the correlations on 504, on the regime-adjusted model, in
   one pass as C16 was. Each setting reads its own forecasts' regime
   history, which now differs from the model's in earnest. The same
   rule, against 90. Beside it, C16's tables, with each setting's
   regime multiplier by half. Midoffice11's 45 and 180 were single
   half-lives under ten Newey-West lags, so 90 has never been tested
   on this model.
3. No run: from D16's stored arrays and the model's factor returns,
   what the calm half's market direction is made of -- the realised
   factor correlations over the half against the forecast's (the mean
   correlation and the market direction's share, as section 4 of
   analyse_correlation has them for the forecasts), and each factor's
   own statistic by half, the market direction's heaviest factors
   first. Your reading is that what is left is on the volatility side;
   this says whether it is.
4. If both levers pass the rule, don't combine them this piece: say
   what the combination's test would be.

Runs one at a time through the runner. Change no default beyond ruling
4. Report, and say which you would choose and why. Same discipline:
retro with its git add and the state file in the same commit, then stop.
```
