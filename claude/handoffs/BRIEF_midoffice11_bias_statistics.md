# Brief: Mid-office 11 -- midoffice10's rulings, then the out-of-sample bias statistics

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-10 (EST; 2026-11-11 UTC)
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-11T04:07Z (23:07 EST on 2026-11-10), the first clock reading after it; transcribed
verbatim below, the paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice10 (4f27f23), checked in `git log` before any
file was touched; then come his rulings on midoffice10's five decisions and one clean-up, then
the brief.
**Retro:** `claude/retros/RETRO_midoffice11_bias_statistics.md`
**Builds on:** 4f27f23 (PCG spared, spans ended off an exchange, the noise share measured).

---

## The brief, verbatim

```text
committed -- midoffice10. Your context was compacted while idle: read
claude/MIDOFFICE_STATE.md and the midoffice10 retro before starting.

On its decisions:

1. The correction: an option, off, as you chose. One more reason, for
   you to check. Under the regression's own assumption -- each name's
   residual variance inversely proportional to its weight -- the noise
   in the factor returns is the members' specific returns,
   (X'WX)^-1 X'W u, and the residuals lack exactly that part: a
   member's residual variance is its specific variance times
   (1 - its leverage). For a single member the two errors cancel. For
   an industry's own portfolio they do not: much of its specific risk
   is counted twice, once as the factor's noise. The correction as
   built puts the portfolio right and leaves each member short by its
   leverage, about an eighth of its specific variance in an 8-member
   industry. If the algebra holds, build the specific side's half as an
   option, off: each residual divided by sqrt(1 - h) for a member and by
   sqrt(1 + h) for a name outside the regression, whose residual
   carries the noise instead, h its leverage w x' (X'WX)^-1 x. The
   eigenvalue adjustment waits for the bias statistics.
2. The diagnostic: the weighted noise share always computed, reported
   beside the equal-weighted one, and the qualification quoting it.
3. The thin industries: no fold yet. Floor 15 by the Pearson rule is
   the candidate the bias statistics judge.
4. Size Nonlinearity: kept, as ruled.
5. FRCB's 2023-05-04 return: dropped, as built. It is an OTC price from
   after the index let it go, which is what ruling 2 removes.

And remove the two unused imports ruff has flagged since before
midoffice10.

Next piece, midoffice11: the out-of-sample bias statistics. The rulings
first, then:

1. Forecasts as of each date. The forecast made at t's close uses only
   data through t: the factor returns, the specific-risk model's fit
   and histories, the exposures. Refit daily if that is affordable,
   otherwise at each month-end with daily forecasts from the latest
   fit; say which. The industry scheme is held fixed over the sample,
   as Barra fixes a model's industry factors; say so. The first
   forecast once the covariance has a year of factor returns. Guard:
   the forecast as of t on the panel cut at t, the scheme held, equals
   the full run's, for a handful of dates.
2. The statistic. Each forecast, taken to a daily scale, against the
   next day's realized total return. The bias statistic is the standard
   deviation of their ratio: 1 if the forecasts are right, with a band
   of about 1 +/- sqrt(2/T). Over the whole test and over rolling
   12-month windows, with the windows' mean absolute deviation from 1.
   A 21-day horizon, non-overlapping, as a check.
3. The portfolios, each held from one forecast to the next:
   - every member singly: the distribution of their statistics, and
     the shares below and above the band;
   - the index cap-weighted and equally weighted, and the size pair;
   - each industry's members cap-weighted, all 20, their statistics
     set against their weighted noise shares;
   - random portfolios of members, seeded;
   - the minimum-variance portfolio the model builds as of each date,
     long-only if a solver is at hand, else unconstrained: the test of
     directions the model calls nearly riskless;
   - the factor covariance's eigenportfolios against the factor
     returns: the test that would call for the eigenvalue adjustment.
     These realize against estimated factor returns, which carry the
     noise, so they cannot judge the correction; real portfolios can.
4. The defaults first. If their statistics sit far from 1 across the
   board, find out why before going on. Then each candidate against the
   defaults, one change at a time:
   - floor 15 by the Pearson rule;
   - the correction as built, and with the specific side's half if
     built;
   - the specific-risk history at two or three half-lives against
     equal weights, its blend weight from the effective number of
     observations;
   - the factor covariance at a shorter and a longer half-life than 90,
     and the two-half-life estimator.

Change no default. Report, and say which you would choose and why.
Same discipline: retro with its git add and the state file in the same
commit, then stop.
```
