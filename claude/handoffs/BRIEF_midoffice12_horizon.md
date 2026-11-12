# Brief: Mid-office 12 -- midoffice11's rulings, then the horizon, measured

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-12 (EST and UTC)
**Source:** written in Chat and pasted into the Claude Code session before 2026-11-12T14:10Z
(09:10 EST), the first clock reading after it (the session's window was closed once between the
paste and that reading); transcribed verbatim below, the paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice11 (7c7350e), checked in `git log` before any
file was touched; then come his rulings on midoffice11's six decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice12_horizon.md`
**Builds on:** 7c7350e (the out-of-sample bias statistics, the leverage correction, the weighted
share).

---

## The brief, verbatim

```text
committed -- midoffice11. Your context was compacted while idle: read
claude/MIDOFFICE_STATE.md and the midoffice11 retro before starting.

On its decisions:

1. The specific-risk history: a 63-day half-life, the default.
2. The factor covariance: the two-half-life estimator, volatilities 90
   and correlations 504, both Newey-West, the default. With ruling 1
   that is the run H63T504; the new defaults' bias run must equal it.
3. The horizon: named, and a parameter. Newey-West with Bartlett
   weights at L lags is exactly the per-day variance of (L+1)-day
   returns, so 10 lags is an 11-day horizon -- not a month, and no
   standard one. The default stays 11 until this piece's measurements
   are in.
4. The eigenvalue adjustment: after the horizon is settled, not with
   it, since the lags that set the horizon are most of the bias it
   would correct.
5. The correction and its specific side: options, off.
6. No fold. Construction's split, and the mega caps' weight in their
   industries (TSLA's leverage 0.357), are a scheme question for later.

Next piece, midoffice12: the horizon, measured. The rulings first,
then:

1. --horizon h, in trading days: Newey-West at h - 1 lags on the
   volatilities and the correlations alike, --newey-west-lags still
   overriding it. The report states the horizon.
2. The bias statistics at the model's horizon: every day's forecast
   against the next h days' return, the windows overlapping, with the
   band for overlapping windows found by simulation, and the
   non-overlapping statistic beside it. At h = 1 it is today's daily
   statistic.
3. Candidates on the new defaults, each judged at its own horizon:
   h = 1, 5 and 21 against 11. For each, what its lags cost: the
   smallest eigenportfolios and the minimum-variance portfolios, at its
   horizon and at one day.
4. One variant against the lags' noise, at h = 21: each volatility its
   90-day daily volatility times the square root of its variance ratio
   at h, the ratio estimated on 504 days, so the serial correlation
   comes from the long window and the level from the short one.
5. The specific history at a 42-day half-life, to see whether the
   trend continues. 63 stays unless 42 is clearly better.

Runs one at a time through the runner, as the state file records.
Change no default beyond rulings 1 and 2. Report, and say which horizon
you would choose for which use, and why. Same discipline: retro with
its git add and the state file in the same commit, then stop.
```
