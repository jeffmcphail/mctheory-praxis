# Brief: Mid-office 13 -- midoffice12's rulings, then the eigenvalue adjustment

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-12 (EST and UTC)
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-12T16:26Z (11:26 EST), the first clock reading after it; transcribed verbatim below, the
paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice12 (16f2fb2), checked in `git log` before any
file was touched; then come his rulings on midoffice12's six decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice13_eigenvalue_adjustment.md`
**Builds on:** 16f2fb2 (the horizon as a parameter, the bias statistics at any horizon, the new
defaults).

---

## The brief, verbatim

```text
committed -- midoffice12.

On its decisions:

1. The default horizon: one day. The new defaults' bias run must equal
   H1's to the bit, and the guard runs on it.
2. The h-day risk of broad portfolios: a run at the holding period's
   horizon, as a reporting view, and the report says which it is. Its
   gain rests on the mean reversion of one 2.7-year window with the
   April 2025 shock in it; it is re-tested as data accumulates.
3. Lags on the volatilities alone, the correlations one-day: measured
   below, as the reporting view's candidate.
4. The eigenvalue adjustment: next, on the one-day matrix (below).
5. The variance-ratio variant: an option, off.
6. The specific history: 63.

Next piece, midoffice13: the eigenvalue adjustment. The rulings first,
then:

1. The eigenfactor risk adjustment (Menchero, Wang and Orr; USE4), an
   option, off. On each forecast date: diagonalise the factor
   covariance; simulate M histories of factor returns from it,
   independent draws over the periods and weights the estimator used
   that day, the two half-lives included; re-estimate each with the
   same estimator and diagonalise it; for each eigenvalue rank, the
   bias is the root mean, over the draws, of the simulated eigenfactor's
   variance under the day's covariance over its simulated estimate;
   scale each eigenvalue by (a (bias - 1) + 1) squared. One simulation
   per date serves every a.
2. a: 1, the simulation alone, and two or three values above it,
   judged at one day by the eigenportfolios and the minimum-variance
   portfolios. a is tuned on the test that judges it, so choose it on
   the test's first half and report the second half's statistics as
   well.
3. What it should leave alone: the broad portfolios, the members, the
   industries and the random portfolios at one day. Report them.
4. M: enough draws that each rank's bias is stable to 0.01. Report the
   time per refit.
5. The reporting view's candidate, at h = 21: lags on the volatilities
   alone, the correlations one-day, two ways -- Newey-West at 20 lags on
   the volatilities' 90 days, and the variance-ratio variant's
   volatilities. Each against H21 and the one-day model, read at 21
   days and at one: the broad portfolios, the optimiser's, the smallest
   eigenportfolios.

Runs one at a time through the runner, as the state file records.
Change no default beyond ruling 1. Report, and say which you would
choose and why. Same discipline: retro with its git add and the state
file in the same commit, then stop.
```
