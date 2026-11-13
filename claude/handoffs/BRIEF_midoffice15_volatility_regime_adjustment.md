# Brief: Mid-office 15 -- midoffice14's rulings, then the volatility regime adjustment

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-12 (EST and UTC)
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-12T23:20Z (18:20 EST), the first clock reading after it; transcribed verbatim below, the
paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice14 (d2b2864), checked in `git log` before any
file was touched; then come his rulings on midoffice14's decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice15_volatility_regime_adjustment.md`
**Builds on:** d2b2864 (the spin-offs the total return does not carry, ruled; the one-day
correlations above one day; the minimum-variance portfolios' realised volatility standing).

---

## The brief, verbatim

```text
committed -- midoffice14.

On its decisions:

1. The five spin-offs from the public record: keep Jacobs' second
   Amentum distribution, BD's Waters shares, Aptiv's Versigent and
   Honeywell's Aerospace. Hold S&P Global's: a distribution worth about
   5% of SPGI's price, and no gap on 2026-07-01 -- SPGI opened 1.9% down
   and closed up, and the ruling turns its day from +1.0 sd to +3.8.
   Lennar's went ex on its record date; S&P Global's may have too.
   Find the gap in SPGI's raw prices around the record date,
   2026-06-15. If it is there, move the ruling to that day, valued at
   Mobility's first close as Lennar's is. If the data show no gap on
   either day, report what they show and leave the ruling out until it
   is understood.
2. Honeywell: the reverse split stands, and midoffice08's count ruling
   is superseded. That ruling read Honeywell's own post-split count as
   contamination; it was wrong.
3. FedEx Freight's count: grossed up by FedEx's retained 19.9%, to
   about 148.9 million, and checked against FDXF's weighted count.
   Every other count in the panel is the total outstanding, and the
   seeding rule meant that. Vylor, wholly distributed, stays as built.
4. Lennar: at Millrose's first close, as built.
5. The seven Polygon carries: left as they are.
6. Price returns: say which series each descriptor reads. If any reads
   the stitched prices, highs or lows across a ruled ex-date, apply
   the rulings to those series as price factors, as Polygon's own
   fractional records are applied; if none does, leave them.

Next piece, midoffice15: the volatility regime adjustment. The rulings
first, then:

1. USE4's volatility regime adjustment, an option, off, for the factor
   covariance and the specific risk separately. Each day's factor
   cross-sectional bias: the root mean square, over the factors, of
   each factor's return over its forecast volatility from the day
   before. Its multiplier: the square root of the exponentially
   weighted mean of that squared statistic, on a short half-life,
   scaling the factor covariance by its square. The specific side the
   same over the members, cap-weighted, scaling the specific variances.
   Out of sample throughout: a day's standardised returns use the
   forecasts made the day before, unadjusted. Say how long the history
   must be before the multiplier starts, and what it is before then. A
   test that nothing after a date moves the multiplier as of it, as for
   the eigenvalue adjustment.
2. Half-lives 21, 42 and 84 days on each side; one pass can serve
   every half-life, as one served every eigen scale. Chosen on the
   test's first half and checked on the second. Judged by the rolling
   12-month windows' mean absolute deviation from one, the statistics
   by half, the broad portfolios, the members, and the minimum-variance
   portfolios' statistics and realised volatility. Report the
   multipliers' path over the test and how much each setting moves the
   forecasts from day to day.
3. Then the eigenvalue adjustment one-sided on the regime adjustment you
   would choose: gamma = max(1, a (lambda - 1) + 1), a = 1, 2, 3, judged
   by the same test, the minimum-variance portfolios' realised
   volatility first.

Runs one at a time through the runner. Change no default beyond rulings
1 to 6. Report, and say which you would choose and why. Same discipline:
retro with its git add and the state file in the same commit, then stop.
```
