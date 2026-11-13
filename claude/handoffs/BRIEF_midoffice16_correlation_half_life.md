# Brief: Mid-office 16 -- midoffice15's rulings, then the calm half's shape: the correlation half-life on the regime-adjusted model

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-13 (EST and UTC)
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-13T15:56Z (10:56 EST), the first clock reading after it; transcribed verbatim below, the
paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice15 (355b772), checked in `git log` before any
file was touched; then come his rulings on midoffice15's decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice16_correlation_half_life.md`
**Builds on:** 355b772 (the volatility regime adjustment measured; held rulings; the revised
panel).

---

## The brief, verbatim

```text
committed -- midoffice15. Your context was compacted while idle: read
claude/MIDOFFICE_STATE.md and the midoffice15 retro before starting.

On its decisions:

1. The regime adjustment's factor side: on, at 21 days, the model's
   default. The criterion fixed before the run chose it on the first
   half (0.098, against 42 days' 0.111 and the model's 0.172), and it
   is the best on the second (0.123, against 0.131 and 0.141). Its
   price is noise: the index's forecast moves 1.00% a day at the
   median against 0.27%, still less than a member's moves under the
   model (1.34%).
   The whole test's level is not a cost of it. Each portfolio's
   whole-test statistic is, to rounding, the root mean square of its
   two halves': the index's 1.015 of 1.224 and 0.751, its 0.934 of
   1.063 and 0.785, eigenportfolio 1's 1.027 of 1.226 and 0.780. The
   model's in-band 1.015 was a first-half under-forecast cancelling a
   second-half over-forecast; the adjustment removes the first, and
   the 0.934 is the second, the calm half's shape. So open problem 3
   is problem 2. Confirm the identity for every family at one day, and
   say where it fails and why. Put it in docs/FACTOR_MODEL.md beside
   the bias statistics: a whole-test band can pass a model that is
   wrong in both halves, which is why the rolling windows are the
   criterion.
2. Its specific side: off, against the criterion's joint pick of 42
   days, which differs from off by 0.004 on the first half and
   reverses on the second. It is a level, 1.09 in both halves, set by
   the largest members; that fix is by size, in the specific risk
   model.
3. The eigenvalue adjustment, one-sided as two-sided: off. The
   optimiser realises more at every a.
4. S&P Global: the ruling stands; release the hold. Polygon's 407.26
   is the regular-way close of 2026-06-30: the ex-distribution line,
   SPGI WI, closed at 384.00 that day, and 407.26 less 384.00 is
   23.26, one Mobility share at about its when-issued price. A
   regular-way close near 428 would need Mobility near 44; it opened
   at 20.30. The gap was there, and 07-01's rally filled it: SPGI
   closed at 414.97, 8.07% over its when-issued close, its peers'
   median +4.27%. The press read the day the same way. Motley Fool:
   "Why S&P Global Stock Zoomed Almost 8% Higher Today"; Mobility's
   shares "sank by almost 4% on their first day of trading". Kalkine:
   "SPGI shares rose 5.23% today to $436.69, building on Wednesday's
   7.73% gain". 436.69 / 1.0523 is 414.99, Polygon's close; 414.99 /
   1.0773 is 385.21, which is 407.26 less 22.05 of Mobility, and
   Mobility's first close, 21.19, is 3.9% under that. So _spinoffs'
   +7.0962% was right, and SPGI's +3.8 sd that day is the market's.
   Write this into the ruling's evidence with the sources below, as
   quoted from Chat's search results of 2026-11-13 (the pages were not
   opened), and check the arithmetic. Keep the held mechanism. The
   panel with every ruling applied, under a new name, the old ones
   kept, becomes the run's default: against _revised only SPGI's
   2026-07-01 return and its index's rescaling should change, against
   _spinoffs only FDXF's counts.
   https://us.trendlyne.com/us/equity/Split/SPGI.WI/3563800/s-p-global-inc-when-issued-june-2026-split/
   https://www.dividend.com/stocks/communications/media/publishing-broadcasting/spgi_wi-sandp-global-inc-when-issued-june-2026/
   https://nasdaq.com/articles/why-sp-global-stock-zoomed-almost-8-higher-today
   https://kalkine.com/news/financial/sp-global-nysespgi-surges-5-today-following-completion-of-mobility-global-spinoff
5. Next: the correlation half-life on the regime-adjusted model,
   below; then the largest members' specific risk.

Next piece, midoffice16: the calm half's shape. The rulings first,
then:

1. The new defaults. Their bias run on _revised must reproduce V15's
   F 21, S off column, to the bit or to rounding with the reason; then
   on the new panel (D16), where only SPGI's 2026-07-01 should move
   anything. Say what the factor side costs a plain run, in minutes.
2. Correlation half-lives of 63, 126, 252 and 504 days (the default),
   the volatilities' 90 kept, on the regime-adjusted model. The factor
   volatilities do not depend on the correlations, so the regime
   multiplier should be the model's under every setting: confirm it to
   the bit. One pass for all four if the factor returns can be shared,
   as the horizons and the eigen scales were.
3. Judged as VE15 was, the rule in the code before the run: the
   minimum-variance portfolios' realised volatility first -- a setting
   under which either realises more over the whole test than at 504,
   to the hundredth of a point, is out -- then REGIME_CRITERION, chosen
   on the first half and checked on the second. Beside them, the shape
   this piece is for: the eigenportfolios by rank and half, as
   eigen_by_half.txt has them, rank 1 first. Then the index and the
   optimiser by half, the members by capitalisation quintile, and how
   much each setting moves the index's forecast from day to day.
4. If a shorter half-life fixes the shape but the optimiser pays for it
   in the small eigen-directions, say so and propose the test that
   would tell, the one-sided eigen adjustment on it; don't run it.

Runs one at a time through the runner. Change no default beyond rulings
1 and 4. Report, and say which you would choose and why. Same
discipline: retro with its git add and the state file in the same
commit, then stop.
```
