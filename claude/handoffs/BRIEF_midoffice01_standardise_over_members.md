# Brief: Mid-office 01 -- standardise over index members

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-05
**Source:** Jeff's request in the Claude Code session, transcribed verbatim below. No Chat-written
brief preceded it; this file records the request as given, so the brief/retro pair exists.
**Retro:** `claude/retros/RETRO_midoffice01_standardise_over_members.md`
**Builds on:** f72310b (each period's regression held to that day's index members), 17a588d,
c1275ab, 7cfe835 -- the S&P 500 model work before the `midofficeNN` series, which has no retros.

---

## The request, verbatim

```text
Standardise over members now, before the rebuild. Same principle as the
mask: the estimation universe on day t is that day's members passing the
data test, and nothing outside it should set the scale of anything inside
it -- Barra's estimation universe versus its coverage universe.

1. Build the membership mask once, as a date-by-security array, and hand
   that one array to the regression and to standardisation. Re-reading the
   change table on all 1,214 calls costs 13 seconds now and would double
   with a second consumer -- but the real reason is that one array used
   everywhere means the two cannot disagree about who is a member.
   Without --membership, behaviour is unchanged.

2. Per period, over the estimation universe only: the MAD clip bounds, the
   cap-weighted mean and the equal-weighted SD. Both standardisations in
   blend_to_index, fit_index_weights and the descriptor screen use the
   same set.

3. Securities outside the estimation universe that have a characteristic
   row get exposures on the members' scale -- the members' mean, SD and
   clip bounds. Report how many hit a bound; the ex-members sit in the
   tails, so some will.

4. Exact guard on every estimated period: the cap-weighted mean of each
   style exposure over the estimation universe is zero to machine
   precision. That is what gives the index itself zero style exposure, so
   an exposure reads as relative to the S&P 500. It is within 0.013 now.
   If the post-standardisation clip makes it inexact, report by how much
   rather than loosening the tolerance.

Then rerun with --membership, before and after: R-squared, the five style
factor vols, the members' cap-weighted means and equal-weighted SDs on
2026-11-03, and how many non-members were clipped.

What I expect, so a surprise is visible. This is close to a
reparameterisation. A per-period affine change to a style column is
absorbed exactly -- the scale by the column's own coefficient, the shift
by the industry columns, which sum to the column of ones -- so fitted
values move only through the parts that are not affine: the clip bounds,
the descriptor blend, and the few names with no industry row. The MAD
clip is robust to tail names by construction, so the bounds should barely
move. R-squared should move by a few thousandths at most. Style factor
vols should fall roughly in proportion to the members' spread: about 12%
for Size and 15% for Size Nonlinearity. If R-squared moves materially,
the descriptor blend is doing more than rescaling and I want to know why
before it is committed.
```
