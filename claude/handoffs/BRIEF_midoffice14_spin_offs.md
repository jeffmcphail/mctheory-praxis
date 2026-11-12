# Brief: Mid-office 14 -- midoffice13's rulings, then the spin-offs the total return does not carry

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-12 (EST and UTC)
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-12T21:10Z (16:10 EST), the first clock reading after it; transcribed verbatim below, the
paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice13 (cee34ca), checked in `git log` before any
file was touched; then come his rulings on midoffice13's decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice14_spin_offs.md`
**Builds on:** cee34ca (the one-day horizon, the eigenfactor adjustment measured, the
correlations' own lags).

---

## The brief, verbatim

```text
committed -- midoffice13. Your context was compacted while idle: read
claude/MIDOFFICE_STATE.md and the midoffice13 retro before starting.

On its decisions:

1. The eigenfactor risk adjustment: an option, off.
2. CTVA: next, as a data piece (below), with the search for others.
3. After it, the volatility regime adjustment, then the eigenvalue
   adjustment one-sided, re-tested on the regime-adjusted model. The
   regime moves every portfolio more than the eigenvalues do (the index
   1.22 in the first half, 0.75 in the second; the minimum-variance
   portfolios 1.25 to 1.30, then 0.82 to 0.85 without the CTVA day), so
   a shape adjustment is judged once the level is right.
4. The reporting view at 21 days: the lags on the volatilities alone.
   At any horizon above one day the correlations take no lags by
   default; the 21-day view's bias run must equal V21N's to the bit.
5. The minimum-variance portfolios' realised volatility becomes a
   standing line of the bias report, beside their statistic: the test
   the adjustment failed, kept for every covariance candidate after it.

Next piece, midoffice14: spin-offs the total return does not carry.
The rulings first, then:

1. CTVA. Corteva distributed its seed business as Vylor (VYLR): one
   VYLR share for each CTVA share, record date 2026-09-24, distributed
   2026-10-01, VYLR trading regular-way from that day. CTVA's total
   return that day is (CTVA's close + VYLR's close) / CTVA's prior
   close - 1, plus any dividend. Carry it through an overrides file
   with its evidence, as the count rulings are carried. Vylor's
   missing count: Corteva's on the record date, at the ratio, until
   Polygon carries one.
2. The others. A distribution the cache records as a fractional ratio
   is carried already; one it does not record looks like CTVA: a raw
   price gap, the count unchanged, no record, and often a new security
   whose first day is that day. Smaller spin-offs move the parent far
   less than CTVA's did, so a threshold scan misses them. Check every
   security's first day in the panel against the members' raw moves
   that day, and check these parent and child pairs from 2022 on, each
   against the cache and the panel: EXC CEG, ZBH ZIMV, BDX EMBC, T WBD,
   FBHS MBC, GE GEHC, LH FTRE, BWA PHIN, DHR VLTO, K KLG, MMM SOLV,
   GE GEV, J AMTM, LEN MRP, WDC SNDK, FTV RAL, HON SOLS, DD Q,
   CMCSA VSNT, FDX FDXF. The list is from memory and not exhaustive:
   the data and the public record are the authority, and a pair whose
   parent was no member needs nothing. For each: carried, carried
   wrongly, or not carried, with the fix for the last two. FDXF's
   missing count the same way as Vylor's. LEN's adjustment of
   2025-01-21, left as Polygon has it, is checked with the rest.
3. The rebuilt panel under a new name, the old one kept; it becomes
   the run's default. Measured: every changed member-day; the defaults'
   run and their bias run against midoffice13's, the minimum-variance
   portfolios' statistics and realised volatility first.

Runs one at a time through the runner. Change no default beyond
rulings 4 and 5 and the rebuilt panel. Report, and say what you would
do next and why. Same discipline: retro with its git add and the state
file in the same commit, then stop.
```
