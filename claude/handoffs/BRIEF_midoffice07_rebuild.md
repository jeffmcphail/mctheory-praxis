# Brief: Mid-office 07 -- the rest of the rebuild

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-10
**Source:** written in Chat and pasted into the Claude Code session at 2026-11-10T14:14:57Z
(09:14 EST); transcribed verbatim below, the paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice06 (208ca80), checked in `git log` before any
file was touched; then come his rulings on midoffice06's seven decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice07_rebuild.md`
**Builds on:** 208ca80 (the rebuild's fetch and gate), 701794e (the identity table) and
1b9d43b (the size pair's standing line).

---

## The brief, verbatim

```text
committed -- midoffice06, stopped at its gate.

On its seven decisions:

1. The identity corrections: all as proposed. Spin-off spans start at
   their first price; an identifier-less record is no evidence of
   another security; Bunge and Vylor are reorganisations, their
   boundaries dated by the identifier the table keys on; II-VI's IIVI
   history joins Coherent Corp.; S600, CEGV and CRHC are dropped, and
   CRH's ADR history before 2023-09-25 stays out, a different
   security, as you judged.
2. The split lag rule: as proposed -- a split applies on its execution
   date; a later sample still at the pre-split count is stale and does
   not reset the carried count; the first that follows the ratio does.
   Add a third test where the counts cannot speak: confirm every split
   by its raw price gap on the execution date, as you did for the
   spin-offs, within the same tolerance of the ratio's reciprocal, so
   MNST's and APH's rest on their prices rather than on their ratios
   alone. A split whose raw price does not gap stops the gate.
3. WBD: accepted as its own security. It is the combined company, not
   Discovery renamed.
4. Reorganisation counts:
   - Carnival: carry the predecessor's count until the successor's
     first sample. A split's ratio is an event record dated on its
     execution date, so applying it then is point-in-time; Carnival's
     1.1054 is known only from a sample three months later, and taking
     it from the boundary would be back-filling, which the share-count
     fill never does. The cost is one company's cap 10.5% low for a
     quarter.
   - OKE and XOM: carry the predecessor's count 1:1 and flag it. The
     four reorganisations the samples can test -- BLK, J, LH and BG --
     are each exactly 1.0000, which is what a holding-company exchange
     is.
5. TEL: keep one of the pair.
6. LEN 2025-01-21: leave it as Polygon has it, flagged.
7. docs/FACTOR_MODEL.md's split paragraph: with the build, rewritten
   for the new rule, with the old panel's measured staleness, 4,196
   member-days over 37 splits.

Next piece, midoffice07: the rest of the rebuild. Same discipline --
retro and git add lines at the end, then stop.

1. Make the identity corrections -- the 27 or so requests -- regenerate
   the plan, and run the gate again with the lag rule and the price-gap
   test. The same stop conditions as before, plus the price-gap one.
2. If it passes, build the new panel: securities as columns, prices
   stitched across tickers by the identity table, splices separated,
   reorganisations continuous, share counts per span with splits
   applied on their execution dates, membership read through the
   renames and mapped to securities, and sp500's current_stale
   warnings printed. The identity guards and the split guard run on
   the built panel.
3. Total returns: each distribution put on the prices' split basis by
   the ex-date's ratio of adjusted to raw close, one of TEL's pair
   kept, and SPY's too.
4. Run the model and compare in two steps, so each effect is measured
   on its own: the old panel against the new one, both on price
   returns; then the new panel on price returns against total returns.
   Report what midoffice06's brief asked for in its item 6, and for the
   dividends step the factor and industry means especially, since that
   is where price returns were biased.
5. The docs: the split paragraph as in decision 7, and a section on the
   identity table and the rebuild.

Not in this piece: residuals for every covered name, and the
model-wide noise-share question. Both come after, on the new panel.
```
