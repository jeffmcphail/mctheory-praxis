# Brief: Mid-office 09 -- the count rulings, the defaults, and residuals for every covered name

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-10
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-10T17:57Z (12:57 EST), the first clock reading after it; transcribed verbatim below, the
paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice08 (cfe1245), checked in `git log` before any
file was touched; then come his rulings on midoffice08's six decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice09_counts_residuals.md`
**Builds on:** cfe1245 (the panel on securities and the three-step comparison).

---

## The brief, verbatim

```text
committed -- midoffice08.

On midoffice08's decisions:

1. Transitional first counts: (b). A first count stands only if the
   span's next sample confirms it within 10%; otherwise the search goes
   on to the first record the next sample confirms, and the days before
   it have no count.
2. Single bad samples: repair the isolated ones. A sample whose two
   samples before and two after all agree with each other within 10%,
   and from which it differs beyond 10% with no split to explain it, is
   an error: it is not carried, and the count before it carries
   through. Where the samples alternate, as CEG's and PCG's do, nothing
   is repaired -- the rule cannot tell which count is the true one --
   and they stay flagged in the count-move report. Report every sample
   the rule removes.
   Rulings 1 and 2 both use a later sample to recognise an error. That
   is hindsight in cleaning the data, not in what the model knows: the
   true counts were public at the time, in the companies' filings, and
   a wrong count carries no information about the day's return, so
   repairing it cannot flatter the fit the way same-day exposures did.
   Real changes -- mergers' issuance, Carnival's unification -- are
   still adopted when the samples show them and never back-filled.
3. HON and SKYD: an overrides file, committed in the repo rather than
   in data/, since it holds rulings, not vendor data. Each entry: a
   security, a date range, what the build does there, and the evidence.
   - HON from 2026-10-12: carry 635,675,701. The halved count matches
     Honeywell Aerospace's to 0.06%, with no split event and no price
     gap: a contaminated record.
   - SKYD from 2026-10-06: no count. After it took WBD the records carry
     no share-class count, and the pre-merger one is 4.6 times under
     the weighted count; no count is better than a wrong one.
   The build applies the file and lists every entry it used. A new
   anomaly goes in the file with its evidence, not into a rule written
   for one case.
4. Returns: total by default, price still runnable.
5. The panel on securities by default, the old one runnable for
   comparison.
6. Next, as below.

Next piece, midoffice09. Same discipline: in this order, each change
measured against the one before, the retro with its git add and
MIDOFFICE_STATE.md brought up to date at the end, then stop.

1. The count rulings, 1 to 3: rebuild the panel and report the
   member-days each one changes, and the model against midoffice08's
   run D.
2. The defaults, 4 and 5: the run at its new defaults should equal
   item 1's run exactly; say whether it does.
3. Residuals for every covered name. Factor returns still come from
   the estimation universe alone, and the specific-risk model's
   estimation -- the V fit, kappa, the blend's model error -- still
   uses only its residuals. But every security with a return and a
   complete characteristic row gets that period's specific return,
   member or not, and each security's own specific-risk history is
   built from all of them. A joiner then arrives with its history from
   before it joined, and a leaver keeps forecasting from its own
   history after. Report the fallback list (11 now), the blend weights
   of the joiners in their first year as members, the specific-risk
   percentiles, and the names whose forecasts move most.

After this: the model-wide noise-share question, then out-of-sample
bias statistics.
```
