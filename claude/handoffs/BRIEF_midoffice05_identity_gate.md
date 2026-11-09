# Brief: Mid-office 05 -- the identity gate

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-09
**Source:** written in Chat and pasted into the Claude Code session; transcribed verbatim below.
Its first line is Jeff's "committed" for midoffice04 (8c76c59), checked in `git log` before any
file was touched; then come his rulings on midoffice04's five decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice05_identity_gate.md`
**Builds on:** 8c76c59 (midoffice04: USE4's market model, one blend per factor, USE4's Size
Nonlinearity, the MAD spread) and the identity probe of 2026-11-04, which scoped the problem
and built nothing.

---

## The brief, verbatim

```text
committed -- midoffice04.

On its five decisions:

1. Keep Size Nonlinearity. Its R-squared contribution, 0.0043, is more
   than a regressor of pure noise would add on about 450 names --
   roughly 0.0016 to 0.0026, depending on how the weights cut the
   effective count -- so it carries some signal, if not much. Its 41.8%
   noise share is the same problem as Construction's and Wholesale's,
   and that gets one model-wide answer after the rebuild, not a
   deletion per factor. Keep "none" runnable beside "use3" and "use4".
2. Yes: orthogonal in the regression's own weights is the intended
   sense. It is the inner product the regression uses, and Size's
   0.9998 against the model without the factor is the proof it worked.
3. Accepted: USE4's market model on a daily panel and USE3's 60 months
   otherwise; None as the default and inf as equal weights.
4. Accepted. The MAD is the right scale -- the standard deviation was
   FRC's -- and how much weight history should get against the model
   is for out-of-sample bias statistics to settle, not the scale.
5. Noted: the linear size effect on this panel is about 2.9% a year;
   the raw cube was adding 1.8 points by splitting it.

Next piece, midoffice05: the identity gate, the first part of the
rebuild. Offline except one endpoint; no prices fetched and no model
change. Same discipline: retro and git add lines at the end, then stop.

First, two small items as their own change with their own git add:
the size pair's portfolio line -- the smallest 85 members, the largest
50 and the index, with their exposures and the pair's risk -- becomes
a standing line in the run; and the two stale figures in
docs/FACTOR_MODEL.md, 3,043 and 100,011, are refreshed from the
current run.

Then the gate. The panel identifies a security by a ticker string fixed
at 2022-01-03, and the probe in this thread found what that costs:
splices (META was a Roundhill ETF until 2022-01-28; BNY was a BlackRock
municipal fund for 753 of its 867 periods; DOC joins Physicians Realty
to Healthpeak with no price gap; INFO's ETF inherited IHS Markit's
share count), truncations (BALL, COR, CPAY, EG, ELV, GEN, MRSH, RVTY,
SKYD, SW, and FISV's 610 days as FI), and phantoms (ECHO in the 2022
universe; AvalonBay twice, as AVB and VMRK).

1. Build an identity table: one row per security per span of dates,
   with the share-class identifier (composite FIGI where the records
   carry it; CIK as the cross-check, since two share classes of one
   company, GOOGL and GOOG, are both members), the ticker it traded
   under in that span, and the evidence for each boundary.
2. Sources, cheapest first: the reference records already cached from
   the panel fetch (no requests), then /vX/reference/tickers/{t}/events
   for the 591-name union (it worked on this key, and covers only
   listed securities). Report what each source resolved and what
   neither did.
3. The checks that build it, across each ticker's cached snapshots: a
   change of name, of CIK or FIGI, or of type (common stock against
   ETF or fund), and a list date after the ticker's first price.
4. Guards, each exact: every (ticker, date) the panel holds maps to
   exactly one security; no security is held under two tickers on the
   same date; every member membership() returns on every date in the
   window maps to a security, or is named as missing.
5. membership()'s three known inconsistencies -- KVUE kept after its
   removal, OKTA never added, ECHO a member back to 1976: fix them at
   the source if the change table supports it, with tests, or report
   why it cannot.
6. Report: the table's size; every splice, truncation, phantom and
   duplicate found, with dates; the identity of each of the 591 union
   names; and the fetch the rebuild will need -- securities, date
   spans, tickers to request and calls, including share counts sampled
   inside each security's own span.

Write the table where the panel's other derived data lives,
gitignored, with the script that regenerates it in the repo.
```
