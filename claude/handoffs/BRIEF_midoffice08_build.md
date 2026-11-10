# Brief: Mid-office 08 -- the build

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-10
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-10T16:17Z (11:17 EST), the session's first clock reading; transcribed verbatim below,
the paste wrapper stripped.
It opens by sending the session to `claude/MIDOFFICE_STATE.md`'s Start here; then come Jeff's
rulings on midoffice07's four decisions, then the brief. It has no "committed" line: `git log`
shows midoffice07 committed (5ccbb70) and the state file after it (7228b01), checked before any
file was touched.
**Retro:** `claude/retros/RETRO_midoffice08_build.md`
**Builds on:** 5ccbb70 (the identity corrections and the gate), 7228b01 (the state file),
208ca80 (the rebuild's fetch) and 701794e (the identity table).

---

## The brief, verbatim

```text
Read claude/MIDOFFICE_STATE.md and do what its Start here section says.
Then midoffice08, below.

On midoffice07's decisions:

1. COHR: (a), the share-class rule. The share-class FIGI is the level
   that names a security across its composite listings, all seven such
   records carry the expected security's, and it needs no fetch. Keep
   it a check, not a re-keying: it reconciles a record to the security
   the table already expects, and only where that security's own
   records carry the share-class FIGI; the table stays keyed by
   composite FIGI. Add Coherent, Inc.'s row under COHR, BBG000BG1DH3,
   2022-01-03 to 2022-07-01.
2. Share counts the records do not carry: search each span's first
   quarter for its first record with a count, the 70 or so requests,
   and carry from that record's own date -- forward only, so nothing is
   back-filled; the days before it stay without a count. Do not use
   weighted counts as data. Use them as a standing check instead:
   wherever a carried share-class count meets a weighted count, report
   the ratio and flag any beyond the 10% tolerance. That is the check
   that just confirmed OKE's and XOM's 1:1 carries.
3. Volumes: yes. The new panel stores traded volumes, consistent with
   point-in-time counts and traded prices, and the comparison measures
   the basis as its own step.
4. Rule 43's ordering: leave the wording. The state file's add came
   after the piece's this once because it was written after the piece;
   from now on it travels in the retro's own commit.

midoffice08: the build. Run the gate first, offline -- with 1 above it
should pass -- then items 2 to 5 of midoffice07's brief, with the
comparison in three steps, each measured on its own:

- the old panel against the new one, on price returns and adjusted
  volumes;
- the new panel, adjusted volumes against traded ones;
- the new panel, price returns against total returns.

Retro and git add lines at the end, MIDOFFICE_STATE.md brought up to
date in the same commit, then stop.
```
