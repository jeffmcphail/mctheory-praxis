# Brief: Mid-office 06 -- the rebuild

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-09
**Source:** written in Chat and pasted into the Claude Code session; transcribed verbatim below,
with the two messages that amended it mid-piece after it.
Its first line is Jeff's "committed" for midoffice05's two changes (1b9d43b, 701794e), checked
in `git log` before any file was touched; then come his rulings on midoffice05's five
decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice06_rebuild.md`
**Builds on:** 701794e (the identity table) and 1b9d43b (the size pair's standing line).

---

## The brief, verbatim

```text
committed -- the size pair line, then the identity gate.

On midoffice05's decisions:

1. Reorganisations: one security with one continuous return series,
   keyed by the successor's FIGI with the predecessor linked in the
   table. Sample the share count on both sides of each boundary and
   report the ratio: a holding-company reorganisation is 1:1, and any
   that is not -- Carnival's could be its dual listing folding in --
   takes the new count from the boundary rather than carrying the old
   one.
2. The 4,813 named member-days: yes, use the reference endpoint. The
   public-record candidates -- CDAY for DAY, DISCA for WBD, WRK for SW,
   FLT for CPAY, PARA for SKYD, Bunge Limited for BG, SMCI itself --
   are hypotheses for it to confirm; only what the endpoint confirms
   enters the table, and anything it does not stays named.
3. Renames in the run: yes, in the rebuild. Have the run print sp500's
   current_stale warnings too, so the roll-forward is visible where
   the model is.
4. The fetch: the whole window for every member security, as planned.
   It costs more requests once and gives every joiner its history
   before joining and every leaver its record after; keep a span-plus-
   look-back cut as a parameter for future panels.
5. Casey's: the first share sample's identity check is the right test.

Next piece, midoffice06: the rebuild. Same discipline -- retro and git
add lines at the end, then stop -- with one gate in the middle.

1. Resolve the named member-days through the reference endpoint, and
   update the identity table and the fetch plan.
2. Run the fetch: resumable, cached, at the original fetch's rate.
   Prices, adjusted and not; share counts only inside each security's
   own spans; split events. The first share sample of every span
   checks the identity the table expects -- CRHC and CASY among them.
3. Gate: if any span's identity check fails, or any split other than
   DD's disagrees with the next share sample, stop here and report
   before building anything. Member-days the endpoint leaves
   unresolved are not a stop; they stay named and are reported.
4. Splits: apply each split's ratio to the carried share count on its
   execution date, from the split events. A split the next share
   sample disagrees with is not applied -- the samples rule there --
   and is flagged. Guard, exact: no member-day pairs a post-split
   price with a pre-split count.
5. Build the new panel with securities, not tickers, as its columns:
   prices stitched across each security's tickers by the identity
   table, splices separated, reorganisations continuous, share counts
   per span. Membership comes from membership() read through the
   renames and mapped to securities. The identity guards run on the
   built panel.
6. Run the model on the new panel at today's defaults and report it
   against the old panel's run: per-period cross-section sizes, which
   should rise toward the index's 503 or so, with what still keeps a
   member out; R-squared; every factor's vol, mean and noise share,
   Construction's and Wholesale's especially; the residual spread; the
   specific risk percentiles; the fallback list; and the size pair's
   portfolio line. Name anything the old panel had right that the new
   one loses.

Not in this piece: residuals for every covered name rather than only
regression members, and the model-wide noise-share question. Both come
after, on the new panel.
```

---

## Amended mid-piece, 2026-11-10

Two further messages from Jeff, written in Chat and pasted in while the piece was under way;
transcribed verbatim.

**1. The split rule, before the gate ran.** Sent at 2026-11-10T06:07:25Z (01:07 EST), after
Code reported that the first fetch had been stopped for low memory:

```text
Restart the fetch; it resumes from the cache, so nothing is lost. Have
it log its own memory every 500 requests, so that if it is stopped
again we know whether it was the cause -- a network-bound fetch should
need little, and the pressure at 15:30 may have come from elsewhere on
the machine.

Before the gate runs, change its split rule. You're right that it would
stop on spin-offs, and stopping there would be wrong: Polygon records a
spin-off as a fractional "split" to adjust the parent's price history,
and the parent's share count does not move across it. Classify every
split event first, by two independent tests -- is the ratio one of
whole numbers, and do the share samples either side move by it, within
10% (a parameter):

- Whole-number ratio, and the counts move by it: a split. Apply it to
  the carried share count on its execution date.
- Fractional ratio, and the counts stay within 10% of unchanged: a
  price-only adjustment, a spin-off or similar. Do not apply it to
  share counts; the adjusted prices already carry it, which for a
  spin-off is the right total-return treatment. Report each, with the
  spin-off or distribution it corresponds to where you can name it.
- The two tests disagree -- a whole-number ratio the counts do not
  follow, or a fractional one they do: stop and report. Those are the
  contradictions the gate exists for.

Where there is no share sample on one side of an event, classify it by
the ratio alone and report it as untested. If the split events carry a
field saying what kind of adjustment each is, use it as a third test
and report where it disagrees with the other two.

The guard is unchanged, counting only events classified as splits: no
member-day pairs a post-split price with a pre-split count.
```

**2. The rulings on the split rule, dividends, and LEN.** Queued at 2026-11-10T06:38:19Z
(01:38 EST) and surfaced mid-turn, while the gate's report was being read:

```text
On your open reading: yes, a fractional ratio whose share counts move
by neither the ratio nor nothing stops the gate. A case the two tests
cannot place is a contradiction by definition, and none of the
window's 55 events is one, so it costs nothing here. Reducing the terms
and capping them at 100 is the right reading of "whole numbers"; keep
the cap a parameter, as you have it.

Two additions to the rebuild:

1. Dividends. If Polygon's adjusted bars are split-adjusted only, the
   model has been estimated on price returns, and every ex-date reads
   as a loss of the dividend: a negative bias in the means of the
   high-yield factors and industries, and a small, regular addition to
   high yielders' specific variance. Barra estimates on total returns.
   Fetch the dividend events for every security in the plan, every
   cash distribution including special ones, and for SPY, since the
   market model regresses on it. Build total returns from them, on the
   same split basis as the prices, and correct prices.py's docstring.
   Run the fetch once the share samples finish -- it does not depend on
   the gate -- and keep its effect separate in the comparison: identity
   first, then dividends, so each one's effect on the model is
   measured on its own.

2. LEN, 2025-01-21. Check LEN's raw prices around Millrose's
   distribution for the ex-date gap the 2025-01-21 adjustment should
   have been. If it is there, Polygon misdated the adjustment, and the
   adjusted series carries a wrong return on both days. Report what
   you find; do not correct it yet.

If the gate stops on identity checks, as you expect from COHR: for each
failing span, give the FIGI the table expects, the FIGIs its samples
name with their dates, what the events and records said, the
correction you would make, and any extra fetch it needs -- COHR may
need IIVI's history, if II-VI is the company that took the ticker in
2022. Then stop, as you planned.
```
