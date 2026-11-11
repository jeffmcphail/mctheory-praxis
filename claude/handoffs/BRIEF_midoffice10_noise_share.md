# Brief: Mid-office 10 -- PCG and the leavers, then the noise-share question, measured

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-10
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-10T18:51Z (13:51 EST), the first clock reading after it; transcribed verbatim below, the
paste wrapper stripped.
Its first line is Jeff's "committed" for midoffice09 (fccc480), checked in `git log` before any
file was touched; then come his rulings on midoffice09's three decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice10_noise_share.md`
**Builds on:** fccc480 (the count rulings, the defaults, residuals for every covered name).

---

## The brief, verbatim

```text
committed -- midoffice09.

On its decisions:

1. PCG: spare it, by the general addition you proposed -- a sample its
   own record's weighted count confirms within 10% is not an error. Two
   independent fields of one record agreeing is the evidence the
   isolated-error rule lacks, and it separates PCG from the six repairs
   correctly: the weighted counts of DGX, EG and SBUX confirm their
   repairs, and NWS, RDDT and XYZ are multi-class. It reads the same
   day's record, so nothing in it looks ahead. PCG's series stays
   flagged as alternating.
2. Leavers after they leave: end a security's span when it stops
   trading on an exchange. A failed bank's shares at fractions of a
   cent off-exchange are not an instrument this model is for, and that
   is where SBNY's 396% comes from. If the reference records say where
   a ticker trades, use them, and report every span the rule ends and
   the specific returns it removes; if they do not, an overrides entry
   for SBNY with its evidence, and report the other candidates the
   prices suggest. Guard: no member-day changes, since the index holds
   only listed shares. A leaver that stays listed keeps forecasting
   from its own history, as ruled.
3. History weighting: leave it equal for now. The factor side uses a
   90-day half-life and the specific side none, which is inconsistent,
   but which half-life is right is what the bias statistics are for;
   they will test equal weighting against several.

Next piece, midoffice10. Rulings 1 and 2 first, each measured. Then the
noise-share question, measured before anything is chosen, as
midoffice03 measured the definitions:

1. The sampling-error correction: the factor covariance less the mean
   sampling covariance of the factor returns, weighted as the
   covariance is, with the PSD repair where it is needed. Report every
   factor's vol before and after, whether and where the repair binds,
   and the risk of a few reference portfolios before and after: the
   index, an equal-weighted index, the size pair's two, and a
   single-industry portfolio for each thin industry.
2. The thin industries: Construction (8 members), Wholesale (12),
   Finance & Real Estate (12). Each is already a whole division, so it
   cannot merge further under the scheme. For each, the industry factor
   its own factor return correlates with most, and the model with it
   merged there: R-squared, noise shares, and its members' specific
   risk. Then a floor of 15, 20 and 25 members applied the same way to
   every industry.
3. Size Nonlinearity at 46.6%: the correction's effect on it, and the
   model without it, already runnable as none.

Change no default. Report, and say which you would choose and why.
Same discipline as before: retro with its git add and the state file
in the same commit, then stop.
```
