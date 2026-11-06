# Brief: Mid-office 02 -- empty windows, and the regression lagged a day

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-05
**Source:** written in Chat and pasted into the Claude Code session; transcribed verbatim below.
It was the second part of the message that ruled "Keep D" on midoffice01 and set Rule 43, to be
started only on Jeff's "committed" -- given after 8cdd037 and f0ae2a9 landed.
**Retro:** `claude/retros/RETRO_midoffice02_empty_windows_and_lag.md`
**Builds on:** 8cdd037 (exposures standardised over the estimation universe), f0ae2a9 (Rule 43).

---

## The brief, verbatim

```text
== Next part: start only after I reply "committed" ==

Two correctness fixes to the regression's inputs, both already found.
Make them in this order and decompose the effect of each, as you did
for A to D.

1. Window-based descriptors return missing, not zero, when the window
   has no valid observation: RSTR, CMRA, STOM, STOQ and STOA, 14,000 to
   23,000 values each, 4,669 of them on members before their first
   price. Add a minimum-coverage parameter -- the fraction of the window
   with valid data -- whose default changes only the empty case, and
   report how many values a 50% rule would remove so I can decide that
   separately.
   Guard: a synthetic security whose prices start after the window gets
   missing for every window descriptor; on the real panel, no window
   descriptor value is computed from an empty window.

2. Lag the estimation by one day. Everything that defines period t's
   regression is fixed at t-1 -- the exposures, their standardisation
   and the estimation universe -- and only the returns come from day t.
   That is Barra's convention and the descriptor screen's existing rule,
   applied to the model. It moves the membership mask by a day; report
   how many security-periods change. Forecast exposures stay at the
   last date, since they are as of today and forecast forward.
   - Guard: multiply every price dated t by a random factor, rerun with
     the descriptor weights held fixed, and assert that period t's
     exposures and estimation universe are bit-identical and only its
     return vector moved. That turns look-ahead into a test failure
     instead of something someone has to notice.
   - If fit_index_weights draws on returns from the whole sample, say so
     in the retro. That is in-sample weight selection: ordinary for a
     research model, but look-ahead in the strict sense, and I want it
     stated rather than discovered.
   - Check how sampled share counts fill the dates between samples.
     Carried forward from the last sample is point-in-time; back-filled
     or interpolated from a later sample is look-ahead in the cap, and
     so in LNCAP and the regression weights. Report which it is, but do
     not change it here; the rebuild resamples shares anyway.
   - SERDP exists only at the last date, so after this it enters no
     regression, only the forecast exposures. Report whether that leaves
     the forecast Volatility exposure built from a different descriptor
     set than the one its factor was estimated with.

Report per change and combined: R-squared, the five style factor vols
and their mean returns annualised, noise shares, the residual spread,
the specific risk percentiles and the floor list. You measured the lag
alone at 0.312 to 0.295 R-squared and Volatility at 9.55% to 7.79%, so
the combined result should land near that. I expect Size's mean return
to fall as well: today's cap containing today's return makes big stocks
look as though they outperformed, every day.

Do not touch the size pair. Retro and git add command at the end, as
above, then stop.
```
