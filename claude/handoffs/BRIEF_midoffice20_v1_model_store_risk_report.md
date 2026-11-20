# Brief: Mid-office 20 -- midoffice19's rulings, then v1 frozen, the model store and the first risk report

**Series:** praxis, mid-office (`midofficeNN`, Rule 43 of `claude/CLAUDE_CODE_RULES.md`)
**Date:** 2026-11-17 (EST; 2026-11-18 UTC)
**Source:** written in Chat and pasted into the Claude Code session shortly before
2026-11-18T01:33Z (20:33 EST), the first clock reading after it; transcribed verbatim below, the
paste wrapper stripped, with the message that amended it mid-piece after it.
Its first line is Jeff's "committed" for midoffice19 (aacd3b2), checked in `git log` before any
file was touched; then come his rulings on midoffice19's decisions, then the brief.
**Retro:** `claude/retros/RETRO_midoffice20_v1_model_store_risk_report.md`
**Builds on:** aacd3b2 (midoffice19: every member on the market; two blocks rejected; surface
seed fix).

---

## The brief, verbatim

```text
committed -- midoffice19.

midoffice20 -- freeze the model as v1, the model store, the first risk report.

Rulings on midoffice19's decisions:
1. D19 is the model: v1. The rule stands on the blocks. They fixed the
   calm half's level but failed the optimiser's test, and the rule was
   fixed before the run so a narrow miss can't be argued in afterwards.
   They go in the write-up as the leading candidate for a v2, with
   their price.
2. Agriculture stays in SIC28: industry by business, as Barra assigns,
   known at the time. The returns check stays in the record.
3. VYLR keeps its code.
4. Fix the two other unseeded batches (surface._compute_multi_point,
   monte_carlo.generate_adf_critical_values) the same way: the first
   batch's draws unchanged, each later batch seeded from the seed. The
   later batches were never reproducible, so nothing reproducible
   changes. Done when a test for each shows the same seed giving the
   same values twice, and the suite passes.
5. The next piece is not specific risk. The model freezes as v1 with
   this piece, and the largest members' specific risk goes in the
   write-up as a known limitation. In the state file, "What is next"
   becomes the risk GUI on Core, once Core's rework lands.

This piece, in order. Change no model default.

1. The v1 write-up, docs/FACTOR_MODEL_V1.md, two pages at most, for a
   reader who knows Barra's models: what v1 is; how it was validated
   (the out-of-sample bias statistics, the halves, the rule, the
   guard); what was tried and rejected, one line each with the number
   that decided it; the known limitations. Every number comes from the
   record (docs/FACTOR_MODEL.md, the state file, the retros) and can be
   traced to its section. docs/FACTOR_MODEL.md stays the full record.

2. The model store. One build from v1's full fit writes, for every
   date the fit covers, in Parquet:
   - the factor covariance as the engine uses it (the regime
     adjustment applied), keyed (model_id, covariance_date);
   - the exposures, keyed (model_id, exposure_date);
   - the specific risk, keyed (model_id, covariance_date);
   - the factor returns, keyed by model_id, for historical VaR.
   These keys are the lookup columns Core's DataTables will use when
   the GUI reads the store. Nothing here uses Core yet; it is being
   reworked in parallel. After the build, any date's lookup takes
   milliseconds. A model variant is one more build under its own
   model_id: build v1, and the two-block setting as a second model if
   that is cheap, saying which in the retro. Check: as of the guard's
   five dates, the store's covariance and specific risk equal the
   model refitted as of the date.

3. Three sample portfolios against the cap-weighted index, $100m each,
   built by rule so that every date has holdings:
   - LOWVOL, the long-only minimum-variance portfolio from v1's
     forecast as of the date;
   - MOM40, the 40 members with the highest Momentum exposure,
     cap-weighted;
   - EQ50, the 50 largest members, equally weighted.

4. The report, from the store and the existing risk core
   (src/praxis/midoffice/risk/): total risk, split into factor and
   specific; tracking error against the index; the decomposition by
   factor group (industries, styles); the ten largest position
   contributions to total risk and to tracking error; one-day VaR and
   ES at 95% and 99%, parametric from the covariance and historical
   from the factor returns. One command, for example:
     python scripts/risk_report.py --portfolio LOWVOL --date 2025-06-30 --model v1
   It prints its checks with the report: factor and specific variance
   sum to the total; the contributions sum to the total and to the
   tracking error; and on a test date, LOWVOL's forecast equals the
   bias run's stored forecast for the long-only minimum-variance
   portfolio.

Done when: the write-up exists; the store is built and checked; the
report prints for any date, portfolio and model in seconds with its
checks passing; the seed fixes are in; the suite passes.

Retro: two pages at most. No verbatim appendix except one report
(LOWVOL on a date of your choice) and the checks' output; everything
else stays in the scratch folder, named. The state file goes in the
same commit, and the git add line ends the retro as before.
```

---

## Amended mid-piece, 2026-11-18

One further message from Jeff, written in Chat and pasted in while the piece was under way;
transcribed verbatim.

**Rulings on the retro draft's four decisions.** Pasted shortly before 2026-11-19T00:12Z (19:12
EST on 2026-11-18), the first clock reading after it. Code had reported the store and portfolio
builds waiting on memory; the log showed both had finished at 21:25 EST on 2026-11-17, checks
passing, so nothing was restarted.

```text
The runner hasn't finished. Check claude/scratch/midoffice20/runs/runner_log.txt and whether the runner process (pid 22076) is still alive. If it has stopped, start it again the same way, with the same 3 GB bar. Meanwhile, apply these rulings on the retro's four decisions:

1. The index: as built, members without a forecast left out, as the
   bias test does. Have the report print the share of the index's
   capitalisation left out on its date, so the coverage is in front of
   the reader.
2. Historical VaR: a fixed window, 252 days by default, with
   --var-days still setting it. Every forecast date has at least a
   year of factor returns behind it, so every date's VaR rests on the
   same length of history, and a year is the Basel minimum. The
   specific part stays out, as built; the report keeps labelling it
   the factor part.
3. Fractional shares, as built.
4. The store starts at the first forecast date, as built.

Then stop and tell me the runner's state. I'll tell you when it has finished, and then you do the final checks, the numbers and the git add, as planned. Keep the retro to the brief's two pages: the appendices are the checks' output and one report, nothing else.
```
