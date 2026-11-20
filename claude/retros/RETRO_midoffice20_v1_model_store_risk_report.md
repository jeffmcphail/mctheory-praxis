# Retro: midoffice20 -- v1 frozen, the model store, the first risk report

**Date:** 2026-11-17 (EST)
**Brief:** `claude/handoffs/BRIEF_midoffice20_v1_model_store_risk_report.md`, written in Chat and
pasted in.
**Builds on:** aacd3b2 (midoffice19), checked in `git log` before any file was touched.
**State file:** `claude/MIDOFFICE_STATE.md`, brought up to date in this commit.

## Summary

- **v1 is frozen, and written up in two pages** (`docs/FACTOR_MODEL_V1.md`). It covers what v1 is,
  how it was validated, the ten things tried and rejected with the number that decided each, and
  seven known limitations. Every number cites its section of the record.
- **The model store is built and checked:** v1 and `v1-blocks`, @@ dates each, in
  `data/model_store/`, @@ MB a model. A date's forecast is read in @@ ms at the median.
  - **The guard passes:** on its five dates, the stored covariance, specific risk and exposures
    equal the model refitted from the panel cut at the date, to the bit.
  - **The stored bias runs agree:** every date's covariance equals D19's (@@ of @@), and the
    blocks' covariance and multipliers equal C19's, to the bit.
- **Three $100m sample books and the index are built on every date.** LOWVOL equals D19's stored
  long-only minimum-variance holdings to the bit on @@ of the @@ dates whose universe is the bias
  test's.
- **The risk report prints for any book, date and model in about @@ s, with its three checks
  passing.** LOWVOL's forecast equals the bias run's stored one to @@.
- **Ruling 4: both seed fixes are in, each with a test.** The tests fail on HEAD's code. The suite:
  @@ passed, 1 skipped.

## What was done

1. **The rulings.** All five are recorded in the state file. Ruling 4 is built:
   `surface._compute_multi_point` and `monte_carlo.generate_adf_critical_values` now seed each
   later batch with `SeedSequence([seed, batch])`, the first batch by the seed itself, as 19
   fixed `_compute_single_point`. `tests/test_seeded_batches.py` (3 tests) runs each twice on
   seed 42, over three and five batches. The same checks on HEAD's code differ on all 50
   t-values and on the multi-point scalars (`seed_tests_on_head.txt`).
2. **The write-up**, 106 lines, sourced from `docs/FACTOR_MODEL.md`, the state file and the retros.
   The noise shares are v1's own (its run's factor table), not the older ones in the state file.
3. **The store.** `src/praxis/midoffice/store.py` holds the writer and the reader: one directory a
   model, one row group a date. `scripts/build_model_store.py` runs v1's full fit through the run's
   own code, by a new `after_fit` hook on `run_sp500_model.main`, which changes nothing else. It
   then writes each date's forecast from the model refitted as of the date.
   - **Why the refits, when the brief says "from v1's full fit":** the full fit forecasts specific
     risk only as of its last date. Its specific risk model is fitted once, on every residual
     through 2026-11-03, so an earlier date's specific risk is not in it.
   - **What the full fit supplies:** the factor returns and the regime multipliers, read from the
     full fit as a plain run reads them. The covariance could have come from it too; the refits
     give the same to the bit.
   - **What the store holds:** exactly the forecasts D19 tested, plus 2026-11-03, the full fit's
     own. The refits took @@ min, 0.4 s each.
4. **v1-blocks, cheap, from the same pass:** v1's exposures and specific risk, and the blocks'
   covariance formed as C19's bias test forms it. It is a model of its own in the store, and a
   separate build would repeat the refits for nothing.
5. **The books** (`scripts/build_sample_portfolios.py`). Each date's universe is its members with a
   v1 forecast and a positive capitalisation. INDEX is that universe cap-weighted, LOWVOL is
   `bias.minimum_variance` on the store's forecast, MOM40 and EQ50 follow their rules. Each book
   is $100m, in fractional shares at the raw close.
6. **The report** (`scripts/risk_report.py`) goes through `engine.compute_risk`: `AsOf` resolves
   the four dates against the store's calendars, the books are valued positions, and Industries
   and Styles form a `FactorTree`. 5 more tests cover the store and 7 the books and report end to
   end on a synthetic store.

## Readings and decisions taken

- **"Every date the fit covers" is read as every date the model forecasts:** 2024-02-08, the first
  date with a year of estimated periods, to 2026-11-03. That is @@ dates, not the fit's 938
  periods: before a year of periods the model makes no forecast, and the bias test starts there.
- **The universe leaves out the next day.** The bias test drops a member with no return the next
  day, which a book formed at the close cannot know, so LOWVOL is the test's portfolio only where
  no member drops. That holds on @@ of 686 dates. Check 3 uses the report's date when it is one of
  them, else the nearest such date before it.
- **The index is the covered members, cap-weighted**, as the bias test forms it. Members without a
  forecast are left out: at most @@ of the members' capitalisation on any date, @@ at the median.
  The report prints the date's share (ruling 1, below).
- **Historical VaR is the factor part's:** the book's exposures as of the date applied to the
  last 252 days' factor returns through it (ruling 2). The report prints it beside the parametric
  factor part's, not as a total.
- **Check 3 is to rounding (1e-12), not to the bit.** The engine sums positions and issuers one by
  one; the bias test sums them by matrix product. The weights themselves are bit-equal in the
  store.
- **The engine never repairs these covariances.** The smallest eigenvalue is at least 3.4e-4 of the
  largest, against the 1e-8 floor; none of D19's 686 forecasts was repaired.

## Jeff's rulings on the draft's four decisions, mid-piece

Jeff ruled before the final checks (the brief's "Amended mid-piece"). Nothing in the store or the
books moved; the report gained two things.

1. **The index** leaves out its members without a forecast, as built. The report now prints, on
   every date, how many it leaves out and their share of the members' capitalisation. Every
   member left out lacks a forecast; one with no capitalisation that day counts at none.
2. **Historical VaR** takes a fixed window, 252 days by default (`--var-days` sets another), the
   specific part still out. The first forecast date has exactly 252 days behind it, so every
   date's window is full. A window longer than a date's history is refused (exit 2), not
   measured on a shorter one. Two new tests cover the window's VaR and the refusal.
3. **Fractional shares**, as built.
4. **The store starts at the first forecast date**, as built.

No decision is left for Jeff.

## Scratch, named (`claude/scratch/midoffice20/`)

- `runner.py` and `runs/`: the runner, its log, and each run's report and progress files
  (`smoke`, `store`, `portfolios`).
- `store_smoke/`: the first 30 dates' store, both models, with its books.
- `seed_tests_on_head.py` and `.txt`: ruling 4's tests' checks on HEAD's code.
- `patch_*.py`: the state file's patches.
- `final_checks.sh` and `.txt`, and `final/`: the final checks and every report they ran.

## Appendix: the checks' output, verbatim

@@CHECKS@@

## Appendix: one report, verbatim

@@REPORT@@

## git add (always last -- Rule 43)

@@GITADD@@
