# Mid-office state

**Updated:** 2026-11-12, at the end of midoffice11 (midoffice10's rulings; the out-of-sample
bias statistics, the defaults and ten candidates). Brought up to date at the end of every
piece, in the same commit as its retro (Rule 43). Not a history: the state a new session needs to carry on. The detail is in the
retros named below.

---

## Start here

1. Read `claude/CLAUDE_CODE_RULES.md` (Rule 43 above all), then this file, then the latest retro:
   `claude/retros/RETRO_midoffice11_bias_statistics.md`.
2. **How the work runs.**
   - **The brief.** Each piece, `midofficeNN`, is a brief written in Chat and pasted into the Code
     session. Code transcribes it verbatim to `claude/handoffs/BRIEF_midofficeNN_<slug>.md`; its
     Source line says it was written in Chat and pasted in.
   - **The retro.** The piece ends with a full retro, `claude/retros/RETRO_midofficeNN_<slug>.md`,
     whose last item is the exact `git add`. The terminal gets a short summary and the retro's
     path, then Code stops.
   - **The commit.** Jeff commits with Chat's message; Code never commits or pushes. The next
     brief opens with "committed" (check it in `git log`) and Jeff's rulings on the decisions the
     last retro left. This file travels in the retro's own commit (08 ruling 4).
3. **The discipline every brief restates.**
   - Changes go in the order given, each measured against the one before.
   - A gate means stop and report before building anything.
   - Nothing in `data/` is overwritten or deleted without Jeff: each stage writes files of its
     own (`_resolved`, `_corrected`, `_amended`, `_repaired`), and the panel build refuses to
     overwrite without `--overwrite`.
   - Scratch goes to `claude/scratch/midofficeNN/` (gitignored), its outputs copied into the retro.
   - Python is ASCII and keeps the file's line endings (most `.py` here are CRLF in the working
     copy; the `claude/` briefs and retros LF).
   - The suite runs as `python -m pytest tests --ignore=tests/test_market_data.py -q`: 2,338 passed
     and 1 skipped at midoffice11. That file looks for `../core_repo/src`, which isn't on this
     machine.
4. **Working notes, from the pieces so far.**
   - A bash heredoc has failed in this harness on some apostrophes (06, 07, 08), though not
     always (09), and on backslash escapes inside a patch (10): a patch script written to a file,
     or the Edit tool, is the safe way.
   - The harness's shell can drift into another directory between calls (10): cd to the repo by
     its absolute path.
   - An `.npz` member is decompressed afresh on every access: read each array once into a dict.
   - The Write tool writes LF; `Path.write_text` without `newline=""` writes CRLF on Windows.
   - Git Bash's process substitution, `diff <(...) file`, fails silently here: diff two files.
   - **Memory gates parallel runs (11).** This machine has 25.8 GB usable and sits above 20 GB in
     use, so parallel runs are gated on available memory. A full run peaks near 1.2 GB, a `--bias`
     run too; eleven at once (about 13 GB) tripped Claude Code's memory guard. That guard stops
     `run_in_background` shells when memory runs low, and on Windows it stops only the shell and
     leaves the Python running. Long runs go in a runner script started in its own minimized
     console (`Start-Process -WindowStyle Minimized`), one at a time, each started only while 3
     GB is available, resumable from its results files, watched by a light background check of
     those files (`claude/scratch/midoffice11/runner.py`).

---

## Where the work stands

**The model's foundations** (generic, before the S&P 500 data):

- cb19460 (2026-11-01): the time-series (Fama-French) factor-model builder.
- 232b582: covariance estimators -- exponential weighting, Ledoit-Wolf shrinkage.
- 449ac95 (11-02): the cross-sectional (Barra) estimator and split-horizon covariance.
- 4fef869: the specific risk forecast from characteristics (Barra EQ 3-12 to 3-14).
- 7b160b5: the specific risk model wired into the cross-sectional estimator.
- 6dc0a99: Newey-West, Bartlett weights.
- 8a271d1: USE3's descriptors, the risk indices, the end-to-end pipeline.

Under them, not the model's: the risk core (98e7fdd to 260759b, 2026-10-21/22) and the value
layer (c92d343 to e859b4e, 2026-11-01).

**The S&P 500 model:**

- dc94fa6 (2026-10-16): point-in-time membership from Wikipedia's tables (`refdata/sp500.py`).
- 666599b (11-04): real data -- the Polygon loader, the 506-ticker daily panel, the model run.
- b9071d3: unit-basis exposures for readability, the covariance moved to match.
- c1275ab: industry members counted over the regression universe; 7cfe835, the SIC date
  fallback it needed.
- 17a588d: specific risk estimated over the regression universe; fallback forecasts labelled.
- f72310b (11-05): each period's regression held to that day's index members.
- 8cdd037: **midoffice01** -- exposures standardised over the estimation universe.
- f0ae2a9: Rule 43. 6e29b2c: its correction, the briefs written in Chat.
- 956331e (11-06): **midoffice02** -- each regression reads its inputs from t-1; an empty window
  gives no value.
- 08087bf (11-09): midoffice01's brief provenance corrected.
- 992f11a: **midoffice03** -- min_coverage 0.5; SERDP in no blend; the blend guard.
- 8c76c59: **midoffice04** -- USE4's market model; one blend per factor; USE4's Size
  Nonlinearity; the residual spread by the MAD.
- 1b9d43b: **midoffice05**, first change -- the size pair's standing line;
  `--size-nonlinearity none`.
- 701794e: **midoffice05** -- the identity table: members as securities, not tickers.
- 208ca80 (11-10): **midoffice06** -- the rebuild's fetch, the split classification, the
  distributions; stopped at its gate.
- 5ccbb70: **midoffice07** -- the identity corrections, the lag rule and the price-gap test;
  stopped at its gate.
- 7228b01: this file, and Rule 43's step that keeps it.
- cfe1245: **midoffice08** -- the build: the panel on securities, total returns, traded volumes,
  the three-step comparison.
- fccc480: **midoffice09** -- the count rulings, total returns by default, residuals for every
  covered name.
- 4f27f23: **midoffice10** -- PCG spared, spans ended off an exchange, the noise share measured.
- Next: **midoffice11**. Jeff commits it after 4f27f23 (`git log --oneline -3`).

**The current piece: midoffice11, COMPLETE (2026-11-12).**

- **midoffice10's rulings, each measured.**
  - **The correction's specific side** (`--leverage-correction`, off): the algebra holds, and
    the estimator's uniform inflation n/(n - k) is it at the mean leverage, 0.050. Construction's
    members average 0.156; the largest leverages are the mega caps (TSLA 0.357, AAPL 0.319, NVDA
    0.306), whose specific risk it raises 16 to 21%.
  - **The weighted noise share** is always computed, reported beside the equal-weighted one,
    and quoted by the qualification (Wholesale's 69%).
  - The two unused imports removed; a test that never ran (a duplicate name) now runs.
- **The out-of-sample bias statistics** (`--bias`, about ten minutes): 686 forecasts from
  2024-02-08, each the estimator refitted on data through its date alone, the scheme held; the
  guard (the inputs rebuilt from the panel cut at five dates, to the bit) passed.
  - **The defaults read 1.07 to 1.12 for broad portfolios at the daily horizon,** and that is
    Newey-West's long horizon: daily returns mean-revert, 10 lags hold the index's factor
    variance at 0.805 of its one-day variance (1.114 predicted); the 21-day check reads 0.88 to
    1.02, and without lags the daily reads 1.00 to 1.02. The members read 1.075 at the median,
    the largest names worst.
  - **The lag terms are most of the eigenvalue bias:** the smallest eigenportfolios read 2.2 with
    them, 1.25 without; the unconstrained minimum-variance portfolio 1.58 and 1.24.
  - **Floor 15 hurts** (Construction's portfolio 1.07 to 1.32); **the correction** hurts the
    optimiser's portfolio at both horizons (1.58 to 1.93) and centres the industries only at 21
    days; **a 63-day history half-life** helps every specific-side measure (members 1.075 to
    1.042); **the two-half-life covariance** (90 / 504) is the best covariance candidate; the two
    together add up.
- **Waiting on Jeff:** the decisions in the retro. My choices: the 63-day history and the two
  half-lives as the defaults; the correction and its specific side left off; no fold; the
  eigenvalue adjustment next, measured together with the Newey-West lag count.

---

## What is settled -- do not reopen

**The retros.** "NN" is `claude/retros/RETRO_midofficeNN_*.md`; "NN ruling k" is Jeff's ruling k
at the top of the brief for the piece after NN. Pre-series commits have no retro; their detail is
in `docs/FACTOR_MODEL.md`.

**Membership and the estimation universe**

- The estimation universe on day t is that day's members passing the data test. It alone sets
  every scale: clip bounds, cap-weighted mean, equal-weighted sd. Outsiders get exposures on the
  members' scale. One membership array goes to every consumer. (01)
- Each period's regression holds to that day's members. Industry counts are taken over the
  regression universe, and the specific risk model is fitted on the regression's residuals
  alone. (f72310b, c1275ab, 17a588d; 09)
- membership():
  - reads the constituent list brought up to the change table (current_stale);
  - with dated renames from the identity table, reads the change table in today's tickers --
    ECHO's phantom days go. (05, 07)
- The panel on securities carries membership read through the table's renames and mapped to
  securities, a reorganisation's member-days under its successor's column; the run uses it, and
  prints current_stale on either panel. (05 ruling 3; built in 08)

**Time and coverage**

- **The lag.** Period t's regression reads its exposures, their standardisation, its weights and
  its membership from t-1, and only its returns from t; the forecast exposures are the last
  date's. The estimator's lag defaults to 1, and test_lookahead's repricing guard fails at 0.
  (02)
- **Coverage.** A window with no valid observation gives no value. min_coverage is 0.5: a window
  descriptor needs half its window valid; 0 stays runnable. (02, 03)
- **Share counts** are carried forward from the last sample, never back-filled. (02)
- **fit_index_weights** is lagged and whole-sample, and the run does not call it. (03)

**Definitions: USE3's by default, USE4's where USE3's choice measures badly, each switchable (04)**

- **The market model** (BTSG, HALPHA, SERDP, VLVR's residual sd): on a daily panel USE4's, 252
  trading days at half-life 63; on other panels USE3's 60 months. `--beta-half-life` takes inf
  for equal weights. (04; 04 ruling 3)
- **Size Nonlinearity**, USE4's: Size cubed, standardised and clipped at +/-4, orthogonalised
  against Size by the regression weights over the members, standardised again. It stays in the
  model, with "use3" and "none" runnable; a guard stops the run unless it is orthogonal. (04; 04
  rulings 1-2)
- **One definition per factor:** a period is estimated only if every index's blend has exactly
  the forecast's descriptors. It is asserted; 938 periods from 2023-02-08; `--mixed-blends`
  overrides. (04)
- **SERDP** is in no blend. A descriptor in the forecast and in no estimated period stops the run.
  (03)
- **RSTR** is USE3's twelve months, skip 0; `--rstr-skip` cuts the academic 12-1. USE4's lag was
  measured and not built. (03, 04)
- **EQ 3-14's residual spread** is 1.4826 x the MAD; `--residual-spread sd` stays runnable. (04;
  04 ruling 4)
- **Weights and industries.** Regression weights are sqrt(cap), and the standardisation's mean
  is cap-weighted. Industries come from SIC, groups under 15 merged into their division, with no
  market factor beside them. Newey-West takes 10 lags at half-life 90. (666599b; docs) On the
  panel on securities, each security's SIC code is its own latest record's. (08)
- **Noise shares** get one model-wide answer after the rebuild, never a deletion per factor (04
  ruling 1). Both are always computed and the run's factor table reports them side by side --
  equal-weighted (Construction 38.2%, Wholesale 46.2%, Finance & Real Estate 31.0%, Size
  Nonlinearity 46.6%) and weighted as the covariance weights the periods (48.6%, 68.8%, 46.5%,
  67.4%) -- and the qualification quotes the weighted one. (10 ruling 2; built in 11)
- **The sampling-error correction** is an option, off. It needs a covariance estimator that is
  a quadratic form -- sample, exponential or Newey-West -- and refuses the others; the
  two-half-life estimator's diagonal is one, so it has a weighted share but not the correction.
  The eigenvalue adjustment waited for the bias statistics. (10; 10 ruling 1)
- **The correction's specific side** (`leverage_correction`, `--leverage-correction`) is an
  option, off: each member's residual over sqrt(1 - h) in place of the uniform inflation, each
  specific return outside the regression over sqrt(1 + h). (10 ruling 1; built in 11)
- **An industry fold** (`merge_industries`, `--merge-industry A=B`) is an option, none by
  default; folds are recorded in `folded`, apart from the grouping's `merged`. No fold for now;
  floor 15 by the Pearson rule was the candidate the bias statistics judged. (10; 10 ruling 3)
- **Size Nonlinearity** stays (04 rulings 1-2; 10 ruling 4). **FRCB's 2023-05-04 return** is
  dropped, an OTC price after the index let it go (10 ruling 5).
- **The bias statistics' method** (11's brief): each forecast the estimator refitted on data
  through its date alone, daily (`cross_sectional_models_as_of`); the industry scheme held fixed,
  as Barra fixes a model's industry factors; the first forecast once the covariance has 252
  estimated periods; the guard on a handful of dates (`--bias-guard`, five); each forecast at a
  daily scale against the next day's total return, the 21-day horizon non-overlapping as a
  check.
- **A security's own specific-risk history** is every specific return it has, member or not:
  in each estimated period, the return of every security with a return and a complete
  characteristic row less its exposures times the members' factor returns, with no
  degrees-of-freedom inflation outside the regression. A security in no regression with a
  history of its own is blended, not a fallback. `--specific-history regression` keeps the
  regression's residuals alone. (09; `specific_history="covered"`, the library's default still
  "regression")

**The identity table (05, 06, 07, 08)**

- **Keys.** A security is a share class, keyed by composite FIGI, else CIK; the CIK is the
  cross-check. A record with neither is no evidence. (05; 06 ruling 1)
- **The share-class rule.** A record naming another composite FIGI names the expected security
  if its share-class FIGI is one that security's own records carry (and no CIK on it
  contradicts theirs). It is a check in the gate, not a re-keying: the table stays keyed by
  composite FIGI. (08 ruling 1; `rebuild.reconciles`)
- **Sources.** The table is built from the panel fetch's own records (its 252-day grid), Polygon's
  ticker events and the panel's prices. The rebuild's share samples check it and never build it.
  (05, 07)
- **The endpoint.** Member-days neither source names go to the reference endpoint on public-record
  hypotheses, and only what it confirms enters. WBD is its own security. (05 ruling 2; 06; 06
  ruling 3)
- **Splices and mergers.** A splice is two securities. A merger's predecessor (WestRock,
  Paramount) is another security, linked for membership. (05, 06)
- **A reorganisation** (a new identifier, the same ticker, the prices unbroken) is one security
  with one return series. It is keyed by the successor, the predecessor linked, and its boundary
  is dated by the key's identifier: Bunge by CIK, Vylor by FIGI. (05 ruling 1; 06 ruling 1)
- **Spans:**
  - a span starts at its first price under its ticker;
  - a span with no price goes, and so does one its records refute: S600, CEGV, CRHC;
  - CRH's ADR history before 2023-09-25 stays out;
  - II-VI's IIVI history, 2022-01-03 to 2022-09-07, is Coherent Corp.'s;
  - Coherent, Inc. holds COHR from 2022-01-03 to 2022-07-01, BBG000BG1DH3, its last price
    2022-06-30 (`Confirmation.last_price`). (06 ruling 1; 07; 08 ruling 1)
- **The identity check** reads each span's first sample with an identifier. It reads past a
  change-day record naming the ticker's previous holder (code since 07). (05 ruling 5; 06, 07)
- **The fetch** covers the whole window for every member security. A span-plus-look-back cut is
  to be a parameter for future panels; it is not built. (05 ruling 4)

**Share counts and splits (06, 07, 08, 09)**

- **Sampling.** Counts are sampled inside each security's own spans: each span's first and last
  day, and a 63-day grid kept in its phase.
- **The first count.** A span that starts without a count, none carried into it, has its first
  quarter searched by bisection for its first record with one. That count stands only if the
  span's next sample confirms it within 10%; otherwise the search goes on to the first record
  the next sample confirms, and the days before it have no count. Only the found record enters,
  not the search's others. (08 ruling 2; 08 ruling 1 for midoffice09; `--first-counts`)
- **Isolated errors.** A sample whose two samples before and two after agree with each other
  within 10%, and which differs from each beyond it with no applied split between, is not
  carried; the count before it carries through. One pass; alternating counts are left and stay
  in the count-move report. Every one found is listed. (08 ruling 2 for midoffice09;
  `--repair`)
- **The weighted spare.** A sample its own record's weighted count confirms within 10% is not
  an error: PCG's of 2026-01-09. Its series stays flagged as alternating. (09 ruling 1; built in
  10, `--spare`)
- **The overrides file**, `config/sp500_share_count_overrides.json`, in the repo, not `data/`:
  rulings on single securities, each a key, a range, "carry" a count or "none", and the
  evidence; samples inside a range are not read. The build lists every entry it uses. A new
  anomaly goes in the file with its evidence, not into a rule for one case. (08 ruling 3 for
  midoffice09; `--overrides`)
- **The hindsight.** Rulings 1 and 2 recognise an error by a later sample: hindsight in cleaning
  the data, not in what the model knows, since the true counts were public at the time and a
  wrong count carries no information about the day's return. Real changes are adopted when the
  samples show them and never back-filled. (08 rulings, for midoffice09)
- **Weighted counts are never data.** They are a standing check: wherever a record carries one,
  against the count the panel carries that day, flagged beyond 10%. (08 ruling 2)
- **Reorganisations' counts.** A successor carries its predecessor's count until its own first
  sample, never back-filled: Carnival's 1.1054. OKE's and XOM's carry 1:1, flagged -- and the
  weighted check confirms them, 1.0001 and 0.9920. (06 ruling 4; 08)
- **Split events** are classified by the ratio -- "whole" means no term over 100 in lowest terms
  -- and two tests within 10%: the counts either side, and the raw opening gap on the execution
  date against 1/ratio. (06 and its two amendments; 06 ruling 2)
  - **A split** is whole, and the counts follow it (or, where the counts cannot speak, the price
    gaps by it). It is applied to the carried count on its execution date.
  - **Price-only** is fractional with the counts unchanged: a spin-off, carried by the adjusted
    prices.
  - **A contradiction** -- a test disagreeing, or a split whose price does not gap -- stops the
    gate.
- **The lag rule.** A sample on or after a split still at the pre-split count is stale and does
  not reset the carried count; the first that follows the ratio does. (06 ruling 2)
- **The split guard**, built and exact: no day on or after a split carries a pre-split count,
  and the execution date carries the count before it times the ratio. (08)

**Prices, returns and volumes**

- **The panel on securities** (08):
  - its columns are securities, labelled by their last ticker;
  - its calendar is SPY's adjusted bars, the old panel's;
  - prices are stitched: within one ticker's run, its adjusted closes times one scale; across a
    change of ticker, the raw closes' return times any split between;
  - splices are separate columns, a reorganisation one.

  It is the run's panel by default; the old one, on tickers, stays runnable for comparison.
  (08 ruling 5 for midoffice09)
- **Total returns** are the run's default; price returns stay runnable (08 ruling 4 for
  midoffice09). They are:
  - every cash distribution, specials included, put on the prices' split basis by the ex-date's
    adjusted over raw close;
  - an exact duplicate kept once, TEL's of 2024-11-22 the only one; SPY's distributions too;
  - in the comparison, identity first, then dividends. (06's second amendment; 06 ruling 5;
    07's brief)
- **Volumes.** The panel on securities stores traded volumes, the raw bars', with the adjusted
  bars' beside them for comparison; the old panel stored the adjusted ones. (07; 08 ruling 3)
- **LEN's 2025-01-21 adjustment** is left as Polygon has it, flagged. (06 ruling 6)
- **Spans end off an exchange.** A span ends where its security stops trading on an exchange: its
  first sample whose record's `market` is not "stocks", but for any day the index still held it.
  So no member-day changes; every span ended is listed with the priced days it removes. A leaver
  that stays listed keeps forecasting from its own history. (09 ruling 2; built in 10,
  `--listed`)
- **A specific-risk history is equally weighted** over the window, for now. (09 ruling 3)

---

## The data

All under `data/`, which is gitignored. Nothing there is overwritten or deleted without Jeff.

- **The old panel:** `data/external/polygon/sp500_daily_2022-01-03_2026-11-04.npz`.
  - 506 tickers -- the index on 2022-01-03 -- over 1,214 days to 2026-11-03, share counts
    sampled every 252 days.
  - Regenerate: `python scripts/fetch_sp500_panel.py --start 2022-01-03 --end 2026-11-04
    --shares-every 252`. From the cache it reproduces every array exactly (checked 2026-11-10).
- **The membership snapshot:** `data/external/sp500/sp500_20261104T033741Z_660dc8159b0a.json`.
  - Runs use the newest snapshot in the directory; the table and the panel on securities record
    the hash they were built on, and the build and the run stop on another.
  - Validate: `python -m praxis.midoffice.refdata.sp500 check --dir data/external/sp500`.
  - `... fetch --out data/external/sp500` writes a new one, which every run would then use: only
    to roll membership forward, and the table and the panel rebuilt after.
- **The Polygon cache:** `data/external/polygon/cache/`, about 15,360 answers, found or not:
  `bars-`, `ref-`, `events-`, `splits-`, `dividends-`.
  - Every script reads it first, so reruns ask nothing; midoffice09 added the 9 records the
    confirmed first-count search asked for.
  - It holds the evidence; it can only be rebuilt by asking again.
- **The identity table and fetch plan, current: `_amended`** (midoffice08).
  - `data/external/polygon/sp500_identity_2022-01-03_2026-11-03_amended*`: the table `.csv`,
    `.json`, `_members.csv`, `_fetch.csv` (the plan), `_links.csv` (reorganisations and mergers),
    `_membership.csv`.
  - Regenerate: `python scripts/build_sp500_identity.py
    data/external/polygon/sp500_daily_2022-01-03_2026-11-04.npz --resolve --correct --amend
    --offline`. Without `--amend` it writes midoffice07's `_corrected` files; without `--correct`,
    midoffice06's `_resolved`; without either, midoffice05's. Each rewrites its own byte for byte
    (checked 2026-11-10).
- **The rebuild's fetch and gate:** `python scripts/fetch_sp500_rebuild.py --offline`.
  - The default plan is the amended one; without `--offline` it fetches what is missing.
  - Report: `data/external/polygon/sp500_rebuild_gate_2022-01-03_2026-11-03_amended.json`.
    midoffice07's is `_corrected`, midoffice06's unsuffixed. Run on another plan, it writes that
    plan's report: given the `_corrected` plan, it would overwrite midoffice07's.
- **The panel on securities, current: `_listed`** (midoffice10):
  `data/external/polygon/sp500_securities_2022-01-03_2026-11-03_listed.npz`, and its report
  `.json`.
  - 592 securities -- each a share class, its key in `securities`, its label its last ticker --
    over the same 1,214 days.
  - Arrays: prices, total_return_prices, unadjusted_prices, volumes (traded), adjusted_volumes,
    shares_outstanding, highs, lows, tickers (each cell's), membership, market_returns,
    market_total_returns, sic_codes and sic_dates, with the snapshot hash and the plan.
  - Regenerate: `python scripts/build_sp500_securities.py --offline`, about 40 s. It runs the gate
    offline first and builds only if it passes, and it refuses to write over an existing panel
    without `--overwrite`.
  - midoffice09's panel is `_repaired`: `--spare none --listed any --suffix _repaired` rebuilds
    its arrays exactly (checked 2026-11-10).
  - midoffice08's panel, before the count rulings, is the unsuffixed one: `--first-counts found
    --repair none --overrides none --spare none --listed any --suffix ""`.
- **The overrides file**, in the repo: `config/sp500_share_count_overrides.json` -- HON from
  2026-10-12 carries 635,675,701; SKYD from 2026-10-06 has no count. Each entry's `last` is the
  last day its evidence reaches.

---

## The run

```
python scripts/run_sp500_model.py --industries --membership
```

That is the panel on securities with its counts repaired and its spans listed
(`DEFAULT_PANEL`, `_listed`), total returns, traded volumes, and covered histories. The old panel, for comparison, runs as below, and with
`--specific-history regression` too it reproduces midoffice08's run A exactly:

```
python scripts/run_sp500_model.py data/external/polygon/sp500_daily_2022-01-03_2026-11-04.npz --industries --membership --returns price
```

**The switches:**

- `--returns total` (the default) or `price`, the market's to match.
- `--volumes traded` (on securities, the default, the stored) or `adjusted`.
- `--specific-history covered` (the default) or `regression`.
- `--sampling-correction`: the factor covariance less its sampling covariance, weighted as the
  covariance is, and repaired; its table and the repair are reported. Off by default.
- `--merge-industry A=B`, repeatable: fold industry A into B once the scheme is built. None by
  default.
- `--leverage-correction`: the correction's specific side. Off by default.
- `--specific-half-life H`: each specific-risk history weighted by a half-life of H trading days,
  the blend told its effective number of observations. Equal weights by default.
- `--correlation-half-life H`: the two-half-life factor covariance, correlations on H,
  volatilities on `--half-life`, both Newey-West at `--newey-west-lags`. One half-life by
  default.
- `--bias`: the out-of-sample bias statistics after the report, about ten minutes alone on this
  machine; `--bias-out PATH` keeps every forecast and outcome in an `.npz`; `--bias-guard N`
  (default 5, 0 skips) is the guard's date count, about four minutes at five.
- On the panel on securities, `--membership` is its own, checked against the newest snapshot's
  hash, and `--industries` reads its own SIC codes and needs no key.
- The old panel carries price returns and adjusted volumes alone. Its `--industries` reads the
  cache and needs `.env`'s `POLYGON_API_KEY`.

**At the defaults:**

- **Time:** about 50 to 75 s.
- **Mean R-squared:** 0.293203 over 938 periods from 2023-02-08.
- **Members in the regression a period:** median 502, range 499 to 503.
- **Specific risk:** p10 15.58%, median 24.76%, p90 40.49%, 550 forecasts.
- **The size pair:** 11.87% for the smallest 85, 0.62% for the largest 50, 0.01% for the index.

The old panel, `--returns price --specific-history regression`: 0.298076, 451 (422-476),
11.94 / 0.38 / 0.01%.

**Its standing lines, and what each is there to catch:**

- **The snapshot.** Its hash and coverage; current_stale, the roll-forward; the members in the
  panel on the first and last days. A change membership() does not reproduce is the change table
  disagreeing with itself. A date the snapshot cannot answer stops the run, and so does a panel
  on securities built on another snapshot.
- **The market model's window and half-life:** which beta the model is using.
- **Coverage:** values from an empty window, or from under min_coverage, counted against each
  descriptor's own inputs independently. It must read "none".
- **Outsiders at the members' bounds:** the estimation universe setting the scale.
- **Each index's blend over the estimated periods.** Guard: it stops if a descriptor feeds the
  forecast and no estimated period, or if an estimated period's blend is not the forecast's.
- **Style exposures over the members on every date read:** cap-weighted mean zero and sd one, to
  rounding. The index at zero exposure; `!` if not.
- **Size Nonlinearity's regression-weighted covariance with Size.** Guard: it stops past
  rounding.
- **The size pair's portfolio line** -- smallest 85, largest 50, the index: where size risk sits.
- **Same-day membership:** the security-periods that move against it. The lag in effect.
- **The industry universe against the regression's:** counts taken over the wrong set; `!` if
  they differ.
- **Securities per estimated period** -- members, in the panel, in the regression -- **and the
  members kept out, by why:**
  - not in the panel;
  - no weight;
  - no return;
  - an exposure missing, by the first factor without one.
- **The factor table's noise shares, equal-weighted and weighted, and its exactly-1 and none
  columns:** thin and one-member factors, and how much of the current regime's variance is noise.
- **Under `--bias`, its guard** (the run stops if the inputs rebuilt from the panel cut at a date
  are not the full run's through it) and **its statistics**: the portfolios' table at both
  horizons, the industries against their weighted noise shares, the random portfolios, every
  member singly, the eigenportfolios.
- **Specific risk:**
  - the percentiles;
  - the shares of V-squared;
  - the residual spread, both MAD and sd, and its shares;
  - under covered histories, the specific returns measured outside the regressions and the
    securities blended outside them;
  - the characteristics' weight in a blended forecast.
  Together: who drives the specific risk model.
- **Qualifications:** dropped periods, excluded securities, fallback forecasts, and the
  securities in no regression blended with their own history.

**The other guards and standing checks:**

- `build_sp500_identity.py` exits non-zero on a failed guard:
  - each priced ticker-date belongs to one security;
  - no security trades under two tickers on one date;
  - no ticker names two securities on one date;
  - every member-day is mapped or named.
- `fetch_sp500_rebuild.py` exits non-zero when its gate stops.
- `build_sp500_securities.py` refuses to build unless the gate passes, and exits non-zero on a
  failed guard:
  - each (ticker, date) price feeds one column;
  - no column takes two tickers on one date;
  - every member-day is a column's, inside its spans or on a day the table maps outside them;
  - every bar of a span is its column's, and only those;
  - the walk through the renames and the panel's membership are the same member-days;
  - the split guard, exact.

  Its reports list:
  - every first count it passes over;
  - every isolated error it finds, removed or spared, with its weighted count;
  - every span it ends off an exchange, with the priced days removed;
  - every override it uses.

  Its standing checks are:
  - the weighted check -- carried counts listed in full, own records by security;
  - every count move beyond 10% that no split explains, 86 now.

---

## What is next

1. **Jeff's rulings on midoffice11** (its retro, "Decisions left for Jeff"):
   - **The specific-risk history:** a 63-day half-life as the default (my choice), 126, or equal
     weights; or shorter ones measured first.
   - **The factor covariance:** the two-half-life estimator, volatilities 90 and correlations
     504, as the default (my choice), or the single half-life 90.
   - **The horizon:** keep Newey-West's long horizon, its daily forecasts about 10% short for
     factor risk, as the model's horizon (my choice), or not.
   - **The eigenvalue adjustment:** commission it next (my choice), measured together with the
     Newey-West lag count, since the lag terms are most of the eigenvalue bias.
   - **The correction and its specific side:** left options, off (my choice).
   - **The thin industries:** no fold (my choice); Construction is two industries if a scheme
     change is wanted.
2. **The bias statistics are standing:** every later candidate is judged by `--bias` against
   the defaults, run one at a time through a runner like midoffice11's.

**Open, not queued:**

- fit_index_weights is whole-sample, though unused;
- VLVR runs the market model a second time;
- the span-plus-look-back cut;
- `tests/test_market_data.py`'s missing `../core_repo/src`;
- six ruff findings at HEAD in files no mid-office piece has touched since --
  `characteristics.py` (an E501, an unused `Sequence`), `exposure_basis.py` (an unused
  `Mapping`, two N806 `T`), `timeseries.py` (an unused `Sequence`); checked in 11
  (`ruff_head_untouched.txt`);
- shorter specific-risk half-lives than 63, and whether the latest half's slight over-forecast
  at 63 grows (11);
- the sampling covariance is the classical `sigma^2 (X'WX)^-1`; a heteroskedasticity-robust
  sandwich would say whether it misstates a thin industry's noise (10);
- a member floor is not a noise floor: SIC20, with 21 members, is 55% noise, weighted (10);
- FDXF and Vylor carry no share-class count in the window, so they enter no regression;
- PCG's alternating counts, which no rule repairs.
