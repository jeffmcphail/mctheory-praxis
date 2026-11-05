# Retro: Mid-office 01 -- standardise over index members (the estimation universe)

**Brief:** `claude/handoffs/BRIEF_midoffice01_standardise_over_members.md` -- Jeff's request,
transcribed verbatim; it was given in the Code session, not as a Chat-written brief
**Date:** 2026-11-05
**Duration:** about 45 minutes of implementation and measurement (12:05-12:45 EST by the file
times), then this retro
**Status:** COMPLETE, uncommitted. The one decision left open (LCAPCB's normalisation) was ruled
on by Jeff after the work: keep it over members (variant D).
**Series:** the first retro of the `midofficeNN` series (Rule 43). The S&P 500 model work before
it has no retros: c1275ab + 7cfe835 (industry counts over the regression universe, SIC
fallback), 17a588d (specific risk over the regression universe), f72310b (each period's
regression held to that day's index members).

---

## Summary

Every standardisation in the S&P 500 model now runs over that day's index members -- the
estimation universe -- and every other security with a value is put on the members' scale. One
membership array, built once, goes to LCAPCB's normalisation, the descriptor standardisation
(clip bounds, cap-weighted mean, equal-weighted SD), the descriptor screen, each risk index's
second standardisation, `fit_index_weights`, the regression universe, the estimator (its own
standardise path included) and a new exact guard. The guard holds: on all 939 estimated periods
the cap-weighted mean of each style exposure over members is within 2.0e-16 of zero (0.9 eps),
and the equal-weighted SD within 3.3e-16 of one; on HEAD's exposures it fails on all 4,695
style-periods, by up to 0.0142. Mean R-squared moves -0.000735, inside Jeff's "a few
thousandths". His volatility prediction held exactly for the changes he listed (variant C: Size
-12.0%, Size Nonlinearity -15.0%); the surprise came from the one step he had not listed,
re-centring the LCAPCB cube on the members, which re-splits the size effect between two factors
whose exposures are 0.93 correlated (net, variant D: Size -3.3%, Size Nonlinearity +0.0%). Jeff
ruled to keep it. Without `--membership` the run report is byte-identical to HEAD's apart from
the new guard lines.

---

## What was asked (verbatim)

```text
Standardise over members now, before the rebuild. Same principle as the
mask: the estimation universe on day t is that day's members passing the
data test, and nothing outside it should set the scale of anything inside
it -- Barra's estimation universe versus its coverage universe.

1. Build the membership mask once, as a date-by-security array, and hand
   that one array to the regression and to standardisation. Re-reading the
   change table on all 1,214 calls costs 13 seconds now and would double
   with a second consumer -- but the real reason is that one array used
   everywhere means the two cannot disagree about who is a member.
   Without --membership, behaviour is unchanged.

2. Per period, over the estimation universe only: the MAD clip bounds, the
   cap-weighted mean and the equal-weighted SD. Both standardisations in
   blend_to_index, fit_index_weights and the descriptor screen use the
   same set.

3. Securities outside the estimation universe that have a characteristic
   row get exposures on the members' scale -- the members' mean, SD and
   clip bounds. Report how many hit a bound; the ex-members sit in the
   tails, so some will.

4. Exact guard on every estimated period: the cap-weighted mean of each
   style exposure over the estimation universe is zero to machine
   precision. That is what gives the index itself zero style exposure, so
   an exposure reads as relative to the S&P 500. It is within 0.013 now.
   If the post-standardisation clip makes it inexact, report by how much
   rather than loosening the tolerance.

Then rerun with --membership, before and after: R-squared, the five style
factor vols, the members' cap-weighted means and equal-weighted SDs on
2026-11-03, and how many non-members were clipped.

What I expect, so a surprise is visible. This is close to a
reparameterisation. A per-period affine change to a style column is
absorbed exactly -- the scale by the column's own coefficient, the shift
by the industry columns, which sum to the column of ones -- so fitted
values move only through the parts that are not affine: the clip bounds,
the descriptor blend, and the few names with no industry row. The MAD
clip is robust to tail names by construction, so the bounds should barely
move. R-squared should move by a few thousandths at most. Style factor
vols should fall roughly in proportion to the members' spread: about 12%
for Size and 15% for Size Nonlinearity. If R-squared moves materially,
the descriptor blend is doing more than rescaling and I want to know why
before it is committed.
```

### The ruling received after the work (verbatim)

```text
Keep D. If the full panel sets the centre of the cube, the ex-members
are still shaping the model through the one door left open. The
instability you found is real, but it belongs to the size pair, and
that decision is being made separately.
```

---

## Files modified

| File | Change | Lines (new file) |
|---|---|---|
| `src/praxis/midoffice/model/characteristics.py` | `_checked_membership`, the one membership validator; `CharacteristicPanel.clipped_outside`; `standardise(membership=)` -- bounds, mean and SD over members, everyone else on their scale, a day with no member value left missing, outsiders counted by either bound; the +/-4 guard comment corrected; module docstring paragraph | 33-39, 53-69, 92-100, 193-197, 235-246, 257-333 |
| `src/praxis/midoffice/model/descriptors.py` | `membership=` on `compute_descriptors` (LCAPCB's normalisation), `_restandardise`, `blend_to_index`, `build_risk_indices` and `fit_index_weights`, with docstrings | 39, 262-297, 350-363, 549-579, 590-625, 649-671, 697-748 |
| `src/praxis/midoffice/model/crosssectional.py` | `_membership_block` uses the shared validator; the estimator's own standardise path takes the same membership; docstrings | 59-61, 87, 426-432, 530-533, 566-581, 594 |
| `scripts/run_sp500_model.py` | membership block moved ahead of the descriptors; the one array handed to every consumer; `screen(membership=)`; new `clipped_outside_on` and `estimation_moments`; the outsiders' bound report; the exact guard and the moments table | 77-100, 142-204, 282-361, 377-392, 473-508 |
| `tests/midoffice/test_crosssectional.py` | 8 new tests, section "the estimation universe: whose values set the scale" | 1145-1293 |
| `tests/midoffice/test_descriptors.py` | `import dataclasses`; 5 new tests, same section name | 16, 576-687 |
| `docs/FACTOR_MODEL.md` | new section "Whose values set the scale: the estimation universe"; "nearly always a no-op" corrected; LCAPCB bullet | 352-356, 362-383, 571-573 |
| `claude/handoffs/BRIEF_midoffice01_standardise_over_members.md` | new: the request, verbatim | new |
| `claude/retros/RETRO_midoffice01_standardise_over_members.md` | new: this retro | new |

---

## What changed and why

### The principle, as implemented

The estimation universe on day t is that day's members passing the data test, and nothing
outside it sets the scale of anything inside it. Given a membership:

- **`CharacteristicPanel.standardise`.** Per date and characteristic, the raw clip bounds (median
  +/- 4 x 1.4826 x MAD, falling back to the SD when the MAD is zero), the cap-weighted mean and the
  equal-weighted SD (ddof=1) are computed over the members with a value. Every security with a
  value, member or not, is then clipped to those bounds, centred on that mean, divided by that SD
  and clipped at +/-4. The data test per characteristic is unchanged -- a value present --
  and membership is one more condition, as in the regression mask.
- **A day on which no member has a value** leaves that characteristic missing that day: otherwise
  it would be standardised against the outsiders alone. In the second pass, fewer than two
  members with a value leaves the index missing that day.
- **`_restandardise`**, each risk index's second pass: the cap-weighted mean and SD over members
  with a value; everyone with a value is put on that scale. No clip follows it, which is why the
  style exposures come out exact.
- **`compute_descriptors(membership=)`**: LCAPCB = ((log cap - members' mean) / members' SD)^3, the
  one cross-sectional step inside a descriptor definition (equal-weighted mean, as before).
- **`fit_index_weights(membership=)`**: the pooled fit runs over member cells only.
- **The descriptor screen** regresses members only, by membership on the return's day (t+1).
- **`estimate_cross_sectional_model(standardise=True, membership=)`**: the estimator's own
  standardisation takes the same array, mapped onto the characteristic panel; characteristic
  cells outside the return panel have no membership and set no scale.
- **One validator** (`_checked_membership`) behind every entry point: boolean only -- a 0/1 float
  is refused, since a NaN in one is truthy -- and the shape checked.

Without a membership every path is unchanged: the no-membership run report equals HEAD's except
for the new guard lines, and in the tests a membership holding everyone gives bit-identical
values.

### The run script

- The membership block moved ahead of the descriptors, so the one array exists before anything
  is standardised. It is still built once -- 1,214 `membership()` calls, about 13 s -- and every
  consumer takes that same array.
- New report lines: outsiders' bound hits, panel-wide and per descriptor on the exposure date
  (read off `standardise` run on that day and descriptor alone, which is exact, because no day's
  or descriptor's standardisation looks at another's); the exact guard; the members' cap-weighted
  mean, equal-weighted SD and n per style on the last estimated period.
- The guard is worked out in the script from the final exposures, independently of the code that
  standardised them. A mean past the first-order rounding bound -- n x eps x the cap-weighted mean
  |x| -- prints a `!` line with the count, the largest, its factor and its date. The tolerance was
  not loosened; nothing after the second standardisation clips, so nothing should move it.

### Docs

`docs/FACTOR_MODEL.md` gains "Whose values set the scale: the estimation universe". Its claim
that the post-standardisation clip "is nearly always a no-op" is corrected: true for members
(3,043 of their values over 1,214 days), not for the securities put on their scale. The LCAPCB
bullet now says the normalisation is over members, and why centring matters for a cube.

---

## Results

Every run is `python scripts/run_sp500_model.py
data/external/polygon/sp500_daily_2022-01-03_2026-11-04.npz --industries --membership`, unless
stated. "HEAD" is f72310b's code, "now" this change.

### The four things asked for

| | HEAD (f72310b) | now | change |
|---|---|---|---|
| Mean cross-sectional R-squared | 0.312985 | 0.312250 | -0.000735 |
| Volatility vol | 10.6489% | 9.5468% | -10.3% |
| Momentum vol | 7.5021% | 7.0575% | -5.9% |
| Size vol | 7.7614% | 7.5062% | -3.3% |
| Size Nonlinearity vol | 6.5487% | 6.5506% | +0.0% |
| Trading Activity vol | 5.6630% | 5.1644% | -8.8% |

| Members on 2026-11-03 | HEAD cap-wtd mean | HEAD eq-wtd SD | now cap-wtd mean | now eq-wtd SD | n |
|---|---|---|---|---|---|
| Volatility | -9.8e-04 | 0.891546 | -7.7e-18 | 1.000000 | 426 |
| Momentum | +4.8e-03 | 0.957262 | +2.3e-17 | 1.000000 | 426 |
| Size | +1.3e-02 | 0.882585 | -3.1e-17 | 1.000000 | 425 |
| Size Nonlinearity | +1.2e-02 | 0.855386 | -7.7e-18 | 1.000000 | 425 |
| Trading Activity | -3.5e-03 | 0.875530 | +3.9e-17 | 1.000000 | 426 |

The guard, over 939 estimated periods x 5 styles = 4,695 style-periods:

- **now:** cap-weighted mean at most 2.0e-16 from zero (0.9 eps); equal-weighted SD at most
  3.3e-16 from one.
- **HEAD's exposures, same guard:** `! 4695 of 4695 style-period means are past rounding`; the
  largest +0.0142, Size on 2026-08-04; SDs up to 1.8e-01 from one.
- **now, without `--membership`:** at most 2.1e-16 (0.9 eps), SD at most 2.2e-16 from one --
  exact over everyone too.

How many non-members were clipped:

- **100,018** values of securities outside the index hit one of the members' bounds, over every
  date: the raw clip bounds or the +/-4 bound after standardising, each value counted once.
- Raw values winsorised in all: 348,002 now against 314,921 at HEAD. The bounds now come from
  the members, so the outsiders in the tails are clipped more often.
- On 2026-11-03, per descriptor -- outsiders that hit a bound / outsiders with a value:

| LNCAP | LCAPCB | LPRI | DASTD | HILO | CMRA | BTSG |
|---|---|---|---|---|---|---|
| 19/45 | 45/45 | 4/45 | 4/47 | 8/45 | 5/77 | 8/66 |

| HALPHA | SERDP | RSTR | STOM | STOQ | STOA | VLVR |
|---|---|---|---|---|---|---|
| 7/66 | 1/66 | 3/80 | 13/80 | 14/80 | 16/80 | 18/54 |

- **LNCAP is where the second bound binds.** Of its 45 outsiders with a price, 19 hit the -4
  bound after standardising, one of them the raw bound too. The cap-weighted mean sits so far
  above the median that the lower raw bound itself standardises past -4.
- **LCAPCB: every outsider, 45 of 45, hits its bound** (see the LCAPCB findings).

### Every factor, HEAD -> now

| Factor | vol HEAD | vol now | change | noise share HEAD | noise share now | change |
|---|---|---|---|---|---|---|
| Construction | 17.29% | 17.38% | +0.6% | 49.7% | 49.3% | -0.41 pt |
| Finance & Real Estate | 15.65% | 15.70% | +0.3% | 13.5% | 13.5% | -0.06 pt |
| Manufacturing | 11.54% | 11.46% | -0.7% | 18.3% | 18.5% | +0.26 pt |
| Mining | 21.11% | 20.95% | -0.8% | 14.2% | 14.5% | +0.25 pt |
| Retail | 13.57% | 13.69% | +0.8% | 9.7% | 9.6% | -0.14 pt |
| SIC20 | 15.42% | 15.38% | -0.3% | 20.7% | 20.9% | +0.20 pt |
| SIC28 | 14.85% | 14.77% | -0.5% | 7.8% | 7.9% | +0.09 pt |
| SIC35 | 16.98% | 17.10% | +0.7% | 6.5% | 6.4% | -0.08 pt |
| SIC36 | 20.35% | 20.34% | -0.0% | 5.3% | 5.4% | +0.02 pt |
| SIC37 | 14.21% | 14.29% | +0.5% | 17.6% | 17.4% | -0.21 pt |
| SIC38 | 14.40% | 14.46% | +0.4% | 10.9% | 10.8% | -0.07 pt |
| SIC49 | 14.61% | 14.56% | -0.3% | 14.0% | 14.2% | +0.19 pt |
| SIC60 | 20.81% | 20.87% | +0.3% | 8.0% | 8.0% | +0.00 pt |
| SIC63 | 15.91% | 15.85% | -0.4% | 11.7% | 11.9% | +0.15 pt |
| SIC67 | 15.51% | 15.66% | +1.0% | 16.2% | 15.9% | -0.24 pt |
| SIC73 | 16.63% | 16.55% | -0.5% | 3.7% | 3.8% | +0.05 pt |
| Services | 13.92% | 13.87% | -0.3% | 22.2% | 22.4% | +0.20 pt |
| Transport & Utilities | 13.81% | 13.73% | -0.5% | 15.6% | 15.8% | +0.16 pt |
| Wholesale | 15.65% | 15.73% | +0.5% | 35.0% | 34.7% | -0.31 pt |
| Volatility | 10.65% | 9.55% | -10.3% | 3.5% | 3.6% | +0.05 pt |
| Momentum | 7.50% | 7.06% | -5.9% | 3.9% | 4.0% | +0.04 pt |
| Size | 7.76% | 7.51% | -3.3% | 9.6% | 9.8% | +0.23 pt |
| Size Nonlinearity | 6.55% | 6.55% | +0.0% | 18.0% | 17.3% | -0.67 pt |
| Trading Activity | 5.66% | 5.16% | -8.8% | 10.8% | 10.8% | -0.02 pt |

### Specific risk and the floor list

| | HEAD | now |
|---|---|---|
| Specific risk percentiles | median 23.12%   p10 14.86%   p90 36.06% | median 23.10%   p10 14.87%   p90 36.12% |
| Residual spread and its largest shares | relative residual sd 0.377, largest shares of its square: FRC 60.6%, ECHO 4.1%, LUMN 2.5%, BNY 2.2%, SIVB 2.1% | relative residual sd 0.380, largest shares of its square: FRC 61.1%, ECHO 4.0%, LUMN 2.7%, SIVB 2.6%, BNY 2.2% |
| EQ 3-14 V fit, largest shares of V^2 | MRNA 2.2%, FRC 2.1%, ENPH 1.3%, ALB 0.9%, CTVA 0.9% | MRNA 2.2%, FRC 2.1%, ENPH 1.3%, ALB 1.0%, CTVA 0.9% |
| Floor list (characteristic forecast below the floor) | BNY, JPM | JPM |
| Delivered floor | none | none |
| Fallback forecasts (the report names the first five) | 7: INFO, IPGP, PENN, PVH, UA | 7: INFO, IPGP, PENN, PVH, UA |

BNY leaves the floor list at variant C (the clip bounds over members); JPM, the real linear-V
case, stays. Unchanged: the regression universe (485 securities), the per-period cross-section
sizes (index members 503 / 504 / 506 / 504 / 503 for min / median / max / first / last, in the
panel 426 / 458 / 486 / 486 / 426, in the regression 422 / 451 / 477 / 477 / 425), 0 collinear
periods, 275 thin periods.

### The descriptor screen, HEAD (everyone) -> now (members)

| Descriptor | cover | periods | mean R2 HEAD | mean R2 now | factor vol HEAD | factor vol now | t-stat HEAD | t-stat now |
|---|---|---|---|---|---|---|---|---|
| BTSG | 97% | 386 | 7.03% | 8.39% | 9.64% | 9.76% | 1.91 | 1.88 |
| DASTD | 93% | 1,149 | 5.88% | 6.13% | 8.30% | 7.78% | 1.80 | 1.35 |
| RSTR | 100% | 938 | 4.71% | 5.29% | 7.49% | 7.08% | 0.31 | 1.11 |
| HILO | 93% | 1,213 | 3.65% | 3.60% | 6.84% | 6.25% | 1.79 | 1.13 |
| CMRA | 99% | 938 | 3.43% | 3.38% | 6.17% | 5.69% | 1.31 | 1.39 |
| HALPHA | 97% | 386 | 3.35% | 3.32% | 6.76% | 6.09% | -0.37 | -0.28 |
| STOQ | 100% | 1,145 | 3.14% | 2.99% | 5.97% | 5.22% | 0.67 | 1.26 |
| STOM | 100% | 1,191 | 3.10% | 2.95% | 5.99% | 5.26% | 0.71 | 1.27 |
| STOA | 100% | 938 | 2.80% | 2.56% | 6.05% | 4.86% | 1.89 | 1.52 |
| LNCAP | 93% | 1,213 | 2.05% | 1.83% | 4.85% | 3.95% | -0.19 | 1.04 |
| LPRI | 93% | 1,213 | 2.02% | 1.77% | 4.92% | 3.77% | -1.73 | -0.40 |
| LCAPCB | 93% | 1,213 | 1.88% | 1.68% | 4.47% | 3.78% | -0.28 | 0.64 |
| VLVR | 95% | 386 | 1.47% | 1.50% | 4.17% | 3.92% | 0.14 | 0.64 |
| SERDP | - | 0 | too few usable periods | | | | | |

### Against Jeff's expectation: the A-to-D decomposition

Five runs of the same script (`decompose.py`), each holding one more ingredient to the members;
the regressions are over members in all five:

- **A**: HEAD -- every standardisation over everyone.
- **B**: A's style exposures, then a pure per-period affine map of each onto the members' scale
  (cap-weighted mean 0, equal-weighted SD 1 over members). Jeff's reparameterisation.
- **B2**: A's standardised descriptors, the same affine map applied per descriptor, then blended
  and re-standardised over members. Adds the blend's non-affinity; clip bounds unchanged.
- **C**: descriptors standardised over members (clip bounds, mean, SD), blended over members;
  LCAPCB's own normalisation still over everyone. Exactly the changes Jeff listed.
- **D**: LCAPCB normalised over members too. The code as it stands, and as ruled.

Checks on the harness: variant D's report is byte-identical to the real run's; variant A
reproduces HEAD's model exactly (its text differs only in the moved membership block, the new
report lines and the members-only screen).

| Variant | R-squared | vs A | Volatility | Momentum | Size | Size Nonlinearity | Trading Activity |
|---|---|---|---|---|---|---|---|
| A: HEAD | 0.312985 | +0.000000 | 10.65% (+0.0%) | 7.50% (+0.0%) | 7.76% (+0.0%) | 6.55% (+0.0%) | 5.66% (+0.0%) |
| B: pure affine onto members | 0.312986 | +0.000001 | 9.66% (-9.3%) | 7.12% (-5.1%) | 6.97% (-10.2%) | 5.76% (-12.1%) | 5.00% (-11.7%) |
| B2: affine per descriptor | 0.312546 | -0.000439 | 9.69% (-9.0%) | 7.01% (-6.5%) | 6.99% (-9.9%) | 5.77% (-11.8%) | 5.02% (-11.4%) |
| C: + clip bounds over members | 0.312369 | -0.000615 | 9.56% (-10.2%) | 7.12% (-5.1%) | 6.83% (-12.0%) | 5.57% (-15.0%) | 5.20% (-8.2%) |
| D: + LCAPCB over members (kept) | 0.312250 | -0.000735 | 9.55% (-10.3%) | 7.06% (-5.9%) | 7.51% (-3.3%) | 6.55% (+0.0%) | 5.16% (-8.8%) |

| Variant | Volatility noise | Momentum noise | Size noise | Size Nonlinearity noise | Trading Activity noise | residual spread | median specific |
|---|---|---|---|---|---|---|---|
| A: HEAD | 3.5% | 3.9% | 9.6% | 18.0% | 10.8% | 0.3767 | 23.12% |
| B: pure affine onto members | 3.4% | 3.9% | 10.1% | 19.5% | 11.2% | 0.3798 | 23.12% |
| B2: affine per descriptor | 3.5% | 4.0% | 10.0% | 19.4% | 11.2% | 0.3792 | 23.13% |
| C: + clip bounds over members | 3.6% | 3.9% | 10.2% | 21.1% | 10.7% | 0.3804 | 23.08% |
| D: + LCAPCB over members (kept) | 3.6% | 4.0% | 9.8% | 17.3% | 10.8% | 0.3804 | 23.10% |

Reading it:

- **A -> B: R-squared +0.000001.** Jeff's argument holds exactly: a per-period affine change is
  absorbed, and the residue is CTVA, FRC and SBNY, the three names with no industry factor, for
  whom the industry columns do not span the constant. Vols fall about in proportion to the
  members' spread: Size -10.2%, Size Nonlinearity -12.1%.
- **B -> B2: -0.000440.** The blend. Members' spreads differ inside an index -- on the last day,
  on everyone's scale, RSTR 1.021 against HALPHA 0.869 in Momentum -- so rescaling each
  descriptor reweights the blend. RSTR's spread above one is the empty-window zeros (problem 2).
- **B2 -> C: -0.000177.** The clip bounds and the +/-4 bound over members. Jeff expected the
  bounds to barely move; in raw units they moved about as much as the SD -- on the last day
  LNCAP's MAD scale is 0.896 of everyone's, STOM / STOQ / STOA 0.836 / 0.837 / 0.846 -- because
  the outsiders are a sixth of the cross-section, all in one tail (plus the dead names' zeros),
  which is not the few outliers a MAD is robust to. In exposure units the bounds barely moved,
  so they clip nearly the same members, and R-squared barely notices.
- **C -> D: -0.000119 R-squared, and the Size pair:** Size 6.83% -> 7.51% (+10.0%), Size
  Nonlinearity 5.57% -> 6.55% (+17.6%). The cube is re-centred on the members' mean of log cap,
  0.21 member SD above everyone's on the last day (members 24.716, sd 1.183; everyone with a cap
  24.466, sd 1.408).

### The Size pair, C -> D

- **Exposures:** Size and Size Nonlinearity among members correlate +0.920 (C) and +0.931 (D) on
  the last day; the medians over the window are +0.931 and +0.934 (A: +0.920 / +0.931).
- **Re-centring moves the cube's clipped mass:** members at its lower bound 3% (C) -> 8% (D), at
  its upper bound 16% -> 11%; its range among members [-3.44, +0.47] (C) -> [-3.25, +0.57] (D).
  A: 3% / 12%, [-3.16, +0.47].
- **Factor returns:** correlation -0.854 (A), -0.841 (C), -0.865 (D). With exposures that
  collinear the regression re-splits the size effect between the two.
- **What it does to portfolios** -- the pair's risk, x' Sigma x over the 2x2 block, with the
  portfolio's average exposures in brackets (Size, SNL):

| Variant | Size vol | SNL vol | Corr | Smallest 85 members, EW | Largest 50 members, EW | The index (cap-weighted) |
|---|---|---|---|---|---|---|
| A | 7.76% | 6.55% | -0.854 | 11.69% (-2.745, -2.211) | 3.00% (+0.011, +0.469) | 0.05% (+0.013, +0.012) |
| C | 6.83% | 5.57% | -0.841 | 11.97% (-3.112, -2.601) | 2.53% (+0.012, +0.466) | 0.00% (-0.000, -0.000) |
| D | 7.51% | 6.55% | -0.865 | 11.83% (-3.112, -2.819) | 3.53% (+0.012, +0.551) | 0.00% (-0.000, -0.000) |

Mostly a re-split: a small-cap tilt's combined risk moves 12.0% -> 11.8%. Not neutral at the
top: the largest names sit at Size about 0 and SNL about +0.5, and their pair risk goes 2.5% ->
3.5%.

### The LCAPCB findings

1. **Normalising over members changes the cube's shape, not only its scale.** A cube is not
   affine: centring it 0.21 SD higher adds an even (z^2) component that no per-period affine map
   removes. That is C -> D.
2. **The 4-MAD raw clip is very aggressive on a cube.** The MAD of z^3 is small against its tails,
   so the bounds sit at |z| of about 1.1-1.2: cube bounds +/-1.21 over everyone on the last day,
   +/-1.78 over members once the cube is re-normalised (|z| = 1.21). On 2026-11-03 the clip moves
   81 of 425 members (19%) and every outsider; over the panel, 139,554 of the 348,002 raw clips
   (40%) are LCAPCB's. Size Nonlinearity is mostly a clipped cube. Pre-existing; not changed.
3. **Ruled: keep D.** The pair's instability belongs to the size-pair decision, made separately.

### The post-standardisation clip, by how much

There is no clip after the second standardisation, so the style exposures are exact (above). The
+/-4 clip after the first standardisation does bite, at the descriptor level:

- **Among members:** 3,043 values over 1,214 days -- LNCAP 1,408, VLVR 775, RSTR 543, HALPHA
  290, LPRI 17, HILO 9, SERDP 1. Among outsiders: 19,588.
- **What it leaves:** the members' cap-weighted mean of a descriptor off zero by at most VLVR
  9.6e-04, RSTR 2.8e-04, HILO 1.8e-04, HALPHA 1.4e-04, LNCAP 3.6e-05, LPRI 1.9e-05, SERDP
  1.6e-05 (about 1e-15 elsewhere), and its equal-weighted SD below one by at most 0.018 (VLVR).
  The second standardisation removes both.

---

## Key decisions taken (Code)

1. **LCAPCB normalised over members**, beyond the four listed items, under the stated principle
   ("nothing outside it should set the scale of anything inside it"). Reported as the cause of the
   volatility surprise, with the decomposition; Jeff ruled keep (D).
2. **A boolean mask, not a snapshot, everywhere**, as in f72310b: the model package does not
   import refdata.
3. **One validator for every entry point** (`_checked_membership`), shared with crosssectional's
   `_membership_block`, so a wrong array is refused the same way everywhere.
4. **Outsiders' bound hits counted by either bound** (`clipped_outside`), because for LNCAP the
   +/-4 bound binds, not the raw one.
5. **Counts, not a per-cell array.** A `clipped` dates x securities x characteristics array was
   tried first and dropped: `dataclasses.replace` slices and edits panels routinely and would
   carry a stale array (an existing test broke on exactly that; see the trail).
6. **A day with no member value leaves the characteristic missing** rather than standardising
   it against outsiders; fewer than two members leaves a risk index missing.
7. **The estimator's own standardise path takes the same membership**; characteristic cells
   outside the return panel set no scale.
8. **The screen and `fit_index_weights` hold to members** ("one array used everywhere"). The
   screen's numbers changed as a result (table above).
9. **The guard reports and flags rather than stopping the run**; its tolerance is the first-order
   rounding bound, not a fixed number, and it is recomputed independently of the standardisation.
10. **Per-descriptor outsider counts on the exposure date** come from one-day, one-descriptor
    `standardise` calls -- exact -- rather than from a second implementation of the bounds.
11. **The membership block moved ahead of the descriptors** in the run script.

## Decisions left for Jeff

1. **The size pair** -- whether to orthogonalise the cube against Size, as USE4 does, and what to
   do about its clip. Being decided separately; nothing here touches it.
2. **Whether the descriptor screen should stay members-only** (decision 8). It follows from "one
   array used everywhere", but it changes a table Jeff reads.
3. **Whether `clipped_outside` should keep counting both bounds** (decision 4).

---

## Tests

- **Counts:** `tests/midoffice` 756 passed, 1 skipped (743 before, plus 13 new). Full suite: 2,207
  passed, 1 skipped, 19 warnings, run with `--ignore=tests/test_market_data.py`, which fails
  collection with `No module named 'mctheory'` -- pre-existing and unrelated.
- **Hygiene:** every edited Python file parses (AST) and is ASCII-only (0 non-ASCII bytes);
  `docs/FACTOR_MODEL.md` likewise. Ruff: 6 findings, identical to HEAD's, all pre-existing --
  `scripts/run_sp500_model.py` F401 `HorizonCovariance`; `characteristics.py` E501 (line 30)
  and F401 `Sequence`; `crosssectional.py` F401 `dataclasses.field` and F401 `typing.Mapping`;
  `tests/midoffice/test_crosssectional.py` I001.
- **Failing first:** the first twelve failed against HEAD (`TypeError: unexpected keyword
  argument 'membership'`). The thirteenth, and the both-bounds count in the third, were shown to
  fail with the +/-4 count disabled (a mutation check) and to pass with it restored.

| New test | What it guards |
|---|---|
| **tests/midoffice/test_crosssectional.py** | |
| `test_standardising_over_members_centres_and_scales_on_them_alone` | Members' cap-weighted mean 0 (< 1e-14) and equal-weighted SD 1 (rel 1e-14) on every date and characteristic; over everyone the members' SD is below 0.8, so the test is not vacuous |
| `test_a_security_outside_the_index_is_put_on_the_members_scale_and_moves_none_of_it` | An outsider's value equals the hand computation from the members' bounds, mean and SD (rel 1e-14); pushing the outsiders deeper into the tail moves no member's value by a bit; without a membership it moves everyone's |
| `test_the_members_bounds_bind_an_outsider_in_the_tail_and_are_counted` | `clipped_outside` equals the hand count over both bounds, `winsorised` the raw-clip count; outsiders clipped to one bound share one exposure |
| `test_an_outsider_inside_the_raw_bound_can_still_hit_the_standardised_one` | LNCAP's case -- cap-weighted mean far above the median, an outsider inside the raw bound standardising past -4: `winsorised` 0, `clipped_outside` 1, the value -4.0 |
| `test_a_membership_that_holds_everyone_standardises_exactly_as_none` | Bit-identical values and counts; `clipped_outside` 0 |
| `test_a_day_with_no_member_value_is_left_missing_not_standardised_on_the_rest` | That day all missing, the other days finite |
| `test_standardise_refuses_a_membership_of_the_wrong_shape_or_kind` | ValueError on the shape, TypeError on a float array |
| `test_the_estimator_standardises_over_the_membership_it_regresses_on` | `standardise=True` with a membership equals pre-standardising with it -- covariance, R-squared and exposures exactly -- and differs from standardising over everyone |
| **tests/midoffice/test_descriptors.py** | |
| `test_risk_indices_built_over_members_hold_the_index_at_zero_style_exposure` | The exact guard in the suite: on every period and index the members' cap-weighted mean is < 1e-14 and their SD 1 (rel 1e-13), over more than 240 checks; over everyone the members' Size mean is off by more than 0.01 |
| `test_a_security_outside_the_index_sets_no_part_of_any_members_exposure` | End to end: shrinking the leavers' share counts after they leave moves their caps, LNCAP, LCAPCB and turnover but no member's risk index by a bit; without a membership every member's moves |
| `test_lcapcb_is_the_cube_of_log_cap_normalised_over_the_members` | LCAPCB = ((log cap - members' mean) / members' SD)^3 (rel 1e-12); identical to the old one while everyone is a member, different once the leavers leave |
| `test_fitted_weights_learn_from_the_members_alone` | Equal, exactly, to the fit on returns with the non-member cells blanked; different from the unmasked fit |
| `test_a_membership_of_the_wrong_shape_or_kind_is_refused_wherever_it_is_taken` | `compute_descriptors`, `blend_to_index`, `build_risk_indices` and `fit_index_weights` refuse a wrong shape; `blend_to_index` refuses a float |

---

## Failures and debugging trail

1. **A per-cell `clipped` array on the panel.** The first design recorded which cells the clip
   moved. `test_specific_risk_is_estimated_over_the_regression_universe_and_nothing_else` then
   failed: it slices securities with `dataclasses.replace(chars, securities=kept,
   values=chars.values[:, 2:, :])`, which carried the old array along -- `ValueError: clipped must
   be a boolean array of shape (60, 78, 2), got bool of shape (60, 80, 2)`. A replace that edits
   values would carry a stale one silently. Replaced by counts, which have the same provenance
   semantics as the existing `winsorised`.
2. **Counting the raw bound only.** The first count missed outsiders that the +/-4 bound moved.
   Measured on the real panel before wiring the report (`measure_clips.py`): on 2026-11-03, 19 of
   45 outsiders hit LNCAP's -4 bound and 1 the raw one. The count now takes both; the dedicated
   test and the both-bounds test fail if the +/-4 count is disabled.
3. **A draft test compared one call with itself.** The first draft of
   `test_the_estimator_standardises_over_the_membership_it_regresses_on` ended with the same fit
   twice; caught on reading it back, before it ran, and replaced by the comparison with
   standardising over everyone.
4. **Shell heredocs with mixed quotes failed** (`unexpected EOF while looking for matching`), as
   in f72310b's session. The descriptor and run-script patches were written as scripts with the
   Write tool and run from the scratchpad, each replacement asserted to apply exactly once.
5. **A `git stash` to re-check a failing test was meaningless**: the stash removed the new tests
   too (0 selected). The fail-first evidence is the earlier run (12 failed) and the mutation
   check. The stash and pop also rewrote the working files with CRLF (`core.autocrlf=true`);
   their content is unchanged, and git normalises on commit.
6. **One backup went to `/tmp`** during the mutation check instead of the scratchpad; removed.
7. **The decomposition harness was validated before its numbers were used**: variant D's report
   byte-identical to the real run's, variant A identical to HEAD's model.

---

## Problems found, not fixed

1. **Same-day exposures, a look-ahead.** The run script hands the estimator exposures and returns
   on the same dates, and `_align` pairs them exactly, so day t's return is regressed on
   exposures built from day t's prices. The descriptor screen lags by a day for exactly that
   reason; the model does not. With every exposure taken from the day before (`lag_probe.py`):
   R-squared 0.3123 -> 0.2949, Volatility 9.55% -> 7.79%, Momentum 7.06% -> 7.24%, Size 7.51% ->
   7.58%, Size Nonlinearity 6.55% -> 6.67%, Trading Activity 5.16% -> 5.23%; 939 -> 938
   periods. Queued as the next part's second fix.
2. **Empty-window zeros.** RSTR, CMRA, STOM, STOQ and STOA are window sums, and they come out 0
   rather than missing when the window has no price at all: RSTR and CMRA 18,634 values each,
   STOM 22,512, STOQ 20,780, STOA 14,300. On index members: RSTR / CMRA 4,669, STOM 705, STOQ
   567, STOA 335 -- ten tickers the membership walk makes members before their first price: COR
   (141 member-days), CPAY (283), ECHO (846), EG (104), FISV (335, its FI gap), MRSH (736),
   RVTY (68), SKYD (918), SW (354), VMRK (884). Members-only standardisation still counts those
   zeros; on everyone's scale they are also why RSTR's members' spread was above one. Queued as
   the next part's first fix.
3. **LCAPCB's clip** -- the LCAPCB findings, point 2.
4. **Style exposures run past the unit-basis bound.** `exposure_basis.DEFAULT_BOUND`'s docstring
   says [-4, 4] maps onto [0, 1] "and nothing falls outside it", but the second standardisation
   does not clip. On the last day (now): Trading Activity among members [-5.40, +4.03], Momentum
   outsiders down to -4.20. Values past +/-4 over every day: Size 10,215 (7,152 at HEAD),
   Momentum 2,296 (1,300), Volatility 672 (421), Trading Activity 236 (194), Size Nonlinearity 0
   (0). The transform is affine, so no risk number changes; the [0, 1] promise is not kept. The
   run script does not use it.
5. **Pre-existing:** the six ruff findings (under Tests) and `tests/test_market_data.py`'s
   `No module named 'mctheory'`.

---

## Commits

None. The change is uncommitted; Jeff commits with Chat's message (Rule 43).

## Open items for Chat

1. **The size pair:** the cube is 0.93 collinear with Size; USE4 orthogonalises it against Size;
   its 4-MAD clip binds at |z| of about 1.2. The decision is being made separately.
2. **The next part, queued until Jeff replies "committed":** (1) window-based descriptors return
   missing for an empty window, with a minimum-coverage parameter and the count a 50% rule would
   remove; (2) the one-day lag of the estimation, with the price-perturbation guard, the
   `fit_index_weights` in-sample note, the share-count fill check and the SERDP note. Not started.
3. **The unit-basis bound claim** (problem 4).

---

## Appendix: scratch outputs, copied in full

The session scratch directory (under Temp) can vanish. Every output this retro draws on is
reproduced here verbatim. The scripts that made them, and these outputs, are also kept locally
in `claude/scratch/midoffice01/` (gitignored). Run order to regenerate: `measure_clips.py`
(writes `inputs.pkl` and `clips.npz`), then `decompose.py` (`decompose.pkl`, `decompose.json`),
then any of the rest.

### final_with.txt

The run report, `--industries --membership`, this change.

```text
Panel
  1214 trading days, 506 securities
  2022-01-03 .. 2026-11-03
  583,081 of 614,284 return observations present (94.9%)

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC, covering 1976-07-01 .. 2026-11-04
  one array for every standardisation and every regression below, each held to that day's members: 506 of the panel's 506 on 2022-01-03, 426 on 2026-11-03
  3 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
  - 2026-03-23 SATS recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 157 panel day(s) from then
  - 2026-10-30 OKTA recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then
  - 2026-10-30 KVUE recorded as removed, but membership() still has it after that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then

Descriptors
  computed 14: LNCAP, LCAPCB, LPRI, DASTD, HILO, CMRA, BTSG, HALPHA, SERDP, RSTR, STOM, STOQ, STOA, VLVR
  - CMRA: computed on raw rather than excess returns: no risk_free supplied
  - RSTR: computed on raw rather than excess returns: no risk_free supplied
  - SERDP: computed at the final period only; the market-model residual series is retained for one window rather than all of them, so SERDP is a current exposure and not a time series
  - STO5: needs 1380 period(s) of history and the panel has 1214, so every value would be missing
  348,002 raw values winsorised before standardising
  100,018 values of securities outside the index hit one of the members' bounds, over every date; on 2026-11-03, of those with a value:
    LNCAP 19/45  LCAPCB 45/45  LPRI 4/45  DASTD 4/47  HILO 8/45  CMRA 5/77  BTSG 8/66
    HALPHA 7/66  SERDP 1/66  RSTR 3/80  STOM 13/80  STOQ 14/80  STOA 16/80  VLVR 18/54

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         97%      386    8.39%      9.76%     1.88
  DASTD        93%    1,149    6.13%      7.78%     1.35
  RSTR        100%      938    5.29%      7.08%     1.11
  HILO         93%    1,213    3.60%      6.25%     1.13
  CMRA         99%      938    3.38%      5.69%     1.39
  HALPHA       97%      386    3.32%      6.09%    -0.28
  STOQ        100%    1,145    2.99%      5.22%     1.26
  STOM        100%    1,191    2.95%      5.26%     1.27
  STOA        100%      938    2.56%      4.86%     1.52
  LNCAP        93%    1,213    1.83%      3.95%     1.04
  LPRI         93%    1,213    1.77%      3.77%    -0.40
  LCAPCB       93%    1,213    1.68%      3.78%     0.64
  VLVR         95%      386    1.50%      3.92%     0.64
  SERDP           -        0   too few usable periods

Risk indices
  Volatility           BTSG, DASTD, HILO, LPRI, CMRA, SERDP
  Momentum             RSTR, HALPHA
  Size                 LNCAP
  Size Nonlinearity    LCAPCB
  Trading Activity     STOM, STOQ, STOA, VLVR

Industries
  33 classification(s) carried from an earlier date (delisted inside the window): ABMD, ANSS, ATVI, AVB, CDAY
  485 of 506 securities are in at least one estimated period's regression while an index member; every count below is over them
  19 industries over 485 securities in the universe (506 classified); largest SIC73 with 53 (11%), 45 thin group(s) merged into their division
    SIC73                      53
    SIC38                      39
    SIC28                      39
    SIC49                      33
    Manufacturing              33
    Retail                     30
    Transport & Utilities      29
    SIC35                      27
  - 1 industry/industries have no member in the universe and are not factors: SIC24
  - 2 industry/industries had fewer than 5 members even after merging into their division and were pooled into Other Industries, which is heterogeneous by construction: Agriculture, Unclassified
  - Other Industries pooled only 3, below the hard floor of 5, so it is not a factor. 3 security(ies) have no industry factor: the industry effect is unmodelled and falls into specific risk: CTVA, FRC, SBNY
  - 3 industry/industries still have fewer than 15 members after merging and their factor returns are largely their members' specific returns: Construction (6), Mining (14), Wholesale (11)
  - 2 security(ies) have no usable SIC code and are left with no industry factor: FRC, SBNY
  - 21 classified security(ies) are outside the universe and count toward no industry: ABMD, CERN, CTXS, DISCA, DISCK

Model
  regression weights: square root of capitalisation
  24-factor model, annualised, estimated on 939 daily period(s) for 470 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 939
  mean cross-sectional R-squared 0.312
  0 collinear period(s) of 939 estimated; 0 factor-period(s) left out for having no member present

  Style exposures over each estimated period's index members with a value, all 939 periods: cap-weighted mean at most 2.0e-16 from zero (0.9 eps), equal-weighted sd at most 3.3e-16 from one
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  -7.7e-18   1.000000   426
  Momentum                    +2.3e-17   1.000000   426
  Size                        -3.1e-17   1.000000   425
  Size Nonlinearity           -7.7e-18   1.000000   425
  Trading Activity            +3.9e-17   1.000000   426

  Securities per estimated period, over 939
                            min  median    max   2023-02-07   2026-11-03
  index members             503     504    506          504          503
  of them in the panel      426     458    486          486          426
  in the regression         422     451    477          477          425

  Over the 939 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                    periods with a member present
  factor                  annual vol  noise share      >=1  exactly 1   none
  Construction               17.38%       49.3%      939          0      0
  Finance & Real Estate      15.70%       13.5%      939          0      0
  Manufacturing              11.46%       18.5%      939          0      0
  Mining                     20.95%       14.5%      939          0      0
  Retail                     13.69%        9.6%      939          0      0
  SIC20                      15.38%       20.9%      939          0      0
  SIC28                      14.77%        7.9%      939          0      0
  SIC35                      17.10%        6.4%      939          0      0
  SIC36                      20.34%        5.4%      939          0      0
  SIC37                      14.29%       17.4%      939          0      0
  SIC38                      14.46%       10.8%      939          0      0
  SIC49                      14.56%       14.2%      939          0      0
  SIC60                      20.87%        8.0%      939          0      0
  SIC63                      15.85%       11.9%      939          0      0
  SIC67                      15.66%       15.9%      939          0      0
  SIC73                      16.55%        3.8%      939          0      0
  Services                   13.87%       22.4%      939          0      0
  Transport & Utilities      13.73%       15.8%      939          0      0
  Wholesale                  15.73%       34.7%      939          0      0
  Volatility                  9.55%        3.6%      939          0      0
  Momentum                    7.06%        4.0%      939          0      0
  Size                        7.51%        9.8%      939          0      0
  Size Nonlinearity           6.55%       17.3%      939          0      0
  Trading Activity            5.16%       10.8%      939          0      0

  specific risk   median 23.10%   p10 14.87%   p90 36.12%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.2%, FRC 2.1%, ENPH 1.3%, ALB 1.0%, CTVA 0.9%
                    relative residual sd 0.380, largest shares of its square: FRC 61.1%, ECHO 4.0%, LUMN 2.7%, SIVB 2.6%, BNY 2.2%

Qualifications
  - 275 period(s) dropped for having fewer than 30 securities: 2022-01-03, 2022-01-04, 2022-01-05 and 272 more
  - 36 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CDAY
  - 49% of the estimated variance of Construction is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 1 security(ies) have a characteristic forecast below the specific-risk floor, their linear V having run past the end of its validity; blended with their own residual history, no forecast delivered is the floor: JPM
  - 7 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: INFO, IPGP, PENN, PVH, UA
  - 275 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### head_with.txt

The run report, `--industries --membership`, HEAD (f72310b).

```text
Panel
  1214 trading days, 506 securities
  2022-01-03 .. 2026-11-03
  583,081 of 614,284 return observations present (94.9%)

Descriptors
  computed 14: LNCAP, LCAPCB, LPRI, DASTD, HILO, CMRA, BTSG, HALPHA, SERDP, RSTR, STOM, STOQ, STOA, VLVR
  - CMRA: computed on raw rather than excess returns: no risk_free supplied
  - RSTR: computed on raw rather than excess returns: no risk_free supplied
  - SERDP: computed at the final period only; the market-model residual series is retained for one window rather than all of them, so SERDP is a current exposure and not a time series
  - STO5: needs 1380 period(s) of history and the panel has 1214, so every value would be missing
  314,921 raw values winsorised before standardising

Which descriptors carry signal
  Ranked by the cross-sectional return variance each explains on its own.
  descriptor  cover  periods   mean R2  factor vol   t-stat
  BTSG         97%      386    7.03%      9.64%     1.91
  DASTD        93%    1,149    5.88%      8.30%     1.80
  RSTR        100%      938    4.71%      7.49%     0.31
  HILO         93%    1,213    3.65%      6.84%     1.79
  CMRA         99%      938    3.43%      6.17%     1.31
  HALPHA       97%      386    3.35%      6.76%    -0.37
  STOQ        100%    1,145    3.14%      5.97%     0.67
  STOM        100%    1,191    3.10%      5.99%     0.71
  STOA        100%      938    2.80%      6.05%     1.89
  LNCAP        93%    1,213    2.05%      4.85%    -0.19
  LPRI         93%    1,213    2.02%      4.92%    -1.73
  LCAPCB       93%    1,213    1.88%      4.47%    -0.28
  VLVR         95%      386    1.47%      4.17%     0.14
  SERDP           -        0   too few usable periods

Risk indices
  Volatility           BTSG, DASTD, HILO, LPRI, CMRA, SERDP
  Momentum             RSTR, HALPHA
  Size                 LNCAP
  Size Nonlinearity    LCAPCB
  Trading Activity     STOM, STOQ, STOA, VLVR

Membership
  snapshot 660dc8159b0a, fetched 2026-11-04 03:37 UTC, covering 1976-07-01 .. 2026-11-04
  each period's regression holds only that day's members: 506 of the panel's 506 on 2022-01-03, 426 on 2026-11-03
  3 change(s) inside the panel's window that membership() does not reproduce, used as it reads them:
  - 2026-03-23 SATS recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 157 panel day(s) from then
  - 2026-10-30 OKTA recorded as added, but membership() does not have it from that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then
  - 2026-10-30 KVUE recorded as removed, but membership() still has it after that date: not in the panel, so no cross-section changes on the 3 panel day(s) from then

Industries
  33 classification(s) carried from an earlier date (delisted inside the window): ABMD, ANSS, ATVI, AVB, CDAY
  485 of 506 securities are in at least one estimated period's regression while an index member; every count below is over them
  19 industries over 485 securities in the universe (506 classified); largest SIC73 with 53 (11%), 45 thin group(s) merged into their division
    SIC73                      53
    SIC38                      39
    SIC28                      39
    SIC49                      33
    Manufacturing              33
    Retail                     30
    Transport & Utilities      29
    SIC35                      27
  - 1 industry/industries have no member in the universe and are not factors: SIC24
  - 2 industry/industries had fewer than 5 members even after merging into their division and were pooled into Other Industries, which is heterogeneous by construction: Agriculture, Unclassified
  - Other Industries pooled only 3, below the hard floor of 5, so it is not a factor. 3 security(ies) have no industry factor: the industry effect is unmodelled and falls into specific risk: CTVA, FRC, SBNY
  - 3 industry/industries still have fewer than 15 members after merging and their factor returns are largely their members' specific returns: Construction (6), Mining (14), Wholesale (11)
  - 2 security(ies) have no usable SIC code and are left with no industry factor: FRC, SBNY
  - 21 classified security(ies) are outside the universe and count toward no industry: ABMD, CERN, CTXS, DISCA, DISCK

Model
  regression weights: square root of capitalisation
  24-factor model, annualised, estimated on 939 daily period(s) for 470 of 506 security(ies); covariance Newey-West, 10 lag(s), half-life 90, 259.3 effective observation(s) of 939
  mean cross-sectional R-squared 0.313
  0 collinear period(s) of 939 estimated; 0 factor-period(s) left out for having no member present

  Securities per estimated period, over 939
                            min  median    max   2023-02-07   2026-11-03
  index members             503     504    506          504          503
  of them in the panel      426     458    486          486          426
  in the regression         422     451    477          477          425

  Over the 939 estimated periods. noise share is blind to a one-member factor; the exactly-1 column is not.
                                                    periods with a member present
  factor                  annual vol  noise share      >=1  exactly 1   none
  Construction               17.29%       49.7%      939          0      0
  Finance & Real Estate      15.65%       13.5%      939          0      0
  Manufacturing              11.54%       18.3%      939          0      0
  Mining                     21.11%       14.2%      939          0      0
  Retail                     13.57%        9.7%      939          0      0
  SIC20                      15.42%       20.7%      939          0      0
  SIC28                      14.85%        7.8%      939          0      0
  SIC35                      16.98%        6.5%      939          0      0
  SIC36                      20.35%        5.3%      939          0      0
  SIC37                      14.21%       17.6%      939          0      0
  SIC38                      14.40%       10.9%      939          0      0
  SIC49                      14.61%       14.0%      939          0      0
  SIC60                      20.81%        8.0%      939          0      0
  SIC63                      15.91%       11.7%      939          0      0
  SIC67                      15.51%       16.2%      939          0      0
  SIC73                      16.63%        3.7%      939          0      0
  Services                   13.92%       22.2%      939          0      0
  Transport & Utilities      13.81%       15.6%      939          0      0
  Wholesale                  15.65%       35.0%      939          0      0
  Volatility                 10.65%        3.5%      939          0      0
  Momentum                    7.50%        3.9%      939          0      0
  Size                        7.76%        9.6%      939          0      0
  Size Nonlinearity           6.55%       18.0%      939          0      0
  Trading Activity            5.66%       10.8%      939          0      0

  specific risk   median 23.12%   p10 14.86%   p90 36.06%
  EQ 3-14's V fit   largest shares of V^2: MRNA 2.2%, FRC 2.1%, ENPH 1.3%, ALB 0.9%, CTVA 0.9%
                    relative residual sd 0.377, largest shares of its square: FRC 60.6%, ECHO 4.1%, LUMN 2.5%, BNY 2.2%, SIVB 2.1%

Qualifications
  - 275 period(s) dropped for having fewer than 30 securities: 2022-01-03, 2022-01-04, 2022-01-05 and 272 more
  - 36 security(ies) excluded for an incomplete characteristic row at the exposure date: ABMD, ANSS, ATVI, AVB, CDAY
  - 50% of the estimated variance of Construction is the sampling error of its own estimation; the factor covariance diagonal is biased upward and is not corrected here
  - 2 security(ies) have a characteristic forecast below the specific-risk floor, their linear V having run past the end of its validity; blended with their own residual history, no forecast delivered is the floor: BNY, JPM
  - 7 security(ies) are in no estimated period's regression, so the specific risk model was not estimated on them and their forecast is a fallback, not a model one -- the characteristic forecast extrapolated to them: INFO, IPGP, PENN, PVH, UA
  - 275 period(s) had no usable residuals and were skipped when forming the average absolute specific return
  - Newey-West combined with exponential weighting is not guaranteed positive semi-definite the way the equal-weighted estimator is; psd_relative_floor is what repairs it if it is not
```

### final_without.txt vs head_without.txt

`diff head_without.txt final_without.txt`: without `--membership` the only difference is the new guard block.

```text
--- head_without.txt
+++ final_without.txt
@@ -63,2 +63,10 @@
 
+  Style exposures over each estimated period's securities with a value, all 939 periods: cap-weighted mean at most 2.1e-16 from zero (0.9 eps), equal-weighted sd at most 2.2e-16 from one
+  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
+  Volatility                  -7.7e-18   1.000000   506
+  Momentum                    +1.5e-17   1.000000   506
+  Size                        +7.7e-17   1.000000   470
+  Size Nonlinearity           +0.0e+00   1.000000   470
+  Trading Activity            +2.3e-17   1.000000   506
+
   Securities per estimated period, over 939
```

### variant_A.txt (guard block)

HEAD's exposures under the new guard: it fires.

```text
  Style exposures over each estimated period's index members with a value, all 939 periods: cap-weighted mean at most 1.4e-02 from zero (63746696177335.2 eps), equal-weighted sd at most 1.8e-01 from one
  ! 4695 of 4695 style-period means are past rounding, so the index is not at zero exposure; the largest, +0.0142, is Size on 2026-08-04
  on 2026-11-03           cap-wtd mean  eq-wtd sd     n
  Volatility                  -9.8e-04   0.891546   426
  Momentum                    +4.8e-03   0.957262   426
  Size                        +1.3e-02   0.882585   425
  Size Nonlinearity           +1.2e-02   0.855386   425
  Trading Activity            -3.5e-03   0.875530   426
```

### decompose.txt

decompose.py: R-squared and style vols per variant.

```text
  A  R2 0.312985  Volatili 10.6489%  Momentum 7.5021%  Size 7.7614%  Size Non 6.5487%  Trading  5.6630%
  B  R2 0.312986  Volatili 9.6609%  Momentum 7.1218%  Size 6.9735%  Size Non 5.7575%  Trading  5.0010%
 B2  R2 0.312546  Volatili 9.6890%  Momentum 7.0108%  Size 6.9906%  Size Non 5.7727%  Trading  5.0197%
  C  R2 0.312369  Volatili 9.5577%  Momentum 7.1187%  Size 6.8263%  Size Non 5.5692%  Trading  5.1972%
  D  R2 0.312250  Volatili 9.5468%  Momentum 7.0575%  Size 7.5062%  Size Non 6.5506%  Trading  5.1644%
```

### followup.txt

followup.py: variant table with changes, noise shares, Size vs SNL exposures, log-cap centre shift, empty-window zeros.

```text
R2 and the five style vols by variant (change from A):
    A R2 0.312985 (+0.000000)  Volati 10.65% (+0.0%)  Moment 7.50% (+0.0%)  Size 7.76% (+0.0%)  Size N 6.55% (+0.0%)  Tradin 5.66% (+0.0%)
    B R2 0.312986 (+0.000001)  Volati 9.66% (-9.3%)  Moment 7.12% (-5.1%)  Size 6.97% (-10.2%)  Size N 5.76% (-12.1%)  Tradin 5.00% (-11.7%)
   B2 R2 0.312546 (-0.000439)  Volati 9.69% (-9.0%)  Moment 7.01% (-6.5%)  Size 6.99% (-9.9%)  Size N 5.77% (-11.8%)  Tradin 5.02% (-11.4%)
    C R2 0.312369 (-0.000615)  Volati 9.56% (-10.2%)  Moment 7.12% (-5.1%)  Size 6.83% (-12.0%)  Size N 5.57% (-15.0%)  Tradin 5.20% (-8.2%)
    D R2 0.312250 (-0.000735)  Volati 9.55% (-10.3%)  Moment 7.06% (-5.9%)  Size 7.51% (-3.3%)  Size N 6.55% (+0.0%)  Tradin 5.16% (-8.8%)
  noise shares: A: Volati 3.5%, Moment 3.9%, Size 9.6%, Size N 18.0%, Tradin 10.8%; D: Volati 3.6%, Moment 4.0%, Size 9.8%, Size N 17.3%, Tradin 10.8%
  spread / median specific: A 0.3767/23.12%, B 0.3798/23.12%, B2 0.3792/23.13%, C 0.3804/23.08%, D 0.3804/23.10%

Size vs Size Nonlinearity among members with both, equal-weighted correlation:
  A: last day +0.920, median over the window +0.931; SNL range [-3.16, +0.47], 3% of members at the bottom, 12% at the top
  C: last day +0.920, median over the window +0.931; SNL range [-3.44, +0.47], 3% of members at the bottom, 16% at the top
  D: last day +0.931, median over the window +0.934; SNL range [-3.25, +0.57], 8% of members at the bottom, 11% at the top

log cap on 2026-11-03: everyone with a cap (470) mean 24.466 sd 1.408; members (425) mean 24.716 sd 1.183; the old centre sits 0.21 member sd below the members' mean
  RSTR: 18,634 values from windows with no price at all (all [0.0]); on the last day 26, of them members 0
  CMRA: 18,634 values from windows with no price at all (all [0.0]); on the last day 26, of them members 0
  STOM: 22,512 values from windows with no price at all (all [0.0]); on the last day 34, of them members 0
  STOQ: 20,780 values from windows with no price at all (all [0.0]); on the last day 32, of them members 0
  STOA: 14,300 values from windows with no price at all (all [0.0]); on the last day 26, of them members 0
```

### size_pair.txt

size_pair.py: Size-pair covariance and portfolio risk under A, C, D.

```text
A: Size vol 7.76%, SNL vol 6.55%, factor-return correlation -0.854
    smallest 85 members, equal-weighted    exposures Size -2.745 SNL -2.211  size-pair risk 11.69%
    largest 50 members, equal-weighted     exposures Size +0.011 SNL +0.469  size-pair risk 3.00%
    the index (members, cap-weighted)      exposures Size +0.013 SNL +0.012  size-pair risk 0.05%
C: Size vol 6.83%, SNL vol 5.57%, factor-return correlation -0.841
    smallest 85 members, equal-weighted    exposures Size -3.112 SNL -2.601  size-pair risk 11.97%
    largest 50 members, equal-weighted     exposures Size +0.012 SNL +0.466  size-pair risk 2.53%
    the index (members, cap-weighted)      exposures Size -0.000 SNL -0.000  size-pair risk 0.00%
D: Size vol 7.51%, SNL vol 6.55%, factor-return correlation -0.865
    smallest 85 members, equal-weighted    exposures Size -3.112 SNL -2.819  size-pair risk 11.83%
    largest 50 members, equal-weighted     exposures Size +0.012 SNL +0.551  size-pair risk 3.53%
    the index (members, cap-weighted)      exposures Size -0.000 SNL -0.000  size-pair risk 0.00%
```

### measure_clips.txt

measure_clips.py: by-hand reproduction check; raw clip and +/-4 counts and bounds by descriptor, last day and panel-wide; members' moments of each descriptor under everyone's scale.

```text
descriptors twice in 40s
by-hand reproduces the library over everyone: True; raw clip count 314,921 vs library 314,921
by-hand reproduces the library over members: True; raw clip count 348,002 vs library 348,002
library outside count over members: 86,458

last date 2026-11-03: 426 members of 506
                          ---- over everyone ----                           ---- over members ----
descr     raw clip mem/non  guard mem/non             bounds    raw clip mem/non  guard mem/non             bounds nonmem w/ value
LNCAP           0/1             0/9       [    19.4,    29.4]          2/1             0/19      [    20.1,    29.1]       45
LCAPCB         64/41            0/0       [   -1.21,    1.21]         81/45            0/0       [   -1.78,    1.78]       45
LPRI            0/1             0/1       [   0.675,    9.11]          0/1             0/4       [    1.09,    8.94]       45
DASTD           5/4             0/0       [ -0.0439,   0.217]         10/4             0/0       [ -0.0334,   0.201]       47
HILO           12/7             0/0       [ -0.0198,  0.0672]         15/8             0/0       [ -0.0181,  0.0638]       45
CMRA           15/5             0/0       [  -0.346,    1.15]         19/5             0/0       [  -0.269,    1.07]       77
BTSG           11/7             0/0       [  -0.029,  0.0529]         13/8             0/0       [ -0.0264,  0.0491]       66
HALPHA          2/3             0/1       [-0.00228, 0.00229]          3/7             0/5       [-0.00192, 0.00202]       66
SERDP           0/1             0/1       [   0.733,    1.23]          0/1             1/1       [   0.743,    1.22]       66
RSTR            9/3             0/0       [  -0.758,   0.913]          7/3             0/0       [  -0.799,       1]       80
STOM           12/13            0/0       [  -0.257,   0.647]         17/13            0/0       [  -0.184,   0.572]       80
STOQ           10/11            0/0       [   -2.86,    7.26]         16/14            0/0       [   -2.05,    6.42]       80
STOA            9/12            0/0       [   -2.74,    7.47]         12/16            0/0       [      -2,    6.64]       80
VLVR            5/1             1/7       [    22.4,    30.1]          6/3             2/18      [    22.9,    29.9]       54

panel-wide, every date:
  over everyone: raw clip 314,921 (non-members 78,077); guard 12,249 (non-members 10,421)
    raw/guard by descriptor: LNCAP 4,087/6,368, LCAPCB 132,467/0, LPRI 2,253/1,208, DASTD 12,554/0, HILO 16,502/10, CMRA 18,802/0, BTSG 6,539/0, HALPHA 2,774/832, SERDP 1/1, RSTR 6,561/822, STOM 41,050/0, STOQ 38,547/0, STOA 29,743/0, VLVR 3,041/3,008
  over members : raw clip 348,002 (non-members 86,458); guard 22,631 (non-members 19,588)
    raw/guard by descriptor: LNCAP 4,713/10,406, LCAPCB 139,554/0, LPRI 2,444/1,730, DASTD 15,265/0, HILO 18,662/18, CMRA 21,265/0, BTSG 8,851/0, HALPHA 3,440/2,505, SERDP 1/2, RSTR 6,423/1,908, STOM 45,648/0, STOQ 43,310/0, STOA 35,165/0, VLVR 3,261/6,062

members' cw mean / eq sd of each descriptor standardised over everyone, 2026-11-03:
  LNCAP    n=425  mean +0.0124  sd 0.8606
  LCAPCB   n=425  mean +0.0120  sd 0.8554
  LPRI     n=426  mean +0.0060  sd 0.8954
  DASTD    n=426  mean -0.0026  sd 0.9229
  HILO     n=426  mean -0.0029  sd 0.9190
  CMRA     n=425  mean -0.0033  sd 0.8912
  BTSG     n=426  mean -0.0018  sd 0.9170
  HALPHA   n=426  mean +0.0058  sd 0.8686
  SERDP    n=426  mean +0.0009  sd 0.9380
  RSTR     n=426  mean +0.0025  sd 1.0207
  STOM     n=425  mean -0.0072  sd 0.8225
  STOQ     n=425  mean -0.0072  sd 0.8199
  STOA     n=425  mean -0.0072  sd 0.8100
  VLVR     n=426  mean +0.0115  sd 0.8638
```

### scale_ratios.txt

scale_ratios.py: members' MAD scale over everyone's, and the centre shift, per descriptor.

```text
descr     scale ratio last day  median over estimated  centre shift, scales
LNCAP                    0.896                  0.960                +0.164
LCAPCB                   1.471                  1.334                -0.003
LPRI                     0.930                  0.967                +0.115
DASTD                    0.899                  0.928                -0.081
HILO                     0.941                  0.951                -0.080
CMRA                     0.896                  0.925                -0.003
BTSG                     0.922                  0.896                -0.057
HALPHA                   0.862                  0.907                +0.083
SERDP                    0.965                  0.965                +0.024
RSTR                     1.079                  1.000                +0.121
STOM                     0.836                  0.909                -0.002
STOQ                     0.837                  0.906                -0.013
STOA                     0.846                  0.898                -0.033
VLVR                     0.903                  0.933                +0.161
```

### guard_members.txt

guard_members.py: the +/-4 bound's moves among members alone.

```text
descriptors twice in 39s
guard moves among members: 3,043; outsiders 19,588; days 1214
  LNCAP    members  1,408  outsiders  8,998
  LCAPCB   members      0  outsiders      0
  LPRI     members     17  outsiders  1,713
  DASTD    members      0  outsiders      0
  HILO     members      9  outsiders      9
  CMRA     members      0  outsiders      0
  BTSG     members      0  outsiders      0
  HALPHA   members    290  outsiders  2,215
  SERDP    members      1  outsiders      1
  RSTR     members    543  outsiders  1,365
  STOM     members      0  outsiders      0
  STOQ     members      0  outsiders      0
  STOA     members      0  outsiders      0
  VLVR     members    775  outsiders  5,287
```

### descriptor_means.txt

descriptor_means.py: how far the +/-4 bound leaves each descriptor's members' mean from zero (new code).

```text
  LNCAP    members' cap-weighted mean at most 3.6e-05 from zero (2022-01-12); eq-weighted sd at most 0.016 from one
  LCAPCB   members' cap-weighted mean at most 5.1e-16 from zero (2026-10-26); eq-weighted sd at most 0.000 from one
  LPRI     members' cap-weighted mean at most 1.9e-05 from zero (2023-03-16); eq-weighted sd at most 0.002 from one
  DASTD    members' cap-weighted mean at most 1.1e-15 from zero (2023-10-04); eq-weighted sd at most 0.000 from one
  HILO     members' cap-weighted mean at most 1.8e-04 from zero (2023-03-24); eq-weighted sd at most 0.006 from one
  CMRA     members' cap-weighted mean at most 8.9e-16 from zero (2024-11-14); eq-weighted sd at most 0.000 from one
  BTSG     members' cap-weighted mean at most 8.2e-16 from zero (2025-05-22); eq-weighted sd at most 0.000 from one
  HALPHA   members' cap-weighted mean at most 1.4e-04 from zero (2025-05-02); eq-weighted sd at most 0.005 from one
  SERDP    members' cap-weighted mean at most 1.6e-05 from zero (2026-11-03); eq-weighted sd at most 0.001 from one
  RSTR     members' cap-weighted mean at most 2.8e-04 from zero (2025-01-07); eq-weighted sd at most 0.008 from one
  STOM     members' cap-weighted mean at most 6.3e-16 from zero (2025-02-14); eq-weighted sd at most 0.000 from one
  STOQ     members' cap-weighted mean at most 6.0e-16 from zero (2022-04-13); eq-weighted sd at most 0.000 from one
  STOA     members' cap-weighted mean at most 6.8e-16 from zero (2025-07-02); eq-weighted sd at most 0.000 from one
  VLVR     members' cap-weighted mean at most 9.6e-04 from zero (2025-11-13); eq-weighted sd at most 0.018 from one
```

### lag_probe.txt

lag_probe.py: same-day against one-day-lagged exposures (new code); empty-window zeros on members.

```text
same day (as run)               R2 0.3123  periods 939  Volati 9.55%  Moment 7.06%  Size 7.51%  Size N 6.55%  Tradin 5.16%
exposures from the day before   R2 0.2949  periods 938  Volati 7.79%  Moment 7.24%  Size 7.58%  Size N 6.67%  Tradin 5.23%
  RSTR: empty-window zeros 18,634, on index members 4,669
  CMRA: empty-window zeros 18,634, on index members 4,669
  STOM: empty-window zeros 22,512, on index members 705
  STOQ: empty-window zeros 20,780, on index members 567
  STOA: empty-window zeros 14,300, on index members 335
```

### zombie_members.txt

zombie_members.py: the members with an RSTR value from a window with no price.

```text
  COR     141 member-days, 2023-02-07 .. 2023-08-29; first price 2023-08-30
  CPAY    283 member-days, 2023-02-07 .. 2024-03-22; first price 2024-03-25
  ECHO    846 member-days, 2023-02-07 .. 2026-06-23; first price 2026-06-24
  EG      104 member-days, 2023-02-07 .. 2023-07-07; first price 2023-07-10
  FISV    335 member-days, 2024-07-12 .. 2025-11-10; first price 2022-01-03
  MRSH    736 member-days, 2023-02-07 .. 2026-01-13; first price 2026-01-14
  RVTY     68 member-days, 2023-02-07 .. 2023-05-15; first price 2023-05-16
  SKYD    918 member-days, 2023-02-07 .. 2026-10-05; first price 2026-10-06
  SW      354 member-days, 2023-02-07 .. 2024-07-05; first price 2024-07-08
  VMRK    884 member-days, 2023-02-07 .. 2026-08-17; first price 2026-08-18
```

### exposure_range.txt

exposure_range.py: style exposures past +/-4 (A and D).

```text
variant A:
  Volatility         last day: members [-2.17, +2.61], outside [-2.94, +2.23]; |x|>4 on the last day 0, over every day 421
  Momentum           last day: members [-3.61, +3.33], outside [-4.24, +2.48]; |x|>4 on the last day 2, over every day 1,300
  Size               last day: members [-3.37, +1.67], outside [-4.10, -2.83]; |x|>4 on the last day 11, over every day 7,152
  Size Nonlinearity  last day: members [-3.16, +0.47], outside [-3.16, -2.44]; |x|>4 on the last day 0, over every day 0
  Trading Activity   last day: members [-5.64, +3.84], outside [-2.22, +2.89]; |x|>4 on the last day 1, over every day 194
variant D:
  Volatility         last day: members [-2.45, +2.82], outside [-3.23, +2.46]; |x|>4 on the last day 0, over every day 672
  Momentum           last day: members [-3.80, +3.41], outside [-4.20, +2.43]; |x|>4 on the last day 2, over every day 2,296
  Size               last day: members [-3.82, +1.77], outside [-4.00, -3.21]; |x|>4 on the last day 0, over every day 10,215
  Size Nonlinearity  last day: members [-3.25, +0.57], outside [-3.25, -3.25]; |x|>4 on the last day 0, over every day 0
  Trading Activity   last day: members [-5.40, +4.03], outside [-2.46, +3.04]; |x|>4 on the last day 2, over every day 236
```

### timings

Wall-clock for the reference runs (`time`).

```text
head_with.time: real 0m51.311s
head_without.time: real 0m37.241s
new_with.time: real 0m42.166s
new_without.time: real 0m31.032s
```


---

## A separate change made in the same session

Jeff's standing rule that every piece of work ends with a full retro, recorded where this repo
keeps Code's standing instructions: `claude/CLAUDE_CODE_RULES.md` v1.5 (new Rule 43, the retro
template's closing `git add` section, a Key Principles line) and `.claude/commands/retro.md`.
It is its own change, with its own commit and file list, and is not in the `git add` below:

```
git add claude/CLAUDE_CODE_RULES.md .claude/commands/retro.md
```

---

## git add -- this change

Every file of the standardisation change, tracked and new, this retro included, and nothing
else:

```
git add scripts/run_sp500_model.py src/praxis/midoffice/model/characteristics.py src/praxis/midoffice/model/crosssectional.py src/praxis/midoffice/model/descriptors.py tests/midoffice/test_crosssectional.py tests/midoffice/test_descriptors.py docs/FACTOR_MODEL.md claude/handoffs/BRIEF_midoffice01_standardise_over_members.md claude/retros/RETRO_midoffice01_standardise_over_members.md
```
