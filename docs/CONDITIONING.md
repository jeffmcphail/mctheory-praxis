# Conditioning and the PSD floor

*Mid-office market risk engine. Written 2026-10-21.*

## The defect

`nearest_psd` repairs a non-PSD covariance by raising eigenvalues below a floor, where the
floor is a fraction of the largest eigenvalue. The default was `0.0`, which clips negative
eigenvalues to exactly zero and leaves the matrix **singular**.

A book lying in one of those null directions then has its common-factor variance computed as
a total cancellation. The risk comes back as `0.0`, or as rounding noise around it. That is
not an imprecise number; it is a false one -- the model asserting that a factor-neutral
long-short book cannot lose money.

This is not an exotic case. It is what a factor-neutral construction aims at, and the floor
is what manufactures it: clipping to zero creates the null direction.

## What was measured

Ground truth was mpmath at 50 decimal places, fed the *same* float64 inputs as the float64
path, so the comparison isolates rounding error rather than input differences. The covariance
was estimated from fewer observations than factors -- the realistic mechanism for negative
eigenvalues, from short history or from blocks pasted together out of separately estimated
models -- across five shapes (40x30, 70x50, 70x200, 150x100, 30x29 factors-by-observations)
and three seeds each.

Two books per matrix: a long-only one, and a factor-neutral one placed along the covariance's
smallest-eigenvalue direction.

| `relative_floor` | factor-neutral book, digits (worst of 15) | long-only distortion (worst) | margin vs 0.01 bps |
|---|---|---|---|
| `0.0` | **-0.7** | -- | -- |
| `1e-8` | 9.3 | 9.9e-06 bps | **1011x** |
| `1e-6` | 11.4 | 9.9e-04 bps | 10x |
| `1e-4` | 13.4 | 9.9e-02 bps | 0.1x -- exceeds |

Negative digits mean the error exceeded the value. That held on 11 of the 15 configurations;
the four exceptions were the `70x200` cases, which have more observations than factors and
therefore no negative eigenvalues at all.

## Why `1e-8`

**The asymmetry.** Distortion is *linear* in the floor -- each 100x step in the floor is a
100x step in the distortion -- while precision gained is *logarithmic*, about +2 digits per
100x. So there is a wide band where digits come almost free, and `1e-8` sits at its knee: it
captures nearly all the precision benefit of `1e-6` for one hundredth of the distortion.

**It cannot regress a healthy matrix.** `nearest_psd` only touches eigenvalues already below
the floor. On the `70x200` matrices the distortion was *exactly* zero at every setting tested,
and `test_floor_is_a_no_op_on_a_well_conditioned_matrix` asserts bit-identical equality rather
than approximate. The change alters numbers only where those numbers were previously noise.

**Nine digits is ample.** Reporting needs four or five significant figures.

Jeff had independently found the same fix in production -- a floor one or two orders of
magnitude above near-zero cured the precision problem without moving the risk numbers. The
measurement confirms the direction and says the step can be much larger: at `1e-12` the
factor-neutral book still holds only 5.4 digits.

Past roughly `1e-4` the floor is changing the risk numbers rather than conditioning them, so
`RiskConfig` refuses anything above `1e-3` outright. Both sides of that boundary are pinned by
tests, for the same reason the parity harness pins its tolerances from above and below: a
limit asserted from one side only drifts.

## The configuration layer

`src/praxis/midoffice/risk/config.py` holds `RiskConfig`, a frozen dataclass whose every field
carries its own calibration in its docstring. `config/midoffice_risk.toml` is the file form.

Resolution order, most specific first:

1. an explicit `RiskConfig` passed to the function
2. the file named by `$PRAXIS_MIDOFFICE_RISK_CONFIG`
3. `config/midoffice_risk.toml`, searched upward from the working directory
4. the built-in calibrated defaults

A missing file is not an error: the defaults *are* the calibrated values, so deleting the file
changes nothing. `source` records which of the four applied and travels onto the result with
the rest of the provenance.

Three deliberate choices:

* **No import-time file read.** Functions default to a module-level `DEFAULTS` singleton
  rather than loading a file when imported, because an import-time read makes behaviour depend
  on the working directory and makes tests order-dependent. A caller wanting file-based
  configuration calls `RiskConfig.load()` and passes the result.
* **Unknown keys are refused, not ignored.** A silently ignored key looks exactly like a
  setting that applied and did not.
* **Values that would produce wrong numbers are refused, not clamped.** A floor of `0.5`
  would flatten the covariance toward a multiple of the identity; the error says so.

Passing `relative_floor=0.0` explicitly still restores the old singular behaviour, for anyone
who needs to reproduce an old number.

## The diagnostic

Fixing the floor removes the common case. It does not remove the hazard, because a book can
sit in a cancellation-heavy direction of a perfectly well-conditioned matrix. So
`RiskDecomposition` now carries the conditioning alongside the number, the way it already
carries the PSD repair and the proxied factors:

```
amplification  = (|x| |F| |x| + specific_variance) / total_variance     >= 1
digits_intact  = 15.65 - log10(amplification) - log10(log2(n_factors))
```

The numerator is the sum of the magnitudes the quadratic form adds and subtracts; its ratio to
the variance is how much cancellation the arithmetic performs, and therefore how much
precision it costs. Specific variance appears on top because it is a sum of squares and
cancels nothing -- which also means **specific risk improves conditioning**, and a
factor-neutral book with real specific risk has a trustworthy total even when its
common-factor component does not.

Below `conditioning_warn_digits` (default 10), `qualifications()` gains a note saying the
amplification, the surviving digits, and what to do about it. `digits_intact` is a conservative
bound -- it assumes every rounding error aligns, which measurement shows understates reality by
about 2.6 digits -- and conservative is the right direction for a warning.

Cost is one extra matrix-vector product per `decompose`, the same order as the variance it
accompanies.

### What it looks like

On a rank-deficient 40-factor matrix, the same two books:

| | floor `0.0` | floor `1e-8` |
|---|---|---|
| long-only risk | `1.69563276e-01` | `1.69563276e-01` (identical) |
| long-only amplification | 1.81, 14.7 digits, clean | 1.81, 14.7 digits, clean |
| factor-neutral risk | **`0.0`** | `3.16543902e-06` |
| factor-neutral amplification | **inf**, 0.0 digits | 9.14e+07, 7.0 digits |

The long-only book is unaffected to every digit it reports. The factor-neutral book goes from
a false zero to a real number -- and is still flagged, correctly, because that book genuinely
does live in a cancellation-heavy direction. The floor fixes the falsehood; the diagnostic
reports the residual.

## Scope

Settled: why `1e-8`, both boundaries of the admissible range, that the floor is a no-op on a
healthy matrix, and that the diagnostic fires on the pathological book and stays quiet on a
normal one.

Not covered:

* **Which floor a model vendor recommends.** Barra or another provider may publish one with
  the matrix, and a published floor should win over a measured default. There is nowhere to put
  that yet -- `FactorCovariance` carries no vendor metadata.
* **Amplification on contributions and decomposition rows.** Those divide by the aggregate
  risk, so they inherit its conditioning; whether they need their own figure is untested.
* **Tail numbers.** VaR and expected shortfall are `risk x constant`, so they inherit
  `digits_intact` unchanged, but nothing asserts that.

## Tests

`tests/midoffice/test_config.py` (27) covers loading, precedence, the upward search, unknown-key
refusal, every validation bound, and that the shipped TOML sets exactly the tunables the code
reads -- a file listing a key the code ignores being worse than no file at all.

`tests/midoffice/test_conditioning.py` (16) covers the calibration properties, chosen so that
none needs a high-precision reference: the no-op on a well-conditioned matrix, the false zero at
floor `0.0`, monotonic amplification and digits across the floor sweep, insensitivity of a
long-only book up to `1e-5` and distortion past it, and the diagnostic's behaviour including
that specific risk dilutes the amplification.

## Cross-platform reproducibility

The first run of this work on Windows failed two tests, and the reason is worth recording
because it bears directly on the parity harness.

The same factor-neutral book at `relative_floor = 0.0` produces:

| platform | risk | amplification | digits |
|---|---|---|---|
| Linux x86-64, OpenBLAS | `0.0` exactly | `inf` | none |
| Windows, MKL | `1.2473336370178445e-10` | `6.574010e+16` | -1.89 |

Both are correct reports of the same defect. A total cancellation has no defined debris, so
which one a machine produces depends on the BLAS's summation order and blocking. The Windows
result is the more dangerous of the two to receive: a plausible-looking small number instead
of an obvious zero.

The general statement is that **the engine is not bit-reproducible across BLAS
implementations in the ill-conditioned regime, and `digits_intact` tells you where that
regime begins.** Where conditioning is healthy, cross-platform differences sit at float noise,
about 1e-16 relative -- six orders inside the parity harness's 1e-10 tolerance. Where
amplification is large, they are unbounded.

Two consequences:

* **Golden files are platform-bound in the ill-conditioned regime.** A parity run against
  numbers captured on a different machine is meaningful only for results whose
  `digits_intact` is comfortable. The diagnostic is therefore also a reproducibility
  predictor, and a parity report that disagrees on a flagged number is telling you about the
  BLAS rather than about the port.
* **A test must assert the property, not the value.** The two failures were both assertions
  on one platform's debris (`total_risk == 0.0`, `amplification == inf`) rather than on the
  property that holds everywhere (no digit of the number is correct). They now assert the
  latter, and the severity ordering was fixed at the same time: a total cancellation reports
  `-inf` digits rather than `0.0`, because `0.0` ranked the worst possible case above a
  merely astronomical one.
