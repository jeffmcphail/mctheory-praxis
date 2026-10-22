"""Tests for the PSD floor's calibration and the conditioning diagnostic.

The floor was chosen by measuring float64 against mpmath at 50 digits over 15 matrix shapes.
Those measurements are recorded in `RiskConfig.psd_relative_floor` and in
docs/CONDITIONING.md; repeating them here would make the suite slow and add a dependency on
mpmath. What the tests pin instead are the structural properties that make the chosen value
defensible, each of which needs no high-precision reference:

* On a well-conditioned matrix the floor is a measured no-op, so raising the default cannot
  regress a healthy model. This is the property that made the change safe to make.
* A zero floor reports a factor-neutral book's risk as exactly 0.0. That is the defect the
  change fixes, and it is a wrong answer rather than an imprecise one.
* Amplification falls, and digits rise, monotonically as the floor rises -- which is what
  makes "pick the knee of the curve" a meaningful instruction.
* A long-only book's risk is insensitive to the floor across the whole admissible range.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from praxis.midoffice.risk.basis import FactorCovariance, active_basis, nearest_psd
from praxis.midoffice.risk.config import DEFAULTS, RiskConfig
from praxis.midoffice.risk.exposures import ExposureVector, SpecificRisk
from praxis.midoffice.risk.metrics import MACHINE_DIGITS, common_factor_abs_term, decompose

N_FACTORS = 40
IDS = tuple(range(100, 100 + N_FACTORS))
FLOORS = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4)


def rank_deficient(seed=17, n_obs=30):
    """A covariance estimated from fewer observations than factors.

    This is the realistic mechanism for negative eigenvalues -- short history, or blocks
    pasted together from separately estimated models -- rather than an artificial one.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(size=(n_obs, N_FACTORS)) * 0.05
    matrix = np.cov(returns, rowvar=False)
    return (matrix + matrix.T) / 2


def well_conditioned(seed=17, n_obs=400):
    return rank_deficient(seed=seed, n_obs=n_obs)


def book(x, specific=0.0):
    return ExposureVector(
        {IDS[i]: float(x[i]) for i in range(N_FACTORS)},
        SpecificRisk({"ISS": specific}) if specific else SpecificRisk({}),
    )


def neutral_book(matrix):
    """A book lying along the covariance's smallest-eigenvalue direction."""
    _, vectors = np.linalg.eigh(matrix)
    return book(vectors[:, 0] * 0.3)


def long_only(seed=5, specific=0.15):
    rng = np.random.default_rng(seed)
    return book(np.abs(rng.normal(size=N_FACTORS)) * 0.25, specific=specific)


def risk_at(matrix, vector, floor):
    basis = active_basis(FactorCovariance(IDS, matrix), vector.exposures, relative_floor=floor)
    return decompose(vector, basis)


# --- the property that made the change safe -----------------------------------------


def test_floor_is_a_no_op_on_a_well_conditioned_matrix():
    """Exact equality, not approximate.

    `nearest_psd` only touches eigenvalues already below the floor. With none below it, the
    returned matrix is the input, so every number downstream is bit-identical. This is why
    raising the default cannot regress a healthy model.
    """
    matrix = well_conditioned()
    assert (np.linalg.eigvalsh(matrix) > 0).all(), "fixture should be positive definite"
    vector = long_only()
    baseline = risk_at(matrix, vector, 0.0).total_risk
    for floor in FLOORS:
        assert risk_at(matrix, vector, floor).total_risk == baseline, floor


def test_no_repair_is_reported_when_nothing_is_below_the_floor():
    matrix = well_conditioned()
    _, repair = nearest_psd(matrix, relative_floor=DEFAULTS.psd_relative_floor)
    assert not repair.repaired
    assert repair.n_negative == 0


# --- the defect the change fixes ------------------------------------------------------


def test_zero_floor_leaves_a_factor_neutral_book_with_no_correct_digits():
    """The wrong answer, pinned as a property rather than as a value.

    Clipping to exactly zero leaves a null direction in the covariance, and a book lying in
    it has its variance computed as a total cancellation. What that *produces* is
    platform-dependent debris: exactly 0.0 where the cancellation is total (OpenBLAS on
    Linux x86-64) and 1.2e-10 where the summation order leaves a residue (MKL on Windows).
    An earlier version of this test asserted `== 0.0` and so pinned one platform's artifact.

    The property that holds everywhere is that not one digit of the number is correct. Note
    the Windows case is the *worse* of the two to receive: a plausible-looking small number
    rather than an obvious zero.
    """
    matrix = rank_deficient()
    result = risk_at(matrix, neutral_book(matrix), 0.0)
    assert result.digits_intact <= 0.0, "a surviving digit here would mean the floor is unnecessary"
    assert result.amplification > 1e14
    assert not result.is_well_conditioned
    # Whatever debris it produced, it is orders away from the honest answer.
    floored = risk_at(matrix, neutral_book(matrix), DEFAULTS.psd_relative_floor)
    assert result.total_risk < floored.total_risk / 100.0


def test_total_cancellation_is_ranked_worse_than_a_large_amplification():
    """Severity must order correctly, or the diagnostic misleads at its own extreme.

    An infinite amplification is the worst possible case. Reporting 0.0 digits for it while a
    finite-but-astronomical amplification reports a negative number would rank the worst case
    above a merely terrible one.
    """
    matrix = rank_deficient()
    vector = neutral_book(matrix)
    total = risk_at(matrix, vector, 0.0)
    if math.isfinite(total.amplification):
        pytest.skip("this platform's BLAS leaves a residue rather than cancelling exactly")
    assert total.digits_intact == float("-inf")
    assert total.digits_intact < risk_at(matrix, vector, 1e-12).digits_intact


def test_the_note_says_no_digit_is_correct_when_none_is():
    matrix = rank_deficient()
    result = risk_at(matrix, neutral_book(matrix), 0.0)
    note = " ".join(result.qualifications())
    assert "correct" in note
    assert "psd_relative_floor" in note


def test_the_default_floor_gives_that_book_a_real_number():
    matrix = rank_deficient()
    result = risk_at(matrix, neutral_book(matrix), DEFAULTS.psd_relative_floor)
    assert result.total_risk > 0.0
    assert math.isfinite(result.amplification)
    assert result.digits_intact > 5.0


# --- the calibration curve ------------------------------------------------------------


def test_amplification_falls_monotonically_as_the_floor_rises():
    """Including the floor-0 case, whose amplification may be infinite or merely enormous.

    Both outcomes exceed every later one, so the monotonicity holds without special-casing
    the first element -- which is what the earlier version did, by asserting it was exactly
    infinite.
    """
    matrix = rank_deficient()
    vector = neutral_book(matrix)
    amplifications = [risk_at(matrix, vector, f).amplification for f in FLOORS]
    assert all(a > b for a, b in zip(amplifications, amplifications[1:])), amplifications


def test_digits_rise_monotonically_as_the_floor_rises():
    matrix = rank_deficient()
    vector = neutral_book(matrix)
    digits = [risk_at(matrix, vector, f).digits_intact for f in FLOORS]
    assert all(a < b for a, b in zip(digits, digits[1:])), digits


def test_a_long_only_book_is_insensitive_to_the_floor_up_to_1e_minus_5():
    """Up to 1e-5 the distortion stays below reporting precision.

    0.01 bps is two decimal places in basis points, which is how risk is reported, so the
    assertion is in those terms rather than in relative error. 1e-5 is the top of the safe
    range; the next test pins what happens past it.
    """
    matrix = rank_deficient()
    vector = long_only()
    baseline = risk_at(matrix, vector, 0.0).total_risk
    for floor in [f for f in FLOORS if f <= 1e-5]:
        drift_bps = abs(risk_at(matrix, vector, floor).total_risk - baseline) * 1e4
        assert drift_bps < 0.01, (floor, drift_bps)


def test_a_floor_of_1e_minus_4_does_distort_a_long_only_book():
    """The other side of the boundary, pinned.

    This is the measurement behind the validation bound in `RiskConfig`: past roughly 1e-4
    the floor is changing the risk numbers rather than conditioning them. Asserting only the
    safe side would let the boundary drift upward unnoticed, so both sides are tested -- the
    same reason the parity harness pins its tolerances from above and below.
    """
    matrix = rank_deficient()
    vector = long_only()
    baseline = risk_at(matrix, vector, 0.0).total_risk
    drift_bps = abs(risk_at(matrix, vector, 1e-4).total_risk - baseline) * 1e4
    assert drift_bps > 0.01, drift_bps


def test_the_default_floor_distorts_a_long_only_book_by_far_less_than_reporting_precision():
    matrix = rank_deficient()
    vector = long_only()
    baseline = risk_at(matrix, vector, 0.0).total_risk
    drift_bps = abs(risk_at(matrix, vector, DEFAULTS.psd_relative_floor).total_risk - baseline) * 1e4
    assert drift_bps < 0.0001, drift_bps


# --- the diagnostic -------------------------------------------------------------------


def test_amplification_is_never_below_one():
    """It is a magnification factor; below one would mean the arithmetic gained precision."""
    matrix = well_conditioned()
    for vector in (long_only(), neutral_book(matrix), long_only(seed=9, specific=0.0)):
        assert risk_at(matrix, vector, DEFAULTS.psd_relative_floor).amplification >= 1.0


def test_a_long_only_book_is_well_conditioned_and_carries_no_conditioning_note():
    result = risk_at(well_conditioned(), long_only(), DEFAULTS.psd_relative_floor)
    assert result.is_well_conditioned
    assert result.digits_intact > 12.0
    assert not any("cancellation" in q for q in result.qualifications())


def test_a_factor_neutral_book_is_flagged_with_the_numbers_in_the_note():
    matrix = rank_deficient()
    result = risk_at(matrix, neutral_book(matrix), DEFAULTS.psd_relative_floor)
    assert not result.is_well_conditioned
    notes = [q for q in result.qualifications() if "cancellation" in q]
    assert len(notes) == 1
    note = notes[0]
    assert "amplification" in note
    assert "digits" in note
    assert "psd_relative_floor" in note, "the note should say what to do about it"


def test_the_warn_threshold_comes_from_the_config():
    matrix = rank_deficient()
    vector = neutral_book(matrix)
    basis = active_basis(FactorCovariance(IDS, matrix), vector.exposures)
    strict = decompose(vector, basis, config=RiskConfig(conditioning_warn_digits=16.0))
    lenient = decompose(vector, basis, config=RiskConfig(conditioning_warn_digits=1.0))
    assert not strict.is_well_conditioned
    assert lenient.is_well_conditioned
    assert strict.digits_intact == lenient.digits_intact, "the number is the same; only the bar moves"


def test_digits_intact_is_bounded_by_the_machine():
    """No configuration can claim more precision than float64 carries."""
    matrix = well_conditioned()
    for vector in (long_only(), neutral_book(matrix)):
        assert risk_at(matrix, vector, DEFAULTS.psd_relative_floor).digits_intact <= MACHINE_DIGITS


def test_an_empty_basis_is_not_flagged():
    """A book with no factor exposure has no quadratic form to cancel in."""
    basis = active_basis(FactorCovariance(IDS, well_conditioned()), {})
    result = decompose(ExposureVector({}, SpecificRisk({"X": 0.2})), basis)
    assert result.amplification == 1.0
    assert result.is_well_conditioned
    assert common_factor_abs_term(ExposureVector({}, SpecificRisk({})), basis) == 0.0


def test_specific_risk_dilutes_the_amplification():
    """Specific variance is a sum of squares and cancels nothing, so it improves conditioning.

    A factor-neutral book with real specific risk has a trustworthy TOTAL risk even though
    its common-factor component is computed by near-total cancellation. The amplification is
    reported against the total, so it reflects that.
    """
    matrix = rank_deficient()
    _, vectors = np.linalg.eigh(matrix)
    bare = book(vectors[:, 0] * 0.3)
    with_specific = book(vectors[:, 0] * 0.3, specific=0.15)
    floor = DEFAULTS.psd_relative_floor
    assert risk_at(matrix, with_specific, floor).amplification < risk_at(matrix, bare, floor).amplification
    assert risk_at(matrix, with_specific, floor).is_well_conditioned
