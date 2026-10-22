"""Tests for the parity harness.

These divide into three kinds, and the second kind is the one that matters.

*Agreement* tests check that the engine and the independent reference produce the same
numbers. *Detection* tests check that the harness would notice if they did not -- by
perturbing one input at a time and asserting both that a mismatch is reported and that it
lands on the metrics that input should influence. A harness that only ever reports agreement
is indistinguishable from a harness that compares nothing, and the detection tests are what
separate the two. *Calibration* tests pin the tolerance boundary from both sides, so a
tolerance cannot be quietly widened until the suite goes green.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from praxis.midoffice.parity.case import (
    FactorSpec,
    InvalidCase,
    ParityCase,
    PositionSpec,
    TailCase,
)
from praxis.midoffice.parity.compare import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    Tolerance,
    TolerancePolicy,
    compare,
    policy_for,
)
from praxis.midoffice.parity.engine import engine_records
from praxis.midoffice.parity.harness import KNOWN_GAPS, run_parity
from praxis.midoffice.parity.record import (
    COMMON_FACTOR_RISK,
    COVERAGE_COUNT,
    COVERAGE_WEIGHT,
    CTR_COMMON_FACTOR,
    CTR_SPECIFIC,
    CTR_TOTAL,
    DuplicateRecord,
    EXPECTED_SHORTFALL,
    ROW_CONTRIBUTION,
    SPECIFIC_RISK,
    TOTAL_RISK,
    VALUE_AT_RISK,
    WEIGHT,
    Record,
    index,
)
from praxis.midoffice.parity.reference import UnsupportedCase, reference_records
from praxis.midoffice.parity.synthetic import shapes_present, synthetic_case

SEEDS = (1, 7, 13, 42, 99, 404, 2718)


def tiny_case(**overrides) -> ParityCase:
    """Two factors, two names, every number checkable by hand."""
    defaults = dict(
        name="tiny",
        factors=(FactorSpec(1, "F1", group="G"), FactorSpec(2, "F2", group="G")),
        covariance=((0.04, 0.01), (0.01, 0.09)),
        positions=(
            PositionSpec("AAA", 0.6, 0.5, {1: 1.0}, 0.2),
            PositionSpec("BBB", 0.4, 0.5, {2: 1.2}, 0.3),
        ),
        groups={"G": "Total Risk"},
        tails=(TailCase(0.95),),
        source="synthetic:tiny_case",
    )
    defaults.update(overrides)
    return ParityCase(**defaults)


def mutate_position(case: ParityCase, idx: int, **changes) -> ParityCase:
    positions = list(case.positions)
    positions[idx] = replace(positions[idx], **changes)
    return replace(case, positions=tuple(positions))


def mutate_covariance(case: ParityCase, i: int, j: int, value: float) -> ParityCase:
    matrix = [list(row) for row in case.covariance]
    matrix[i][j] = value
    matrix[j][i] = value  # keep it symmetric, or validate() rejects it for the wrong reason
    return replace(case, covariance=tuple(tuple(r) for r in matrix))


# --------------------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------------------


def test_tiny_case_agrees():
    assert run_parity(tiny_case()).ok


@pytest.mark.parametrize("seed", SEEDS)
def test_synthetic_cases_agree(seed):
    report = run_parity(synthetic_case(seed=seed))
    assert report.ok, report.describe()


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_agreement_holds_on_covered_weight_basis(seed):
    report = run_parity(synthetic_case(seed=seed, weight_basis="covered"))
    assert report.ok, report.describe()


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_agreement_holds_with_refined_z(seed):
    """With z refined, both sides are at float precision and tolerances tighten."""
    case = synthetic_case(seed=seed, refine=True)
    policy = policy_for(case)
    assert policy.by_metric == {}, "a refined case should need no special tail tolerance"
    assert run_parity(case).ok


def test_agreement_covers_a_meaningful_number_of_records():
    """A harness comparing a handful of numbers would pass these tests too."""
    report = run_parity(synthetic_case(seed=7))
    assert report.compared > 500, f"only {report.compared} records compared"


def test_generator_actually_produces_the_awkward_shapes():
    """The agreement tests are only as good as the shapes the fixtures contain."""
    shapes = shapes_present(synthetic_case(seed=7))
    for shape, count in shapes.items():
        assert count > 0, f"generator produced no {shape} positions: {shapes}"


def test_hand_computed_benchmark_risk():
    """One number verified outside both implementations.

    x_bench = {F1: 0.5, F2: 0.6}; x'Fx = 0.01 + 0.003 + 0.003 + 0.0324 = 0.0484; sqrt = 0.22.
    """
    records = index(reference_records(tiny_case()))
    assert records[(COMMON_FACTOR_RISK, "benchmark", "", "")] == pytest.approx(0.22, abs=1e-15)
    engine = index(engine_records(tiny_case()))
    assert engine[(COMMON_FACTOR_RISK, "benchmark", "", "")] == pytest.approx(0.22, abs=1e-15)


def test_hand_computed_beta():
    """cov(p,b) = 0.030 specific + 0.04392 factor; var_b = 0.0325 + 0.0484; ratio below."""
    expected = (0.030 + 0.04392) / (0.0325 + 0.0484)
    for records in (index(reference_records(tiny_case())), index(engine_records(tiny_case()))):
        assert records[("beta", "portfolio", "", "")] == pytest.approx(expected, rel=1e-14)


# --------------------------------------------------------------------------------------
# Detection -- one input at a time, with the metrics it must move
# --------------------------------------------------------------------------------------

MUTATIONS = [
    pytest.param(
        lambda c: mutate_position(c, 10, weight=c.positions[10].weight * 1.5),
        {WEIGHT, TOTAL_RISK, CTR_TOTAL},
        id="portfolio_weight",
    ),
    pytest.param(
        lambda c: mutate_position(c, 10, benchmark_weight=c.positions[10].benchmark_weight * 1.5),
        {WEIGHT, TOTAL_RISK, CTR_TOTAL},
        id="benchmark_weight",
    ),
    pytest.param(
        lambda c: mutate_position(c, 10, specific_volatility=c.positions[10].specific_volatility * 1.5),
        {SPECIFIC_RISK, CTR_SPECIFIC},
        id="specific_volatility",
    ),
    pytest.param(
        lambda c: mutate_position(
            c, 10, exposures={f: v * 1.5 for f, v in c.positions[10].exposures.items()}
        ),
        {COMMON_FACTOR_RISK, CTR_COMMON_FACTOR, ROW_CONTRIBUTION},
        id="factor_exposure",
    ),
    pytest.param(
        lambda c: mutate_covariance(c, 0, 0, c.covariance[0][0] * 1.5),
        {COMMON_FACTOR_RISK, ROW_CONTRIBUTION},
        id="covariance_diagonal",
    ),
    pytest.param(
        lambda c: mutate_covariance(c, 0, 1, c.covariance[0][1] * 0.5),
        {COMMON_FACTOR_RISK, ROW_CONTRIBUTION},
        id="covariance_offdiagonal",
    ),
    pytest.param(
        lambda c: mutate_position(c, 10, covered=False),
        {COVERAGE_COUNT, COVERAGE_WEIGHT, TOTAL_RISK, CTR_TOTAL},
        id="coverage_flag",
    ),
    pytest.param(
        lambda c: mutate_position(c, 10, issuer_key="ISS1"),
        {SPECIFIC_RISK, CTR_SPECIFIC},
        id="issuer_key_merge",
    ),
    pytest.param(
        lambda c: replace(c, weight_basis="covered"),
        {TOTAL_RISK, WEIGHT},
        id="weight_basis",
    ),
    pytest.param(
        lambda c: replace(c, tails=(TailCase(0.99),)),
        {VALUE_AT_RISK, EXPECTED_SHORTFALL},
        id="tail_confidence",
    ),
]


@pytest.mark.parametrize("mutate,expected_metrics", MUTATIONS)
def test_mutating_one_input_is_detected(mutate, expected_metrics):
    """Perturb one input and confirm the harness notices, on the right metrics.

    This is the test that gives the agreement tests their meaning. If a mutation produced no
    mismatch, that input would not be influencing any number the harness checks -- the
    agreement would be real but vacuous over that part of the surface.
    """
    case = synthetic_case(seed=7)
    mutated = mutate(case)
    report = compare(
        reference_records(case),
        engine_records(mutated),
        policy=policy_for(case),
        known_gaps=KNOWN_GAPS,
        case_name=f"{case.name}-mutated",
    )
    assert not report.ok, "mutation produced no disagreement at all"
    # Detection happens through either channel: a number that differs, or a key that appears
    # or vanishes. Changing a tail confidence does the latter -- the confidence is part of the
    # record key -- and an assertion that only looked at numeric mismatches would call that
    # an undetected mutation when the harness had in fact caught it.
    moved = (
        {m.key[0] for m in report.mismatches}
        | {k[0] for k in report.missing}
        | {k[0] for k in report.extra}
    )
    assert expected_metrics <= moved, (
        f"mutation did not move {sorted(expected_metrics - moved)}; it moved {sorted(moved)}"
    )


def test_every_mutation_is_distinguishable():
    """No two mutations produce the same mismatch signature.

    If two different inputs moved exactly the same numbers by the same amounts, the harness
    could not tell them apart and one of them would be untested in practice.
    """
    case = synthetic_case(seed=7)
    signatures = {}
    for param in MUTATIONS:
        mutate, _ = param.values
        report = compare(
            reference_records(case),
            engine_records(mutate(case)),
            policy=policy_for(case),
            known_gaps=KNOWN_GAPS,
        )
        signature = (
            tuple(sorted((m.key, round(m.actual, 12)) for m in report.mismatches)),
            tuple(sorted(report.missing)),
            tuple(sorted(report.extra)),
        )
        assert signature not in signatures, (
            f"{param.id} is indistinguishable from {signatures[signature]}"
        )
        signatures[signature] = param.id


# --------------------------------------------------------------------------------------
# Calibration -- the tolerance boundary, from both sides
# --------------------------------------------------------------------------------------


def _shift(records, key, factor):
    return tuple(
        replace(r, value=r.value * factor) if r.key == key else r for r in records
    )


def test_perturbation_just_above_tolerance_is_caught():
    case = tiny_case()
    records = reference_records(case)
    key = (TOTAL_RISK, "portfolio", "", "")
    report = compare(records, _shift(records, key, 1.0 + DEFAULT_RTOL * 10.0))
    assert [m.key for m in report.mismatches] == [key]


def test_perturbation_just_below_tolerance_is_accepted():
    case = tiny_case()
    records = reference_records(case)
    key = (TOTAL_RISK, "portfolio", "", "")
    report = compare(records, _shift(records, key, 1.0 + DEFAULT_RTOL * 0.1))
    assert report.ok, report.describe()


def test_absolute_floor_catches_a_near_zero_number():
    """A relative test is useless at zero, so the floor has to do the work there."""
    records = (Record(TOTAL_RISK, "portfolio", "", "", 0.0),)
    actual = (Record(TOTAL_RISK, "portfolio", "", "", DEFAULT_ATOL * 10.0),)
    assert not compare(records, actual).ok
    inside = (Record(TOTAL_RISK, "portfolio", "", "", DEFAULT_ATOL * 0.1),)
    assert compare(records, inside).ok


def test_expected_shortfall_tolerance_exceeds_value_at_risk_tolerance():
    """ES inherits z^2 of the z error where VaR inherits z^1; the policy must reflect that."""
    case = tiny_case(tails=(TailCase(0.99),))
    policy = policy_for(case)
    var_tol = policy.for_metric(VALUE_AT_RISK).rtol
    es_tol = policy.for_metric(EXPECTED_SHORTFALL).rtol
    z = 2.3263478740408408  # inv_cdf(0.99)
    assert es_tol == pytest.approx(var_tol * z * z, rel=1e-6)


def test_acklam_error_bound_is_real_and_is_not_zero():
    """Verify the documented claim rather than tolerating it.

    The engine's unrefined Acklam z and the reference's AS241 z genuinely differ. The test
    asserts the difference is non-zero -- so the two really are different algorithms, not the
    same one twice -- and that it sits inside the bound the engine documents.
    """
    from praxis.midoffice.risk.tail import ACKLAM_RELATIVE_ERROR

    case = tiny_case(tails=(TailCase(0.95, refine=False),))
    ref = index(reference_records(case))
    eng = index(engine_records(case))
    key = (VALUE_AT_RISK, "portfolio", "", "c=0.95,h=1")
    expected, actual = ref[key], eng[key]
    relative = abs(actual - expected) / abs(expected)
    assert relative > 0.0, "the two sides computed z identically; the reference is not independent"
    assert relative <= ACKLAM_RELATIVE_ERROR * 1.5, f"relative difference {relative:.3e}"


def test_refining_z_closes_the_gap():
    """The same case with refinement on should agree far more tightly."""
    unrefined = tiny_case(tails=(TailCase(0.95, refine=False),))
    refined = tiny_case(tails=(TailCase(0.95, refine=True),))
    key_u = (VALUE_AT_RISK, "portfolio", "", "c=0.95,h=1")

    def gap(case, key):
        ref, eng = index(reference_records(case)), index(engine_records(case))
        return abs(eng[key] - ref[key]) / abs(ref[key])

    assert gap(refined, key_u) < gap(unrefined, key_u) / 100.0


# --------------------------------------------------------------------------------------
# Absence, duplication and the gap list
# --------------------------------------------------------------------------------------


def test_a_dropped_record_is_reported_not_ignored():
    records = reference_records(tiny_case())
    key = (TOTAL_RISK, "portfolio", "", "")
    thinned = tuple(r for r in records if r.key != key)
    report = compare(records, thinned)
    assert not report.ok
    assert report.missing == (key,)


def test_an_unexpected_extra_record_is_reported():
    records = reference_records(tiny_case())
    extra = records + (Record("invented_metric", "portfolio", "", "", 1.0),)
    report = compare(records, extra)
    assert not report.ok
    assert report.extra == (("invented_metric", "portfolio", "", ""),)


def test_a_declared_gap_is_excused_but_still_counted():
    records = reference_records(tiny_case())
    key = (TOTAL_RISK, "portfolio", "", "")
    thinned = tuple(r for r in records if r.key != key)
    report = compare(records, thinned, known_gaps={key: "declared for this test"})
    assert report.ok
    assert report.expected_gaps == (key,)


def test_a_stale_gap_fails():
    """A gap that has been closed must fail, or the list silently rots."""
    records = reference_records(tiny_case())
    key = (TOTAL_RISK, "portfolio", "", "")
    report = compare(records, records, known_gaps={key: "no longer true"})
    assert not report.ok
    assert report.stale_gaps == (key,)


def test_known_gaps_are_all_live():
    """Every entry in KNOWN_GAPS must still be a real gap, with a reason attached."""
    report = run_parity(synthetic_case(seed=7))
    assert report.stale_gaps == (), f"stale: {report.stale_gaps}"
    assert set(report.expected_gaps) == set(KNOWN_GAPS)
    for key, reason in KNOWN_GAPS.items():
        assert len(reason) > 40, f"{key} needs a real explanation, got {reason!r}"


def test_duplicate_keys_are_refused():
    r = Record(TOTAL_RISK, "portfolio", "", "", 1.0)
    with pytest.raises(DuplicateRecord):
        index((r, replace(r, value=2.0)))


# --------------------------------------------------------------------------------------
# Fixture integrity
# --------------------------------------------------------------------------------------


def test_case_round_trips_exactly(tmp_path):
    case = synthetic_case(seed=7)
    path = case.save(tmp_path / "case.json")
    assert ParityCase.load(path).dumps() == case.dumps()


def test_case_round_trip_preserves_every_number(tmp_path):
    """Byte equality is not enough on its own: check the numbers survive too."""
    case = synthetic_case(seed=13)
    reloaded = ParityCase.load(case.save(tmp_path / "c.json"))
    assert reference_records(reloaded) == reference_records(case)


def test_records_round_trip(tmp_path):
    from praxis.midoffice.parity import record as R

    case = synthetic_case(seed=7)
    records = reference_records(case)
    path = R.save(records, tmp_path / "golden.json", case_name=case.name, source=case.source)
    loaded, header = R.load(path)
    assert loaded == tuple(sorted(records))
    assert header["case_name"] == case.name
    assert header["source"] == case.source


def test_truncated_golden_file_is_refused(tmp_path):
    import json

    from praxis.midoffice.parity import record as R

    path = R.save(reference_records(tiny_case()), tmp_path / "g.json")
    payload = json.loads(path.read_text())
    payload["records"] = payload["records"][:-1]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="truncated"):
        R.load(path)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"covariance": ((0.04, 0.01),)}, "shape"),
        ({"covariance": ((0.04, 0.02), (0.01, 0.09))}, "symmetric"),
        ({"covariance": ((-0.04, 0.0), (0.0, 0.09))}, "negative variance"),
        (
            {
                "positions": (
                    PositionSpec("AAA", 0.6, 0.5, {1: 1.0}, 0.2),
                    PositionSpec("AAA", 0.4, 0.5, {2: 1.2}, 0.3),
                )
            },
            "duplicate security",
        ),
        ({"positions": (PositionSpec("AAA", 0.6, 0.5, {99: 1.0}, 0.2),)}, "neither in the model"),
        ({"positions": (PositionSpec("AAA", 0.6, 0.5, {1: 1.0}, -0.2),)}, "negative specific"),
        ({"weight_basis": "renormalised"}, "not"),
        ({"tails": (TailCase(1.5),)}, "strictly between"),
        ({"tails": (TailCase(0.95, horizon_days=0),)}, "at least one day"),
        ({"groups": {"G": "G"}}, "its own parent"),
        ({"groups": {"A": "B", "B": "A"}}, "cycle"),
        ({"schema_version": 99}, "schema_version"),
    ],
)
def test_invalid_cases_are_refused(overrides, match):
    with pytest.raises(InvalidCase, match=match):
        tiny_case(**overrides)


def test_reference_refuses_proxies_rather_than_guessing():
    case = tiny_case(
        positions=(PositionSpec("AAA", 1.0, 1.0, {1: 1.0, 7: 0.5}, 0.2),),
        proxies={7: 1},
    )
    with pytest.raises(UnsupportedCase, match="proxies"):
        reference_records(case)


def test_reference_refuses_a_non_psd_covariance():
    case = tiny_case(covariance=((0.04, 0.5), (0.5, 0.09)))
    with pytest.raises(UnsupportedCase, match="not PSD"):
        reference_records(case)


def test_reference_accepts_a_floor_that_triggers_no_repair():
    """A non-zero floor is fine; what the reference cannot handle is an actual repair.

    This matters because the engine's configured default floor is non-zero. A reference that
    refused any non-zero floor would only ever exercise a configuration the engine no longer
    runs, which is a harness quietly testing the wrong thing.
    """
    case = tiny_case(relative_floor=1e-8)
    assert reference_records(case), "no eigenvalue is below this floor, so nothing is repaired"


def test_reference_refuses_a_floor_that_would_repair_the_matrix():
    """tiny_case has eigenvalues 0.0381 and 0.0919, so a floor of 0.5 clips the smaller one."""
    case = tiny_case(relative_floor=0.5)
    with pytest.raises(UnsupportedCase, match="below its floor"):
        reference_records(case)


def test_a_case_with_no_source_is_still_valid_but_visible():
    """Provenance is not enforced by the schema, so a test documents the convention."""
    assert synthetic_case(seed=1).source.startswith("synthetic:")
    assert tiny_case(source="").source == ""
