"""Unit tests for engines/cpo_core.py::compute_allocation.

Locks in the equal_weight portfolio gross-cap arithmetic so a future
refactor that subtly breaks the cap behavior fails loudly. See
docs/CPO_ALLOCATION.md for the underlying arithmetic.

Each test constructs a synthetic list[dict] of model predictions and
asserts on the resulting weights dict.
"""

from __future__ import annotations

import pytest

from engines.cpo_core import compute_allocation


def _preds(probs: list[float], *, base_rate: float = 0.50,
           expected_return: float = 0.001) -> list[dict]:
    """Build a synthetic model_predictions list with the requested
    p_profitable values. Identifier indices match the input order."""
    return [
        {
            "model_id": f"M{i}",
            "p_profitable": p,
            "expected_return": expected_return,
            "base_rate": base_rate,
        }
        for i, p in enumerate(probs)
    ]


def test_per_model_cap_binds_small_n() -> None:
    """5 models above gate, max_leverage=2.0, max_weight_per_model=0.05.

    max_leverage/N = 0.40 > 0.05, so the per-model cap binds.
    Each weight = 0.05; total gross = 5 * 0.05 = 0.25.
    """
    weights = compute_allocation(
        _preds([0.60] * 5),
        max_leverage=2.0,
        max_weight_per_model=0.05,
        prob_threshold=0.50,
        mode="equal_weight",
    )
    assert len(weights) == 5
    assert all(w == pytest.approx(0.05) for w in weights.values())
    assert sum(weights.values()) == pytest.approx(0.25)


def test_leverage_cap_binds_large_n() -> None:
    """50 models above gate, max_leverage=0.5, max_weight_per_model=0.05.

    max_leverage/N = 0.01 < 0.05, so the leverage cap binds.
    Each weight = 0.01; total gross = 0.5 exactly.
    """
    weights = compute_allocation(
        _preds([0.60] * 50),
        max_leverage=0.5,
        max_weight_per_model=0.05,
        prob_threshold=0.50,
        mode="equal_weight",
    )
    assert len(weights) == 50
    assert all(w == pytest.approx(0.01) for w in weights.values())
    assert sum(weights.values()) == pytest.approx(0.5)


def test_gate_filters_models_below_threshold() -> None:
    """10 models; 5 above gate (0.60), 5 below (0.40).

    Only the 5 above-gate models receive a weight entry; the 5 below
    are absent from the returned dict (no zero-weight entries).
    """
    probs = [0.60] * 5 + [0.40] * 5
    weights = compute_allocation(
        _preds(probs),
        max_leverage=2.0,
        max_weight_per_model=0.05,
        prob_threshold=0.50,
        mode="equal_weight",
    )
    assert len(weights) == 5
    assert all(w > 0 for w in weights.values())
    assert sum(weights.values()) <= 2.0 + 1e-9


def test_exp10_runaway_reproduction() -> None:
    """35 models above gate, max_leverage=2.0, max_weight_per_model=0.05.

    max_leverage/N = 0.0571 > 0.05, so the per-model cap binds.
    Total gross = 35 * 0.05 = 1.75 -- the documented Exp 10 runaway.
    """
    weights = compute_allocation(
        _preds([0.60] * 35),
        max_leverage=2.0,
        max_weight_per_model=0.05,
        prob_threshold=0.50,
        mode="equal_weight",
    )
    assert len(weights) == 35
    assert sum(weights.values()) == pytest.approx(1.75)


def test_exp10_fix_with_max_leverage_half() -> None:
    """35 models above gate, max_leverage=0.5, max_weight_per_model=0.05.

    max_leverage/N = 0.0143 < 0.05, so the leverage cap binds.
    Total gross = 0.5 exactly -- the Cycle 36c fix target.
    """
    weights = compute_allocation(
        _preds([0.60] * 35),
        max_leverage=0.5,
        max_weight_per_model=0.05,
        prob_threshold=0.50,
        mode="equal_weight",
    )
    assert len(weights) == 35
    assert sum(weights.values()) == pytest.approx(0.5)


def test_zero_gating_models_returns_empty() -> None:
    """All 10 models below the 0.50 gate -> empty allocation dict."""
    weights = compute_allocation(
        _preds([0.40] * 10),
        max_leverage=2.0,
        max_weight_per_model=0.05,
        prob_threshold=0.50,
        mode="equal_weight",
    )
    assert weights == {}


def test_single_model_per_model_cap_binds() -> None:
    """Edge case n=1. max_leverage/N = 2.0 > 0.05, per-model cap binds.

    Weight = 0.05; total gross = 0.05.
    """
    weights = compute_allocation(
        _preds([0.60]),
        max_leverage=2.0,
        max_weight_per_model=0.05,
        prob_threshold=0.50,
        mode="equal_weight",
    )
    assert len(weights) == 1
    assert list(weights.values())[0] == pytest.approx(0.05)
    assert sum(weights.values()) == pytest.approx(0.05)
