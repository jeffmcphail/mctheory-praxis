"""Unit tests for engines/atlas_sync.py::_classify_result_token.

Covers Cycle 35.5's pattern-ordering fix: compound verdict phrases
(WEAK POSITIVE, STRONG NEGATIVE, etc.) must classify as PARTIAL before
the bare POSITIVE/NEGATIVE branches can fire on the contained substring.
"""

from __future__ import annotations

import pytest

from engines.atlas_sync import _classify_result_token


TRUTH_TABLE = [
    ("**POSITIVE** (Sharpe +4.65, ...)", "POSITIVE"),
    ("**NEGATIVE** (Sharpe -0.94)", "NEGATIVE"),
    ("**PARTIAL** (weak; primary +1.91% Sharpe +0.545)", "PARTIAL"),
    ("**PARTIAL** (WEAK POSITIVE; ...)", "PARTIAL"),
    ("**PARTIAL** (STRONG NEGATIVE; ...)", "PARTIAL"),
    ("**INCONCLUSIVE** (-83.78% portfolio leverage runaway)", "INCONCLUSIVE"),
    ("**INCONCLUSIVE** (BLOCKED on data infrastructure)", "INCONCLUSIVE"),
    ("WEAK POSITIVE -- confirmed improvement", "PARTIAL"),
    ("PROMISING -- best result in atlas", "PARTIAL"),
    ("**NEGATIVE after TC**", "NEGATIVE"),
    ("Confirmed POSITIVE.", "POSITIVE"),
    ("Confirmed NEGATIVE.", "NEGATIVE"),
]


@pytest.mark.parametrize("text,expected", TRUTH_TABLE)
def test_classify_result_token(text: str, expected: str) -> None:
    assert _classify_result_token(text) == expected


def test_empty_string_returns_none() -> None:
    assert _classify_result_token("") is None


def test_no_verdict_words_returns_none() -> None:
    assert _classify_result_token("hello world, no verdict here") is None
