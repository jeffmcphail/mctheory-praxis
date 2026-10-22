"""Tests for the risk engine's configuration.

The settings in `RiskConfig` are numerical choices with measured justifications, so the
tests here check two things beyond loading: that a value which would produce wrong numbers
is refused rather than accepted, and that the file and the code agree on what is tunable. A
config file listing a key the code ignores is worse than no file, because it looks like a
setting that applied.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from praxis.midoffice.risk.config import (
    BUILT_IN,
    CONFIG_FILENAME,
    DEFAULTS,
    ENV_VAR,
    InvalidConfig,
    RiskConfig,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / CONFIG_FILENAME


def write_config(directory: Path, body: str) -> Path:
    path = directory / CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# --- defaults -----------------------------------------------------------------------


def test_defaults_are_the_calibrated_values():
    assert DEFAULTS.psd_relative_floor == 1e-8
    assert DEFAULTS.weight_basis == "full"
    assert DEFAULTS.refine_inverse_normal is False
    assert DEFAULTS.trading_days == 252.0
    assert DEFAULTS.source == BUILT_IN


def test_the_engine_works_with_no_config_file_at_all():
    """A missing file is not an error: the defaults are the calibrated values."""
    cfg = RiskConfig.load(start=Path("/"))
    assert cfg.source == BUILT_IN
    assert cfg.psd_relative_floor == DEFAULTS.psd_relative_floor


def test_psd_floor_default_is_not_zero():
    """Pinning the behavioural change.

    Zero leaves the covariance singular and reports a factor-neutral book's risk as exactly
    0.0 -- a false number rather than an imprecise one. If someone restores 0.0 as the
    default, this fails and the message says why.
    """
    assert DEFAULTS.psd_relative_floor > 0.0, (
        "a zero floor leaves the matrix singular; a factor-neutral book's risk then comes "
        "back as 0.0 or as rounding noise"
    )


# --- the shipped file ---------------------------------------------------------------


def test_shipped_config_loads_and_matches_the_defaults():
    """The file in the repo must agree with the code's defaults.

    They are allowed to diverge in a deployment, but a repo whose own file disagrees with its
    own defaults means one of the two is stale and nobody can tell which.
    """
    if not REPO_CONFIG.is_file():
        pytest.skip(f"{CONFIG_FILENAME} not present in this tree")
    loaded = RiskConfig.load(REPO_CONFIG)
    for name in RiskConfig.tunable_names():
        assert getattr(loaded, name) == getattr(DEFAULTS, name), name


def test_shipped_config_sets_every_tunable_and_nothing_else():
    """A key in the file that the code ignores looks like a setting that applied."""
    if not REPO_CONFIG.is_file():
        pytest.skip(f"{CONFIG_FILENAME} not present in this tree")
    with open(REPO_CONFIG, "rb") as handle:
        raw = tomllib.load(handle)["risk"]
    assert sorted(raw) == sorted(RiskConfig.tunable_names())


# --- loading ------------------------------------------------------------------------


def test_load_from_an_explicit_path(tmp_path):
    path = write_config(tmp_path, "[risk]\npsd_relative_floor = 1e-6\n")
    cfg = RiskConfig.load(path)
    assert cfg.psd_relative_floor == 1e-6
    assert cfg.source == str(path)
    # Unspecified settings keep their calibrated defaults rather than becoming zero.
    assert cfg.trading_days == 252.0


def test_load_walks_upward_from_the_working_directory(tmp_path):
    write_config(tmp_path, "[risk]\npsd_relative_floor = 1e-7\n")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert RiskConfig.load(start=deep).psd_relative_floor == 1e-7


def test_a_bare_table_without_the_risk_section_also_loads(tmp_path):
    path = tmp_path / "flat.toml"
    path.write_text("psd_relative_floor = 1e-5\n", encoding="utf-8")
    assert RiskConfig.load(path).psd_relative_floor == 1e-5


def test_env_var_takes_precedence(tmp_path, monkeypatch):
    write_config(tmp_path, "[risk]\npsd_relative_floor = 1e-7\n")
    other = tmp_path / "other.toml"
    other.write_text("[risk]\npsd_relative_floor = 1e-5\n", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(other))
    assert RiskConfig.load(start=tmp_path).psd_relative_floor == 1e-5


def test_env_var_pointing_nowhere_fails_loudly(tmp_path, monkeypatch):
    """Falling back silently would run with different numbers than the operator intended."""
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "absent.toml"))
    with pytest.raises(InvalidConfig, match="not a file"):
        RiskConfig.load(start=tmp_path)


def test_source_records_where_the_values_came_from(tmp_path):
    path = write_config(tmp_path, "[risk]\npsd_relative_floor = 1e-6\n")
    assert RiskConfig.load(path).source == str(path)
    assert RiskConfig.load(start=Path("/")).source == BUILT_IN


def test_overrides_record_themselves_in_the_source():
    cfg = DEFAULTS.with_overrides(psd_relative_floor=1e-5)
    assert cfg.psd_relative_floor == 1e-5
    assert "psd_relative_floor=1e-05" in cfg.source
    assert BUILT_IN in cfg.source


def test_overrides_refuse_unknown_settings():
    with pytest.raises(InvalidConfig, match="unknown setting"):
        DEFAULTS.with_overrides(psd_floor=1e-5)


# --- validation ---------------------------------------------------------------------


def test_unknown_key_in_a_file_is_refused(tmp_path):
    path = write_config(tmp_path, "[risk]\npsd_floor = 1e-6\n")
    with pytest.raises(InvalidConfig, match="unknown key"):
        RiskConfig.load(path)


@pytest.mark.parametrize(
    "settings,match",
    [
        ({"psd_relative_floor": -1e-9}, r"must be in \[0, 1\)"),
        ({"psd_relative_floor": 1.0}, r"must be in \[0, 1\)"),
        ({"psd_relative_floor": 0.01}, "exceeds 1e-3"),
        ({"symmetry_tolerance": 0.0}, "symmetry_tolerance"),
        ({"risk_tolerance": 2.0}, "risk_tolerance"),
        ({"conditioning_warn_digits": -1.0}, "non-negative"),
        ({"trading_days": 0.0}, "trading_days must be positive"),
        ({"default_confidence": 1.0}, "strictly between"),
        ({"default_confidence": 0.0}, "strictly between"),
        ({"default_horizon_days": 0.0}, "must be positive"),
        ({"weight_basis": "renormalised"}, "must be 'full' or 'covered'"),
    ],
)
def test_values_that_would_produce_wrong_numbers_are_refused(settings, match):
    with pytest.raises(InvalidConfig, match=match):
        RiskConfig(**settings)


def test_a_floor_above_the_measured_distortion_limit_is_refused():
    """1e-4 already distorts a long-only book past a 0.01 bps reported precision.

    The bound is deliberately a refusal rather than a warning: a floor this large is changing
    the risk numbers, not conditioning them, and that should be a decision someone makes in
    the code rather than a value that slips through a config file.
    """
    with pytest.raises(InvalidConfig, match="exceeds 1e-3"):
        RiskConfig(psd_relative_floor=1e-2)
    # Just inside the bound is allowed, so the limit is a real edge rather than a blanket ban.
    assert RiskConfig(psd_relative_floor=1e-3).psd_relative_floor == 1e-3


def test_config_is_immutable():
    with pytest.raises(Exception):
        DEFAULTS.psd_relative_floor = 1e-6  # type: ignore[misc]
