"""Tunable numerical settings for the mid-office risk engine.

Every value here was chosen by measurement, and each carries the measurement in its
docstring.  That is the point of the module: a tolerance whose justification is not written
down becomes folklore within a year, and the next person to meet it either trusts it blindly
or changes it blindly.

Resolution order, most specific first:

1. an explicit `RiskConfig` passed to the function
2. the file named by `PRAXIS_MIDOFFICE_RISK_CONFIG`
3. `config/midoffice_risk.toml`, searched upward from the working directory
4. the calibrated defaults in this module

A missing file is not an error -- the defaults are the calibrated values, so the engine works
with no configuration at all.  `source` records which of the four applied, and travels onto
the result alongside the other provenance, so a number can be reconstructed later rather
than merely re-run.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

CONFIG_FILENAME = "config/midoffice_risk.toml"
ENV_VAR = "PRAXIS_MIDOFFICE_RISK_CONFIG"

BUILT_IN = "built-in calibrated defaults"


class InvalidConfig(ValueError):
    """A configuration value that would produce wrong numbers rather than slow ones."""


@dataclass(frozen=True)
class RiskConfig:
    """Numerical settings, with the calibration behind each one."""

    psd_relative_floor: float = 1e-8
    """Floor for eigenvalue clipping in `nearest_psd`, as a fraction of the largest eigenvalue.

    **Calibrated 2026-10-21 across 15 configurations** (40x30, 70x50, 70x200, 150x100, 30x29
    factors-by-observations, three seeds each), measuring a long-only book and a factor-neutral
    book placed in the covariance's near-null direction, against mpmath at 50 digits.

    | floor | neutral-book digits (worst) | long-only distortion (worst) | margin vs 0.01bp |
    |-------|------------------------------|------------------------------|------------------|
    | 0.0   | -0.7                         | --                           | --               |
    | 1e-8  | 9.3                          | 9.9e-06 bps                  | 1011x            |
    | 1e-6  | 11.4                         | 9.9e-04 bps                  | 10x              |
    | 1e-4  | 13.4                         | 9.9e-02 bps                  | 0.1x (exceeds)   |

    Two things make 1e-8 the right default rather than a compromise.

    First, the asymmetry: distortion is *linear* in the floor while precision is
    *logarithmic* in it, so there is a wide band where digits come almost free.  1e-8 sits at
    the knee -- it captures nearly all the precision benefit of 1e-6 for a hundredth of the
    distortion.

    Second, raising it cannot regress a well-conditioned matrix.  `nearest_psd` only touches
    eigenvalues already below the floor, so on the 70x200 cases -- more observations than
    factors, no negative eigenvalues -- the distortion was *exactly* zero at every setting
    tested.  The change alters numbers only where those numbers were previously noise.

    The old default of 0.0 clipped negatives to exactly zero, leaving the matrix singular. A
    factor-neutral book aligned with a null direction then had its risk reported as 0.0: not
    an imprecise number but a false one, the model declaring a long-short book riskless when
    it could not resolve that direction at all.
    """

    symmetry_tolerance: float = 1e-10
    """Largest `max|A - A.T|` accepted before `nearest_psd` refuses the matrix.

    Deliberately strict.  An asymmetric covariance means the producer is wrong, and silently
    averaging it would hide that while changing every number downstream.  Float asymmetry
    from a matrix reconstruction sits far below this.
    """

    risk_tolerance: float = 1e-12
    """Variance below which a risk number is treated as zero rather than square-rooted.

    Guards the division in every contribution and decomposition, where a near-zero aggregate
    would otherwise produce enormous meaningless shares. Measured float noise on an O(1)
    variance accumulated over a few hundred thousand terms is ~1e-13, so this sits one order
    above it.
    """

    conditioning_warn_digits: float = 10.0
    """Qualify a risk number when fewer than this many decimal digits survive.

    Measured slack between the worst-case bound and reality was 2.25 to 3.05 digits across
    four conditioning cases, so the reported figure is conservative by about 2.6 digits. At
    the default PSD floor the worst observed case retained 9.3 digits, which trips this
    threshold -- deliberately: the point is to say so, not to be silent about it. Reporting
    needs 4 or 5 significant figures, so 10 leaves a wide margin before anything is at risk.
    """

    trading_days: float = 252.0
    """Annualisation basis for horizon scaling. RAT hardcodes 252 (`getVarMultiplier`)."""

    default_confidence: float = 0.95
    """Default tail confidence when a caller does not specify one."""

    default_horizon_days: float = 1.0
    """Default tail horizon in days."""

    refine_inverse_normal: bool = False
    """Whether to apply a Halley refinement to Acklam's inverse normal CDF.

    `False` matches RAT, which uses Acklam unrefined (`PortfolioRiskData.cs:576`) with a
    relative error of 1.15e-9. Keep it false while parity against production output matters;
    `True` is the more accurate choice once it does not. Note the two tail measures do not
    inherit that error equally -- VaR is linear in z and carries 1.15e-9, expected shortfall
    scales as z^2 and carries 2.7x that at 95%, 5.4x at 99%.
    """

    weight_basis: str = "full"
    """`full` keeps uncovered names in the weight denominator; `covered` redistributes.

    `full` was the production convention in RAT and the reason is not technical: portfolio
    managers reconcile weights and market values against their own systems, and a tool whose
    weights do not tie is a tool they will not believe.
    """

    source: str = BUILT_IN
    """Where these values came from. Metadata, not a tunable; travels onto the result."""

    def __post_init__(self) -> None:
        self.validate()

    # -- validation -------------------------------------------------------------------

    def validate(self) -> None:
        """Refuse values that would produce wrong numbers rather than merely slow ones."""
        if not 0.0 <= self.psd_relative_floor < 1.0:
            raise InvalidConfig(
                f"psd_relative_floor must be in [0, 1), got {self.psd_relative_floor!r}; "
                f"it is a fraction of the largest eigenvalue, so 1.0 would flatten the "
                f"covariance to a multiple of the identity"
            )
        if self.psd_relative_floor > 1e-3:
            raise InvalidConfig(
                f"psd_relative_floor {self.psd_relative_floor!r} exceeds 1e-3; measured "
                f"distortion at 1e-4 already exceeds a reported precision of 0.01 bps, so "
                f"anything above 1e-3 is changing the risk numbers rather than conditioning "
                f"them. Raise this bound deliberately if you mean to."
            )
        for name in ("symmetry_tolerance", "risk_tolerance"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise InvalidConfig(f"{name} must be in (0, 1), got {value!r}")
        if self.conditioning_warn_digits < 0.0:
            raise InvalidConfig(
                f"conditioning_warn_digits must be non-negative, got "
                f"{self.conditioning_warn_digits!r}"
            )
        if self.trading_days <= 0.0:
            raise InvalidConfig(f"trading_days must be positive, got {self.trading_days!r}")
        if not 0.0 < self.default_confidence < 1.0:
            raise InvalidConfig(
                f"default_confidence must be strictly between 0 and 1, got "
                f"{self.default_confidence!r}"
            )
        if self.default_horizon_days <= 0.0:
            raise InvalidConfig(
                f"default_horizon_days must be positive, got {self.default_horizon_days!r}"
            )
        if self.weight_basis not in ("full", "covered"):
            raise InvalidConfig(
                f"weight_basis must be 'full' or 'covered', got {self.weight_basis!r}"
            )

    # -- loading ----------------------------------------------------------------------

    @classmethod
    def tunable_names(cls) -> tuple[str, ...]:
        """Every setting a file may set. `source` is metadata and is not settable."""
        return tuple(f.name for f in fields(cls) if f.name != "source")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, source: str) -> "RiskConfig":
        """Build from a mapping, refusing unknown keys.

        An unknown key is almost always a typo or a stale name, and silently ignoring it
        means the setting the author intended never takes effect while the file looks right.
        """
        known = set(cls.tunable_names())
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise InvalidConfig(
                f"{source} sets unknown key(s) {unknown}; known settings are "
                f"{sorted(known)}. A silently ignored key looks like a setting that "
                f"applied and did not."
            )
        return cls(**dict(mapping), source=source)

    @classmethod
    def find_file(cls, start: Path | None = None) -> Path | None:
        """Locate a config file: the env var first, then `config/` walking upward."""
        named = os.environ.get(ENV_VAR)
        if named:
            path = Path(named).expanduser()
            if not path.is_file():
                raise InvalidConfig(
                    f"{ENV_VAR} points at {path}, which is not a file; unset it or fix the "
                    f"path rather than falling back silently to different numbers"
                )
            return path
        here = (start or Path.cwd()).resolve()
        for directory in (here, *here.parents):
            candidate = directory / CONFIG_FILENAME
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def load(cls, path: str | Path | None = None, *, start: Path | None = None) -> "RiskConfig":
        """Load the configuration, falling back to the calibrated defaults."""
        resolved = Path(path) if path is not None else cls.find_file(start)
        if resolved is None:
            return cls()
        with open(resolved, "rb") as handle:
            raw = tomllib.load(handle)
        section = raw.get("risk", raw)
        return cls.from_mapping(section, source=str(resolved))

    # -- reporting --------------------------------------------------------------------

    def with_overrides(self, **changes: Any) -> "RiskConfig":
        unknown = sorted(set(changes) - set(self.tunable_names()))
        if unknown:
            raise InvalidConfig(f"unknown setting(s) {unknown}")
        note = ", ".join(f"{k}={v!r}" for k, v in sorted(changes.items()))
        return replace(self, **changes, source=f"{self.source} with {note}")

    def describe(self) -> str:
        return (
            f"risk config from {self.source}: psd_relative_floor="
            f"{self.psd_relative_floor:g}, weight_basis={self.weight_basis}, "
            f"trading_days={self.trading_days:g}, refine_inverse_normal="
            f"{self.refine_inverse_normal}"
        )


DEFAULTS = RiskConfig()
"""The calibrated defaults, as a module-level singleton.

Functions default to this rather than reading a file at import time: an import-time file read
makes behaviour depend on the working directory and makes tests order-dependent. A caller
that wants file-based configuration calls `RiskConfig.load()` explicitly and passes the
result.
"""
