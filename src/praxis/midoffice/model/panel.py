"""Return panels and factor return series: the raw material a factor model is built from.

Two things here earn their own types rather than being bare arrays.

**Frequency travels with the data.** A covariance estimated from monthly returns is a monthly
covariance, and the engine's risk numbers are annual -- `TailSpec` scales by 252 trading days.
Feeding a monthly covariance to an annual engine understates risk by a factor of 3.5 and
nothing about the number says so.  So a panel knows its frequency, and annualisation is an
explicit step that gets recorded.

**Missing is not zero.**  A security that did not trade, or did not exist, has no return for
that period.  Treating that as a zero return pulls its estimated volatility down and its beta
toward zero, which is a bias toward looking safe.  `nan` marks absence and the estimator
counts what it actually had.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np


class Frequency(Enum):
    """How often the returns are sampled, and the periods per year that implies.

    252 for daily rather than 365: a risk number is per trading day, which is the convention
    RAT hardcodes in `getVarMultiplier` and the port carries in `RiskConfig.trading_days`.
    """

    DAILY = ("daily", 252.0)
    WEEKLY = ("weekly", 52.0)
    MONTHLY = ("monthly", 12.0)
    QUARTERLY = ("quarterly", 4.0)
    ANNUAL = ("annual", 1.0)

    def __init__(self, label: str, periods_per_year: float) -> None:
        self.label = label
        self.periods_per_year = periods_per_year

    @property
    def variance_scale(self) -> float:
        """Multiply a periodic variance by this to annualise it."""
        return self.periods_per_year

    @property
    def volatility_scale(self) -> float:
        """Multiply a periodic standard deviation by this to annualise it.

        The square root, because variance scales with time and a standard deviation is its
        square root. Scaling a volatility by the period count instead is the commonest
        annualisation error and inflates a monthly 4% into an annual 48% rather than 13.9%.
        """
        return float(np.sqrt(self.periods_per_year))


def _as_matrix(values, n_rows: int, n_cols: int, what: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{what} must be two-dimensional, got shape {matrix.shape}")
    if matrix.shape != (n_rows, n_cols):
        raise ValueError(
            f"{what} has shape {matrix.shape} but {n_rows} date(s) and {n_cols} "
            f"column(s) were given"
        )
    return matrix


@dataclass(frozen=True)
class ReturnPanel:
    """Periodic returns for a set of securities over a common set of dates.

    `returns[i, j]` is security `j`'s return over the period ending `dates[i]`, or `nan` when
    there is none.
    """

    dates: tuple[date, ...]
    securities: tuple[str, ...]
    returns: np.ndarray
    frequency: Frequency = Frequency.MONTHLY

    def __post_init__(self) -> None:
        dates = tuple(self.dates)
        securities = tuple(self.securities)
        if len(set(dates)) != len(dates):
            raise ValueError("duplicate dates in the return panel")
        if len(set(securities)) != len(securities):
            dupes = sorted({s for s in securities if securities.count(s) > 1})
            raise ValueError(f"duplicate securities in the return panel: {dupes[:5]}")
        if list(dates) != sorted(dates):
            raise ValueError(
                "return panel dates must be ascending; an unsorted panel makes every "
                "window-based estimate depend on load order"
            )
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "securities", securities)
        object.__setattr__(
            self,
            "returns",
            _as_matrix(self.returns, len(dates), len(securities), "returns"),
        )

    @classmethod
    def of(
        cls,
        dates: Iterable[date],
        series: Mapping[str, Sequence[float]],
        frequency: Frequency = Frequency.MONTHLY,
    ) -> "ReturnPanel":
        """Build from a mapping of security code to its return series."""
        dates = tuple(dates)
        securities = tuple(series)
        matrix = np.full((len(dates), len(securities)), np.nan)
        for j, code in enumerate(securities):
            values = list(series[code])
            if len(values) != len(dates):
                raise ValueError(
                    f"{code} has {len(values)} return(s) against {len(dates)} date(s); "
                    f"pad with nan rather than truncating, so the gap stays visible"
                )
            matrix[:, j] = values
        return cls(dates=dates, securities=securities, returns=matrix, frequency=frequency)

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def n_securities(self) -> int:
        return len(self.securities)

    def column(self, security: str) -> np.ndarray:
        try:
            return self.returns[:, self.securities.index(security)]
        except ValueError:
            raise KeyError(f"{security!r} is not in this panel") from None

    def observations(self, security: str) -> int:
        """How many periods this security actually has a return for."""
        return int(np.count_nonzero(~np.isnan(self.column(security))))

    def coverage(self) -> dict[str, int]:
        return {s: self.observations(s) for s in self.securities}


@dataclass(frozen=True)
class FactorReturns:
    """The factor return series a time-series model is estimated against.

    Taken as given rather than constructed, so a published series -- Fama and French's own,
    say -- can be used directly. Constructing factors from characteristics by sorting is a
    separate step and a separate set of choices.
    """

    dates: tuple[date, ...]
    names: tuple[str, ...]
    returns: np.ndarray
    frequency: Frequency = Frequency.MONTHLY

    def __post_init__(self) -> None:
        dates = tuple(self.dates)
        names = tuple(self.names)
        if len(set(dates)) != len(dates):
            raise ValueError("duplicate dates in the factor returns")
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate factor names: {dupes[:5]}")
        if list(dates) != sorted(dates):
            raise ValueError("factor return dates must be ascending")
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self,
            "returns",
            _as_matrix(self.returns, len(dates), len(names), "factor returns"),
        )
        if np.isnan(self.returns).any():
            missing = int(np.count_nonzero(np.isnan(self.returns)))
            raise ValueError(
                f"{missing} missing factor return(s); a gap in a factor series cannot be "
                f"skipped the way a gap in one security's returns can, because every "
                f"security's estimate depends on that period. Drop the period instead."
            )

    @classmethod
    def of(
        cls,
        dates: Iterable[date],
        series: Mapping[str, Sequence[float]],
        frequency: Frequency = Frequency.MONTHLY,
    ) -> "FactorReturns":
        dates = tuple(dates)
        names = tuple(series)
        matrix = np.empty((len(dates), len(names)))
        for j, name in enumerate(names):
            values = list(series[name])
            if len(values) != len(dates):
                raise ValueError(
                    f"factor {name} has {len(values)} return(s) against {len(dates)} date(s)"
                )
            matrix[:, j] = values
        return cls(dates=dates, names=names, returns=matrix, frequency=frequency)

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def n_factors(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class Alignment:
    """The dates a panel and a factor series share, and what each lost to get there."""

    dates: tuple[date, ...]
    panel_rows: tuple[int, ...]
    factor_rows: tuple[int, ...]
    dropped_from_panel: tuple[date, ...]
    dropped_from_factors: tuple[date, ...]

    @property
    def is_exact(self) -> bool:
        return not (self.dropped_from_panel or self.dropped_from_factors)

    def qualifications(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.dropped_from_panel:
            notes.append(
                f"{len(self.dropped_from_panel)} period(s) of security returns had no "
                f"matching factor return and were dropped from the estimation"
            )
        if self.dropped_from_factors:
            notes.append(
                f"{len(self.dropped_from_factors)} period(s) of factor returns had no "
                f"matching security returns and were dropped"
            )
        return tuple(notes)


def align(panel: ReturnPanel, factors: FactorReturns) -> Alignment:
    """Intersect the two date sets, recording what each side lost.

    Intersecting rather than requiring an exact match, because a factor series and a holdings
    history will rarely start on the same day. Recording the loss rather than intersecting
    silently, because a factor series that overlaps the panel by six months out of sixty is a
    model estimated on six months and nothing else would say so.
    """
    if panel.frequency is not factors.frequency:
        raise ValueError(
            f"panel is {panel.frequency.label} and factor returns are "
            f"{factors.frequency.label}; mixing frequencies would annualise one of them wrong"
        )
    panel_index = {d: i for i, d in enumerate(panel.dates)}
    factor_index = {d: i for i, d in enumerate(factors.dates)}
    shared = tuple(sorted(set(panel_index) & set(factor_index)))
    return Alignment(
        dates=shared,
        panel_rows=tuple(panel_index[d] for d in shared),
        factor_rows=tuple(factor_index[d] for d in shared),
        dropped_from_panel=tuple(d for d in panel.dates if d not in factor_index),
        dropped_from_factors=tuple(d for d in factors.dates if d not in panel_index),
    )
