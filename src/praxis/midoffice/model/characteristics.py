"""Exposures as observable characteristics, standardised period by period.

This is the input to a cross-sectional model, and the thing that makes it different in kind
from a time-series one.  A time-series exposure is an estimated beta: backward-looking over
its window, measured with error, and an average of whatever the security was during it.  A
characteristic is what the security *is* today -- its log market cap, its book-to-price -- and
carries no estimation error at all.

That is why Barra built its models this way and why RAT's exposure table holds loadings that
are not betas.  For a risk *forecast* the question is what the portfolio is exposed to now,
and a beta estimated over five years answers a question about the past.

**Standardisation is not cosmetic.**  Raw characteristics are on incomparable scales -- log
market cap runs from 10 to 30, book-to-price from 0 to 3 -- so the factor returns estimated
against them would be on incomparable scales too, and a factor covariance built from those is
meaningless.  Standardising each characteristic cross-sectionally every period, to zero mean
and unit standard deviation, makes an exposure of 1.0 mean "one standard deviation above the
market" in every factor and every period.  That is what makes the factor returns comparable
across factors and over time, and therefore what makes their covariance mean something.

The mean is **cap-weighted** and the standard deviation **equal-weighted**, which is Barra's
own convention.  The *Global Equity Model Handbook* states it exactly:

    [Normalized Risk Index] = ([Raw Data] - [Capitalization-weighted Mean])
                               / [Equal-weighted Standard Deviation]

and says why: standardisation "sets the zero value at the capitalization-weighted mean of the
companies within each local market".  That is the market-at-zero property.  The asymmetry is
deliberate on both sides -- a cap-weighted mean so the market has no style tilt, an
equal-weighted dispersion so a handful of mega-caps do not set the scale.  Equal-weighting the mean leaves the market with a large negative size exposure,
because there are far more small companies than large ones, and every portfolio's size
exposure is then measured against a benchmark nobody holds.

**Whose values set the scale is a separate question from who gets an exposure.**  Barra
standardises over its estimation universe and applies the result to its whole coverage
universe.  Given a membership, `standardise` does the same: the index's members that day set
the clip bounds, the mean and the standard deviation, and everyone else with a value is put
on their scale.  The market at zero is then the index at zero, which is what lets an exposure
read as relative to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Sequence

import numpy as np

from .panel import Frequency


def _checked_membership(
    membership: np.ndarray | None, shape: tuple[int, int], expected: str
) -> np.ndarray | None:
    """A dates-by-securities membership, refused unless it is boolean and of `shape`."""
    if membership is None:
        return None
    members = np.asarray(membership)
    if members.dtype != bool:
        raise TypeError(
            f"membership must be a boolean array, got {members.dtype}: a 0/1 float is not a "
            f"membership, and a NaN in one is truthy"
        )
    if members.shape != shape:
        raise ValueError(f"membership has shape {members.shape}, expected {shape} {expected}")
    return members


@dataclass(frozen=True)
class CharacteristicPanel:
    """Characteristic values for securities over time: dates x securities x factors.

    Three dimensions rather than two because a characteristic moves: a company's market cap
    changes daily and its book-to-price every quarter. Holding exposures fixed over the window
    would reintroduce exactly the staleness that choosing characteristics over betas avoids.
    """

    dates: tuple[date, ...]
    securities: tuple[str, ...]
    names: tuple[str, ...]
    values: np.ndarray
    frequency: Frequency = Frequency.MONTHLY
    standardised: bool = False
    winsorised: int = 0
    """How many raw values the clip moved, across every date and characteristic.

    Provenance worth keeping. A handful means the clip did its job on genuine outliers; a
    large fraction of the panel means the characteristic itself is wrong -- wrong units, a
    sign error, a vendor placeholder like -999 -- and no amount of clipping will fix it.
    """
    clipped_outside: int = 0
    """How many values of securities outside the estimation universe hit one of the members'
    bounds: their raw clip bounds, or the `winsorise` bound on the standardised value.

    The ex-members sit in the tails, so some will. Both bounds count because either can bind:
    a cap-weighted mean far above the median -- LNCAP's -- puts the second bound inside the
    first in the lower tail. Most of a characteristic's outsiders hitting a bound would mean
    the members' scale does not describe them at all.
    """

    def __post_init__(self) -> None:
        dates = tuple(self.dates)
        securities = tuple(self.securities)
        names = tuple(self.names)
        for label, items in (
            ("dates", dates),
            ("securities", securities),
            ("characteristics", names),
        ):
            if len(set(items)) != len(items):
                raise ValueError(f"duplicate {label} in the characteristic panel")
        if list(dates) != sorted(dates):
            raise ValueError("characteristic panel dates must be ascending")
        values = np.asarray(self.values, dtype=float)
        expected = (len(dates), len(securities), len(names))
        if values.shape != expected:
            raise ValueError(
                f"characteristic values have shape {values.shape}, expected {expected} "
                f"(dates x securities x characteristics)"
            )
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "securities", securities)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)

    @classmethod
    def constant(
        cls,
        dates: Iterable[date],
        characteristics: Mapping[str, Mapping[str, float]],
        frequency: Frequency = Frequency.MONTHLY,
    ) -> "CharacteristicPanel":
        """Build from a security-to-characteristic mapping held fixed over every date.

        A convenience for a single snapshot, not a model of reality: characteristics move, and
        `test_a_constant_panel_is_a_simplification` says so. Useful when only one observation
        date is available.
        """
        dates = tuple(dates)
        securities = tuple(characteristics)
        names = tuple(dict.fromkeys(k for c in characteristics.values() for k in c))
        values = np.full((len(dates), len(securities), len(names)), np.nan)
        for j, code in enumerate(securities):
            for k, name in enumerate(names):
                if name in characteristics[code]:
                    values[:, j, k] = float(characteristics[code][name])
        return cls(dates, securities, names, values, frequency)

    @property
    def n_factors(self) -> int:
        return len(self.names)

    @property
    def n_securities(self) -> int:
        return len(self.securities)

    def __len__(self) -> int:
        return len(self.dates)

    def at(self, index: int) -> np.ndarray:
        """The securities-by-factors slice for one period."""
        return self.values[index, :, :]

    def snapshot(self, index: int = -1) -> dict[str, dict[str, float]]:
        """One period's characteristics as a mapping, skipping what is missing.

        Index -1 by default: the most recent period, which is the one a forecast wants. A
        time-series model's exposures are an average over its window; these are a point in
        time, and that is the whole difference.
        """
        slice_ = self.values[index, :, :]
        out: dict[str, dict[str, float]] = {}
        for j, code in enumerate(self.securities):
            row = {
                name: float(slice_[j, k])
                for k, name in enumerate(self.names)
                if not np.isnan(slice_[j, k])
            }
            if row:
                out[code] = row
        return out

    def coverage(self) -> dict[str, int]:
        """How many periods each security has a complete characteristic row for."""
        complete = ~np.isnan(self.values).any(axis=2)
        return {
            code: int(np.count_nonzero(complete[:, j]))
            for j, code in enumerate(self.securities)
        }

    def standardise(
        self,
        weights: np.ndarray | None = None,
        *,
        winsorise: float | None = 4.0,
        membership: np.ndarray | None = None,
    ) -> "CharacteristicPanel":
        """Zero weighted mean and unit standard deviation, per characteristic per period.

        `weights` is a dates-by-securities array -- market capitalisation, or its square root.
        The weighted mean is what puts the market portfolio at zero exposure to every style
        factor; an equal-weighted mean leaves the market with a large negative size exposure
        because small companies outnumber large ones, and every portfolio's size tilt would
        then be measured against a benchmark nobody holds.

        The standard deviation is deliberately **equal-weighted** even when the mean is not.
        A cap-weighted standard deviation is dominated by the few largest names, so the scale
        of the factor would be set by a handful of companies and would jump whenever one of
        them moved.

        `winsorise` clips the **raw** characteristic to that many robust scale units either
        side of its median, before the mean and standard deviation are computed.

        The default of 4.0 is Barra's own bound: the GEM handbook truncates a factor exposure
        to +/-4 standard deviations, and eliminates outright anything beyond 10.  But Barra
        clips the *standardised* value, which this does not, and the difference is a matter of
        cross-section size rather than disagreement.  Their estimation universe runs to tens of
        thousands of assets, where no single name can move the standard deviation enough to
        matter.  On a few dozen it can, and does.  Clipping the raw value against a robust
        scale gives Barra's behaviour at their n and survives at small n too -- so the order
        here is the whole point. Clipping *standardised* values cannot catch an outlier, because by
        then the outlier has set the scale it would be measured against: a book-to-price of
        twenty inflates the standard deviation to about eight, so it standardises to +0.1 --
        well inside any sane clip -- while every ordinary company is crushed to about -2.3.
        The outlier never looks extreme; it makes everyone else look identical.

        The scale used for the clip is the median absolute deviation, rescaled by 1.4826 so it
        estimates the same quantity as the standard deviation on normal data. MAD is used
        rather than the standard deviation precisely because one extreme value cannot inflate
        it -- which is what lets the clip bite. Where the MAD is zero (more than half the
        securities sharing one value) it falls back to the standard deviation, and where that
        is zero too the characteristic has no cross-sectional variation and standardises to
        zeros.

        **`membership` decides whose values set the scale.**  A dates-by-securities boolean
        array, True where the security is in the estimation universe that day -- for an index
        model, an index member.  Given one, the clip bounds, the weighted mean and the
        standard deviation are taken over that day's members with a value and nobody else,
        and every other security with a value is put on the members' scale: clipped to their
        bounds, centred on their mean, divided by their standard deviation.  A name that left
        the index still gets an exposure, as a holding of it would need, but no say in what an
        exposure of one means -- and it is the ex-members, shrinking in the tails, that would
        otherwise widen the scale for everyone.  A day on which no member has a value leaves
        the characteristic missing that day rather than standardising the rest against
        themselves.
        """
        values = self.values.copy()
        n_dates, n_securities, n_factors = values.shape
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (n_dates, n_securities):
                raise ValueError(
                    f"weights have shape {weights.shape}, expected "
                    f"{(n_dates, n_securities)} (dates x securities)"
                )
        members = _checked_membership(
            membership, (n_dates, n_securities), "(dates x securities)"
        )

        clipped = 0
        clipped_outside = 0
        for t in range(n_dates):
            for k in range(n_factors):
                column = values[t, :, k]
                present = ~np.isnan(column)
                if not present.any():
                    continue
                inside = present if members is None else present & members[t]
                if not inside.any():
                    values[t, present, k] = np.nan
                    continue
                raw = column[present]
                # Which of `raw` may set the scale. Every statistic below is taken over
                # `sample` alone; everything computed from them is applied to all of `raw`.
                among = inside[present]
                sample = raw[among]

                # Clip the RAW values first, against a scale an outlier cannot inflate.
                # Doing this after standardising would be useless: see the docstring.
                moved = np.zeros(raw.size, dtype=bool)
                if winsorise is not None and sample.size > 2:
                    centre = float(np.median(sample))
                    scale = float(np.median(np.abs(sample - centre))) * 1.4826
                    if scale <= 0.0:
                        scale = float(np.std(sample, ddof=1))
                    if scale > 0.0:
                        low = centre - winsorise * scale
                        high = centre + winsorise * scale
                        moved = (raw < low) | (raw > high)
                        clipped += int(np.count_nonzero(moved))
                        raw = np.clip(raw, low, high)
                        sample = raw[among]

                if weights is None:
                    w = np.ones(sample.size)
                else:
                    w = np.nan_to_num(weights[t, inside], nan=0.0)
                    if w.sum() <= 0.0:
                        w = np.ones(sample.size)

                mean = float(np.average(sample, weights=w))
                # Equal-weighted dispersion on purpose: see the docstring.
                spread = float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0
                if spread <= 0.0:
                    # A characteristic with no cross-sectional variation carries no
                    # information this period. Centring it leaves zeros, which is honest.
                    standardised = np.zeros(raw.size)
                else:
                    standardised = (raw - mean) / spread
                    if winsorise is not None:
                        # A guard, not the clip that does the work. The robust clip above
                        # fixed the scale; this one only makes `winsorise` mean what it says,
                        # since the standard deviation computed on clipped data is still a
                        # little larger than the MAD scale the clip used, so a clipped value
                        # lands inside the stated bound rather than on it -- when the mean is
                        # near the median. A cap-weighted one need not be: LNCAP's sits far
                        # above it, so its lower tail reaches this bound, and below the
                        # members' scale the outsiders reach it most of all.
                        moved |= np.abs(standardised) > winsorise
                        standardised = np.clip(standardised, -winsorise, winsorise)
                clipped_outside += int(np.count_nonzero(moved & ~among))
                values[t, present, k] = standardised

        return CharacteristicPanel(
            dates=self.dates,
            securities=self.securities,
            names=self.names,
            values=values,
            frequency=self.frequency,
            standardised=True,
            winsorised=clipped,
            clipped_outside=clipped_outside,
        )
