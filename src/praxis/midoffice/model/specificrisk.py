"""Forecasting specific risk from characteristics, rather than measuring it from history.

The sample standard deviation of a security's residual series is what `crosssectional.py`
currently produces, and the *Barra Risk Model Handbook* names it as the naive approach in
almost those words: "The simplest way to estimate the specific risk matrix is to compute the
historical variances of the specific returns. This, however, assumes that the specific return
variance is stable over time."

It has two failures.  It is noisy for a short history, and it says **nothing at all** about a
security with none -- a new issue gets no forecast, which for a model whose whole appeal is
that exposures are observable today is an embarrassing hole.  And it misses the fact that the
general level of idiosyncratic volatility rises and falls for everyone at once, so in a
turbulent month every security's forecast is stale in the same direction.

Barra's model (Risk Model Handbook EQ 3-12 to 3-14) separates those two problems:

    sigma_it = kappa_it * (1 + V_it) * S_t                                       (EQ 3-12)

    S_t      = alpha + sum_j beta_j S_{t-j}                                      (EQ 3-13)

    V_it     = sum_k Z_ikt gamma_k                                               (EQ 3-14)

Three pieces, each answering a different question:

* **S_t**, the market-wide average level of absolute specific return, forecast from its own
  recent history.  This is the part that moves together, and nothing built from a single
  security's residuals can see it.
* **V_it**, how much riskier asset i is than that average, regressed cross-sectionally on
  observable characteristics.  This is the part that gives a brand-new issue a forecast:
  it has characteristics even with no return history at all.
* **kappa**, which converts a forecast of *absolute* return into a standard deviation.

Everything is built on **absolute** returns rather than squared ones, and the handbook says
why in two words: "for robustness reasons".  A mean absolute deviation is far less sensitive
to one outlying month than a variance is -- the same instinct that puts a median absolute
deviation behind the winsorising in `characteristics.py`.

**kappa is not a constant.**  For a normal distribution E|X| = sigma * sqrt(2/pi), so the
conversion factor is sqrt(pi/2) = 1.2533, and that is the fallback here.  But USE3 calibrates
it as a piecewise-linear function of relative capitalisation, fitted so that the average bias
in predicted specific risk is zero within each capitalisation decile.  That is doing two jobs
at once: converting units, and absorbing whatever systematic bias is left over, which runs
with size because small companies' specific returns are further from normal than large ones'.

**The linear form has an end.**  EQ 3-14 makes V linear in the characteristics, so `1 + V`
turns negative for a security far enough into the tail, and the multiplicative form then
forecasts a negative specific risk.  The model is a local approximation around the average
security and stops being one at the edges.  `SpecificRiskModel.forecast` floors at a fraction
of the market-average specific risk rather than at zero -- a floored forecast is weak, but a
zero would claim the security has no idiosyncratic risk at all, and a portfolio concentrated
in it would be reported as carrying only factor risk.  `floored()` names them.

Identification is worth stating, because three multiplied factors could otherwise absorb each
other's scale.  `S_t` is pinned to an observable -- the realised cross-sectional mean of
|u_it|.  `V_it` is mean-zero by construction, since the mean of |u_it| over assets *is* `S_t`,
so `V_it = |u_it| / S_t - 1` averages to zero exactly.  What is left for kappa is the honest
remainder: the absolute-to-standard-deviation conversion and the size-related bias.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

NORMAL_KAPPA = float(np.sqrt(np.pi / 2.0))
"""sqrt(pi/2) = 1.2533: E|X| = sigma*sqrt(2/pi) for a normal, so this converts back.

The fallback when there is no capitalisation data to calibrate against, and the value the
calibration should land near when specific returns really are normal and size carries no
residual bias -- which `test_calibration_recovers_the_normal_constant` checks.
"""

DEFAULT_AVERAGE_PERIODS = 3
"""The moving-average length for S_t. Barra's ARE1 sheet states J = 3 months."""

MIN_PERIODS = 12
"""Below this there is not enough history to separate a level from a trend in S_t."""

RESIDUAL_SPREADS = ("mad", "sd")
"""How `relative_residual_sd` is measured across securities: 1.4826 times the median absolute
deviation from the median, or the standard deviation."""


@dataclass(frozen=True)
class SpecificRiskNotes:
    """What the fit could not do properly, and under what conventions it ran."""

    n_periods: int = 0
    n_securities: int = 0
    n_characteristics: int = 0
    average_model: str = "moving"
    average_periods: int = DEFAULT_AVERAGE_PERIODS
    kappa_calibrated: bool = False
    kappa_range: tuple[float, float] | None = None
    gamma: Mapping[str, float] = field(default_factory=dict)
    # The securities with the largest shares of the sum of V squared that gamma is fitted to,
    # largest first, at most five: who the fit learns most from.
    v_squared_shares: tuple[tuple[str, float], ...] = ()
    # How `relative_residual_sd` was measured, and both measures: 1.4826 times the MAD of the
    # securities' average relative residuals, and their standard deviation.
    residual_spread: str = "mad"
    residual_spread_mad: float = float("nan")
    residual_spread_sd: float = float("nan")
    # Likewise for the square of the standard deviation: who would set how far each
    # security's own history is trusted, were the spread measured by it. Not the same list.
    residual_spread_shares: tuple[tuple[str, float], ...] = ()
    relative_r_squared: float = float("nan")
    level_r_squared: float = float("nan")
    thin_periods: int = 0
    no_characteristics: tuple[str, ...] = ()
    forecast_level: float = float("nan")

    def qualifications(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.n_periods < MIN_PERIODS:
            out.append(
                f"the specific risk model was fitted on {self.n_periods} period(s), fewer "
                f"than the {MIN_PERIODS} it asks for; the average-level forecast is barely "
                f"distinguishable from the last observation"
            )
        if not self.kappa_calibrated:
            out.append(
                f"kappa was not calibrated against capitalisation and falls back to the "
                f"normal constant {NORMAL_KAPPA:.4f}; supply capitalisations to absorb the "
                f"size-related bias in specific risk that USE3 calibrates away"
            )
        elif self.kappa_range is not None:
            low, high = self.kappa_range
            if high / low > 2.0 if low > 0 else False:
                out.append(
                    f"calibrated kappa runs from {low:.3f} to {high:.3f} across "
                    f"capitalisation deciles, a spread of more than two to one, which is a "
                    f"larger size effect than the normal conversion factor can explain"
                )
        if not np.isnan(self.relative_r_squared) and self.relative_r_squared < 0.01:
            out.append(
                f"the characteristics explain {self.relative_r_squared:.1%} of the "
                f"cross-sectional variation in relative specific risk, so V is close to "
                f"flat and the model is barely better than one average for everyone"
            )
        if self.no_characteristics:
            out.append(
                f"{len(self.no_characteristics)} security(ies) have no characteristic row "
                f"and get the average-level forecast with no relative adjustment: "
                f"{', '.join(self.no_characteristics[:5])}"
            )
        if self.thin_periods:
            out.append(
                f"{self.thin_periods} period(s) had no usable residuals and were skipped "
                f"when forming the average absolute specific return"
            )
        return tuple(out)


@dataclass(frozen=True)
class SpecificRiskModel:
    """A fitted forecast of specific risk: the three factors of EQ 3-12, ready to apply."""

    names: tuple[str, ...]
    gamma: np.ndarray
    forecast_level: float
    relative_residual_sd: float = float("nan")
    kappa_constant: float = NORMAL_KAPPA
    kappa_knots: tuple[tuple[float, float], ...] = ()
    notes: SpecificRiskNotes = field(default_factory=SpecificRiskNotes)

    def kappa(self, relative_cap: float | None = None) -> float:
        """The absolute-to-standard-deviation factor at a given relative capitalisation.

        `relative_cap` is a position in [0, 1] -- the security's capitalisation percentile
        within its universe, which is what "capitalisation decile" means operationally.
        Linear between the fitted knots and flat outside them, because extrapolating a
        size effect past the largest company observed is inventing data.
        """
        if not self.kappa_knots or relative_cap is None:
            return self.kappa_constant
        xs = np.array([k for k, _ in self.kappa_knots])
        ys = np.array([v for _, v in self.kappa_knots])
        return float(np.interp(relative_cap, xs, ys))

    def relative_level(self, exposures: Mapping[str, float]) -> float:
        """V: how much riskier than average this security's characteristics imply."""
        z = np.array([float(exposures.get(name, 0.0)) for name in self.names])
        return float(z @ self.gamma)

    def floor_level(self, floor_fraction: float = 0.1) -> float:
        """The specific risk a forecast is floored at: a fraction of the market-average level.

        One definition for every method that floors and for a caller asking which forecasts
        landed on it, so the comparison is against the same number the floor produced.
        """
        return max(floor_fraction, 0.0) * (self.kappa_constant * self.forecast_level)

    def forecast(
        self,
        exposures: Mapping[str, Mapping[str, float]],
        relative_caps: Mapping[str, float] | None = None,
        *,
        floor_fraction: float = 0.1,
    ) -> dict[str, float]:
        """Specific volatility per security, for anything with characteristics.

        A security with no return history at all gets a forecast here, which is the entire
        point: `(1 + V) * S` needs characteristics, not a residual series.

        **The floor is not a formality.**  EQ 3-14 makes V a linear function of the
        characteristics, so `1 + V` goes negative for a security far enough into the tail --
        with a Size coefficient of 0.5, anything below two standard deviations.  Unfloored,
        such a security is forecast to have *no idiosyncratic risk at all*, which is a far
        worse error than any plausible mis-estimate: a portfolio concentrated in it would be
        reported as carrying only factor risk.  Flooring at zero merely renames the problem.

        `floor_fraction` therefore floors at a fraction of the market-average specific risk,
        defaulting to a tenth.  The claim it encodes -- that no security has less than a tenth
        of the average security's idiosyncratic volatility -- is weak enough to be safe and
        strong enough to stop the linear form producing a zero.  `floored()` names the
        securities it caught, because a security at the floor has a forecast the model did not
        really make.
        """
        floor = self.floor_level(floor_fraction)
        out: dict[str, float] = {}
        for code, row in exposures.items():
            relative = self.relative_level(row)
            cap = None if relative_caps is None else relative_caps.get(code)
            value = self.kappa(cap) * (1.0 + relative) * self.forecast_level
            out[code] = max(float(value), floor)
        return out

    def floored(
        self,
        exposures: Mapping[str, Mapping[str, float]],
        relative_caps: Mapping[str, float] | None = None,
        *,
        floor_fraction: float = 0.1,
    ) -> tuple[str, ...]:
        """Which securities the floor caught -- those whose linear V ran past the end of its
        own validity, and whose forecast is therefore the floor rather than the model.

        The test is on the characteristic forecast alone. Under `blend` that is not the number
        delivered: a security's own history can carry it well clear of the floor."""
        floor = self.floor_level(floor_fraction)
        caught: list[str] = []
        for code, row in exposures.items():
            cap = None if relative_caps is None else relative_caps.get(code)
            value = self.kappa(cap) * (1.0 + self.relative_level(row)) * self.forecast_level
            if value < floor:
                caught.append(code)
        return tuple(caught)

    def blend(
        self,
        exposures: Mapping[str, Mapping[str, float]],
        history: Mapping[str, tuple[float, int]] | None = None,
        relative_caps: Mapping[str, float] | None = None,
        *,
        floor_fraction: float = 0.1,
    ) -> dict[str, float]:
        """Combine the characteristic forecast with each security's own residual history.

        Neither source dominates, and measuring that is what motivates this method.  The
        characteristic model's error is **irreducible**: it is the cross-sectional variation
        the characteristics simply do not contain, and no amount of extra history shrinks it.
        A sample standard deviation's error is **statistical**: it falls as 1/sqrt(2(n-1)),
        so it beats the model outright once a security has enough history -- on a synthetic
        panel where the characteristics explain 20% of the cross-sectional variation, a plain
        15-period trailing standard deviation wins by a wide margin.

        So the model earns its place in exactly two situations: a security with little or no
        history, where the sample estimate is hopeless or undefined, and a shift in the
        market-wide level, which `forecast_level` tracks and a long trailing window cannot
        see.  BIM301 says as much in a footnote -- its specific risk models work "by combining
        assets' specific return history with other information."

        The weights are inverse-variance, in the relative space both estimates live in:

            v_model = relative_residual_sd^2            (irreducible)
            v_hist  = 1 / (2 (n - 1))                   (statistical, per security)
            w_model = v_hist / (v_model + v_hist)

        which has the right limits without being told them.  A security with no history gets
        `v_hist = inf` and so the pure model; a security with a long history gets
        `v_hist -> 0` and so its own sample estimate.  `history` maps a security to its
        `(sample standard deviation, number of observations)`; anything absent from it is
        treated as having no history at all.
        """
        history = history or {}
        base = self.kappa_constant * self.forecast_level
        floor = self.floor_level(floor_fraction)
        v_model = (
            self.relative_residual_sd**2
            if np.isfinite(self.relative_residual_sd)
            else 0.0
        )
        out: dict[str, float] = {}
        for code, row in exposures.items():
            model_relative = self.relative_level(row)
            entry = history.get(code)
            relative = model_relative
            if entry is not None and base > 0.0:
                sample_sd, n_observations = float(entry[0]), int(entry[1])
                if n_observations > 1 and sample_sd > 0.0:
                    history_relative = sample_sd / base - 1.0
                    v_history = 1.0 / (2.0 * (n_observations - 1))
                    total = v_model + v_history
                    weight = v_history / total if total > 0.0 else 1.0
                    relative = weight * model_relative + (1.0 - weight) * history_relative
            cap = None if relative_caps is None else relative_caps.get(code)
            value = self.kappa(cap) * (1.0 + relative) * self.forecast_level
            out[code] = max(float(value), floor)
        return out

    def model_weight(self, n_observations: int) -> float:
        """How much of a blended forecast comes from the characteristics, at this much
        history. One at no history, falling toward zero as the sample estimate sharpens."""
        if n_observations <= 1:
            return 1.0
        v_model = (
            self.relative_residual_sd**2
            if np.isfinite(self.relative_residual_sd)
            else 0.0
        )
        v_history = 1.0 / (2.0 * (n_observations - 1))
        total = v_model + v_history
        return float(v_history / total) if total > 0.0 else 1.0

    def describe(self) -> str:
        head = (
            f"specific risk model on {len(self.names)} characteristic(s), "
            f"forecast average absolute specific return {self.forecast_level:.4%}"
        )
        if self.kappa_knots:
            low = min(v for _, v in self.kappa_knots)
            high = max(v for _, v in self.kappa_knots)
            return f"{head}, kappa {low:.3f}-{high:.3f} by capitalisation"
        return f"{head}, kappa {self.kappa_constant:.4f} (normal constant)"


def _largest_shares(
    squares: np.ndarray, securities: Sequence[str], count: int = 5
) -> tuple[tuple[str, float], ...]:
    """The securities with the largest shares of a sum of squares, largest first."""
    total = float(squares.sum())
    if not total > 0.0:
        return ()
    return tuple(
        (securities[j], float(squares[j] / total))
        for j in np.argsort(-squares, kind="stable")[:count]
        if squares[j] > 0.0
    )


def _forecast_average_level(
    levels: np.ndarray, *, model: str, periods: int
) -> tuple[float, float]:
    """Forecast the next S from its own history. Returns (forecast, in-sample R-squared)."""
    n = len(levels)
    if model == "moving":
        window = levels[-min(periods, n) :]
        return float(window.mean()), float("nan")
    if model != "autoregressive":
        raise ValueError(
            f"average_model must be 'moving' or 'autoregressive', got {model!r}"
        )
    lags = min(periods, max(n - periods - 1, 1))
    rows = n - lags
    if rows < lags + 2:
        # Not enough to fit the autoregression; the moving average is the honest fallback.
        window = levels[-min(periods, n) :]
        return float(window.mean()), float("nan")
    design = np.column_stack(
        [np.ones(rows)] + [levels[lags - j - 1 : n - j - 1] for j in range(lags)]
    )
    target = levels[lags:]
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residual = target - fitted
    total = target - target.mean()
    r_squared = (
        1.0 - float(residual @ residual) / float(total @ total)
        if float(total @ total) > 0.0
        else float("nan")
    )
    latest = np.concatenate([[1.0], levels[n - lags :][::-1]])
    return float(latest @ coefficients), r_squared


def estimate_specific_risk_model(
    residuals: np.ndarray,
    securities: Sequence[str],
    characteristics: np.ndarray,
    names: Sequence[str],
    *,
    capitalisations: np.ndarray | None = None,
    average_model: str = "moving",
    average_periods: int = DEFAULT_AVERAGE_PERIODS,
    deciles: int = 10,
    min_periods: int = MIN_PERIODS,
    annualisation: float = 1.0,
    residual_spread: str = "mad",
) -> SpecificRiskModel:
    """Fit EQ 3-12 to 3-14 from a residual panel and a characteristic panel.

    `residuals` is periods-by-securities, nan where a security has no return that period --
    the `residuals` a cross-sectional estimation produces.  `characteristics` is
    periods-by-securities-by-characteristics and should already be standardised, since EQ 3-14
    has no intercept and relies on the characteristics being centred.

    `capitalisations` is periods-by-securities and is what makes the kappa calibration
    possible.  Without it kappa falls back to the normal constant and the notes say so.

    Only cells with a residual take part. S, V and gamma need one by construction; the kappa
    calibration is held to the same cells on both of its sides, so a characteristic row or a
    capitalisation recorded in a period the security was not in the regression moves nothing.

    `residual_spread` says how `relative_residual_sd` -- v_model's root in every blend weight --
    is measured across the securities' average relative residuals. "mad", the default, is
    1.4826 times their median absolute deviation from the median, which estimates a standard
    deviation on normal data and which one security far out barely moves; it falls back to the
    standard deviation where it is zero. "sd" is the standard deviation, which one security far
    enough out does set: on the S&P 500 run FRC, collapsing while still in the index, is 72
    percent of its square, and starting the sample 24 days earlier moved it from 0.361 to 0.281,
    where the MAD's moved from 0.1333 to 0.1334. The notes carry both.
    """
    if residual_spread not in RESIDUAL_SPREADS:
        raise ValueError(
            f"residual_spread must be one of {RESIDUAL_SPREADS}; got {residual_spread!r}"
        )
    residuals = np.asarray(residuals, dtype=float)
    characteristics = np.asarray(characteristics, dtype=float)
    securities = tuple(securities)
    names = tuple(names)
    n_periods, n_securities = residuals.shape
    if characteristics.shape[:2] != residuals.shape:
        raise ValueError(
            f"characteristics have shape {characteristics.shape[:2]} but residuals are "
            f"{residuals.shape}; they must share periods and securities"
        )
    if characteristics.shape[2] != len(names):
        raise ValueError(
            f"{characteristics.shape[2]} characteristic column(s) but {len(names)} name(s)"
        )
    if len(securities) != n_securities:
        raise ValueError(
            f"{len(securities)} security code(s) for {n_securities} residual column(s)"
        )

    absolute = np.abs(residuals)

    # S_t: the realised cross-sectional mean absolute specific return, period by period.
    with warnings.catch_warnings():
        # A period with no usable residual at all is a legitimate state, named in the notes
        # as a thin period; numpy's "mean of empty slice" is not news.
        warnings.simplefilter("ignore", RuntimeWarning)
        levels = np.nanmean(absolute, axis=1)
    usable = np.isfinite(levels) & (levels > 0.0)
    thin = int(np.count_nonzero(~usable))
    if not usable.any():
        raise ValueError(
            "no period had a usable mean absolute specific return, so there is no level to "
            "forecast; check that the residual panel is not entirely missing"
        )
    forecast_level, level_r2 = _forecast_average_level(
        levels[usable], model=average_model, periods=average_periods
    )
    # The cells the fit learns from: a residual, in a period with a usable level. A residual
    # exists exactly where the security was in that period's regression.
    in_fit = np.isfinite(residuals) & usable[:, None]

    # V_it = |u_it| / S_t - 1, which is mean-zero across assets by construction.
    relative = np.full_like(absolute, np.nan)
    relative[usable, :] = absolute[usable, :] / levels[usable, None] - 1.0

    # gamma: one pooled cross-sectional regression of V on the characteristics, no intercept
    # (EQ 3-14 has none, and V is already centred).
    flat_v = relative.reshape(-1)
    flat_z = characteristics.reshape(-1, len(names))
    keep = np.isfinite(flat_v) & np.isfinite(flat_z).all(axis=1)
    if int(keep.sum()) <= len(names):
        raise ValueError(
            f"only {int(keep.sum())} usable observation(s) to fit {len(names)} "
            f"characteristic coefficient(s); widen the panel or drop characteristics"
        )
    gamma, *_ = np.linalg.lstsq(flat_z[keep], flat_v[keep], rcond=None)
    # Who it learns from. V is bounded below by -1 and not at all above, and a least-squares
    # fit weighs each cell by its square, so one security far enough into the tail can set
    # gamma for everyone: on the S&P 500 run SBNY, trading on at cents after it left the
    # index, was 40 percent of the sum.
    v_squared = np.where(keep.reshape(n_periods, n_securities), relative, 0.0) ** 2
    v_squared_shares = _largest_shares(v_squared.sum(axis=0), securities)
    fitted = flat_z[keep] @ gamma
    residual_v = flat_v[keep] - fitted
    total_v = flat_v[keep] - flat_v[keep].mean()
    relative_r2 = (
        1.0 - float(residual_v @ residual_v) / float(total_v @ total_v)
        if float(total_v @ total_v) > 0.0
        else float("nan")
    )
    # The cross-sectional dispersion the characteristics do NOT explain. This is the model's
    # own irreducible error, and it is what `blend` weighs against a security's history.
    # Averaging over periods first strips out the single-period noise in |u|, which is large
    # and is not part of the model's error at all.
    per_security = np.full(n_securities, np.nan)
    for j in range(n_securities):
        column = relative[:, j] - characteristics[:, j, :] @ gamma
        present = np.isfinite(column)
        if present.any():
            per_security[j] = float(column[present].mean())
    finite = np.isfinite(per_security)
    spread_sd = spread_mad = float("nan")
    if int(finite.sum()) > 1:
        averages = per_security[finite]
        spread_sd = float(np.std(averages, ddof=1))
        spread_mad = 1.4826 * float(np.median(np.abs(averages - np.median(averages))))
        if spread_mad <= 0.0:
            spread_mad = spread_sd
    relative_residual_sd = spread_mad if residual_spread == "mad" else spread_sd
    # A standard deviation across securities, so one average far enough out sets it for all of
    # them: FRC, collapsing while still in the index, was 60 percent of its square.
    residual_spread_shares = (
        _largest_shares(
            np.where(finite, per_security - per_security[finite].mean(), 0.0) ** 2, securities
        )
        if int(finite.sum()) > 1
        else ()
    )

    # kappa: what remains between the absolute-return forecast and realised volatility.
    #
    # Both sides are divided through by S_t first, and that is not cosmetic. kappa describes
    # the *shape* of the specific return distribution -- how far a standard deviation is from
    # a mean absolute deviation -- which is a property of the distribution, not of its level.
    # Comparing an unnormalised realised standard deviation against an unnormalised mean
    # absolute forecast mixes a quadratic measure with a linear one, so a window spanning a
    # change in the market-wide level biases kappa upward: a 3x jump in the last quarter of
    # the window inflates it about 15 percent, because the realised standard deviation
    # weights the loud periods by their square while the mean absolute does not. Normalising
    # by S_t removes the level path entirely and leaves kappa measuring what it names.
    v_hat = characteristics @ gamma
    normalised = np.full_like(absolute, np.nan)
    normalised[usable, :] = residuals[usable, :] / levels[usable, None]
    # Over the realised side's own cells. A predicted level where there is no residual has
    # nothing to be compared with, and averaging it in set kappa from periods it never saw.
    predicted_relative = np.where(in_fit, 1.0 + v_hat, np.nan)

    realised_sd = np.full(n_securities, np.nan)
    mean_predicted = np.full(n_securities, np.nan)
    for j in range(n_securities):
        column = normalised[:, j]
        present = np.isfinite(column)
        if int(present.sum()) >= 2:
            realised_sd[j] = float(np.std(column[present], ddof=1))
        predicted = predicted_relative[:, j]
        ok = np.isfinite(predicted) & (predicted > 0.0)
        if ok.any():
            mean_predicted[j] = float(predicted[ok].mean())

    kappa_knots: tuple[tuple[float, float], ...] = ()
    kappa_constant = NORMAL_KAPPA
    calibrated = False
    ratio_ok = np.isfinite(realised_sd) & np.isfinite(mean_predicted) & (mean_predicted > 0)
    if ratio_ok.any():
        kappa_constant = float(np.median(realised_sd[ratio_ok] / mean_predicted[ratio_ok]))

    if capitalisations is not None:
        caps = np.asarray(capitalisations, dtype=float)
        if caps.shape != residuals.shape:
            raise ValueError(
                f"capitalisations have shape {caps.shape}, expected {residuals.shape}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Size over the same cells: the decile a security's bias is calibrated in is its
            # size while it was being measured, not across periods it was absent from.
            mean_cap = np.nanmean(np.where(in_fit & (caps > 0.0), caps, np.nan), axis=0)
        eligible = ratio_ok & np.isfinite(mean_cap)
        if int(eligible.sum()) >= deciles * 2:
            order = np.argsort(mean_cap[eligible])
            index = np.flatnonzero(eligible)[order]
            buckets = np.array_split(index, deciles)
            knots: list[tuple[float, float]] = []
            for position, bucket in enumerate(buckets):
                if bucket.size == 0:
                    continue
                # Zero the average bias within the decile: the ratio of realised volatility
                # to the pre-kappa absolute forecast, which is what kappa has to supply.
                value = float(
                    np.sum(realised_sd[bucket]) / np.sum(mean_predicted[bucket])
                )
                midpoint = (position + 0.5) / len(buckets)
                knots.append((midpoint, value))
            if len(knots) >= 2:
                kappa_knots = tuple(knots)
                calibrated = True

    no_characteristics = tuple(
        code
        for j, code in enumerate(securities)
        if not np.isfinite(characteristics[:, j, :]).all(axis=1).any()
    )

    notes = SpecificRiskNotes(
        n_periods=int(usable.sum()),
        n_securities=n_securities,
        n_characteristics=len(names),
        average_model=average_model,
        average_periods=average_periods,
        kappa_calibrated=calibrated,
        kappa_range=(
            (min(v for _, v in kappa_knots), max(v for _, v in kappa_knots))
            if kappa_knots
            else None
        ),
        gamma={name: float(g) for name, g in zip(names, gamma)},
        v_squared_shares=v_squared_shares,
        residual_spread=residual_spread,
        residual_spread_mad=spread_mad,
        residual_spread_sd=spread_sd,
        residual_spread_shares=residual_spread_shares,
        relative_r_squared=relative_r2,
        level_r_squared=level_r2,
        thin_periods=thin,
        no_characteristics=no_characteristics,
        forecast_level=forecast_level * annualisation,
    )
    return SpecificRiskModel(
        names=names,
        gamma=np.asarray(gamma, dtype=float),
        forecast_level=forecast_level * annualisation,
        relative_residual_sd=relative_residual_sd,
        kappa_constant=kappa_constant,
        kappa_knots=kappa_knots,
        notes=notes,
    )
