"""USE3's descriptors: the raw measurements underneath the risk indices.

`characteristics.py` takes exposures as given and standardises them.  This builds them, to
the definitions in Appendix A of the *U.S. Equity Model Version 3* handbook, from price,
volume and shares outstanding -- data a trading platform already has, with no fundamentals
needed.

That covers fifteen descriptors across five of the thirteen US-E3 risk indices: Volatility,
Momentum, Size, Size Nonlinearity and Trading Activity.  The remaining eight -- Growth,
Earnings Yield, Value, Earnings Variation, Leverage, Currency Sensitivity, Yield and the
Non-estimation indicator -- need balance sheet and earnings data and are not here.

**One thing is deliberately not reproduced.**  Appendix A states plainly: "The method of
combining these descriptors into risk indices is proprietary to BARRA."  The descriptor
definitions are published; the weights blending them into a risk index are not.  So this
module stops at descriptors, and `blend_to_index` offers the two honest ways forward rather
than a guessed table of weights:

* equal-weighted across the standardised descriptors, which is what you use when you have no
  better information and is at least a stated convention rather than an invented one;
* weights fitted to maximise explanatory power, which is what the handbook says Barra does --
  "blended with other descriptors ... to produce the factor exposure that maximizes
  explanatory power" -- and which is estimable from returns rather than proprietary.

The second recovers Barra's *method* without claiming to recover their numbers. A model built
this way is a reimplementation of the published methodology, not a clone of USE3, and
pretending otherwise would be the one thing worse than not building it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Sequence

import numpy as np

from .characteristics import CharacteristicPanel, _checked_membership
from .panel import Frequency

TRADING_DAYS_PER_MONTH = 23.0
"""USE3's own figure, stated in the DASTD definition."""

USE4_BETA_WINDOW = 252
"""USE4's market-model window, in trading days."""

USE4_BETA_HALF_LIFE = 63.0
"""USE4's market-model half-life, in trading days."""

USE3_BETA_WINDOW_MONTHS = 60.0
"""USE3's market-model window, in months."""

PRICE_DESCRIPTORS = (
    "LNCAP",
    "LCAPCB",
    "LPRI",
    "DASTD",
    "HILO",
    "CMRA",
    "BTSG",
    "HALPHA",
    "SERDP",
    "RSTR",
    "STOM",
    "STOQ",
    "STOA",
    "STO5",
    "VLVR",
)

WINDOW_DESCRIPTORS = (
    "DASTD",
    "CMRA",
    "BTSG",
    "HALPHA",
    "SERDP",
    "RSTR",
    "STOM",
    "STOQ",
    "STOA",
    "STO5",
    "VLVR",
)
"""The descriptors computed over a trailing window of a security's own history, which
`compute_descriptors`' `min_coverage` applies to. HILO, LPRI, LNCAP and LCAPCB are one day's
values and have no window."""

_RISK_INDEX = {
    "Volatility": ("BTSG", "DASTD", "HILO", "LPRI", "CMRA", "SERDP"),
    "Momentum": ("RSTR", "HALPHA"),
    "Size": ("LNCAP",),
    "Size Nonlinearity": ("LCAPCB",),
    "Trading Activity": ("STOM", "STOQ", "STOA", "STO5", "VLVR"),
}
"""Which descriptors belong to which US-E3 risk index, from Table 8-1.

The membership is published; the weights within each index are not. `blend_to_index` is where
that gap is handled rather than papered over.
"""


@dataclass(frozen=True)
class SecurityPanel:
    """Price, volume and shares outstanding, by date and security.

    One frequency throughout.  USE3 mixes frequencies -- DASTD is 65 daily returns, RSTR is
    12 monthly ones -- and a faithful implementation would take both.  This takes one panel
    and computes each descriptor over the equivalent number of its own periods, which is
    correct at daily frequency and an approximation at monthly.

    **Two price series, because the descriptors need different bases.**  `prices` is
    split-adjusted and drives returns, so a four-for-one split is not a -75% day.
    `unadjusted_prices` is what the stock actually traded at, and three things need it:

    * **LNCAP**, because market capitalisation pairs it with a *point-in-time* share count.
      An adjusted price against a pre-split share count undercounts by the split factor.
    * **LPRI**, because the descriptor's economic content is that low-priced stocks are
      riskier.  A stock that traded at $400 before a four-for-one split shows as $100 in
      adjusted terms, and LPRI would record it as a cheap stock in a year when it was not.
    * **VLVR**, whose numerator is the dollars that actually changed hands.

    Supplying only `prices` is allowed and the descriptors still compute, but the three above
    are then on the adjusted basis and `compute_descriptors` says so in its notes rather than
    letting the substitution pass unremarked.
    """

    dates: tuple[date, ...]
    securities: tuple[str, ...]
    prices: np.ndarray
    unadjusted_prices: np.ndarray | None = None
    volumes: np.ndarray | None = None
    shares_outstanding: np.ndarray | None = None
    highs: np.ndarray | None = None
    lows: np.ndarray | None = None
    frequency: Frequency = Frequency.MONTHLY

    def __post_init__(self) -> None:
        dates = tuple(self.dates)
        securities = tuple(self.securities)
        if len(set(dates)) != len(dates):
            raise ValueError("duplicate dates in the security panel")
        if len(set(securities)) != len(securities):
            raise ValueError("duplicate securities in the security panel")
        if list(dates) != sorted(dates):
            raise ValueError("security panel dates must be ascending")
        shape = (len(dates), len(securities))
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "securities", securities)
        for name in (
            "prices",
            "unadjusted_prices",
            "volumes",
            "shares_outstanding",
            "highs",
            "lows",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if array.shape != shape:
                raise ValueError(
                    f"{name} have shape {array.shape}, expected {shape} "
                    f"(dates x securities)"
                )
            object.__setattr__(self, name, array)

    @property
    def n_periods(self) -> int:
        return len(self.dates)

    @property
    def n_securities(self) -> int:
        return len(self.securities)

    def returns(self) -> np.ndarray:
        """Simple returns, nan in the first period and wherever a price is missing."""
        out = np.full_like(self.prices, np.nan)
        previous = self.prices[:-1]
        current = self.prices[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.where(previous > 0.0, current / previous - 1.0, np.nan)
        return out

    @property
    def traded_prices(self) -> np.ndarray:
        """What the stock actually traded at: unadjusted where available, else adjusted."""
        return self.prices if self.unadjusted_prices is None else self.unadjusted_prices

    @property
    def has_unadjusted(self) -> bool:
        return self.unadjusted_prices is not None

    def market_capitalisation(self) -> np.ndarray:
        if self.shares_outstanding is None:
            raise ValueError(
                "market capitalisation needs shares_outstanding, which this panel does not "
                "carry; LNCAP, LCAPCB and every turnover descriptor depend on it"
            )
        return self.traded_prices * self.shares_outstanding

    def periods_for(self, months: float) -> int:
        """How many of this panel's periods correspond to that many months.

        USE3 states every window in months. On a monthly panel this is the identity; on a
        daily one it is `months * 23`, USE3's own trading-day figure.
        """
        if self.frequency is Frequency.DAILY:
            return max(int(round(months * TRADING_DAYS_PER_MONTH)), 1)
        if self.frequency is Frequency.WEEKLY:
            return max(int(round(months * 52.0 / 12.0)), 1)
        if self.frequency is Frequency.QUARTERLY:
            return max(int(round(months / 3.0)), 1)
        return max(int(round(months)), 1)


def _covered(valid: np.ndarray, window: int, min_coverage: float) -> np.ndarray:
    """Dates by securities: whether the trailing window ending there holds at least one valid
    observation, and at least `min_coverage` of the window. False before the first full one."""
    n_periods = valid.shape[0]
    out = np.zeros(valid.shape, dtype=bool)
    if window > n_periods:
        return out
    running = np.zeros((n_periods + 1, valid.shape[1]), dtype=np.int64)
    np.cumsum(valid, axis=0, dtype=np.int64, out=running[1:])
    counts = running[window:] - running[: n_periods - window + 1]
    out[window - 1 :] = (counts > 0) & (counts / window >= min_coverage)
    return out


def _rolling(
    values: np.ndarray,
    window: int,
    fn,
    *,
    min_coverage: float = 0.0,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Apply `fn` to each trailing window, leaving nan where there is not enough history --
    and where the window holds no valid observation, or less than `min_coverage` of it.

    `valid` marks the observations that count, the finite `values` by default. A sum over a
    window with none is a zero that measures nothing, so it is not taken.
    """
    n_periods, n_securities = values.shape
    out = np.full((n_periods, n_securities), np.nan)
    enough = _covered(np.isfinite(values) if valid is None else valid, window, min_coverage)
    for t in range(window - 1, n_periods):
        block = values[t - window + 1 : t + 1, :]
        out[t, :] = np.where(enough[t], fn(block), np.nan)
    return out


def _nan_std(block: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    counts = np.count_nonzero(~np.isnan(block), axis=0)
    out = np.full(block.shape[1], np.nan)
    usable = counts > ddof
    if usable.any():
        sub = block[:, usable]
        mean = np.nanmean(sub, axis=0)
        centred = np.where(np.isnan(sub), 0.0, sub - mean)
        out[usable] = np.sqrt(
            np.sum(centred**2, axis=0) / (counts[usable] - ddof)
        )
    return out


def market_model_window(
    panel: SecurityPanel,
    beta_window_months: float | None = None,
    beta_half_life: float | None = None,
) -> tuple[int, float | None]:
    """The market model `compute_descriptors` fits, in the panel's own periods: its window,
    capped at the panel's length, and its half-life -- None where every period of the window
    is weighted equally.

    The defaults are USE4's on a daily panel, 252 trading days with a half-life of 63, and
    USE3's on any other, 60 months weighted equally: USE4's beta is a daily estimator, and a
    year of monthly returns is too few to fit one on. `beta_window_months` sets the window in
    its place, converted by `periods_for`; `beta_half_life` the half-life, in the panel's own
    periods, `math.inf` weighting every period equally.
    """
    daily = panel.frequency is Frequency.DAILY
    if beta_window_months is None:
        window = USE4_BETA_WINDOW if daily else panel.periods_for(USE3_BETA_WINDOW_MONTHS)
    else:
        window = panel.periods_for(beta_window_months)
    if beta_half_life is None:
        half_life = USE4_BETA_HALF_LIFE if daily else None
    elif np.isposinf(beta_half_life):
        half_life = None
    elif np.isfinite(beta_half_life) and beta_half_life > 0.0:
        half_life = float(beta_half_life)
    else:
        raise ValueError(
            f"beta_half_life is a number of periods, so positive -- math.inf for equal "
            f"weights -- or None for the panel's default; got {beta_half_life!r}"
        )
    return min(window, panel.n_periods), half_life


def _market_model(
    returns: np.ndarray,
    market: np.ndarray,
    window: int,
    min_coverage: float = 0.0,
    half_life: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling regression of each security on the market: alpha, beta, residual sd, residuals.

    Residuals come back full-length with nan outside the last window, because SERDP needs the
    series rather than its summary. A window needs three observations to fit, and at least
    `min_coverage` of it.

    Ordinary least squares by default. Given a `half_life` in periods, weighted least squares
    with each observation's weight halving every `half_life` periods back from the window's
    end; the residual sd is then the weighted one, its degrees of freedom taken on Kish's
    effective number of observations -- which is n - 2 again when the weights are equal.
    """
    n_periods, n_securities = returns.shape
    alpha = np.full((n_periods, n_securities), np.nan)
    beta = np.full((n_periods, n_securities), np.nan)
    residual_sd = np.full((n_periods, n_securities), np.nan)
    residuals = np.full((n_periods, n_securities), np.nan)
    decay = None
    if half_life is not None:
        decay = 0.5 ** (np.arange(window - 1, -1, -1, dtype=float) / half_life)
    for t in range(window - 1, n_periods):
        lo = t - window + 1
        x = market[lo : t + 1]
        for j in range(n_securities):
            y = returns[lo : t + 1, j]
            ok = np.isfinite(y) & np.isfinite(x)
            if int(ok.sum()) < 3 or int(ok.sum()) / window < min_coverage:
                continue
            design = np.column_stack([np.ones(int(ok.sum())), x[ok]])
            if decay is None:
                coefficients, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
            else:
                root = np.sqrt(decay[ok])
                coefficients, *_ = np.linalg.lstsq(
                    design * root[:, None], y[ok] * root, rcond=None
                )
            fitted = design @ coefficients
            error = y[ok] - fitted
            alpha[t, j] = coefficients[0]
            beta[t, j] = coefficients[1]
            if decay is None:
                if int(ok.sum()) > 2:
                    residual_sd[t, j] = float(
                        np.sqrt(error @ error / (int(ok.sum()) - 2))
                    )
            else:
                w = decay[ok]
                effective = float(w.sum() ** 2 / (w @ w))
                if effective > 2.0:
                    residual_sd[t, j] = float(
                        np.sqrt(
                            (w @ error**2) / w.sum() * effective / (effective - 2.0)
                        )
                    )
            if t == n_periods - 1:
                slot = np.full(window, np.nan)
                slot[np.flatnonzero(ok)] = error
                residuals[lo : t + 1, j] = slot
    return alpha, beta, residual_sd, residuals


def compute_descriptors(
    panel: SecurityPanel,
    *,
    market_returns: np.ndarray | None = None,
    risk_free: np.ndarray | None = None,
    names: Sequence[str] | None = None,
    beta_window_months: float | None = None,
    volatility_window_months: float | None = None,
    membership: np.ndarray | None = None,
    min_coverage: float = 0.5,
    rstr_skip: int = 0,
    beta_half_life: float | None = None,
) -> tuple[CharacteristicPanel, dict[str, str]]:
    """Compute USE3's price-and-volume descriptors over the panel.

    Returns the descriptors as a `CharacteristicPanel` -- unstandardised, because
    standardisation is `CharacteristicPanel.standardise`'s job and doing it here would make
    the raw values unavailable -- and a mapping from each descriptor that could **not** be
    computed to the reason why, so a missing column is explained rather than silently absent.

    `market_returns` and `risk_free` are per-period series over the same dates. Without the
    market, BTSG, HALPHA and SERDP cannot be computed; without the risk-free rate, RSTR and
    CMRA fall back to raw returns, which is a different descriptor and is reported as such.

    `volatility_window_months` defaults to USE3's own 65 trading days on a daily panel and to
    twelve months otherwise. That substitution is not free and is reported in the notes: 65
    daily returns is a usable sample, while the same span of monthly data is three
    observations, and a standard deviation of three points is noise. A longer monthly window
    measures something real but it is no longer DASTD -- it is a slower descriptor wearing
    the same name.

    `membership`, a dates-by-securities boolean array, is the estimation universe LCAPCB's
    normalisation is taken over -- the one descriptor here whose definition is cross-sectional.
    Everything else is a security's own history.

    **A window mostly empty gives no value, and an empty one never does.**  The window
    descriptors -- `WINDOW_DESCRIPTORS` -- are missing wherever their window holds less than
    `min_coverage` of valid data: the fraction of its periods with a valid observation of what
    the descriptor is computed from -- a return for DASTD, RSTR and CMRA, a return with the
    market's for the market-model ones, a volume for the turnovers. The default, 0.5, is the
    line below which a descriptor is not the one it is named: a twelve-month RSTR summed over
    five months of returns is a five-month momentum. 0 keeps any window holding at least one
    observation. An empty window gives no value at any setting -- a sum over one is zero, and
    RSTR, CMRA and the turnovers once recorded that zero as a value: every day before a
    security's first price read as no momentum and no trading. The market model's own floor of
    three observations, and DASTD's of two, still apply.

    `rstr_skip` is how many of the panel's periods -- trading days on a daily panel -- are cut
    from the end of RSTR's twelve-month window: the academic twelve-minus-one reading, so that
    a month's short-term reversal does not cancel the momentum being measured. The window
    keeps its start, so RSTR has a value from the same date; the sum covers its earlier
    periods, and the coverage rule is applied to those. The default, 0, cuts nothing. At 21 on
    the S&P 500 run nothing material moved -- Momentum's mean rose from +2.44% to +3.23% a year
    at a t of 1.09 -- and R-squared fell 0.0006. USE4's lag is a different reading, a full
    twelve-month window ending `rstr_skip` periods back; it was measured in midoffice03 and is
    not built.

    **The market model** behind BTSG, HALPHA, SERDP and VLVR's residual sd is USE4's on a
    daily panel by default: 252 trading days, each period's weight halving every 63 back from
    the window's end. USE3's 60 months need more days than a five-year daily panel has, so a
    regression over them fits once, at the last date; the 36 months the S&P 500 run once used
    instead kept beta, the screen's strongest descriptor, out of 552 of its 938 estimated
    periods. On a panel of any other frequency the default is USE3's 60 months, weighted
    equally. `beta_window_months` sets the window in its place, and `beta_half_life` the
    half-life, in the panel's own periods -- trading days on a daily panel; `math.inf` weights
    every period equally, by ordinary least squares. `market_model_window` says what a call
    fits.
    """
    wanted = tuple(names) if names is not None else PRICE_DESCRIPTORS
    unknown = set(wanted) - set(PRICE_DESCRIPTORS)
    if unknown:
        raise ValueError(
            f"unknown descriptor(s): {sorted(unknown)}; this module computes "
            f"{list(PRICE_DESCRIPTORS)}"
        )
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(
            f"min_coverage is the fraction of a window with valid data, so it lies in "
            f"[0, 1]; got {min_coverage}"
        )
    year = panel.periods_for(12.0)
    if (
        isinstance(rstr_skip, bool)
        or not isinstance(rstr_skip, (int, np.integer))
        or not 0 <= rstr_skip < year
    ):
        raise ValueError(
            f"rstr_skip is a whole number of periods excluded from the end of RSTR's "
            f"{year}-period window, so at least 0 and fewer than {year}; got {rstr_skip!r}"
        )
    rstr_skip = int(rstr_skip)
    beta_window, beta_half_life = market_model_window(panel, beta_window_months, beta_half_life)

    n_periods, n_securities = panel.prices.shape
    members = _checked_membership(
        membership, (n_periods, n_securities), "(dates x securities)"
    )
    returns = panel.returns()
    skipped: dict[str, str] = {}
    computed: dict[str, np.ndarray] = {}

    market = None
    if market_returns is not None:
        market = np.asarray(market_returns, dtype=float).reshape(-1)
        if market.size != n_periods:
            raise ValueError(
                f"market_returns has {market.size} period(s), expected {n_periods}"
            )
    rate = None
    if risk_free is not None:
        rate = np.asarray(risk_free, dtype=float).reshape(-1)
        if rate.size != n_periods:
            raise ValueError(
                f"risk_free has {rate.size} period(s), expected {n_periods}"
            )

    has_shares = panel.shares_outstanding is not None
    has_volume = panel.volumes is not None

    if not panel.has_unadjusted:
        for name in ("LNCAP", "LPRI", "VLVR"):
            if name in wanted:
                skipped[name] = (
                    "computed on split-adjusted prices: the panel carries no "
                    "unadjusted_prices, so a stock that split reads as cheaper and smaller "
                    "than it was"
                )

    if volatility_window_months is None:
        if panel.frequency is Frequency.DAILY:
            volatility_window_months = 65.0 / TRADING_DAYS_PER_MONTH
        else:
            volatility_window_months = 12.0

    # --- Size -----------------------------------------------------------------------
    log_cap = None
    if "LNCAP" in wanted or "LCAPCB" in wanted:
        if not has_shares:
            skipped["LNCAP"] = "no shares_outstanding on the panel"
            skipped["LCAPCB"] = "no shares_outstanding on the panel"
        else:
            cap = panel.market_capitalisation()
            with np.errstate(divide="ignore", invalid="ignore"):
                log_cap = np.where(cap > 0.0, np.log(cap), np.nan)
            if "LNCAP" in wanted:
                computed["LNCAP"] = log_cap

    if "LCAPCB" in wanted and log_cap is not None:
        # "the cube of the NORMALIZED log of market capitalization" -- the standardisation
        # is part of the definition, so it is done here rather than left to the caller. Over
        # the members when there is a membership, like every other standardisation: the cube
        # is not affine, so an ex-member that moved the centre would change its shape, not
        # just its scale.
        cube = np.full_like(log_cap, np.nan)
        for t in range(n_periods):
            row = log_cap[t, :]
            present = np.isfinite(row)
            inside = present if members is None else present & members[t]
            if int(inside.sum()) > 1:
                sample = row[inside]
                spread = float(np.std(sample, ddof=1))
                if spread > 0.0:
                    cube[t, present] = ((row[present] - sample.mean()) / spread) ** 3
        computed["LCAPCB"] = cube

    # --- Volatility and Momentum from prices ----------------------------------------
    if "LPRI" in wanted:
        traded = panel.traded_prices
        with np.errstate(divide="ignore", invalid="ignore"):
            computed["LPRI"] = np.where(traded > 0.0, np.log(traded), np.nan)

    if "DASTD" in wanted:
        window = panel.periods_for(volatility_window_months)
        if window < 12 and panel.frequency is not Frequency.DAILY:
            skipped["DASTD"] = (
                f"computed over {window} period(s): USE3 defines it on 65 daily returns and "
                f"this panel cannot supply a comparable sample, so it is noisier than the "
                f"descriptor it stands in for"
            )
        scale = (
            np.sqrt(TRADING_DAYS_PER_MONTH)
            if panel.frequency is Frequency.DAILY
            else 1.0
        )
        computed["DASTD"] = (
            _rolling(returns, window, _nan_std, min_coverage=min_coverage) * scale
        )

    if "HILO" in wanted:
        if panel.highs is None or panel.lows is None:
            skipped["HILO"] = "no highs/lows on the panel"
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                computed["HILO"] = np.where(
                    (panel.highs > 0.0) & (panel.lows > 0.0),
                    np.log(panel.highs / panel.lows),
                    np.nan,
                )

    if "CMRA" in wanted or "RSTR" in wanted:
        excess = returns if rate is None else returns - rate[:, None]
        if rate is None:
            for name in ("CMRA", "RSTR"):
                if name in wanted:
                    skipped[name] = (
                        "computed on raw rather than excess returns: no risk_free supplied"
                    )
        with np.errstate(invalid="ignore"):
            log_excess = np.log1p(np.where(excess > -1.0, excess, np.nan))
        window = year
        if "RSTR" in wanted:
            # Summed over the window less its last `rstr_skip` periods: the sum over the
            # shorter span, read `rstr_skip` periods later.
            summed = _rolling(
                log_excess,
                window - rstr_skip,
                lambda b: np.nansum(b, axis=0),
                min_coverage=min_coverage,
            )
            if rstr_skip:
                later = np.full_like(summed, np.nan)
                later[rstr_skip:] = summed[:-rstr_skip]
                summed = later
            computed["RSTR"] = summed
        if "CMRA" in wanted:

            def _range(block: np.ndarray) -> np.ndarray:
                cumulative = np.nancumsum(np.where(np.isnan(block), 0.0, block), axis=0)
                top = np.nanmax(cumulative, axis=0)
                bottom = np.nanmin(cumulative, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.where(
                        (1.0 + bottom) > 0.0,
                        np.log((1.0 + top) / (1.0 + bottom)),
                        np.nan,
                    )

            computed["CMRA"] = _rolling(
                log_excess, window, _range, min_coverage=min_coverage
            )

    # --- Market-model descriptors ---------------------------------------------------
    needs_market = {"BTSG", "HALPHA", "SERDP"} & set(wanted)
    if needs_market:
        if market is None:
            for name in sorted(needs_market):
                skipped[name] = "no market_returns supplied"
        else:
            alpha, beta, residual_sd, residuals = _market_model(
                returns, market, beta_window, min_coverage, beta_half_life
            )
            if "HALPHA" in wanted:
                computed["HALPHA"] = alpha
            if "BTSG" in wanted:
                # "If beta is negative, then the descriptor is set equal to zero."
                product = beta * residual_sd
                computed["BTSG"] = np.where(beta < 0.0, 0.0, product)
            if "SERDP" in wanted:
                # Only the final period. The market-model residual *series* is kept for the
                # last window alone -- keeping it for every window would mean storing
                # (periods x window x securities) residuals, which for a daily panel is
                # hundreds of millions of values. So SERDP is a point-in-time descriptor
                # here: usable as a current exposure, not as a history.
                skipped["SERDP"] = (
                    "computed at the final period only; the market-model residual series "
                    "is retained for one window rather than all of them, so SERDP is a "
                    "current exposure and not a time series"
                )
                serial = np.full((n_periods, n_securities), np.nan)
                for j in range(n_securities):
                    column = residuals[:, j]
                    present = np.isfinite(column)
                    if int(present.sum()) < 4:
                        continue
                    error = column[present]
                    triples = error[2:] + error[1:-1] + error[:-2]
                    squares = error[2:] ** 2 + error[1:-1] ** 2 + error[:-2] ** 2
                    denominator = float(np.sum(squares))
                    if denominator > 0.0:
                        serial[-1, j] = float(np.sum(triples**2)) / denominator
                computed["SERDP"] = serial

    # --- Trading Activity -----------------------------------------------------------
    turnover_windows = {"STOM": 1.0, "STOQ": 3.0, "STOA": 12.0, "STO5": 60.0}
    for name, months in turnover_windows.items():
        if name not in wanted:
            continue
        if not (has_volume and has_shares):
            skipped[name] = "needs both volumes and shares_outstanding"
            continue
        window = panel.periods_for(months)
        if window > n_periods:
            # Otherwise this is an all-nan column that quietly disappears downstream, and
            # the reader is left wondering whether the descriptor failed or was never asked
            # for. A five-year turnover needs five years.
            skipped[name] = (
                f"needs {window} period(s) of history and the panel has {n_periods}, so "
                f"every value would be missing"
            )
            continue
        # Covered by the days that traded: a window of share counts and no volume is a
        # turnover nobody measured, not a turnover of zero.
        traded = _rolling(
            panel.volumes,
            window,
            lambda b: np.nansum(b, axis=0),
            min_coverage=min_coverage,
        )
        with warnings.catch_warnings():
            # A security with no shares reported anywhere in the window is a legitimate
            # state -- it is named in the load report -- and numpy's "mean of empty slice"
            # adds nothing to that.
            warnings.simplefilter("ignore", RuntimeWarning)
            outstanding = _rolling(
                panel.shares_outstanding, window, lambda b: np.nanmean(b, axis=0)
            )
        annualiser = 12.0 / months if months > 1.0 else 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            computed[name] = np.where(
                outstanding > 0.0, annualiser * traded / outstanding, np.nan
            )

    if "VLVR" in wanted:
        if not has_volume:
            skipped["VLVR"] = "needs volumes"
        elif market is None:
            skipped["VLVR"] = "needs market_returns for the residual standard deviation"
        else:
            _, _, residual_sd, _ = _market_model(
                returns, market, beta_window, min_coverage, beta_half_life
            )
            months = panel.periods_for(12.0)
            dollar = _rolling(
                panel.volumes * panel.traded_prices,
                months,
                lambda b: np.nansum(b, axis=0),
                min_coverage=min_coverage,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(
                    (residual_sd > 0.0) & (dollar > 0.0),
                    (12.0 / months) * dollar / residual_sd,
                    np.nan,
                )
                computed["VLVR"] = np.where(ratio > 0.0, np.log(ratio), np.nan)

    ordered = [name for name in wanted if name in computed]
    if not ordered:
        raise ValueError(
            f"no descriptor could be computed; reasons: {skipped or 'no names requested'}"
        )
    values = np.stack([computed[name] for name in ordered], axis=2)
    return (
        CharacteristicPanel(
            dates=panel.dates,
            securities=panel.securities,
            names=tuple(ordered),
            values=values,
            frequency=panel.frequency,
        ),
        skipped,
    )


def risk_index_members(index: str) -> tuple[str, ...]:
    """Which descriptors US-E3 Table 8-1 assigns to a risk index."""
    if index not in _RISK_INDEX:
        raise ValueError(
            f"unknown risk index {index!r}; this module covers {sorted(_RISK_INDEX)}"
        )
    return _RISK_INDEX[index]


SIZE_NONLINEARITY = ("use4", "use3")
"""How Size Nonlinearity is built: USE4's way, from the Size exposure, or USE3's, the blend of
LCAPCB. See `nonlinear_size`."""


def _checked_size_nonlinearity(size_nonlinearity: str) -> str:
    if size_nonlinearity not in SIZE_NONLINEARITY:
        raise ValueError(
            f"size_nonlinearity must be one of {SIZE_NONLINEARITY}; got {size_nonlinearity!r}"
        )
    return size_nonlinearity


def index_descriptors(index: str, size_nonlinearity: str = "use4") -> tuple[str, ...]:
    """The descriptors a risk index's exposure is built from: Table 8-1's, as
    `risk_index_members` gives them, except Size Nonlinearity built USE4's way -- a function of
    the Size exposure, and so of Size's descriptors."""
    if _checked_size_nonlinearity(size_nonlinearity) == "use4" and index == "Size Nonlinearity":
        return risk_index_members("Size")
    return risk_index_members(index)


def nonlinear_size(
    size: np.ndarray,
    *,
    cap_weights: np.ndarray | None = None,
    regression_weights: np.ndarray | None = None,
    membership: np.ndarray | None = None,
    bound: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """USE4's Size Nonlinearity, date by date, from the Size exposure: the exposure, and a
    dates-by-securities mask of the values its clip moved.

    The Size exposure is cubed, and the cube treated as any descriptor is but for the raw MAD
    clip. On a cube that clip binds at |z| of about 1.2 -- the median |z| of a normal, 0.6745,
    cubed is 0.31, four times 1.4826 times that is 1.82, and 1.82 is the cube of 1.22 -- so it
    would flatten the tails the cube is there to measure. The cube is standardised instead over
    the estimation universe -- cap-weighted mean, equal-weighted standard deviation -- and
    clipped at +/-`bound`. It is then orthogonalised against Size: each date's least-squares
    fit on Size and a constant, weighted by `regression_weights` over the estimation universe,
    is taken out, and what is left standardised again, exactly. Its regression-weighted
    covariance with Size over the estimation universe is then zero to rounding on every date,
    so the pair no longer splits one size effect between two collinear exposures.

    `membership` is the estimation universe, as everywhere: the members set the moments and
    the fit, and everyone else with a Size exposure is put on their scale. Without
    `regression_weights` the fit weighs every security equally.
    """
    size = np.asarray(size, dtype=float)
    n_dates, n_securities = size.shape
    held = _checked_membership(membership, (n_dates, n_securities), "(dates x securities)")
    weights = np.ones(size.shape)
    if regression_weights is not None:
        weights = np.asarray(regression_weights, dtype=float)
        if weights.shape != size.shape:
            raise ValueError(
                f"regression_weights have shape {weights.shape}, expected {size.shape} "
                f"(dates x securities)"
            )
    standardised = _restandardise(size**3, cap_weights, held)
    with np.errstate(invalid="ignore"):
        moved = np.abs(standardised) > bound
    clipped = np.clip(standardised, -bound, bound)
    residual = np.full_like(size, np.nan)
    for t in range(n_dates):
        present = np.isfinite(size[t]) & np.isfinite(clipped[t])
        inside = present & np.isfinite(weights[t]) & (weights[t] > 0.0)
        if held is not None:
            inside &= held[t]
        if int(inside.sum()) < 3:
            continue
        w = weights[t, inside]
        x, y = size[t, inside], clipped[t, inside]
        x_mean = float(np.average(x, weights=w))
        y_mean = float(np.average(y, weights=w))
        spread = float(w @ (x - x_mean) ** 2)
        if spread <= 0.0:
            continue
        slope = float(w @ ((x - x_mean) * (y - y_mean))) / spread
        residual[t, present] = (
            clipped[t, present] - y_mean - slope * (size[t, present] - x_mean)
        )
    return _restandardise(residual, cap_weights, held), moved


def _restandardise(
    values: np.ndarray,
    cap_weights: np.ndarray | None,
    membership: np.ndarray | None = None,
) -> np.ndarray:
    """Cross-sectional standardisation, period by period: the convention used throughout.

    Cap-weighted mean so the market has no tilt, equal-weighted dispersion so a handful of
    large names cannot set the scale. The same asymmetry `characteristics.py` explains, and
    the same estimation universe: given a membership, the members set the mean and the
    dispersion and everyone else is put on their scale. Fewer than two members with a value
    leaves the day missing.
    """
    out = np.full_like(values, np.nan)
    for t in range(values.shape[0]):
        row = values[t, :]
        present = np.isfinite(row)
        inside = present if membership is None else present & membership[t]
        if int(inside.sum()) < 2:
            continue
        sample = row[inside]
        if cap_weights is None:
            w = np.ones(sample.size)
        else:
            w = np.nan_to_num(cap_weights[t, inside], nan=0.0)
            if w.sum() <= 0.0:
                w = np.ones(sample.size)
        spread = float(np.std(sample, ddof=1))
        if spread <= 0.0:
            out[t, present] = 0.0
            continue
        out[t, present] = (row[present] - float(np.average(sample, weights=w))) / spread
    return out


def blend_to_index(
    panel: CharacteristicPanel,
    index: str,
    *,
    weights: Mapping[str, float] | None = None,
    standardise: bool = True,
    cap_weights: np.ndarray | None = None,
    membership: np.ndarray | None = None,
) -> np.ndarray:
    """Combine a risk index's descriptors into one exposure, from a standardised panel.

    USE3's own weights are proprietary -- Appendix A says so in as many words -- so the
    default here is equal weighting across whichever of the index's descriptors the panel
    carries. That is a stated convention rather than a guess at theirs. Supply `weights` to
    use a fitted set instead; `fit_index_weights` estimates them the way the handbook
    describes, by explanatory power.

    The panel must be standardised, because averaging descriptors on different scales is
    meaningless -- a log market capitalisation of 17 would swamp a turnover of 0.4.

    **The result is standardised again, and that second pass is not redundant.**  A weighted
    average of standardised descriptors has a standard deviation *below* one unless they are
    perfectly correlated: 0.707 for two uncorrelated ones, 0.894 at a correlation of 0.6. The
    shrinkage grows with the number of descriptors blended, so without the second pass Size
    (one descriptor) would come out at scale 1.0 while Trading Activity (five) came out at
    0.38 and Volatility (six) at 0.47.  A one-sigma move in the underlying characteristic
    would then register as 0.38 in one factor and 1.00 in another, the factor returns would be
    scaled inversely, and the covariance between them would mean nothing -- which is the exact
    failure standardising exists to prevent.  Barra standardises twice for this reason: the
    GEM handbook's normalisation formula is written for the *risk index*, not the descriptor.

    `membership` is the estimation universe the second pass is taken over, and should be the
    one the descriptors were standardised over: then the index's members carry a cap-weighted
    mean of exactly zero, so the index itself has no exposure to its own style factors.
    """
    if not panel.standardised:
        raise ValueError(
            "blend_to_index needs a standardised panel: averaging raw descriptors on "
            "different scales is meaningless. Call CharacteristicPanel.standardise first."
        )
    held = _checked_membership(
        membership, (len(panel.dates), len(panel.securities)), "(dates x securities)"
    )
    members = [name for name in risk_index_members(index) if name in panel.names]
    if not members:
        raise ValueError(
            f"the panel carries none of {index}'s descriptors "
            f"({', '.join(risk_index_members(index))})"
        )
    columns = [panel.names.index(name) for name in members]
    if weights is None:
        chosen = np.ones(len(members))
    else:
        chosen = np.array([float(weights.get(name, 0.0)) for name in members])
        total = np.abs(chosen).sum()
        if total <= 0.0:
            raise ValueError(f"the supplied weights for {index} are all zero")
    chosen = chosen / chosen.sum()
    block = panel.values[:, :, columns]
    mask = np.isfinite(block)
    weighted = np.where(mask, block, 0.0) * chosen[None, None, :]
    denominator = (mask * chosen[None, None, :]).sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        blended = np.where(
            denominator > 0.0, weighted.sum(axis=2) / denominator, np.nan
        )
    return _restandardise(blended, cap_weights, held) if standardise else blended


def build_risk_indices(
    panel: CharacteristicPanel,
    *,
    weights: Mapping[str, Mapping[str, float]] | None = None,
    cap_weights: np.ndarray | None = None,
    indices: Iterable[str] | None = None,
    membership: np.ndarray | None = None,
    size_nonlinearity: str = "use4",
    regression_weights: np.ndarray | None = None,
) -> CharacteristicPanel:
    """Descriptors in, risk indices out: the exposures a cross-sectional model consumes.

    This is the step that turns measurements into factors.  RSTR and HALPHA are not factors
    and never enter the regression; Momentum is, and does.  Collapsing roughly sixty
    descriptors into thirteen indices is what keeps the factor covariance matrix estimable --
    115 factors would carry 6,555 free covariance parameters against 68 factors' 2,346 -- and
    what stops two measurements of the same thing splitting their shared signal arbitrarily
    between two unstable coefficients.

    The result is standardised and drops straight into
    `estimate_cross_sectional_model(..., standardise=False)`. Pass the `membership` the
    descriptors were standardised over, and the estimator the same.

    `size_nonlinearity` says how Size Nonlinearity is built. "use4", the default, builds it
    from the Size exposure, cubed and orthogonalised against Size -- see `nonlinear_size` --
    and needs only Size's descriptors; pass the `regression_weights` the model will be
    estimated with, which the orthogonalisation is weighted by. "use3" blends LCAPCB, USE3's
    raw cube, which is 0.93 correlated with Size on the S&P 500 run.
    """
    if not panel.standardised:
        raise ValueError(
            "build_risk_indices needs a standardised descriptor panel; call "
            "CharacteristicPanel.standardise first"
        )
    _checked_size_nonlinearity(size_nonlinearity)
    wanted = list(indices) if indices is not None else list(_RISK_INDEX)
    unknown = set(wanted) - set(_RISK_INDEX)
    if unknown:
        raise ValueError(f"unknown risk index/indices: {sorted(unknown)}")
    available = [
        name
        for name in wanted
        if any(
            member in panel.names for member in index_descriptors(name, size_nonlinearity)
        )
    ]
    if not available:
        raise ValueError(
            "the descriptor panel carries no member of any requested risk index"
        )

    def blended(name: str) -> np.ndarray:
        return blend_to_index(
            panel,
            name,
            weights=None if weights is None else weights.get(name),
            cap_weights=cap_weights,
            membership=membership,
        )

    columns = {
        name: blended(name)
        for name in available
        if not (name == "Size Nonlinearity" and size_nonlinearity == "use4")
    }
    if "Size Nonlinearity" in available and size_nonlinearity == "use4":
        size = columns["Size"] if "Size" in columns else blended("Size")
        columns["Size Nonlinearity"], _ = nonlinear_size(
            size,
            cap_weights=cap_weights,
            regression_weights=regression_weights,
            membership=membership,
        )
    return CharacteristicPanel(
        dates=panel.dates,
        securities=panel.securities,
        names=tuple(available),
        values=np.stack([columns[name] for name in available], axis=2),
        frequency=panel.frequency,
        standardised=True,
    )


def blend_composition(
    panel: CharacteristicPanel,
    index: str,
    rows: Sequence[int],
    counted: np.ndarray | None = None,
    *,
    size_nonlinearity: str = "use4",
) -> dict[str, np.ndarray]:
    """Which of a risk index's descriptors its blend drew on, on each of `rows`.

    `blend_to_index` averages whichever of the index's descriptors a security has, so the
    blend changes composition whenever one starts or stops having values -- on the S&P 500 run,
    the market model's three, a long window after the rest. Per descriptor of `index` the panel
    carries, in Table 8-1 order: a boolean per row, True where it holds a value for at least
    one security `counted` there. `counted`, rows by securities, is whose exposures the
    question is about -- a period's regression, say -- and everyone by default. The
    descriptors are `index_descriptors`', so Size Nonlinearity built USE4's way, the default,
    is read as built from Size's.
    """
    rows = [int(row) for row in rows]
    members = [
        name for name in index_descriptors(index, size_nonlinearity) if name in panel.names
    ]
    shape = (len(rows), len(panel.securities))
    if counted is None:
        counted = np.ones(shape, dtype=bool)
    counted = np.asarray(counted)
    if counted.dtype != bool or counted.shape != shape:
        raise ValueError(
            f"counted must be a boolean array of shape {shape} (rows x securities); got "
            f"{counted.dtype} {counted.shape}"
        )
    return {
        name: (np.isfinite(panel.values[rows, :, panel.names.index(name)]) & counted).any(
            axis=1
        )
        for name in members
    }


def fit_index_weights(
    panel: CharacteristicPanel,
    index: str,
    returns: np.ndarray,
    *,
    membership: np.ndarray | None = None,
    lag: int = 1,
) -> dict[str, float]:
    """Fit a risk index's descriptor weights by explanatory power, pooled over periods.

    The handbook describes the real procedure as blending descriptors "to produce the factor
    exposure that maximizes explanatory power". This is that, in its simplest honest form: a
    pooled cross-sectional regression of returns on the index's standardised descriptors,
    with the coefficients normalised to sum to one.

    **Each descriptor is paired with the return `lag` periods after it** -- descriptor on t,
    return on t+1 by default, as the model's own regressions read their exposures. At a lag of
    0 a descriptor built from day t's prices explains day t's return, which it holds: RSTR
    sums it, so the fit would hand RSTR the blend for knowing the answer. 0 is there for a
    synthetic world whose descriptors are known before their returns.

    **It is fitted on the whole sample, and that is look-ahead too.**  Every period's returns
    go into one regression, so the weights -- and every exposure blended with them, the
    earliest period's included -- are chosen knowing returns that come after. Out of sample
    would mean refitting on each date from the returns before it alone. Nothing here does
    that, and `test_lookahead.py` records what the whole-sample fit costs: repricing one day
    moves the exposures of every period. The run script does not call this.

    It recovers Barra's *method*, not their numbers. A model built this way is a
    reimplementation of a published methodology and should be described as one.

    `membership` holds the fit to the estimation universe, as the model's regressions are
    held -- read on the descriptor's date, as theirs is: otherwise the names outside the index
    help choose the blend that measures the ones inside it.
    """
    if not panel.standardised:
        raise ValueError("fit_index_weights needs a standardised panel")
    members = [name for name in risk_index_members(index) if name in panel.names]
    if not members:
        raise ValueError(f"the panel carries none of {index}'s descriptors")
    returns = np.asarray(returns, dtype=float)
    if returns.shape != panel.values.shape[:2]:
        raise ValueError(
            f"returns have shape {returns.shape}, expected {panel.values.shape[:2]}"
        )
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError(f"lag must be a whole number of periods, zero or more; got {lag!r}")
    n_periods = returns.shape[0]
    if lag >= n_periods:
        raise ValueError(
            f"a lag of {lag} leaves none of the {n_periods} period(s) with an earlier one to "
            f"read its descriptors from"
        )
    held = _checked_membership(membership, returns.shape, "(dates x securities)")
    columns = [panel.names.index(name) for name in members]
    design = panel.values[: n_periods - lag, :, columns].reshape(-1, len(members))
    target = returns[lag:].reshape(-1)
    keep = np.isfinite(target) & np.isfinite(design).all(axis=1)
    if held is not None:
        keep &= held[: n_periods - lag].reshape(-1)
    if int(keep.sum()) <= len(members):
        raise ValueError(
            f"only {int(keep.sum())} usable observation(s) to fit {len(members)} weight(s)"
        )
    coefficients, *_ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
    total = coefficients.sum()
    if abs(total) < 1e-15:
        raise ValueError(
            f"the fitted weights for {index} sum to zero, so they cannot be normalised; "
            f"the descriptors are cancelling rather than combining"
        )
    return {name: float(c / total) for name, c in zip(members, coefficients)}
