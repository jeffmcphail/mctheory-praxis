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


def _rolling(values: np.ndarray, window: int, fn) -> np.ndarray:
    """Apply `fn` to each trailing window, leaving nan where there is not enough history."""
    n_periods, n_securities = values.shape
    out = np.full((n_periods, n_securities), np.nan)
    for t in range(window - 1, n_periods):
        block = values[t - window + 1 : t + 1, :]
        out[t, :] = fn(block)
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


def _market_model(
    returns: np.ndarray, market: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling OLS of each security on the market: alpha, beta, residual sd, residuals.

    Residuals come back full-length with nan outside the last window, because SERDP needs the
    series rather than its summary.
    """
    n_periods, n_securities = returns.shape
    alpha = np.full((n_periods, n_securities), np.nan)
    beta = np.full((n_periods, n_securities), np.nan)
    residual_sd = np.full((n_periods, n_securities), np.nan)
    residuals = np.full((n_periods, n_securities), np.nan)
    for t in range(window - 1, n_periods):
        lo = t - window + 1
        x = market[lo : t + 1]
        for j in range(n_securities):
            y = returns[lo : t + 1, j]
            ok = np.isfinite(y) & np.isfinite(x)
            if int(ok.sum()) < 3:
                continue
            design = np.column_stack([np.ones(int(ok.sum())), x[ok]])
            coefficients, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
            fitted = design @ coefficients
            error = y[ok] - fitted
            alpha[t, j] = coefficients[0]
            beta[t, j] = coefficients[1]
            if int(ok.sum()) > 2:
                residual_sd[t, j] = float(
                    np.sqrt(error @ error / (int(ok.sum()) - 2))
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
    beta_window_months: float = 60.0,
    volatility_window_months: float | None = None,
    membership: np.ndarray | None = None,
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
    """
    wanted = tuple(names) if names is not None else PRICE_DESCRIPTORS
    unknown = set(wanted) - set(PRICE_DESCRIPTORS)
    if unknown:
        raise ValueError(
            f"unknown descriptor(s): {sorted(unknown)}; this module computes "
            f"{list(PRICE_DESCRIPTORS)}"
        )

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
        computed["DASTD"] = _rolling(returns, window, _nan_std) * scale

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
        window = panel.periods_for(12.0)
        if "RSTR" in wanted:
            computed["RSTR"] = _rolling(
                log_excess, window, lambda b: np.nansum(b, axis=0)
            )
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

            computed["CMRA"] = _rolling(log_excess, window, _range)

    # --- Market-model descriptors ---------------------------------------------------
    needs_market = {"BTSG", "HALPHA", "SERDP"} & set(wanted)
    if needs_market:
        if market is None:
            for name in sorted(needs_market):
                skipped[name] = "no market_returns supplied"
        else:
            window = min(panel.periods_for(beta_window_months), n_periods)
            alpha, beta, residual_sd, residuals = _market_model(returns, market, window)
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
        traded = _rolling(panel.volumes, window, lambda b: np.nansum(b, axis=0))
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
            window = min(panel.periods_for(beta_window_months), n_periods)
            _, _, residual_sd, _ = _market_model(returns, market, window)
            months = panel.periods_for(12.0)
            dollar = _rolling(
                panel.volumes * panel.traded_prices, months, lambda b: np.nansum(b, axis=0)
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
    """
    if not panel.standardised:
        raise ValueError(
            "build_risk_indices needs a standardised descriptor panel; call "
            "CharacteristicPanel.standardise first"
        )
    wanted = list(indices) if indices is not None else list(_RISK_INDEX)
    unknown = set(wanted) - set(_RISK_INDEX)
    if unknown:
        raise ValueError(f"unknown risk index/indices: {sorted(unknown)}")
    available = [
        name
        for name in wanted
        if any(member in panel.names for member in _RISK_INDEX[name])
    ]
    if not available:
        raise ValueError(
            "the descriptor panel carries no member of any requested risk index"
        )
    columns = [
        blend_to_index(
            panel,
            name,
            weights=None if weights is None else weights.get(name),
            cap_weights=cap_weights,
            membership=membership,
        )
        for name in available
    ]
    return CharacteristicPanel(
        dates=panel.dates,
        securities=panel.securities,
        names=tuple(available),
        values=np.stack(columns, axis=2),
        frequency=panel.frequency,
        standardised=True,
    )


def fit_index_weights(
    panel: CharacteristicPanel,
    index: str,
    returns: np.ndarray,
    *,
    membership: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit a risk index's descriptor weights by explanatory power, pooled over periods.

    The handbook describes the real procedure as blending descriptors "to produce the factor
    exposure that maximizes explanatory power". This is that, in its simplest honest form: a
    pooled cross-sectional regression of returns on the index's standardised descriptors,
    with the coefficients normalised to sum to one.

    It recovers Barra's *method*, not their numbers. A model built this way is a
    reimplementation of a published methodology and should be described as one.

    `membership` holds the fit to the estimation universe, as the model's regressions are
    held: otherwise the names outside the index help choose the blend that measures the ones
    inside it.
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
    held = _checked_membership(membership, returns.shape, "(dates x securities)")
    columns = [panel.names.index(name) for name in members]
    design = panel.values[:, :, columns].reshape(-1, len(members))
    target = returns.reshape(-1)
    keep = np.isfinite(target) & np.isfinite(design).all(axis=1)
    if held is not None:
        keep &= held.reshape(-1)
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
