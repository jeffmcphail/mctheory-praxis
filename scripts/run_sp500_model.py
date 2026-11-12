"""Build a factor model from a fetched S&P 500 panel, and say which descriptors earn a place.

    python scripts/run_sp500_model.py --industries --membership
    python scripts/run_sp500_model.py data/external/polygon/sp500_daily_*.npz --returns price

By default the panel is on securities (`build_sp500_securities.py`, `DEFAULT_PANEL`), which
carries total returns and traded volumes, and its own membership and industry codes besides;
the returns are total, Barra's, and price returns stay runnable (midoffice08's rulings 4 and
5). The first panel, on tickers (`fetch_sp500_panel.py`), stays runnable for comparison: it
carries price returns alone, so it runs with `--returns price`.

The descriptor screen follows the US-E3 handbook's own method -- "asset returns are regressed
against industries and descriptors, one descriptor at a time, after the normalization step.
Each descriptor is tested for statistical significance" -- with one change of emphasis that
matters for a risk model.

A premium and a risk factor are different things.  The Fama-MacBeth t-statistic asks whether a
descriptor has been *paid*, which is a question about expected return.  A risk model does not
care: a factor whose return oscillates around zero carries plenty of risk and no premium, and
excluding it because its t-statistic is small would leave that risk in the specific term where
nothing can hedge it.  So the column to read first is **mean R-squared** -- how much
cross-sectional return variance the descriptor explains, period by period -- and the second is
the **volatility of its factor return**, which is what actually enters the covariance matrix.
The t-statistic is reported third, as information about returns rather than risk.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from praxis.midoffice.model.bias import (  # noqa: E402
    OVERLAP_DRAWS,
    BiasTest,
    RiskForecast,
    bias_statistics,
)
from praxis.midoffice.model.characteristics import CharacteristicPanel  # noqa: E402
from praxis.midoffice.model.covariance import (  # noqa: E402
    HorizonCovariance,
    NeweyWestCovariance,
)
from praxis.midoffice.model.crosssectional import (  # noqa: E402
    cross_sectional_models_as_of,
    estimate_cross_sectional_model,
    regression_design,
    regression_universe,
)
from praxis.midoffice.model.descriptors import (  # noqa: E402
    SIZE_NONLINEARITY,
    TRADING_DAYS_PER_MONTH,
    SecurityPanel,
    blend_composition,
    build_risk_indices,
    compute_descriptors,
    index_descriptors,
    market_model_window,
    risk_index_members,
)
from praxis.midoffice.model.eigen import EIGEN_DRAWS  # noqa: E402
from praxis.midoffice.model.panel import Frequency, ReturnPanel  # noqa: E402
from praxis.midoffice.model.specificrisk import RESIDUAL_SPREADS  # noqa: E402
from praxis.midoffice.refdata import sp500  # noqa: E402
from praxis.midoffice.refdata.industries import (  # noqa: E402
    build_scheme,
    combine,
    industry_exposures,
    merge_industries,
)
from praxis.midoffice.refdata.polygon import (  # noqa: E402
    PolygonClient,
    PolygonError,
    read_api_key,
)

DEFAULT_PANEL = "data/external/polygon/sp500_securities_2022-01-03_2026-11-03_spinoffs.npz"
"""The panel on securities: its share counts repaired by midoffice08's rulings, a sample its own
weighted count confirms spared and every span ended where it leaves an exchange by midoffice09's,
and every spin-off the cache does not carry carried in the total return by midoffice14's rulings
(`config/sp500_spin_off_overrides.json`). The run's panel when none is given; midoffice13's,
before the spin-offs, is `_listed`."""

UNBLENDED = {
    "SERDP": "its residual series is kept for the last window alone, so it has a value on the "
             "last date and no other; no estimated period reads it, and blended it would enter "
             "the forecast exposures alone",
}
"""Descriptors computed, screened and reported, and kept out of every blend, with why. A
forecast exposure must be built from the descriptors the factor was estimated on. Computing
SERDP historically waits for a reason to; it has never had a usable period in the screen."""

SPECIFIC_HALF_LIFE = 63.0
"""Each security's specific-risk history weighted by a half-life of 63 trading days, the blend
told its effective number of observations (midoffice11's ruling 1): the shortest half-life
measured, and the best on every specific-side bias statistic. `--specific-half-life none` weighs
every specific return the same, as before."""

CORRELATION_HALF_LIFE = 504.0
"""The factor covariance's correlations on a half-life of 504 trading days, its volatilities on
`--half-life`'s 90, both Newey-West: the two-half-life estimator (midoffice11's ruling 2),
BIM301S's two years. `--correlation-half-life none` is one half-life for the whole matrix, as
before."""

HORIZON = 1
"""The model's horizon in trading days: one day (midoffice12's ruling 1), the factor covariance
the one-day covariance. Newey-West with Bartlett weights at L lags is exactly the per-day variance
of (L + 1)-day returns (midoffice11's ruling 3), so `--horizon H` puts H - 1 lags on the
volatilities -- a reporting view of broad portfolios' risk over a holding period of H days, not
the model (midoffice12's ruling 2) -- and none on the correlations, by default: the lags on the
volatilities alone (midoffice13's ruling 4). `--correlation-lags horizon` gives the correlations
the horizon's lags too, as before."""


def half_life_or_none(text: str) -> float | None:
    """A half-life in trading days, or `none` for the option's other form."""
    if text.strip().lower() == "none":
        return None
    value = float(text)
    if not value > 0.0:
        raise argparse.ArgumentTypeError(f"a half-life must be positive, or none; got {text!r}")
    return value


def horizon_list(text: str) -> tuple[int, ...]:
    """Horizons in trading days, comma-separated, each at least one."""
    try:
        values = tuple(sorted({int(part) for part in text.split(",") if part.strip()}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"horizons are whole trading days; got {text!r}")
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(f"each horizon must be at least one day; got {text!r}")
    return values


def lags_or_horizon(text: str) -> int | str:
    """A number of Newey-West lags, or `horizon` for the horizon's own."""
    if text.strip().lower() == "horizon":
        return "horizon"
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"lags are whole periods, or horizon; got {text!r}")
    if value < 0:
        raise argparse.ArgumentTypeError(f"lags must be non-negative; got {text!r}")
    return value


def scale_list(text: str) -> tuple[float, ...]:
    """Eigenfactor adjustment scales, comma-separated, each positive."""
    try:
        values = tuple(sorted({float(part) for part in text.split(",") if part.strip()}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"scales are numbers; got {text!r}")
    if not values or min(values) <= 0.0:
        raise argparse.ArgumentTypeError(f"each scale must be positive; got {text!r}")
    return values


def without(panel: CharacteristicPanel, names) -> CharacteristicPanel:
    """`panel` less the named characteristics."""
    keep = [k for k, name in enumerate(panel.names) if name not in names]
    return dataclasses.replace(
        panel, names=tuple(panel.names[k] for k in keep), values=panel.values[:, :, keep]
    )


def load_panel(
    path: Path, returns: str = "price", volumes: str | None = None
) -> tuple[SecurityPanel, np.ndarray, dict]:
    """The panel, the market series and what else the file carries.

    Two kinds of file. `fetch_sp500_panel.py`'s, on tickers: price returns and the adjusted
    bars' volumes, nothing else. `build_sp500_securities.py`'s, on securities: each column a
    share class, labelled by its last ticker; price returns or total returns, the market's to
    match; traded volumes, or the adjusted ones beside them; and its own membership, read
    through the identity table's renames and mapped to securities, and each security's SIC code
    from its own records."""
    data = np.load(path, allow_pickle=False)
    dates = tuple(dt.date.fromisoformat(str(d)) for d in data["dates"])
    on_securities = "membership" in data.files
    extras: dict = {"on_securities": on_securities}
    if on_securities:
        securities = tuple(str(s) for s in data["labels"])
        prices = data["total_return_prices"] if returns == "total" else data["prices"]
        traded = data["adjusted_volumes"] if volumes == "adjusted" else data["volumes"]
        market = data["market_total_returns"] if returns == "total" else data["market_returns"]
        extras.update(
            membership=data["membership"].astype(bool),
            keys=tuple(str(s) for s in data["securities"]),
            sic_codes={label: (str(code) or None)
                       for label, code in zip(securities, data["sic_codes"])},
            snapshot=str(data["snapshot"]),
        )
    else:
        if returns == "total":
            raise SystemExit("  this panel, on tickers, carries price returns alone: run it "
                             "with --returns price; total returns, the default, need "
                             "build_sp500_securities.py's")
        if volumes == "traded":
            raise SystemExit("  this panel, on tickers, stores the adjusted bars' volumes alone: "
                             "--volumes traded needs build_sp500_securities.py's")
        securities = tuple(str(s) for s in data["securities"])
        prices, traded, market = data["prices"], data["volumes"], data["market_returns"]
    panel = SecurityPanel(
        dates=dates,
        securities=securities,
        prices=prices,
        unadjusted_prices=data["unadjusted_prices"],
        volumes=traded,
        shares_outstanding=data["shares_outstanding"],
        highs=data["highs"],
        lows=data["lows"],
        frequency=Frequency.DAILY,
    )
    return panel, market, extras


def kept_out(design, rows: list[int], names: tuple[str, ...]) -> dict:
    """Of the members of each estimated period, those its regression does not hold, and why:
    no weight -- no capitalisation, a count or a price missing, on the day its weights are read;
    else no return on the day; else an exposure missing, by the first factor without one. Read
    from the estimator's own design, so it is the estimator's test that is counted."""
    members = design.members[rows]
    weights = design.weights[rows]
    weighted = np.isfinite(weights) & (weights > 0.0)
    returned = np.isfinite(design.returns[rows])
    exposed = np.isfinite(design.exposures[rows])
    complete = exposed.all(axis=2)
    out = {
        "no weight": members & ~weighted,
        "no return": members & weighted & ~returned,
        "an exposure missing": members & weighted & returned & ~complete,
    }
    first_missing: dict[str, int] = {}
    cells = np.argwhere(out["an exposure missing"])
    for t, j in cells:
        k = int(np.flatnonzero(~exposed[t, j])[0])
        first_missing[names[k]] = first_missing.get(names[k], 0) + 1
    return {"reasons": out, "first_missing": first_missing}


def screen(
    standardised: CharacteristicPanel,
    returns: np.ndarray,
    membership: np.ndarray | None = None,
) -> list[dict]:
    """One univariate cross-sectional regression per descriptor per period.

    Returns are shifted forward by one period against the exposures: a descriptor computed
    from data up to day t can only explain day t+1's return. Regressing it on day t's return
    would use the price that produced it to predict itself.

    Given a membership, each regression holds to the members on the exposure's day, as the
    model's own do: a descriptor earns its place on the securities the model is estimated on,
    and who those are is fixed with the exposures, before the return.
    """
    rows = []
    for k, name in enumerate(standardised.names):
        exposure = standardised.values[:-1, :, k]
        future = returns[1:, :]
        coefficients, r_squared = [], []
        for t in range(exposure.shape[0]):
            x, y = exposure[t], future[t]
            ok = np.isfinite(x) & np.isfinite(y)
            if membership is not None:
                ok &= membership[t]
            if int(ok.sum()) < 30:
                continue
            xs, ys = x[ok], y[ok]
            # With an intercept. Without one the fit cannot absorb the day's market move,
            # so the residuals keep it while the total sum of squares -- taken about the
            # mean -- does not, and R-squared comes out negative for every descriptor. The
            # question here is how much *cross-sectional* dispersion the descriptor
            # explains; the market level is not its job and industries would absorb it in
            # the real regression anyway.
            xc = xs - xs.mean()
            yc = ys - ys.mean()
            variance = float(xc @ xc)
            if variance <= 0.0:
                continue
            slope = float(xc @ yc) / variance
            residual = yc - slope * xc
            total = float(yc @ yc)
            coefficients.append(slope)
            if total > 0.0:
                r_squared.append(1.0 - float(residual @ residual) / total)
        if len(coefficients) < 20:
            rows.append({"name": name, "periods": len(coefficients)})
            continue
        series = np.array(coefficients)
        rows.append(
            {
                "name": name,
                "periods": len(series),
                "mean_r2": float(np.mean(r_squared)) if r_squared else float("nan"),
                "factor_vol": float(series.std(ddof=1) * np.sqrt(252.0)),
                "t_stat": float(
                    series.mean() / (series.std(ddof=1) / np.sqrt(series.size))
                ),
                "coverage": float(
                    np.isfinite(standardised.values[-1, :, k]).mean()
                ),
            }
        )
    return rows


def clipped_outside_on(
    descriptors: CharacteristicPanel,
    caps: np.ndarray,
    membership: np.ndarray,
    t: int,
) -> list[tuple[str, int, int]]:
    """Per descriptor on day t: how many values of securities outside the index hit one of
    the members' bounds, and how many such values there were.

    Read off `standardise` run on that day and descriptor alone, which is exact: no day's or
    descriptor's standardisation looks at any other's.
    """
    out = []
    for k, name in enumerate(descriptors.names):
        day = CharacteristicPanel(
            descriptors.dates[t : t + 1],
            descriptors.securities,
            (name,),
            descriptors.values[t : t + 1, :, k : k + 1],
            descriptors.frequency,
        )
        hit = day.standardise(caps[t : t + 1], membership=membership[t : t + 1])
        have = int(np.count_nonzero(np.isfinite(day.values[0, :, 0]) & ~membership[t]))
        out.append((name, hit.clipped_outside, have))
    return out


def trailing_counts(valid: np.ndarray, window: int) -> np.ndarray:
    """How many of each trailing window's periods are valid; -1 before the first full window."""
    if window > valid.shape[0]:
        return np.full(valid.shape, -1, dtype=np.int64)
    running = np.concatenate(
        [np.zeros((1, valid.shape[1]), dtype=np.int64), np.cumsum(valid, axis=0, dtype=np.int64)]
    )
    out = np.full(valid.shape, -1, dtype=np.int64)
    out[window - 1 :] = running[window:] - running[: valid.shape[0] - window + 1]
    return out


def window_coverage(
    panel: SecurityPanel,
    descriptors: CharacteristicPanel,
    market: np.ndarray,
    beta_window: int,
    membership: np.ndarray | None,
    min_coverage: float,
    rstr_skip: int = 0,
) -> list[tuple[str, int, int, int, int, int]]:
    """Per window descriptor: its values, how many were computed from a window holding no
    valid observation and how many from under `min_coverage` of it -- both of which must be
    none -- and how many from under half their window, all of them and on index members that
    day.

    Worked out from the panel's own prices, volumes and market series, not taken from the
    code that computed the descriptors, so a descriptor still summing an empty window shows.
    Each descriptor's inputs are restated here: a return for DASTD, RSTR and CMRA, a return on
    a day the market has one for the market model, a volume for the turnovers, and for VLVR
    both a dollar volume and the market model's window, `beta_window` periods. RSTR's are the
    periods it sums: its window less the last `rstr_skip`.
    """
    returns = panel.returns()
    has_return = np.isfinite(returns)
    paired = has_return & np.isfinite(market)[:, None]
    traded = np.isfinite(panel.volumes)
    dollars = np.isfinite(panel.volumes * panel.traded_prices)
    year = panel.periods_for(12.0)
    beta = beta_window
    # Each source: what counts as valid, over how many periods, ending how many periods back.
    inputs = {
        "DASTD": [(has_return, panel.periods_for(65.0 / TRADING_DAYS_PER_MONTH), 0)],
        "RSTR": [(has_return, year - rstr_skip, rstr_skip)],
        "CMRA": [(has_return, year, 0)],
        "BTSG": [(paired, beta, 0)],
        "HALPHA": [(paired, beta, 0)],
        "SERDP": [(paired, beta, 0)],
        "STOM": [(traded, panel.periods_for(1.0), 0)],
        "STOQ": [(traded, panel.periods_for(3.0), 0)],
        "STOA": [(traded, year, 0)],
        "VLVR": [(dollars, year, 0), (paired, beta, 0)],
    }
    out = []
    for name, sources in inputs.items():
        if name not in descriptors.names:
            continue
        values = np.isfinite(descriptors.values[:, :, descriptors.names.index(name)])
        empty = np.zeros(values.shape, dtype=bool)
        coverage = np.ones(values.shape)
        for valid, window, back in sources:
            counts = trailing_counts(valid, window)
            if back:
                counts = np.concatenate([np.full((back, counts.shape[1]), -1), counts[:-back]])
            empty |= counts == 0
            coverage = np.minimum(coverage, np.where(counts > 0, counts / window, 1.0))
        below = int((values & ~empty & (coverage < min_coverage)).sum())
        thin = values & ~empty & (coverage < 0.5)
        on_members = int((thin & membership).sum()) if membership is not None else 0
        out.append(
            (name, int(values.sum()), int((values & empty).sum()), below, int(thin.sum()),
             on_members)
        )
    return out


def estimation_moments(
    values: np.ndarray, caps: np.ndarray, membership: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per date and column, over the estimation universe -- the members with a value, or
    everyone with one: the cap-weighted mean, the equal-weighted standard deviation, the
    first-order rounding bound on that mean (n * eps * the cap-weighted mean absolute value),
    and n. A mean inside its bound is zero to machine precision.

    Worked out here from the exposures, not taken from the code that standardised them, so a
    step after it that moved them -- or a different set standardising them -- shows.
    """
    n_dates, _, n_columns = values.shape
    means = np.full((n_dates, n_columns), np.nan)
    spreads = np.full((n_dates, n_columns), np.nan)
    bounds = np.full((n_dates, n_columns), np.nan)
    sizes = np.zeros((n_dates, n_columns), dtype=int)
    eps = np.finfo(float).eps
    for t in range(n_dates):
        for k in range(n_columns):
            x = values[t, :, k]
            inside = np.isfinite(x)
            if membership is not None:
                inside &= membership[t]
            n = int(np.count_nonzero(inside))
            if n < 2:
                continue
            w = np.nan_to_num(caps[t, inside], nan=0.0)
            if w.sum() <= 0.0:
                w = np.ones(n)
            means[t, k] = np.average(x[inside], weights=w)
            spreads[t, k] = np.std(x[inside], ddof=1)
            bounds[t, k] = n * eps * np.average(np.abs(x[inside]), weights=w)
            sizes[t, k] = n
    return means, spreads, bounds, sizes


def size_pair_moments(
    size: np.ndarray,
    nonlinearity: np.ndarray,
    weights: np.ndarray | None,
    caps: np.ndarray,
    membership: np.ndarray | None,
    bound: float = 4.0,
) -> dict[str, np.ndarray]:
    """Per date, over the estimation universe -- the members with both exposures and, if the
    regression is weighted, a positive weight: the regression-weighted covariance of the Size
    and Size Nonlinearity exposures; its first-order rounding bound, n * eps * the weighted
    mean of |S - S_w| |N - N_w|; their equal-weighted correlation; and how many members' Size
    exposures, cubed and standardised over the members with one, lie past `bound` -- the
    values USE4's clip moves. A covariance inside its bound is zero to machine precision.

    Worked out here from the exposures, not taken from the code that built them.
    """
    n_dates, n_securities = size.shape
    eps = np.finfo(float).eps
    out = {key: np.full(n_dates, np.nan) for key in ("covariance", "bound", "correlation")}
    out["clipped"] = np.zeros(n_dates, dtype=int)
    for t in range(n_dates):
        sized = np.isfinite(size[t])
        if membership is not None:
            sized &= membership[t]
        if int(sized.sum()) > 1:
            cube = size[t, sized] ** 3
            w = np.nan_to_num(caps[t, sized], nan=0.0)
            if w.sum() <= 0.0:
                w = np.ones(cube.size)
            z = (cube - np.average(cube, weights=w)) / np.std(cube, ddof=1)
            out["clipped"][t] = int(np.count_nonzero(np.abs(z) > bound))
        row = np.ones(n_securities) if weights is None else weights[t]
        inside = sized & np.isfinite(nonlinearity[t]) & np.isfinite(row) & (row > 0.0)
        if int(inside.sum()) < 3:
            continue
        s, x, w = size[t, inside], nonlinearity[t, inside], row[inside]
        product = (s - np.average(s, weights=w)) * (x - np.average(x, weights=w))
        out["covariance"][t] = float(w @ product) / float(w.sum())
        out["bound"][t] = int(inside.sum()) * eps * float(w @ np.abs(product)) / float(w.sum())
        out["correlation"][t] = float(np.corrcoef(s, x)[0, 1])
    return out


def size_pair_portfolios(
    model, securities: tuple[str, ...], held: np.ndarray, caps: np.ndarray, names: list[str]
) -> list[tuple[str, np.ndarray, float]]:
    """Where the size pair's risk sits, on the forecast date: the smallest 85 of `held` by
    capitalisation and the largest 50, each equally weighted, and all of `held` cap-weighted,
    the index. Per portfolio, its mean forecast exposure to each factor in `names` -- Size and
    Size Nonlinearity, or Size alone -- and their risk together, sqrt(x' S x) over that block
    of the model's annualised covariance.

    `held` is a boolean per security, `caps` the forecast date's capitalisations.
    """
    codes = [code for code, keep in zip(securities, held) if keep]
    cap_of = dict(zip(securities, caps))
    by_cap = sorted(codes, key=cap_of.__getitem__)
    ids = [next(fid for fid in model.factor_ids if model.factor_names[fid] == name)
           for name in names]
    rows = [list(model.factor_ids).index(fid) for fid in ids]
    block = np.asarray(model.covariance.matrix)[np.ix_(rows, rows)]
    out = []
    for label, chosen, weights in (
        ("smallest 85, equally weighted", by_cap[:85], None),
        ("largest 50, equally weighted", by_cap[-50:], None),
        ("the index, cap-weighted", codes, np.array([cap_of[code] for code in codes])),
    ):
        x = np.array([[model.exposures[code].exposures.get(fid, 0.0) for fid in ids]
                      for code in chosen])
        mean = np.average(x, axis=0, weights=weights)
        out.append((label, mean, float(np.sqrt(mean @ block @ mean))))
    return out


def index_membership(
    snapshot: sp500.Snapshot, dates: tuple[dt.date, ...], securities: tuple[str, ...]
) -> tuple[np.ndarray, list[int]]:
    """Dates by securities, True where the security was an index member that day; and the
    index's own size each day.

    Every date is asked, and CoverageError is not caught here. A regression restricted to
    members on most days and unrestricted on the few the snapshot cannot answer is a different
    model, and nothing downstream would show the difference.
    """
    mask = np.zeros((len(dates), len(securities)), dtype=bool)
    sizes: list[int] = []
    for i, day in enumerate(dates):
        members = set(sp500.membership(snapshot, day))
        mask[i] = [code in members for code in securities]
        sizes.append(len(members))
    return mask, sizes


def unreproduced_changes(
    snapshot: sp500.Snapshot, start: dt.date, end: dt.date
) -> list[tuple[dt.date, str, str]]:
    """The changes inside [start, end] that membership() does not reproduce on their own date.

    There the source table disagrees with itself -- an addition the list carried back never
    had, or a removal of a name it still holds -- and membership() answers from the list. They
    are reported with what they do to the cross-sections and not repaired: a quiet repair would
    be a second membership source that nobody reviewed.
    """
    out: list[tuple[dt.date, str, str]] = []
    for change in sorted(snapshot.changes, key=lambda c: c.date):
        if not start <= change.date <= end:
            continue
        members = set(sp500.membership(snapshot, change.date))
        out.extend((change.date, t, "added") for t in change.added if t not in members)
        out.extend((change.date, t, "removed") for t in change.removed if t in members)
    return out


@dataclasses.dataclass(frozen=True)
class ModelInputs:
    """What the run builds from a panel for the estimator, and what its report reads on the
    way: the descriptors and their standardisation, the risk indices, the periods one blend per
    factor allows, and the industries, built and then folded."""

    descriptors: CharacteristicPanel
    skipped: dict
    caps: np.ndarray
    standardised: CharacteristicPanel
    weights: np.ndarray | None
    return_panel: ReturnPanel
    unblended: dict
    blendable: CharacteristicPanel
    styles: CharacteristicPanel
    unlike: dict
    periods: np.ndarray | None
    universe: tuple[str, ...] | None
    built_scheme: object | None
    scheme: object | None
    indices: CharacteristicPanel


def parse_merges(specs: list[str]) -> dict[str, str]:
    """--merge-industry's A=B pairs."""
    merges: dict[str, str] = {}
    for spec in specs:
        source, sep, target = spec.partition("=")
        if not sep or not source.strip() or not target.strip():
            raise SystemExit(f"  --merge-industry takes A=B, got {spec!r}")
        merges[source.strip()] = target.strip()
    return merges


def industry_codes(
    panel: SecurityPanel, extras: dict, args
) -> tuple[dict[str, str | None], list[str]]:
    """Each security's SIC code, and the tickers whose code was carried from an earlier date.

    On securities, each security's own latest record with a code, under its own tickers inside
    its spans: a ticker's record can be another security's. On tickers, the reference cache's,
    which needs `.env`'s key."""
    if extras["on_securities"]:
        return dict(extras["sic_codes"]), []
    try:
        client = PolygonClient(api_key=read_api_key(args.env), cache_dir=Path(args.cache))
    except PolygonError as error:
        raise SystemExit(f"  industries need the reference cache: {error}") from None
    codes: dict[str, str | None] = {}
    carried: list[str] = []
    for ticker in panel.securities:
        code, on = client.sic_code_any(ticker, panel.dates[-1])
        codes[ticker] = code
        if code and on != panel.dates[-1]:
            carried.append(ticker)
    return codes, carried


def model_inputs(
    panel: SecurityPanel,
    market: np.ndarray,
    membership: np.ndarray | None,
    args,
    *,
    codes: dict[str, str | None] | None = None,
    merges: dict[str, str] | None = None,
    scheme=None,
) -> ModelInputs:
    """Everything from the panel to the estimator's inputs, as the run computes it.

    With `codes`, the industries: a scheme built over the regression universe and folded by
    `merges` -- or, given, `scheme` held in its place and folded the same way. One definition
    for the run and for its bias statistics' guard, which rebuilds these from the panel cut at a
    date with the full run's scheme held, and checks they are the full run's through that date.
    """
    # None leaves the window to the library.
    beta_months = (
        args.beta_window if args.beta_days is None
        else args.beta_days / TRADING_DAYS_PER_MONTH
    )
    descriptors, skipped = compute_descriptors(
        panel,
        market_returns=market,
        beta_window_months=beta_months,
        membership=membership,
        min_coverage=args.min_coverage,
        rstr_skip=args.rstr_skip,
        beta_half_life=args.beta_half_life,
    )
    caps = panel.market_capitalisation()
    # Over the estimation universe: the members set each day's clip bounds, cap-weighted mean
    # and equal-weighted dispersion, and every other security with a value is put on their
    # scale. Barra's estimation universe against its coverage universe -- an ex-member still
    # gets an exposure, but no say in what an exposure of one means.
    standardised = descriptors.standardise(caps, membership=membership)
    # "Regression Weighting Scheme: Square root of capitalisation" appears on every model
    # sheet in the Equity Risk Model Reference Guide. Equal weighting lets the smallest,
    # noisiest names set the factor returns -- there are far more of them, and in a
    # cap-weighted world almost none of the money. Needed before the risk indices, because
    # Size Nonlinearity is orthogonalised against Size by them, and before the industries,
    # because which securities a regression sees depends on them.
    weights = None if args.equal_weight else np.sqrt(np.where(caps > 0.0, caps, np.nan))
    return_panel = ReturnPanel(panel.dates, panel.securities, panel.returns(), Frequency.DAILY)

    unblended = dict(UNBLENDED)
    if args.size_nonlinearity == "use4":
        unblended["LCAPCB"] = ("USE3's raw cube; Size Nonlinearity is built from the Size "
                               "exposure instead, USE4's way")
    if args.size_nonlinearity == "none":
        unblended["LCAPCB"] = ("USE3's raw cube; Size Nonlinearity is not in the model "
                               "(--size-nonlinearity none)")
    blendable = without(standardised, unblended)
    available = [
        name
        for name in ("Volatility", "Momentum", "Size", "Size Nonlinearity",
                     "Trading Activity")
        if any(m in blendable.names for m in index_descriptors(name, args.size_nonlinearity))
    ]
    styles = build_risk_indices(
        blendable, cap_weights=caps, indices=available, membership=membership,
        size_nonlinearity=args.size_nonlinearity, regression_weights=weights,
    )

    # One definition per factor across the sample. blend_to_index averages whichever of an
    # index's descriptors a security has, so a blend changes composition where one starts or
    # stops having values, and a factor estimated on one blend and forecast on another is two
    # factors under one name. So a period is estimated only if every index's blend in it, over
    # the securities its regression holds, has exactly the descriptors the forecast's has.
    # Worked out before the industries, whose universe is the estimated periods'.
    design = regression_design(
        return_panel, styles, regression_weights=weights, membership=membership
    )
    forecast_row = len(panel.dates) - 1
    complete = np.isfinite(styles.values[forecast_row]).all(axis=1)[None, :]
    period_rows = np.arange(design.lag, len(panel.dates))
    unlike: dict[int, list[str]] = {}
    for name in styles.names:
        read = blend_composition(
            blendable, name, period_rows - design.lag, design.in_regression[period_rows],
            size_nonlinearity=args.size_nonlinearity,
        )
        wanted = blend_composition(
            blendable, name, [forecast_row], complete, size_nonlinearity=args.size_nonlinearity
        )
        for k, t in enumerate(period_rows):
            have = [d for d, present in read.items() if present[k]]
            missing = [d for d, present in wanted.items() if present[0] and not read[d][k]]
            extra = [d for d in have if not wanted[d][0]]
            if missing or extra:
                unlike.setdefault(int(t), []).append(
                    f"{name} from {', '.join(have) or 'nothing'}"
                    + (f", without {', '.join(missing)}" if missing else "")
                    + (f", with {', '.join(extra)} the forecast's lacks" if extra else "")
                )
    one_blend = np.array([t >= design.lag and t not in unlike for t in range(len(panel.dates))])
    periods = None if args.mixed_blends else one_blend

    universe = built = folded = None
    indices = styles
    if codes is not None:
        # Counted over the securities in at least one estimated period's regression, not
        # over the panel. A member that never enters a regression contributes nothing to its
        # factor, and counting them let a four-member Unclassified -- one member present in
        # most periods -- past a floor of five.
        universe = regression_universe(
            return_panel, styles, regression_weights=weights, membership=membership,
            periods=periods,
        )
        built = scheme if scheme is not None else build_scheme(
            codes, min_members=args.min_industry, universe=universe
        )
        folded = built
        if merges:
            try:
                folded = merge_industries(built, merges)
            except ValueError as error:
                raise SystemExit(f"  --merge-industry: {error}") from None
        indices = combine(
            styles,
            industry_exposures(
                folded, panel.dates, panel.securities, frequency=Frequency.DAILY
            ),
        )
    return ModelInputs(
        descriptors=descriptors,
        skipped=skipped,
        caps=caps,
        standardised=standardised,
        weights=weights,
        return_panel=return_panel,
        unblended=unblended,
        blendable=blendable,
        styles=styles,
        unlike=unlike,
        periods=periods,
        universe=universe,
        built_scheme=built,
        scheme=folded,
        indices=indices,
    )


BIAS_YEAR = 252
"""The estimated periods the covariance has before the first forecast: a year of factor
returns."""

BIAS_MIN_FORECASTS = 63
"""A member singly is tested only on a quarter's forecasts or more."""


def cut_panel(panel: SecurityPanel, rows: int) -> SecurityPanel:
    """The panel through its first `rows` dates, every dated array cut there."""
    cut = {
        name: (None if getattr(panel, name) is None else getattr(panel, name)[:rows])
        for name in ("prices", "unadjusted_prices", "volumes", "shares_outstanding", "highs",
                     "lows")
    }
    return dataclasses.replace(panel, dates=panel.dates[:rows], **cut)


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def _models_differ(a, b) -> list[str]:
    """What two models' forecasts differ in, to the bit."""
    out = []
    if not _same(a.covariance.matrix, b.covariance.matrix):
        out.append("covariance")
    if a.exposures.keys() != b.exposures.keys() or any(
        a.exposures[c].exposures != b.exposures[c].exposures for c in a.exposures
    ):
        out.append("exposures")
    elif any(a.exposures[c].specific.by_issuer != b.exposures[c].specific.by_issuer
             for c in a.exposures):
        out.append("specific risk")
    if not _same(a.notes.factor_returns.returns, b.notes.factor_returns.returns):
        out.append("factor returns")
    return out


def bias_section(
    args, panel, market, membership, codes, merges, inputs, model, options, horizon
) -> None:
    """The out-of-sample bias statistics (midoffice11): the model refitted as of each trading
    day, each forecast against the next day's return, over the standard portfolios; and at the
    model's horizon, every forecast against the next `horizon` days' return (midoffice12)."""
    notes = model.notes
    dates = panel.dates
    at = {d: i for i, d in enumerate(dates)}
    estimated_rows = [at[d] for d in notes.estimated_dates]
    if len(estimated_rows) <= BIAS_YEAR:
        raise SystemExit(f"  --bias needs more than a year ({BIAS_YEAR}) of estimated periods; "
                         f"this run has {len(estimated_rows)}")
    rows = list(range(estimated_rows[BIAS_YEAR - 1], len(dates) - 1))
    held = inputs.built_scheme
    print("\nOut-of-sample bias statistics")
    print(f"  {len(rows)} forecasts, as of each trading day from {dates[rows[0]]} -- the first "
          f"whose covariance has a year ({BIAS_YEAR}) of estimated periods -- to "
          f"{dates[rows[-1]]}: each the whole estimator refitted on data through its date "
          f"alone, daily -- the factor returns, the covariance, the specific risk model's fit "
          f"and every history it blends, the exposures as of the date")
    if held is not None:
        folds = ", ".join(f"{a} into {b}" for a, b in sorted(inputs.scheme.folded.items()))
        print(f"  the industry scheme is held fixed over the sample, as Barra fixes a model's "
              f"industry factors: built once over the full sample's regression universe, "
              f"{len(held.industries)} industries"
              + (f", folded as given ({folds})" if merges else ""))
    print(f"  each forecast taken to a daily scale (annual / {notes.frequency.periods_per_year:g})"
          f" against the next trading day's {args.returns} return; the bias statistic is the "
          f"standard deviation of realised / forecast, one if the forecasts are right, its 95% "
          f"band about 1 +/- sqrt(2/T); over the whole test, and over every 12-month "
          f"({BIAS_YEAR}-forecast) window with the windows' mean absolute deviation from one "
          f"(MRAD)")
    # At the model's horizon (midoffice12): every forecast against the h days after it, the
    # windows overlapping, and every h-th apart; and at any other horizon asked for.
    evaluated = sorted({horizon, *args.bias_horizons} - {1})
    others = [h for h in evaluated if h != horizon]
    if horizon == 1:
        print("  the model's horizon is one trading day, so the daily statistic is the "
              "statistic at its horizon")
    else:
        print(f"  and at the model's horizon, {horizon} trading days: every forecast, at {horizon} "
              f"times its daily variance, against the {horizon} trading days' {args.returns} "
              f"return after it, bought and held, a name that stops trading held as cash -- the "
              f"windows overlapping, so their 95% band is found by simulation, right forecasts of "
              f"normal returns ({OVERLAP_DRAWS:,} draws) -- and beside it every "
              f"{ordinal(horizon)} forecast from the first, the windows apart, its band "
              f"1 +/- sqrt(2/T)")
    if others:
        print(f"  also evaluated at {', '.join(str(h) for h in others)} trading days "
              f"(--bias-horizons), the same way, summarised after the eigenportfolios")
    # The eigenfactor risk adjustment at other scales (midoffice13): the refits simulate the
    # bias -- at a = 0 if the model scales nothing itself -- and one simulation a date serves
    # every scale, each tested beside the model.
    scales = [a for a in args.bias_eigen_scales if a != (options.get("eigen_adjustment") or 0.0)]
    if scales and options.get("eigen_adjustment") is None:
        options = dict(options, eigen_adjustment=0.0, eigen_draws=args.eigen_draws)
    if scales:
        print(f"  the eigenfactor risk adjustment tested beside the model at a = "
              f"{', '.join(f'{a:g}' for a in scales)}: on each forecast date its bias simulated "
              f"once, {options['eigen_draws']} histories from the day's factor covariance "
              f"re-estimated by its own estimator, every scale read from it -- the model's own "
              f"a = {options['eigen_adjustment']:g}")

    # The guard: the forecast as of a date, rebuilt from the panel cut at that date through
    # the run's own pipeline, the scheme held, is the full run's -- its inputs bit for bit
    # through the date, and the model from them.
    if args.bias_guard > 0:
        picks = sorted(set(np.linspace(0, len(rows) - 1, args.bias_guard).round().astype(int)))
        failures: list[str] = []
        checked: list[str] = []
        for i in picks:
            t = rows[i]
            cut = cut_panel(panel, t + 1)
            cut_members = None if membership is None else membership[: t + 1]
            rebuilt = model_inputs(cut, market[: t + 1], cut_members, args, codes=codes,
                                   merges=merges, scheme=held)
            pairs = {
                "returns": (rebuilt.return_panel.returns, inputs.return_panel.returns[: t + 1]),
                "exposures": (rebuilt.indices.values, inputs.indices.values[: t + 1]),
                "weights": (rebuilt.weights,
                            None if inputs.weights is None else inputs.weights[: t + 1]),
                "capitalisations": (rebuilt.caps, inputs.caps[: t + 1]),
                "periods": (rebuilt.periods,
                            None if inputs.periods is None else inputs.periods[: t + 1]),
            }
            differ = [name for name, (a, b) in pairs.items() if not _same(a, b)]
            if rebuilt.indices.names != inputs.indices.names:
                differ.append("factor names")
            (_, as_of), = cross_sectional_models_as_of(
                inputs.return_panel, inputs.indices, [dates[t]], **options
            )
            direct = estimate_cross_sectional_model(
                rebuilt.return_panel, rebuilt.indices,
                **dict(options, capitalisations=rebuilt.caps, regression_weights=rebuilt.weights,
                       membership=cut_members, periods=rebuilt.periods),
            )
            differ += [f"the model's {name}" for name in _models_differ(as_of, direct)]
            checked.append(str(dates[t]))
            if differ:
                failures.append(f"{dates[t]}: {', '.join(differ)}")
        if failures:
            raise SystemExit("  the run stops: the forecast as of a date, rebuilt from the panel "
                             "cut at it with the scheme held, is not the full run's -- "
                             + "; ".join(failures))
        print(f"  guard: as of {', '.join(checked)}, the model's inputs rebuilt from the panel "
              f"cut at the date, the scheme held, equal the full run's through it to the bit -- "
              f"returns, exposures, weights, capitalisations, periods -- and so do the forecasts "
              f"made from them: covariance, exposures, specific risk, factor returns")

    securities = panel.securities
    groups = {} if held is None else {
        name: np.array([held.assignment.get(code) == name for code in securities])
        for name in held.industries
    }
    factor_rows = np.full((len(dates), len(model.factor_ids)), np.nan)
    factor_rows[estimated_rows] = notes.factor_returns.returns
    began = time.monotonic()

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            elapsed = time.monotonic() - began
            print(f"  [bias] {done} of {total} forecasts, {elapsed / 60:.1f} min, about "
                  f"{(total - done) * elapsed / done / 60:.1f} min left", file=sys.stderr,
                  flush=True)

    models = cross_sectional_models_as_of(
        inputs.return_panel, inputs.indices, [dates[t] for t in rows], **options
    )
    setup = dict(
        dates=dates, securities=securities, returns=inputs.return_panel.returns,
        membership=(np.ones((len(dates), len(securities)), dtype=bool) if membership is None
                    else membership),
        capitalisations=inputs.caps, groups=groups, factor_returns=factor_rows,
        horizons=evaluated,
    )
    test = BiasTest(**setup)
    variants = {a: BiasTest(**setup) for a in scales}
    simulated: list[tuple[np.ndarray, np.ndarray, float]] = []
    for count, (t, (_, fitted)) in enumerate(zip(rows, models)):
        test.add(t, RiskForecast.from_model(fitted, securities))
        for a, variant in variants.items():
            variant.add(t, RiskForecast.from_model(fitted, securities, eigen_scale=a))
        if fitted.notes.eigen is not None:
            eigen = fitted.notes.eigen
            simulated.append((eigen.bias, eigen.standard_error, eigen.seconds))
        progress(count + 1, len(rows))
    run = test.result()
    eigen_runs = {a: variant.result() for a, variant in variants.items()}
    minutes = (time.monotonic() - began) / 60
    run_caps = inputs.caps[[at[d] for d in run.dates]]
    print(f"  {len(run.dates)} refits in {minutes:.1f} min"
          + (f", {60 * minutes / len(run.dates):.2f} s a refit with its eigenfactor simulation "
             f"({np.mean([s for _, _, s in simulated]):.2f} s of it) and {len(scales) + 1} "
             f"tests" if simulated else "")
          + f"; the factor covariance as the engine uses it, repaired to positive semi-definite "
          f"at the configured floor, on {int(run.repaired.sum())} of the {len(run.dates)} dates; "
          f"{run.dropped} member-day(s) with a forecast and no next-day return left out of every "
          f"portfolio formed that day")

    def summary(stats) -> str:
        rolled = stats.rolling
        return (f"{stats.n:>5} {stats.statistic:>7.3f} {stats.verdict:>5}  "
                f"{stats.low:.3f}-{stats.high:.3f}  {stats.mrad:>6.3f} "
                f"{(rolled.min() if rolled.size else float('nan')):>6.3f} "
                f"{(rolled.max() if rolled.size else float('nan')):>6.3f}")

    def apart(stats) -> str:
        return (f"{stats.n:>4} {stats.statistic:>6.3f} {stats.verdict:>5}  "
                f"{stats.low:.3f}-{stats.high:.3f}")

    def vols(name, h=1) -> str:
        forecast, realised = _volatility(run, name, horizon=h)
        return f"   {forecast:>8.2%} {realised:>8.2%}"

    broad = run.names[:4] + ("minimum variance, long-only", "minimum variance, unconstrained")
    header = (f"  {'':<34} {'T':>5} {'bias':>7} {'':>5}  {'95% band':>11}  {'MRAD':>6} "
              f"{'min':>6} {'max':>6}")
    vol_header = f"   {'forecast':>8} {'realised':>8}"
    # The realised volatility beside the statistic (midoffice13's ruling 5): the optimiser's
    # portfolios realising more risk is the test the eigenfactor adjustment failed, kept for
    # every covariance candidate after it.
    print("\n  Portfolios at one day, each formed at its forecast date and held to the next; "
          "beside each, its volatility % a year -- the forecast at its median, the realised "
          "from its returns")
    print(header + vol_header)
    for name in broad:
        print(f"  {name:<34} " + summary(run.statistics(name)) + vols(name))
    if horizon > 1:
        print(f"\n  Portfolios at the model's horizon, {horizon} trading days, each bought at its "
              f"forecast date and held {horizon} days: every forecast, the windows overlapping, "
              f"its band simulated; and every {ordinal(horizon)}, the windows apart; the "
              f"volatility % a year from the {horizon}-day returns, the windows overlapping")
        print(header + f"   {'apart: T':>8} {'bias':>6} {'':>5}  {'95% band':>11}" + vol_header)
        for name in broad:
            print(f"  {name:<34} " + summary(run.statistics(name, horizon=horizon)) + "   "
                  + apart(run.statistics(name, horizon=horizon, overlapping=False))
                  + vols(name, horizon))
    halves = _halves(run)
    for label in ("long-only", "unconstrained"):
        name = f"minimum variance, {label}"
        row = run.names.index(name)
        size = run.min_variance_held[label]
        worst = float(run.min_variance_violation[label].max())
        parts = [(part, _volatility(run, name, part)[1],
                  bias_statistics(run.forecast[row, part], run.realised[row, part]).statistic)
                 for part in halves.values()]
        print(f"    {name}: {int(np.median(size))} names held at the median ({size.min()} to "
              f"{size.max()}); its KKT conditions met to {worst:.1e}; realised "
              f"{_volatility(run, name)[1]:.2%} a year, the first half {parts[0][1]:.2%} (bias "
              f"{parts[0][2]:.3f}), the second {parts[1][1]:.2%} ({parts[1][2]:.3f})")

    if groups:
        names = [model.factor_names[fid] for fid in model.factor_ids]
        counts = held.counts()
        print(f"\n  Each industry's members, cap-weighted -- the scheme's {len(groups)} as built "
              f"-- against its factor's noise share in the full run (equal / weighted)")
        columns = f"{'1 day':>7} {'':>5}  {'MRAD':>6}"
        if horizon > 1:
            columns += f"   {f'{horizon} days':>7} {'':>5} {'apart':>6}"
        print(f"  {'industry':<24} {'members':>7} {'T':>5} {columns}   {'noise share':>14}")
        biases, shares = [], []
        for group in groups:
            stats = run.statistics(f"industry: {group}")
            cells = f"{stats.statistic:>7.3f} {stats.verdict:>5}  {stats.mrad:>6.3f}"
            if horizon > 1:
                at = run.statistics(f"industry: {group}", horizon=horizon)
                away = run.statistics(f"industry: {group}", horizon=horizon, overlapping=False)
                cells += f"   {at.statistic:>7.3f} {at.verdict:>5} {away.statistic:>6.3f}"
            if group in names:
                k = names.index(group)
                share = (f"{notes.noise_share[k]:>6.1%} / {notes.weighted_noise_share[k]:.1%}"
                         if notes.weighted_noise_share else f"{notes.noise_share[k]:>6.1%}")
                if notes.weighted_noise_share:
                    biases.append(stats.statistic)
                    shares.append(notes.weighted_noise_share[k])
            else:
                share = f"in {inputs.scheme.folded.get(group, '?')}"
            print(f"  {group:<24} {counts.get(group, 0):>7} {stats.n:>5} {cells}   {share:>14}")
        if len(biases) > 2:
            print(f"  across the {len(biases)} with a factor of their own, the one-day bias "
                  f"statistic's correlation with the weighted noise share: "
                  f"{float(np.corrcoef(biases, shares)[0, 1]):+.3f}")

    for family in ("long-only", "long-short"):
        family_names = [n for n in run.names if n.startswith(f"random {family} ")]
        stats = [run.statistics(n) for n in family_names]
        values = np.array([s.statistic for s in stats])
        below = np.mean([s.verdict == "over" for s in stats])
        above = np.mean([s.verdict == "under" for s in stats])
        print(f"\n  Random portfolios, {family}, seed {run.random_seed}: {len(stats)} of 30 members"
              + (" a side" if family == "long-short" else "")
              + f", weights drawn once per member -- at one day, bias median "
              f"{np.median(values):.3f}, p5 {np.percentile(values, 5):.3f}, p95 "
              f"{np.percentile(values, 95):.3f}; below the band {below:.0%}, above it "
              f"{above:.0%}; MRAD median {np.median([s.mrad for s in stats]):.3f}")
        if horizon > 1:
            at = [run.statistics(n, horizon=horizon) for n in family_names]
            values = np.array([s.statistic for s in at])
            away = [run.statistics(n, horizon=horizon, overlapping=False).statistic
                    for n in family_names]
            print(f"    at {horizon} days, the windows overlapping: median "
                  f"{np.median(values):.3f}, p5 {np.percentile(values, 5):.3f}, p95 "
                  f"{np.percentile(values, 95):.3f}; "
                  f"below the simulated band {np.mean([s.verdict == 'over' for s in at]):.0%}, "
                  f"above it {np.mean([s.verdict == 'under' for s in at]):.0%}; apart, median "
                  f"{np.median(away):.3f}")

    member_stats = []
    for j, code in enumerate(securities):
        stats = bias_statistics(run.member_forecast[:, j], run.member_realised[:, j])
        if stats.n >= BIAS_MIN_FORECASTS:
            member_stats.append((code, stats))
    if not member_stats:
        print(f"\n  Every member singly: none has {BIAS_MIN_FORECASTS} forecasts")
        member_stats = None
    if member_stats is not None:
        _report_members(run, member_stats, securities, run_caps, horizon)
    _report_eigenportfolios(run, horizon)
    if run.horizons and set(run.horizons) != {horizon}:
        _report_horizons(run, groups, member_stats or [], securities)
    if eigen_runs:
        _report_eigen(run, eigen_runs, simulated, options["eigen_adjustment"], groups,
                      member_stats or [], securities)
    if args.bias_out:
        _save_bias(args.bias_out, run, run_caps, horizon, eigen_runs, simulated)


def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th, 11th, 21st."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _member_columns(member_stats, securities) -> list[int]:
    position = {code: j for j, code in enumerate(securities)}
    return [position[code] for code, _ in member_stats]


def _report_members(run, member_stats, securities, caps, horizon) -> None:
    """Every member singly: the distribution of their statistics, the shares outside the band,
    by size, and at the model's horizon."""
    values = np.array([s.statistic for _, s in member_stats])
    below = np.mean([s.verdict == "over" for _, s in member_stats])
    above = np.mean([s.verdict == "under" for _, s in member_stats])
    fat_below = np.mean([s.statistic < s.fat_band[0] for _, s in member_stats])
    fat_above = np.mean([s.statistic > s.fat_band[1] for _, s in member_stats])
    print(f"\n  Every member singly: {len(member_stats)} securities with {BIAS_MIN_FORECASTS} "
          f"forecasts or more -- at one day, bias p5 {np.percentile(values, 5):.3f}, p25 "
          f"{np.percentile(values, 25):.3f}, median {np.median(values):.3f}, p75 "
          f"{np.percentile(values, 75):.3f}, p95 {np.percentile(values, 95):.3f}")
    kurtosis = float(np.median([s.kurtosis for _, s in member_stats]))
    print(f"    below its band (over-forecast) {below:.1%}, above it (under-forecast) {above:.1%}, "
          f"against 2.5% each for right forecasts of normal returns; with each member's band "
          f"widened for its own kurtosis (median {kurtosis:.1f}), {fat_below:.1%} and "
          f"{fat_above:.1%}")
    held = np.isfinite(run.member_forecast)
    # Over the days each was held; a security never held has no mean, and is not ranked.
    mean_caps = np.nansum(np.where(held, caps, 0.0), axis=0) / np.maximum(held.sum(axis=0), 1)
    position = {code: j for j, code in enumerate(securities)}
    ranked = sorted(member_stats, key=lambda item: -mean_caps[position[item[0]]])
    quintiles = np.array_split(np.arange(len(ranked)), 5)
    print("    median by capitalisation quintile over the test, largest first: "
          + ", ".join(f"{np.median([ranked[i][1].statistic for i in q]):.3f}" for q in quintiles
                      if q.size))
    if horizon > 1:
        forecast, realised = run.member_outcomes(horizon)
        at = [bias_statistics(forecast[:, j], realised[:, j], overlap=horizon)
              for j in _member_columns(member_stats, securities)]
        values = np.array([s.statistic for s in at])
        print(f"    at {horizon} days, the windows overlapping: median {np.median(values):.3f}, p5 "
              f"{np.percentile(values, 5):.3f}, p95 {np.percentile(values, 95):.3f}; below its "
              f"simulated band {np.mean([s.verdict == 'over' for s in at]):.1%}, above it "
              f"{np.mean([s.verdict == 'under' for s in at]):.1%} (the band for normal returns)")
        forecast, realised = run.member_outcomes(horizon, overlapping=False)
        away = [bias_statistics(forecast[:, j], realised[:, j]) for j in range(len(securities))]
        away = np.array([s.statistic for s in away if s.n >= 12])
        if away.size:
            print(f"    apart, every {ordinal(horizon)} forecast, members with 12 outcomes or "
                  f"more: median {np.median(away):.3f}")


def _report_eigenportfolios(run, horizon) -> None:
    print("\n  The factor covariance's eigenportfolios against the next period's estimated factor "
          "returns, by rank, 1 the largest eigenvalue"
          + (f"; at {horizon} days against the sum of the {horizon} periods' after the forecast"
             if horizon > 1 else ""))
    columns = f"{'1 day':>7} {'':>5}  {'MRAD':>6}"
    if horizon > 1:
        columns += f"   {f'{horizon} days':>7} {'':>5} {'apart':>6}"
    print(f"  {'rank':>4} {'vol, median':>12} {'T':>5} {columns}")
    for k in range(run.eigen_forecast.shape[0]):
        stats = bias_statistics(run.eigen_forecast[k], run.eigen_realised[k])
        vol = float(np.median(run.eigen_forecast[k]) * np.sqrt(BIAS_YEAR))
        cells = f"{stats.statistic:>7.3f} {stats.verdict:>5}  {stats.mrad:>6.3f}"
        if horizon > 1:
            forecast, realised = run.eigen_outcomes(horizon)
            at = bias_statistics(forecast[k], realised[k], overlap=horizon)
            forecast, realised = run.eigen_outcomes(horizon, overlapping=False)
            away = bias_statistics(forecast[k], realised[k])
            cells += f"   {at.statistic:>7.3f} {at.verdict:>5} {away.statistic:>6.3f}"
        print(f"  {k + 1:>4} {vol:>12.2%} {stats.n:>5} {cells}")


def _report_horizons(run, groups, member_stats, securities) -> None:
    """Every horizon evaluated, side by side: the statistic with the windows overlapping, and
    after the slash with them apart -- each forecast at h times its daily variance."""
    horizons = (1,) + run.horizons
    columns = _member_columns(member_stats, securities)

    def portfolio(name):
        return lambda h, overlapping: run.statistics(
            name, horizon=h, overlapping=overlapping).statistic

    def random(family):
        names = [n for n in run.names if n.startswith(f"random {family} ")]
        return lambda h, overlapping: float(np.median(
            [run.statistics(n, horizon=h, overlapping=overlapping).statistic for n in names]))

    def members(h, overlapping):
        forecast, realised = run.member_outcomes(h, overlapping=overlapping)
        stats = [bias_statistics(forecast[:, j], realised[:, j]) for j in columns]
        return float(np.median([s.statistic for s in stats
                                if s.n >= (2 if overlapping else 12)]))

    def industries(h, overlapping):
        return float(np.mean([abs(run.statistics(f"industry: {g}", horizon=h,
                                                 overlapping=overlapping).statistic - 1.0)
                              for g in groups]))

    def eigen(ranks):
        def read(h, overlapping):
            forecast, realised = run.eigen_outcomes(h, overlapping=overlapping)
            return float(np.mean([bias_statistics(forecast[k], realised[k]).statistic
                                  for k in ranks]))
        return read

    k = run.eigen_forecast.shape[0]
    rows = [(name, portfolio(name)) for name in run.names[:4]]
    rows += [(name, portfolio(name)) for name in ("minimum variance, long-only",
                                                  "minimum variance, unconstrained")]
    rows += [("random long-only, median", random("long-only")),
             ("random long-short, median", random("long-short"))]
    if columns:
        rows.append(("every member singly, median", members))
    if groups:
        rows.append(("industries, mean abs(bias - 1)", industries))
    rows += [(f"eigenportfolios 1-{min(5, k)}, mean", eigen(range(min(5, k)))),
             (f"eigenportfolios {max(k - 4, 1)}-{k}, mean", eigen(range(max(k - 5, 0), k)))]
    print("\n  At every horizon evaluated, each forecast at h times its daily variance: the "
          "statistic with the windows overlapping / apart")
    print(f"  {'':<34}" + "".join(
        f" {'1 day' if h == 1 else f'{h} days':>13}" for h in horizons))
    for label, read in rows:
        cells = []
        for h in horizons:
            if h == 1:
                cells.append(f"{read(1, True):>13.3f}")
            else:
                cells.append(f"{read(h, True):>7.3f}/{read(h, False):<5.3f}")
        print(f"  {label:<34}" + "".join(f" {cell:>13}" for cell in cells))


EIGEN_CRITERION = ("half the eigenportfolios' mean abs(bias - 1) over every rank, half the "
                   "minimum-variance portfolios', long-only and unconstrained")
"""How a is chosen (midoffice13's brief): at one day, by the eigenportfolios and the
minimum-variance portfolios, on the test's first half; the second half is reported beside it."""


def _volatility(run, name, part=slice(None), horizon=1) -> tuple[float, float]:
    """A named portfolio's volatility a year over a part of the test's dates (a slice): the
    forecast at its median, and the realised from its returns -- at a horizon past one day its
    h-day returns', the windows overlapping."""
    row = run.names.index(name)
    forecast, realised = run.portfolio_outcomes(horizon)
    year = np.sqrt(BIAS_YEAR / horizon)
    return (float(np.nanmedian(forecast[row, part]) * year),
            float(np.nanstd(realised[row, part], ddof=1) * year))


def _halves(run) -> dict[str, slice]:
    """The test's two halves by forecast date, labelled: the first n // 2 dates, then the rest."""
    n = len(run.dates)
    half = n // 2
    return {
        f"the first half ({run.dates[0]} to {run.dates[half - 1]}, {half})": slice(0, half),
        f"the second half ({run.dates[half]} to {run.dates[-1]}, {n - half})": slice(half, n),
    }


def _eigen_reading(run, part, groups, columns) -> dict[str, float]:
    """One test's one-day statistics over a part of its forecast dates (a slice)."""
    names = list(run.names)

    def stat(row):
        return bias_statistics(run.forecast[row, part], run.realised[row, part]).statistic

    k = run.eigen_forecast.shape[0]
    eigen = np.array([bias_statistics(run.eigen_forecast[r, part],
                                      run.eigen_realised[r, part]).statistic for r in range(k)])
    out = {f"eigenportfolios {lo + 1}-{hi}, mean": float(eigen[lo:hi].mean())
           for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, k))}
    out["eigenportfolios, mean abs(bias - 1)"] = float(np.mean(np.abs(eigen - 1.0)))
    optimiser = [stat(names.index(f"minimum variance, {label}"))
                 for label in ("long-only", "unconstrained")]
    out["minimum variance, long-only"], out["minimum variance, unconstrained"] = optimiser
    out["minimum variance, mean abs(bias - 1)"] = float(np.mean(np.abs(np.array(optimiser) - 1)))
    # Each scale's own optimiser portfolios' realised risk (midoffice13's ruling 5).
    for label in ("long-only", "unconstrained"):
        out[f"minimum variance, {label}: realised vol, %"] = 100.0 * _volatility(
            run, f"minimum variance, {label}", part)[1]
    out["the criterion"] = 0.5 * (out["eigenportfolios, mean abs(bias - 1)"]
                                  + out["minimum variance, mean abs(bias - 1)"])
    for name in run.names[:4]:
        out[name] = stat(names.index(name))
    members = [bias_statistics(run.member_forecast[part, j], run.member_realised[part, j])
               for j in columns]
    members = np.array([s.statistic for s in members if s.n >= BIAS_MIN_FORECASTS])
    out["members, median"] = float(np.median(members)) if members.size else float("nan")
    out["members, mean abs(bias - 1)"] = (float(np.mean(np.abs(members - 1.0)))
                                          if members.size else float("nan"))
    if groups:
        out["industries, mean abs(bias - 1)"] = float(np.mean(
            [abs(stat(names.index(f"industry: {g}")) - 1.0) for g in groups]))
    for family in ("long-only", "long-short"):
        rows = [i for i, n in enumerate(names) if n.startswith(f"random {family} ")]
        out[f"random {family}, median"] = float(np.median([stat(i) for i in rows]))
    return out


def _report_eigen(run, eigen_runs, simulated, own, groups, member_stats, securities) -> None:
    """The eigenfactor risk adjustment at every scale tested, at one day: what it is for -- the
    eigenportfolios and the minimum-variance portfolios -- and what it should leave alone, over
    the whole test and each half; a chosen on the first half."""
    lam = np.array([b for b, _, _ in simulated])
    errors = np.array([e for _, e, _ in simulated])
    seconds = np.array([s for _, _, s in simulated])
    k = lam.shape[1]
    print(f"\n  The eigenfactor risk adjustment (Menchero, Wang and Orr; USE4) at one day: each "
          f"eigenvalue of the day's factor covariance scaled by (a (lambda - 1) + 1)^2, the "
          f"eigenportfolios tested in the unadjusted covariance's eigenbasis; the simulation "
          f"{seconds.mean():.2f} s a refit ({seconds.min():.2f} to {seconds.max():.2f})")
    print(f"  lambda by rank over the {len(lam)} dates, the mean (min to max): "
          + ", ".join(f"{r + 1}: {lam[:, r].mean():.3f} ({lam[:, r].min():.3f}-"
                      f"{lam[:, r].max():.3f})" for r in (0, 1, 2, 4, 9, 14, 19))
          + ", " + ", ".join(f"{r + 1}: {lam[:, r].mean():.3f}" for r in range(20, k))
          + f"; its Monte Carlo standard error at most {errors.max():.4f}, "
          f"{np.median(errors.max(axis=1)):.4f} at the median date's worst rank")
    tests = {own: run, **eigen_runs}
    order = sorted(tests)
    n = len(run.dates)
    parts = {f"the whole test ({run.dates[0]} to {run.dates[-1]}, {n})": slice(0, n),
             **_halves(run)}
    columns = [list(securities).index(code) for code, _ in member_stats]
    readings = {label: {a: _eigen_reading(tests[a], part, groups, columns) for a in order}
                for label, part in parts.items()}
    rows = list(next(iter(next(iter(readings.values())).values())))
    for label, by_scale in readings.items():
        print(f"\n  {label}, by a ({own:g} the model):")
        print(f"  {'':<42}" + "".join(f" {f'a = {a:g}':>9}" for a in order))
        for row in rows:
            print(f"  {row:<42}" + "".join(f" {by_scale[a][row]:>9.3f}" for a in order))
    first, second = list(readings.values())[1], list(readings.values())[2]
    chosen = min(order, key=lambda a: first[a]["the criterion"])
    print(f"\n  a chosen on the first half by the criterion -- {EIGEN_CRITERION}: a = "
          f"{chosen:g} ({first[chosen]['the criterion']:.3f}; the model's "
          f"{first[own]['the criterion']:.3f}); on the second half it reads "
          f"{second[chosen]['the criterion']:.3f} against the model's "
          f"{second[own]['the criterion']:.3f}, the best there a = "
          f"{min(order, key=lambda a: second[a]['the criterion']):g}")


def _bias_arrays(run, suffix: str = "", members: bool = True) -> dict[str, np.ndarray]:
    """One test's forecasts and outcomes as arrays, keyed with `suffix`: the daily ones and each
    horizon's outcomes with the windows overlapping."""
    out = {
        f"forecast{suffix}": run.forecast, f"realised{suffix}": run.realised,
        f"member_forecast{suffix}": run.member_forecast,
        f"eigen_forecast{suffix}": run.eigen_forecast,
        f"eigen_realised{suffix}": run.eigen_realised,
        f"repaired{suffix}": run.repaired,
        f"held_long_only{suffix}": run.min_variance_held["long-only"],
        f"held_unconstrained{suffix}": run.min_variance_held["unconstrained"],
    }
    if members:
        out[f"member_realised{suffix}"] = run.member_realised
    for h in run.horizons:
        out[f"realised_h{h}{suffix}"] = run.horizon_realised[h]
        if members:
            out[f"member_realised_h{h}{suffix}"] = run.member_horizon_realised[h]
        out[f"eigen_realised_h{h}{suffix}"] = run.eigen_horizon_realised[h]
    return out


def _save_bias(path, run, caps, horizon, eigen_runs=None, simulated=None) -> None:
    """Every forecast and outcome, for comparing runs: the daily ones, and each horizon's
    outcomes with the windows overlapping (its forecast the daily one times sqrt(h); apart,
    every h-th date's from the first). Under the eigenfactor adjustment's other scales, each
    test's arrays too, keyed `_a<scale>` -- but its members' outcomes, which are the model's --
    and the simulation by date."""
    extra: dict[str, np.ndarray] = {}
    for a, variant in (eigen_runs or {}).items():
        # The same members on the same days, only the covariance changed.
        if not (_same(variant.member_realised, run.member_realised)
                and all(_same(variant.member_horizon_realised[h], run.member_horizon_realised[h])
                        for h in run.horizons)):
            raise SystemExit(f"  the run stops: the members' outcomes at a = {a:g} are not the "
                             f"model's")
        extra.update(_bias_arrays(variant, f"_a{a:g}", members=False))
    if eigen_runs:
        extra["eigen_scales"] = np.array(sorted(eigen_runs), dtype=float)
    if simulated:
        extra["eigen_bias"] = np.array([b for b, _, _ in simulated])
        extra["eigen_standard_error"] = np.array([e for _, e, _ in simulated])
        extra["eigen_seconds"] = np.array([s for _, _, s in simulated])
    np.savez_compressed(
        path,
        dates=np.array([str(d) for d in run.dates]),
        names=np.array(run.names),
        securities=np.array(run.securities),
        model_horizon=np.array(horizon),
        horizons=np.array(run.horizons, dtype=int),
        caps=caps,
        **_bias_arrays(run),
        **extra,
    )
    print(f"  every forecast and outcome written to {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", nargs="?", default=DEFAULT_PANEL,
                        help="the .npz written by build_sp500_securities.py, on securities (by "
                             f"default {DEFAULT_PANEL}), or by fetch_sp500_panel.py, on tickers")
    parser.add_argument("--returns", choices=("price", "total"), default="total",
                        help="the panel's total returns, every distribution reinvested (the "
                             "default), or its price returns, the market's to match; a panel on "
                             "tickers carries price returns alone")
    parser.add_argument("--volumes", choices=("traded", "adjusted"), default=None,
                        help="the volumes the turnovers and VLVR read: the panel's stored ones "
                             "by default -- traded on securities, the adjusted bars' on tickers "
                             "-- or the adjusted bars', kept beside them on securities")
    parser.add_argument("--beta-window", type=float, default=None,
                        help="the market model's window in months, in place of the library's "
                             "default of 252 trading days (USE4's); the USE3 value of 60 needs "
                             "more days than a 5-year panel has")
    parser.add_argument("--half-life", type=float, default=90.0,
                        help="volatility half-life in trading days (BIM301S uses 90)")
    parser.add_argument("--horizon", type=int, default=None, metavar="H",
                        help=f"the model's horizon in trading days (default {HORIZON}): the "
                             f"factor covariance Newey-West at H - 1 lags with Bartlett weights, "
                             f"on the volatilities and the correlations alike, which is exactly "
                             f"the per-day variance of H-day returns; 1 is the one-day covariance")
    parser.add_argument("--newey-west-lags", type=int, default=None, metavar="L",
                        help="the Newey-West lags, overriding --horizon's H - 1; the horizon is "
                             "then L + 1")
    parser.add_argument("--industries", action="store_true",
                        help="add SIC-derived industry factors and drop the market factor, "
                             "which they span; codes come from the fetch cache at no cost")
    parser.add_argument("--cache", default="data/external/polygon/cache")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--min-industry", type=int, default=15,
                        help="merge a SIC major group below this size into its division")
    parser.add_argument("--equal-weight", action="store_true",
                        help="weight the cross-sectional regression equally; the default "
                             "is Barra's square root of capitalisation")
    parser.add_argument("--membership", action="store_true",
                        help="hold each period's regression to that day's index members, from "
                             "the newest snapshot in --snapshots; without it every panel "
                             "security with data is regressed")
    parser.add_argument("--snapshots", default="data/external/sp500")
    parser.add_argument("--min-coverage", type=float, default=0.5,
                        help="the fraction of a window descriptor's window that must hold "
                             "valid data for it to have a value (default 0.5; 0 takes any "
                             "data at all, and an empty window never gives one)")
    parser.add_argument("--rstr-skip", type=int, default=0,
                        help="trading days cut from the end of RSTR's twelve-month window, "
                             "the academic twelve-minus-one reading (default 0, every day "
                             "summed)")
    parser.add_argument("--beta-days", type=int, default=None,
                        help="the market model's window in trading days, in place of the "
                             "library's default and of --beta-window")
    parser.add_argument("--beta-half-life", type=float, default=None,
                        help="the market model's half-life in trading days, in place of the "
                             "library's default of 63 (USE4's); inf weights every day of the "
                             "window equally")
    parser.add_argument("--size-nonlinearity", choices=SIZE_NONLINEARITY, default="use4",
                        help="build Size Nonlinearity USE4's way, from the Size exposure, cubed "
                             "and orthogonalised against Size (the default), or USE3's, the "
                             "blend of LCAPCB; none leaves it out of the model")
    parser.add_argument("--residual-spread", choices=RESIDUAL_SPREADS, default="mad",
                        help="how the specific risk model's residual spread, which sets every "
                             "blend weight, is measured: 1.4826 x the MAD (the default) or the "
                             "standard deviation")
    parser.add_argument("--mixed-blends", action="store_true",
                        help="estimate every period with enough securities, whatever its "
                             "blends are built from; by default a period is estimated only if "
                             "every risk index's blend in it has exactly the descriptors the "
                             "forecast's has")
    parser.add_argument("--sampling-correction", action="store_true",
                        help="subtract from the factor covariance the sampling covariance of "
                             "the factor returns, weighted as the covariance carries each "
                             "period's, and repair it to positive semi-definite where that is "
                             "needed; by default the covariance as estimated")
    parser.add_argument("--leverage-correction", action="store_true",
                        help="the correction's specific side: each member's residual divided by "
                             "sqrt(1 - h), its leverage, in place of the uniform degrees-of-"
                             "freedom inflation, and each specific return outside the "
                             "regressions by sqrt(1 + h); by default the uniform inflation")
    parser.add_argument("--merge-industry", action="append", default=[], metavar="A=B",
                        help="fold industry A into industry B once the scheme is built, its "
                             "members labelled B; repeatable, and a chain resolves to its end "
                             "(with --industries)")
    parser.add_argument("--specific-history", choices=("covered", "regression"),
                        default="covered",
                        help="whose residuals each security's own specific-risk history is "
                             "built from: every specific return it has, member or not (the "
                             "default), or its residuals in the regressions alone; factor "
                             "returns and the specific risk model's fit keep to the "
                             "regressions either way")
    parser.add_argument("--specific-half-life", type=half_life_or_none,
                        default=SPECIFIC_HALF_LIFE, metavar="H",
                        help="weight each security's specific-risk history by this half-life "
                             f"in trading days, the blend told its effective number of "
                             f"observations (default {SPECIFIC_HALF_LIFE:g}); none weighs every "
                             f"specific return the same")
    parser.add_argument("--correlation-half-life", type=half_life_or_none,
                        default=CORRELATION_HALF_LIFE, metavar="H",
                        help="the two-half-life factor covariance: correlations on this "
                             "half-life in trading days, volatilities on --half-life's, both "
                             f"Newey-West at the horizon's lags (default "
                             f"{CORRELATION_HALF_LIFE:g}); none is one half-life for the whole "
                             f"matrix")
    parser.add_argument("--correlation-lags", type=lags_or_horizon, default=None, metavar="L",
                        help="the factor correlations' own Newey-West lags, with the "
                             "two-half-life covariance: by default none at any horizon above one "
                             "day, the volatilities keeping the horizon's -- the lags on the "
                             "volatilities alone (midoffice13's ruling 4); horizon gives the "
                             "correlations the horizon's lags too, midoffice12's reporting view")
    parser.add_argument("--variance-ratio-half-life", type=float, default=None, metavar="H",
                        help="each factor volatility its one-day volatility on --half-life "
                             "times the square root of its variance ratio at the horizon -- its "
                             "Newey-West variance over its one-day variance -- estimated on "
                             "this half-life, so the serial correlation comes from the long "
                             "window and the level from the short one (with the two-half-life "
                             "covariance); by default each volatility is Newey-West on "
                             "--half-life")
    parser.add_argument("--bias", action="store_true",
                        help="the out-of-sample bias statistics: the model refitted as of each "
                             "trading day from its first with a year of factor returns, each "
                             "forecast against the next day's return and the next H days', H "
                             "the model's horizon, over the index, the size pair, the "
                             "industries, random and minimum-variance portfolios, every member "
                             "singly and the eigenportfolios; about ten minutes")
    parser.add_argument("--eigen-adjustment", type=float, default=None, metavar="A",
                        help="the eigenfactor risk adjustment (Menchero, Wang and Orr; USE4): "
                             "each eigenvalue of the factor covariance scaled by "
                             "(A (lambda - 1) + 1)^2, lambda by rank from --eigen-draws "
                             "histories simulated from the covariance and re-estimated by the "
                             "model's own estimator; 0 simulates and scales nothing; by default "
                             "none")
    parser.add_argument("--eigen-draws", type=int, default=EIGEN_DRAWS, metavar="M",
                        help=f"the eigenfactor simulation's histories per covariance (default "
                             f"{EIGEN_DRAWS})")
    parser.add_argument("--bias-eigen-scales", type=scale_list, default=(), metavar="A,A",
                        help="with --bias, the eigenfactor risk adjustment at these scales too, "
                             "each tested beside the model and one simulation per forecast date "
                             "serving them all; read at one day over the whole test and each "
                             "half, and kept in --bias-out")
    parser.add_argument("--bias-horizons", type=horizon_list, default=(), metavar="H,H",
                        help="with --bias, more horizons in trading days to evaluate every "
                             "forecast at, beside one day and the model's own, comma-separated; "
                             "summarised side by side and kept in --bias-out")
    parser.add_argument("--bias-out", default=None, metavar="PATH",
                        help="with --bias, every forecast and outcome to this .npz")
    parser.add_argument("--bias-guard", type=int, default=5, metavar="N",
                        help="with --bias, how many forecast dates the guard rebuilds the "
                             "model's inputs on from the panel cut at the date, the scheme "
                             "held (default 5: the first, the last and three between; 0 skips)")
    args = parser.parse_args(argv)
    if args.horizon is not None and args.horizon < 1:
        parser.error(f"--horizon must be at least one trading day; got {args.horizon}")
    if args.newey_west_lags is not None and args.newey_west_lags < 0:
        parser.error(f"--newey-west-lags must be non-negative; got {args.newey_west_lags}")
    if args.eigen_adjustment is not None:
        if not args.eigen_adjustment >= 0.0:
            parser.error(f"--eigen-adjustment must be non-negative; got {args.eigen_adjustment}")
        if args.sampling_correction:
            parser.error("--eigen-adjustment and --sampling-correction are alternative remedies "
                         "for the covariance's estimation error; use one")
    if args.eigen_draws < 2:
        parser.error(f"--eigen-draws must be at least two; got {args.eigen_draws}")
    if args.bias_eigen_scales and args.sampling_correction:
        parser.error("--bias-eigen-scales needs the covariance without --sampling-correction")
    if args.correlation_lags is not None and args.correlation_half_life is None:
        parser.error("--correlation-lags needs the two-half-life covariance, not "
                     "--correlation-half-life none")
    if args.variance_ratio_half_life is not None:
        if args.correlation_half_life is None:
            parser.error("--variance-ratio-half-life needs the two-half-life covariance, not "
                         "--correlation-half-life none")
        if not args.variance_ratio_half_life > 0.0:
            parser.error(f"--variance-ratio-half-life must be positive; got "
                         f"{args.variance_ratio_half_life}")
    # The lags in effect, and the horizon they make: --newey-west-lags overrides --horizon.
    lags = (args.newey_west_lags if args.newey_west_lags is not None
            else (HORIZON if args.horizon is None else args.horizon) - 1)
    horizon = lags + 1
    # The correlations' own lags (midoffice13's ruling 4): at any horizon above one day none by
    # default, the lags on the volatilities alone; `horizon` the horizon's, as the estimator's
    # own default has it. At one day there are no lags to give either.
    if args.correlation_lags == "horizon":
        correlation_lags = None
    elif args.correlation_lags is not None:
        correlation_lags = args.correlation_lags
    elif lags > 0 and args.correlation_half_life is not None:
        correlation_lags = 0
    else:
        correlation_lags = None
    # None leaves the window to the library.
    beta_months = (
        args.beta_window if args.beta_days is None
        else args.beta_days / TRADING_DAYS_PER_MONTH
    )

    path = Path(args.panel)
    if not path.exists():
        raise SystemExit(f"no such panel: {path}")
    panel, market, extras = load_panel(path, args.returns, args.volumes)
    returns = panel.returns()
    on_securities = extras["on_securities"]

    print(f"Panel\n  {len(panel.dates)} trading days, {panel.n_securities} securities")
    print(f"  {panel.dates[0]} .. {panel.dates[-1]}")
    if on_securities:
        print(f"  on securities, each a share class labelled by its last ticker; "
              f"{args.returns} returns, the market's {args.returns} return; "
              f"{args.volumes or 'traded'} volumes")
    else:
        print("  on tickers: price returns, the adjusted bars' volumes")
    usable = np.isfinite(returns).sum()
    print(f"  {usable:,} of {returns.size:,} return observations present "
          f"({usable / returns.size:.1%})")

    # The panel is the index on its first day held fixed. Nothing in it removes a name after
    # it leaves -- SBNY traded on at cents against a pre-failure share count -- so without a
    # membership the regression keeps whatever still has a price.
    membership = None
    index_sizes: list[int] = []
    if args.membership:
        print("\nMembership")
        try:
            snapshot = sp500.load_latest(args.snapshots)
        except FileNotFoundError as error:
            raise SystemExit(f"  --membership needs a snapshot: {error}") from None
        print(f"  snapshot {snapshot.content_hash[:12]}, fetched "
              f"{snapshot.fetched_at:%Y-%m-%d %H:%M} UTC, covering "
              f"{snapshot.coverage_start} .. {snapshot.coverage_end}")
        # The roll-forward, where the model is: a change the constituent list has not reached.
        for issue in sp500.validate(snapshot):
            if issue.code == "current_stale":
                print(f"  {issue.detail}")
        if on_securities:
            if extras["snapshot"] != snapshot.content_hash:
                raise SystemExit(f"  the run stops: the panel's membership is snapshot "
                                 f"{extras['snapshot'][:12]}'s, and the newest is "
                                 f"{snapshot.content_hash[:12]}'s; rebuild the table and the "
                                 f"panel on it")
            # Every member is a column of this panel, so the index is its members.
            membership = extras["membership"]
            index_sizes = [int(n) for n in membership.sum(axis=1)]
            print("  the panel's own: membership() read through the identity table's renames "
                  "and mapped to securities, a reorganisation's under its successor")
        else:
            try:
                membership, index_sizes = index_membership(
                    snapshot, panel.dates, panel.securities
                )
            except sp500.CoverageError as error:
                raise SystemExit(f"  the run stops: {error}") from None
        print(f"  one array for every standardisation and every regression below, each held "
              f"to that day's members: {int(membership[0].sum())} of the panel's "
              f"{panel.n_securities} on {panel.dates[0]}, {int(membership[-1].sum())} on "
              f"{panel.dates[-1]}")
        unreproduced = unreproduced_changes(snapshot, panel.dates[0], panel.dates[-1])
        if unreproduced:
            print(f"  {len(unreproduced)} change(s) inside the panel's window that "
                  f"membership() does not reproduce, used as it reads them:")
        for when, ticker, kind in unreproduced:
            days = [i for i, d in enumerate(panel.dates) if d >= when]
            claim = (
                "recorded as added, but membership() does not have it from that date"
                if kind == "added"
                else "recorded as removed, but membership() still has it after that date"
            )
            if ticker not in panel.securities:
                effect = (f"not in the panel, so no cross-section changes on the "
                          f"{len(days)} panel day(s) from then")
            else:
                priced = int(np.isfinite(returns[days, panel.securities.index(ticker)]).sum())
                effect = (f"in the panel with a return on {priced} of the {len(days)} panel "
                          f"day(s) from then, "
                          f"{'left out of' if kind == 'added' else 'kept in'} each of those "
                          f"days' cross-sections")
            print(f"  - {when} {ticker} {claim}: {effect}")

    # The industries' codes and folds are read first, so a bad --merge-industry stops the run
    # before the descriptors are computed rather than after.
    if args.merge_industry and not args.industries:
        raise SystemExit("  --merge-industry needs --industries: there is no scheme to merge in")
    merges = parse_merges(args.merge_industry)
    codes: dict[str, str | None] | None = None
    carried: list[str] = []
    if args.industries:
        codes, carried = industry_codes(panel, extras, args)
    inputs = model_inputs(panel, market, membership, args, codes=codes, merges=merges)
    descriptors, standardised, caps = inputs.descriptors, inputs.standardised, inputs.caps
    weights, return_panel = inputs.weights, inputs.return_panel
    blendable, unblended = inputs.blendable, inputs.unblended
    use4_size = args.size_nonlinearity == "use4"
    style_names, unlike, periods = inputs.styles.names, inputs.unlike, inputs.periods
    scheme, indices = inputs.scheme, inputs.indices
    forecast_row = len(panel.dates) - 1

    print("\nDescriptors")
    print(f"  computed {len(descriptors.names)}: {', '.join(descriptors.names)}")
    for name, reason in sorted(inputs.skipped.items()):
        print(f"  - {name}: {reason}")
    if args.rstr_skip:
        print(f"  RSTR sums its twelve-month window less the last {args.rstr_skip} trading "
              f"day(s)")
    # The window sets the date the market model's descriptors first have a value, and so
    # when the blends change composition; the half-life how far back a beta remembers. Read
    # from the library, which is what fitted it.
    window, half_life = market_model_window(panel, beta_months, args.beta_half_life)
    given = [f"--{flag} {value:g}" for flag, value in (
        ("beta-window", args.beta_window), ("beta-days", args.beta_days),
        ("beta-half-life", args.beta_half_life)) if value is not None]
    print(f"  the market model -- BTSG, HALPHA, SERDP, and VLVR's residual sd -- is a regression "
          f"on the market over {window} trading days, "
          + ("every day weighted equally" if half_life is None
             else f"half-life {half_life:g} trading days")
          + (f" ({', '.join(given)})" if given else " (the library's default, USE4's)"))

    # An empty window sums to zero, and RSTR, CMRA and the turnovers once recorded that zero as
    # a value: 18,645 of RSTR's, 4,679 of them on index members and 4,334 of those before the
    # security's first price. A window under half full is not the descriptor it is named: a
    # twelve-month RSTR summed over five months is a five-month momentum. Both checked against
    # each descriptor's own inputs, so the check does not share the code it checks.
    coverage = window_coverage(
        panel, descriptors, market, window, membership, args.min_coverage, args.rstr_skip,
    )
    rule = ("any valid observation" if args.min_coverage == 0.0
            else f"at least {args.min_coverage:.0%} of the window valid")
    empty = [(row[0], row[2]) for row in coverage if row[2]]
    below = [(row[0], row[3]) for row in coverage if row[3]]
    report = (f"  window descriptors need {rule}; from a window with none, over "
              f"{sum(row[1] for row in coverage):,} values: "
              f"{'none' if not empty else sum(bad for _, bad in empty)}")
    if args.min_coverage > 0.0:
        report += (f"; from one under {args.min_coverage:.0%} valid: "
                   f"{'none' if not below else sum(bad for _, bad in below)}")
    print(report)
    for name, bad in empty:
        print(f"  ! {name}: {bad:,} values computed from a window holding no valid observation")
    for name, bad in below:
        print(f"  ! {name}: {bad:,} values computed from under {args.min_coverage:.0%} of "
              f"their window")
    if args.min_coverage < 0.5:
        print(f"  from under half their window, which a 50% rule would remove "
              f"({'all/on members' if membership is not None else 'all'}):")
        cells = [f"{row[0]} {row[4]:,}" + (f"/{row[5]:,}" if membership is not None else "")
                 for row in coverage]
        for i in range(0, len(cells), 5):
            print(f"    {'  '.join(cells[i:i + 5])}")

    if standardised.winsorised:
        print(f"  {standardised.winsorised:,} raw values winsorised before standardising")
    if membership is not None:
        last = len(panel.dates) - 1
        print(f"  {standardised.clipped_outside:,} values of securities outside the index hit "
              f"one of the members' bounds, over every date; on {panel.dates[last]}, of those "
              f"with a value:")
        hits = [f"{name} {hit}/{have}"
                for name, hit, have in clipped_outside_on(descriptors, caps, membership, last)]
        for i in range(0, len(hits), 7):
            print(f"    {'  '.join(hits[i:i + 7])}")

    print("\nWhich descriptors carry signal")
    print("  Ranked by the cross-sectional return variance each explains on its own.")
    print(f"  {'descriptor':<10} {'cover':>6} {'periods':>8} {'mean R2':>9} "
          f"{'factor vol':>11} {'t-stat':>8}")
    rows = screen(standardised, returns, membership)
    scored = [r for r in rows if "mean_r2" in r]
    thin = [r for r in rows if "mean_r2" not in r]
    for row in sorted(scored, key=lambda r: -r["mean_r2"]):
        print(f"  {row['name']:<10} {row['coverage']:>5.0%} {row['periods']:>8,} "
              f"{row['mean_r2']:>8.2%} {row['factor_vol']:>10.2%} {row['t_stat']:>8.2f}")
    for row in thin:
        print(f"  {row['name']:<10} {'-':>6} {row['periods']:>8,}   too few usable periods")

    print("\nRisk indices")
    for name in style_names:
        if name == "Size Nonlinearity" and use4_size:
            print(f"  {name:<20} the Size exposure cubed, standardised and clipped at +/-4, "
                  f"orthogonalised against Size with the regression weights, standardised "
                  f"again (USE4)")
            continue
        members = [m for m in risk_index_members(name) if m in blendable.names]
        print(f"  {name:<20} {', '.join(members)}")
    for name, reason in unblended.items():
        if name in standardised.names:
            print(f"  - {name} is in no blend: {reason}")

    if args.industries:
        print("\nIndustries")
        if on_securities:
            missing = sorted(code for code, sic in codes.items() if not sic)
            print(f"  each security's SIC code from its own latest record with one: "
                  f"{len(codes) - len(missing)} of {len(codes)}"
                  + (f"; none for {', '.join(missing)}" if missing else ""))
        elif carried:
            # A company acquired mid-window has no record at the end of it, but its industry
            # while it traded is exactly what the regression needs.
            print(f"  {len(carried)} classification(s) carried from an earlier date "
                  f"(delisted inside the window): {', '.join(sorted(carried)[:5])}")
        held = "" if membership is None else " while an index member"
        print(f"  {len(inputs.universe)} of {panel.n_securities} securities are in at least one "
              f"estimated period's regression{held}; every count below is over them")
        if merges:
            before, after = inputs.built_scheme.counts(), scheme.counts()
            for source, target in sorted(scheme.folded.items()):
                print(f"  {source} ({before[source]}) folded into {target}, which holds "
                      f"{after[target]}")
        print(f"  {scheme.describe()}")
        for name, count in sorted(scheme.counts().items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {name:<24} {count:>4}")
        for note in scheme.qualifications():
            print(f"  - {note}")

    print("\nModel")
    print(f"  regression weights: "
          f"{'equal' if args.equal_weight else 'square root of capitalisation'}")
    # The horizon (midoffice11's ruling 3): Newey-West with Bartlett weights at L lags is the
    # per-day variance of (L + 1)-day returns, so the lags are the horizon, said in days.
    if args.newey_west_lags is None:
        source = ""
    else:
        source = f"; set by --newey-west-lags {lags}" + (
            f", overriding --horizon {args.horizon}"
            if args.horizon is not None and args.horizon != horizon else "")
    ratio = args.variance_ratio_half_life
    # Which it is (midoffice12's ruling 2): the model's own horizon, or a reporting view of
    # broad portfolios' risk over a holding period.
    if horizon == HORIZON:
        role = "the model's horizon"
    else:
        model_days = "one day" if HORIZON == 1 else f"{HORIZON} days"
        role = (f"a reporting view, not the model's horizon of {model_days}: broad "
                f"portfolios' risk over a holding period of {horizon} days. Its gain over the "
                f"one-day model scaled by sqrt(h) was measured on 2024-26 alone, a window with "
                f"the April 2025 shock in it, and is re-tested as data accumulates")
    corr_lags = correlation_lags
    if corr_lags is not None and corr_lags != lags:
        # The lags on the volatilities alone (midoffice13): the correlations their own.
        corr = ("the factor correlations one-day, with no lags" if corr_lags == 0
                else f"the factor correlations Newey-West at {corr_lags} lag(s)")
        if ratio is None:
            vols = (f"the factor volatilities Newey-West at {lags} lag(s) with Bartlett weights, "
                    f"each the per-day variance of {horizon}-day returns")
        else:
            vols = (f"each factor volatility its one-day volatility on a half-life of "
                    f"{args.half_life:g} times the square root of its variance ratio at {lags} "
                    f"lags, the ratio estimated on a half-life of {ratio:g}")
        days = "1 trading day" if horizon == 1 else f"{horizon} trading days"
        print(f"  the horizon: {days}, {role} -- {vols}; {corr}: the lags on "
              f"the volatilities alone; the specific risk one day's variance, with no lags, at "
              f"every horizon{source}")
    elif horizon == 1:
        print(f"  the horizon: 1 trading day, {role} -- the factor covariance the one-day "
              f"covariance, with no Newey-West lags, and the specific risk one day's "
              f"variance{source}")
    elif ratio is None:
        print(f"  the horizon: {horizon} trading days, {role} -- the factor covariance "
              f"Newey-West at {lags} lag(s) with Bartlett weights, on the volatilities and the "
              f"correlations alike, which is exactly the per-day variance of {horizon}-day "
              f"returns; the specific risk one day's variance, with no lags, at every "
              f"horizon{source}")
    else:
        print(f"  the horizon: {horizon} trading days, {role} -- the factor correlations "
              f"Newey-West at {lags} lag(s) with Bartlett weights, which is exactly the per-day "
              f"variance of {horizon}-day returns, and each factor volatility its one-day "
              f"volatility on a half-life of {args.half_life:g} times the square root of its "
              f"variance ratio at {lags} lags, the ratio estimated on a half-life of {ratio:g}: "
              f"the serial correlation from the long window, the level from the short; the "
              f"specific risk one day's variance, with no lags, at every horizon{source}")
    # One half-life for the whole matrix, Newey-West; or the two-half-life estimator (the
    # default, midoffice11's ruling 2): correlations on --correlation-half-life, volatilities
    # on --half-life's, both Newey-West at the horizon's lags, as USE4 corrects both horizons.
    if args.correlation_half_life is None:
        estimator = NeweyWestCovariance(lags=lags, half_life=args.half_life)
    else:
        estimator = HorizonCovariance(
            volatility_half_life=args.half_life,
            correlation_half_life=args.correlation_half_life,
            newey_west_lags=lags,
            ratio_half_life=ratio,
            correlation_lags=correlation_lags,
        )
    options = dict(
        standardise=False,
        annualise=True,
        specific_risk="blend",
        capitalisations=caps,
        regression_weights=weights,
        membership=membership,
        periods=periods,
        residual_spread=args.residual_spread,
        specific_history=args.specific_history,
        # Industries sum to one per security, which IS the unit column. Fitting a market
        # factor alongside them makes the design exactly singular, not nearly so.
        fit_market=scheme is None,
        covariance_estimator=estimator,
        sampling_correction=args.sampling_correction,
        leverage_correction=args.leverage_correction,
        specific_half_life=args.specific_half_life,
    )
    if args.eigen_adjustment is not None:
        options.update(eigen_adjustment=args.eigen_adjustment, eigen_draws=args.eigen_draws)
    model = estimate_cross_sectional_model(return_panel, indices, **options)
    notes = model.notes
    print(f"  {model.describe()}")
    # Barra's convention, and the descriptor screen's: everything that defines period t's
    # regression is fixed the day before, so today's cap -- which holds today's return -- can
    # make no stock look as though it outperformed today.
    print(f"  each period regresses day t's returns on the exposures, weights and membership "
          f"of {'day t' if notes.lag == 0 else f'{notes.lag} trading day(s) earlier'}; the "
          f"forecast exposures are {panel.dates[-1]}'s")
    if notes.leverage_correction:
        print("  each member's residual divided by sqrt(1 - h), its own leverage, in place of "
              "the uniform degrees-of-freedom inflation, and each specific return outside the "
              "regressions by sqrt(1 + h): the sampling-error correction's specific side")
    if notes.eigen is not None:
        eigen, a = notes.eigen, notes.eigen_adjustment
        gamma = eigen.scales(a)
        print(f"  the eigenfactor risk adjustment, a = {a:g}"
              + (" (simulated, nothing scaled)" if not a else "")
              + f": lambda by rank from {eigen.draws} histories simulated from the factor "
              f"covariance over its {eigen.n_periods} periods and re-estimated by its own "
              f"estimator, in {eigen.seconds:.1f} s -- {eigen.bias[0]:.3f} at the largest "
              f"eigenvalue, {eigen.bias[-1]:.3f} at the smallest, {eigen.bias.min():.3f} to "
              f"{eigen.bias.max():.3f} over the ranks, its Monte Carlo standard error at most "
              f"{eigen.standard_error.max():.4f}"
              + (f"; each eigenfactor's volatility multiplied by {gamma.min():.3f} to "
                 f"{gamma.max():.3f}" if a else ""))
    print(f"  mean cross-sectional R-squared {notes.mean_r_squared:.3f}")
    print(f"  {len(notes.rank_deficient_periods)} collinear period(s) of {notes.n_periods} "
          f"estimated; {sum(notes.empty_periods)} factor-period(s) left out for having no "
          f"member present")
    at = {d: i for i, d in enumerate(panel.dates)}
    estimated_rows = [at[d] for d in notes.estimated_dates]
    # The dates the estimated periods read their exposures, weights and membership from.
    read_rows = [row - notes.lag for row in estimated_rows]
    # Who each estimated period's regression held, by the estimator's own test of entry.
    held_in = regression_design(
        return_panel, indices, regression_weights=weights, membership=membership,
        lag=notes.lag,
    ).in_regression[estimated_rows]
    if membership is not None:
        # Read with the exposures, so a name is in the regression on the day it leaves the
        # index and out of it on the day it joins.
        unheld = regression_design(
            return_panel, indices, regression_weights=weights, lag=notes.lag
        ).in_regression[estimated_rows]
        same_day = unheld & membership[estimated_rows]
        mask_moves = int((membership[read_rows] != membership[estimated_rows]).sum())
        print(f"  membership read on the exposures' date: against same-day membership, "
              f"{int((held_in != same_day).sum())} security-period(s) of the regressions "
              f"move -- {int((held_in & ~same_day).sum())} in, on the day a name left the "
              f"index, {int((same_day & ~held_in).sum())} out, on the day one joined; the "
              f"mask itself differs on {mask_moves}")
    if scheme is not None and scheme.universe is not None:
        # The universe was worked out before the industries existed. It can differ from the
        # regression's only if a security has no classification at all, or the model has
        # min_securities factors or more -- neither is the case here, but a count over the
        # wrong set is the defect this exists to prevent, so check it.
        differs = scheme.universe.symmetric_difference(notes.regression_universe)
        if differs:
            print(f"  ! the industry counts were taken over a universe that differs from the "
                  f"regression's by {len(differs)} security(ies): "
                  f"{', '.join(sorted(differs)[:5])}")

    # Each risk index's blend, period by period, by the estimator's own test of who each
    # regression held: the rule above, checked on what was estimated. Every estimated period's
    # blend must be the forecast's -- built from the descriptors the forecast exposures are --
    # or the run stops. Under --mixed-blends a descriptor in the forecast exposures and in no
    # estimated period's -- SERDP, were it blended -- still stops it, and the periods blended
    # otherwise are flagged.
    forecast_held = np.array([code in model.exposures for code in panel.securities])[None, :]
    print(f"  each risk index's blend over the {notes.n_periods} estimated periods: every "
          f"descriptor from the first period whose regression reads it, and in how many")
    stops: list[str] = []
    mismatched: list[str] = []
    for name in style_names:
        estimated = blend_composition(
            blendable, name, read_rows, held_in, size_nonlinearity=args.size_nonlinearity
        )
        forecast = blend_composition(
            blendable, name, [forecast_row], forecast_held,
            size_nonlinearity=args.size_nonlinearity,
        )
        grouped: dict[str, list[str]] = {}
        starts: dict[str, int] = {}
        for descriptor, present in estimated.items():
            where = np.flatnonzero(present)
            if where.size:
                label = f"from {notes.estimated_dates[where[0]]}"
                if where[-1] != present.size - 1:
                    label += f" to {notes.estimated_dates[where[-1]]}"
                label += f" ({where.size})"
            else:
                label = "in none"
            grouped.setdefault(label, []).append(descriptor)
            starts.setdefault(label, int(where[0]) if where.size else present.size)
        print(f"    {name:<20} " + "; ".join(
            f"{', '.join(grouped[label])} {label}" for label in sorted(grouped, key=starts.get)
        ))
        for descriptor, present in forecast.items():
            if present[0] and not estimated[descriptor].any():
                carried = np.isfinite(
                    blendable.values[forecast_row, :, blendable.names.index(descriptor)]
                ) & forecast_held[0]
                stops.append(f"{name}'s {descriptor}, in {int(carried.sum())} securities' "
                             f"forecast exposures and none of the {notes.n_periods} estimated "
                             f"periods'")
        off = np.zeros(notes.n_periods, dtype=bool)
        for descriptor, present in estimated.items():
            off |= present != forecast[descriptor][0]
        if off.any():
            where = np.flatnonzero(off)
            mismatched.append(f"{name}: {where.size} of the {notes.n_periods} estimated periods, "
                              f"from {notes.estimated_dates[where[0]]} to "
                              f"{notes.estimated_dates[where[-1]]}, are blended from other "
                              f"descriptors than the forecast exposures")
    if stops:
        raise SystemExit(
            "  the run stops: a descriptor contributes to a forecast exposure and to no "
            "estimated period's, so its factor would be read on a blend it was never estimated "
            "on -- " + "; ".join(stops)
        )
    if mismatched and periods is not None:
        raise SystemExit(
            "  the run stops: an estimated period's blend is not the forecast's, so a factor "
            "would be estimated on one definition and forecast on another -- "
            + "; ".join(mismatched)
        )
    for note in mismatched:
        print(f"  ! {note}")
    if not mismatched:
        print(f"  every estimated period's blend is the forecast exposures', "
              f"{panel.dates[forecast_row]}'s, in every index: one definition per factor "
              f"across the sample")
    # What the rule left out: the periods that held enough securities to be estimated.
    if notes.excluded_periods:
        dropped = [at[d] for d in notes.excluded_periods]
        print(f"  {len(dropped)} period(s) held enough securities and are not estimated, a "
              f"risk index's blend in each not being the forecast's:")
        runs: list[list[int]] = []
        for t in dropped:
            if runs and t == runs[-1][-1] + 1 and unlike[t] == unlike[runs[-1][-1]]:
                runs[-1].append(t)
            else:
                runs.append([t])
        for run in runs:
            print(f"    {panel.dates[run[0]]} to {panel.dates[run[-1]]} ({len(run)}): "
                  + "; ".join(unlike[run[0]]))

    # The index at zero style exposure, exactly: on every date an estimated period or the
    # forecast reads, the cap-weighted mean of each style exposure over that date's estimation
    # universe is zero to rounding, and the equal-weighted dispersion one. That is what lets an
    # exposure read as relative to the index. Nothing after the second standardisation clips,
    # so nothing may move it.
    checked = sorted(set(read_rows) | {len(panel.dates) - 1})
    columns = [indices.names.index(name) for name in style_names]
    means, spreads, bounds, sizes = estimation_moments(
        indices.values[checked][:, :, columns],
        caps[checked],
        None if membership is None else membership[checked],
    )
    worst = float(np.nanmax(np.abs(means)))
    eps = np.finfo(float).eps
    over = np.abs(means) > bounds
    universe_label = "index members" if membership is not None else "securities"
    in_eps = f" ({worst / eps:.1f} eps)" if worst < 1e3 * eps else ""
    print(f"\n  Style exposures over the {universe_label} with a value, on all {len(checked)} "
          f"dates an estimated period or the forecast reads: cap-weighted mean at most "
          f"{worst:.1e} from zero{in_eps}, equal-weighted sd at most "
          f"{float(np.nanmax(np.abs(spreads - 1.0))):.1e} from one")
    if over.any():
        t, k = np.unravel_index(int(np.nanargmax(np.where(over, np.abs(means), -1.0))),
                                means.shape)
        print(f"  ! {int(over.sum())} of {over.size} style-date means are past rounding, so "
              f"the index is not at zero exposure; the largest, {means[t, k]:+.4f}, is "
              f"{style_names[k]} on {panel.dates[checked[t]]}")
    print(f"  {'on ' + str(panel.dates[checked[-1]]):<22} {'cap-wtd mean':>13} "
          f"{'eq-wtd sd':>10} {'n':>5}")
    for k, name in enumerate(style_names):
        print(f"  {name:<22} {means[-1, k]:>+13.1e} {spreads[-1, k]:>10.6f} "
              f"{sizes[-1, k]:>5}")

    # The size pair. USE3's raw cube is 0.93 correlated with Size, so the regression splits one
    # size effect between two collinear exposures; USE4's is orthogonal to Size by
    # construction, and on every date the regression-weighted covariance of the two over the
    # estimation universe is zero to rounding, or the run stops.
    if "Size" in style_names and "Size Nonlinearity" in style_names:
        pair = size_pair_moments(
            indices.values[checked][:, :, indices.names.index("Size")],
            indices.values[checked][:, :, indices.names.index("Size Nonlinearity")],
            None if weights is None else weights[checked],
            caps[checked],
            None if membership is None else membership[checked],
        )
        last_day = panel.dates[checked[-1]]
        if use4_size:
            worst = float(np.nanmax(np.abs(pair["covariance"])))
            past = np.abs(pair["covariance"]) > pair["bound"]
            print(f"  Size Nonlinearity, USE4's, on the same dates: its clip at +/-4 moved "
                  f"{int(pair['clipped'].sum()):,} values of the {universe_label}, "
                  f"{int(pair['clipped'][-1])} on {last_day}; its regression-weighted covariance "
                  f"with Size over them is at most {worst:.1e} ({worst / eps:.1f} eps), inside "
                  f"rounding on {int((~past).sum())} of the {past.size} dates")
            if past.any():
                t = int(np.nanargmax(np.where(past, np.abs(pair["covariance"]), -1.0)))
                raise SystemExit(
                    f"  the run stops: Size Nonlinearity is not orthogonal to Size on "
                    f"{int(past.sum())} date(s), so the pair splits one size effect between "
                    f"them -- the largest covariance, {pair['covariance'][t]:+.2e}, on "
                    f"{panel.dates[checked[t]]}, against a rounding bound of "
                    f"{pair['bound'][t]:.1e}"
                )
        else:
            print(f"  Size Nonlinearity, USE3's raw cube, on the same dates: correlated with "
                  f"Size over the {universe_label} {float(np.nanmedian(pair['correlation'])):+.3f} "
                  f"at the median, {pair['correlation'][-1]:+.3f} on {last_day}")

    # Where the pair's risk sits: in the smallest members, which both exposures reach, and in
    # the largest, where USE3's raw cube put most of it. The index is at zero in each.
    if "Size" in style_names:
        pair_names = [name for name in ("Size", "Size Nonlinearity") if name in style_names]
        sized = np.array([code in model.exposures for code in panel.securities])
        sized &= np.isfinite(caps[forecast_row]) & (caps[forecast_row] > 0.0)
        if membership is not None:
            sized &= membership[forecast_row]
        portfolios = size_pair_portfolios(
            model, panel.securities, sized, caps[forecast_row], pair_names
        )
        print(f"\n  The size pair on {panel.dates[forecast_row]}, over the {int(sized.sum())} "
              f"{universe_label} with a forecast: "
              + ("each portfolio's mean exposures, and the pair's risk, sqrt(x' S x) over "
                 "their block of the annualised covariance" if len(pair_names) == 2 else
                 "Size alone, there being no Size Nonlinearity (--size-nonlinearity none) -- "
                 "each portfolio's mean exposure, and its risk, sqrt(x' S x) on Size's "
                 "annualised variance"))
        print(f"  {'':<32}" + "".join(f" {name:>18}" for name in pair_names) + f" {'risk':>8}")
        for label, mean, risk in portfolios:
            print(f"  {label:<32}" + "".join(f" {value:>+18.3f}" for value in mean)
                  + f" {risk:>8.2%}")

    # A restriction to members only removes, so the cross-sections shrink and sampling error
    # rises with them; the sizes are where that shows. The index is counted on the date each
    # regression reads its membership from.
    rows = [("in the regression", np.array(notes.cross_section_sizes))]
    if membership is not None:
        rows = [
            ("index members", np.array(index_sizes)[read_rows]),
            ("of them in the panel", membership[read_rows].sum(axis=1)),
            *rows,
        ]
    print(f"\n  Securities per estimated period, over {notes.n_periods}")
    print(f"  {'':<22} {'min':>6} {'median':>7} {'max':>6}   "
          f"{notes.estimated_dates[0]!s:>10}   {notes.estimated_dates[-1]!s:>10}")
    for label, counts in rows:
        print(f"  {label:<22} {counts.min():>6,} {np.median(counts):>7,.0f} "
              f"{counts.max():>6,}   {counts[0]:>10,}   {counts[-1]:>10,}")
    if membership is not None:
        # What keeps a member out of its period's regression, by the estimator's own test.
        absent = np.array(index_sizes)[read_rows] - membership[read_rows].sum(axis=1)
        design_read = regression_design(
            return_panel, indices, regression_weights=weights, membership=membership,
            lag=notes.lag,
        )
        out = kept_out(design_read, estimated_rows, indices.names)
        held = int(held_in.sum())
        print(f"  member-periods over the {notes.n_periods} periods: "
              f"{int(np.array(index_sizes)[read_rows].sum()):,}; in the regression {held:,}; "
              f"out of it, by why:")
        print(f"    {'not in the panel':<22} {int(absent.sum()):>8,}")
        for reason, cells in out["reasons"].items():
            by_security = cells.sum(axis=0)
            worst = np.argsort(-by_security)[:6]
            shown = ", ".join(f"{panel.securities[j]} {int(by_security[j])}" for j in worst
                              if by_security[j])
            print(f"    {reason:<22} {int(cells.sum()):>8,}" + (f"   {shown}" if shown else ""))
        if out["first_missing"]:
            print("    the first exposure missing: " + ", ".join(
                f"{name} {n:,}" for name, n in sorted(out["first_missing"].items(),
                                                     key=lambda kv: -kv[1])))

    # noise_share cannot see a factor with one member: the regression fits that member
    # exactly, its residual is zero, and the sampling variance looks small. Other Industries
    # read 5.7 percent with one member present in every period. The exactly-1 column can.
    print(f"\n  Over the {notes.n_periods} estimated periods. noise share is blind to a "
          f"one-member factor; the exactly-1 column is not.")
    # Two noise shares (midoffice10 ruling 2). Equal: the mean sampling variance over the
    # variance as estimated. Weighted: the sampling variance weighted as the covariance weights
    # the periods -- the consistent ratio, and what the correction would subtract; in a regime
    # noisier than the sample's average it is the larger.
    print("  noise share, equal: the mean sampling variance over the variance as estimated; "
          "weighted: each period's weighted as the covariance weights its return")
    # The mean is each factor's plain average return over the periods, annualised. Under
    # same-day exposures Size's was inflated: today's cap holds today's return.
    print(f"  {'':<22} {'':>11} {'':>12} {'noise share':>19}   periods with a member present")
    print(f"  {'factor':<22} {'annual vol':>11} {'annual mean':>12} {'equal':>8} "
          f"{'weighted':>10}   {'>=1':>6} {'exactly 1':>10} {'none':>6}")
    volatilities = np.sqrt(np.diag(model.covariance.matrix))
    average = (
        notes.factor_returns.returns.mean(axis=0) * return_panel.frequency.periods_per_year
    )
    for i, fid in enumerate(model.factor_ids):
        share = notes.noise_share[i] if notes.noise_share else float("nan")
        weighted_share = (
            notes.weighted_noise_share[i] if notes.weighted_noise_share else float("nan")
        )
        print(f"  {model.factor_names[fid]:<22} {volatilities[i]:>10.2%} {average[i]:>+11.2%} "
              f"{share:>8.1%} {weighted_share:>10.1%}   {notes.present_periods[i]:>6,} "
              f"{notes.single_member_periods[i]:>10,} {notes.empty_periods[i]:>6,}")
    if notes.sampling_correction:
        # What the correction took off each factor: its vol as estimated, the share of that
        # variance the weighted sampling covariance is, what the subtraction left, and the vol
        # the model carries after the PSD repair. The annual vol above is that last one.
        print(f"\n  The sampling-error correction: the factor covariance less the sampling "
              f"covariance of the factor returns, weighted as the covariance carries each "
              f"period's; {notes.psd_repair.describe() if notes.psd_repair else 'no repair'}")
        print(f"  {'factor':<22} {'as estimated':>13} {'weighted noise':>15} "
              f"{'subtracted':>11} {'corrected':>10}")
        weighted = notes.weighted_noise_share
        for i, fid in enumerate(model.factor_ids):
            left = notes.corrected_variance[i]
            print(f"  {model.factor_names[fid]:<22} "
                  f"{np.sqrt(notes.observed_variance[i]):>12.2%} {weighted[i]:>14.1%} "
                  + (f"{np.sqrt(left):>10.2%}" if left > 0.0 else f"{'<= 0':>10}")
                  + f" {volatilities[i]:>10.2%}")

    specific = np.array(
        [ev.specific.by_issuer[code] for code, ev in model.exposures.items()]
    )
    print(f"\n  specific risk   median {np.median(specific):.2%}   "
          f"p10 {np.percentile(specific, 10):.2%}   p90 {np.percentile(specific, 90):.2%}")
    # Who the V fit learns from, and what the residual spread is -- v_model in every blend
    # weight, so how far each security's own history is trusted. Not the same names: FRC was
    # most of the standard deviation's square and 2 percent of V^2, which is why the spread is
    # measured by the MAD unless --residual-spread sd.
    risk_model = notes.specific_risk_model
    if risk_model is not None:
        def named(shares: tuple[tuple[str, float], ...]) -> str:
            return ", ".join(f"{code} {share:.1%}" for code, share in shares)

        spread = risk_model.notes
        print(f"  EQ 3-14's V fit   largest shares of V^2: {named(spread.v_squared_shares)}")
        print(f"  {'':<17} relative residual spread {risk_model.relative_residual_sd:.3f}, "
              + (f"1.4826 x the MAD of the securities' average relative residuals; their sd "
                 f"{spread.residual_spread_sd:.3f}" if spread.residual_spread == "mad" else
                 f"the sd of the securities' average relative residuals; 1.4826 x their MAD "
                 f"{spread.residual_spread_mad:.3f}"))
        print(f"  {'':<17} largest shares of the sd's square: "
              f"{named(spread.residual_spread_shares)}")
        # What the spread is for: the characteristics' weight in a blended forecast, at the
        # history a security typically has -- in the regression, or every specific return.
        if notes.specific_history == "covered":
            lengths = np.array([n for n in notes.history_lengths.values() if n > 1])
            typical = int(np.median(lengths))
            print(f"  {'':<17} each security's own history is every specific return it has, "
                  f"member or not: {notes.outside_residuals:,} measured outside the "
                  f"regressions; {len(notes.blended_outside)} security(ies) in no regression "
                  f"blended with their own")
        else:
            history = held_in.sum(axis=0)
            typical = int(np.median(history[history > 1]))
        if notes.specific_half_life is not None and notes.history_effective:
            # A weighted history is worth its effective number of observations, and the blend
            # is told that.
            effective = float(np.median(list(notes.history_effective.values())))
            print(f"  {'':<17} each history weighted by a half-life of "
                  f"{notes.specific_half_life:g} trading days; the characteristics' weight in a "
                  f"blended forecast, at the median effective history of {effective:.1f} "
                  f"observations of {typical}: {risk_model.model_weight(effective):.2%}")
        else:
            print(f"  {'':<17} the characteristics' weight in a blended forecast, at the "
                  f"median history of {typical} periods: {risk_model.model_weight(typical):.2%}")

    qualifications = model.qualifications()
    if qualifications:
        print("\nQualifications")
        for note in qualifications:
            print(f"  - {note}")
    if args.bias:
        bias_section(args, panel, market, membership, codes, merges, inputs, model, options,
                     horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
