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
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from praxis.midoffice.model.characteristics import CharacteristicPanel  # noqa: E402
from praxis.midoffice.model.covariance import (  # noqa: E402
    HorizonCovariance,
    NeweyWestCovariance,
)
from praxis.midoffice.model.crosssectional import (  # noqa: E402
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

DEFAULT_PANEL = "data/external/polygon/sp500_securities_2022-01-03_2026-11-03_listed.npz"
"""The panel on securities: its share counts repaired by midoffice08's rulings, a sample its own
weighted count confirms spared and every span ended where it leaves an exchange by midoffice09's.
The run's panel when none is given."""

UNBLENDED = {
    "SERDP": "its residual series is kept for the last window alone, so it has a value on the "
             "last date and no other; no estimated period reads it, and blended it would enter "
             "the forecast exposures alone",
}
"""Descriptors computed, screened and reported, and kept out of every blend, with why. A
forecast exposure must be built from the descriptors the factor was estimated on. Computing
SERDP historically waits for a reason to; it has never had a usable period in the screen."""


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
    parser.add_argument("--newey-west-lags", type=int, default=10)
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
    args = parser.parse_args(argv)
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

    print("\nDescriptors")
    descriptors, skipped = compute_descriptors(
        panel,
        market_returns=market,
        beta_window_months=beta_months,
        membership=membership,
        min_coverage=args.min_coverage,
        rstr_skip=args.rstr_skip,
        beta_half_life=args.beta_half_life,
    )
    print(f"  computed {len(descriptors.names)}: {', '.join(descriptors.names)}")
    for name, reason in sorted(skipped.items()):
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

    caps = panel.market_capitalisation()
    # Over the estimation universe: the members set each day's clip bounds, cap-weighted mean
    # and equal-weighted dispersion, and every other security with a value is put on their
    # scale. Barra's estimation universe against its coverage universe -- an ex-member still
    # gets an exposure, but no say in what an exposure of one means.
    standardised = descriptors.standardise(caps, membership=membership)
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

    # "Regression Weighting Scheme: Square root of capitalisation" appears on every model
    # sheet in the Equity Risk Model Reference Guide. Equal weighting lets the smallest,
    # noisiest names set the factor returns -- there are far more of them, and in a
    # cap-weighted world almost none of the money. Needed before the risk indices, because
    # Size Nonlinearity is orthogonalised against Size by them, and before the industries,
    # because which securities a regression sees depends on them.
    weights = None if args.equal_weight else np.sqrt(np.where(caps > 0.0, caps, np.nan))
    return_panel = ReturnPanel(panel.dates, panel.securities, returns, Frequency.DAILY)

    print("\nRisk indices")
    unblended = dict(UNBLENDED)
    use4_size = args.size_nonlinearity == "use4"
    if use4_size:
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
    indices = build_risk_indices(
        blendable, cap_weights=caps, indices=available, membership=membership,
        size_nonlinearity=args.size_nonlinearity, regression_weights=weights,
    )
    style_names = indices.names
    for name in indices.names:
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

    # One definition per factor across the sample. blend_to_index averages whichever of an
    # index's descriptors a security has, so a blend changes composition where one starts or
    # stops having values, and a factor estimated on one blend and forecast on another is two
    # factors under one name. So a period is estimated only if every index's blend in it, over
    # the securities its regression holds, has exactly the descriptors the forecast's has.
    # Worked out before the industries, whose universe is the estimated periods'.
    design = regression_design(
        return_panel, indices, regression_weights=weights, membership=membership
    )
    forecast_row = len(panel.dates) - 1
    complete = np.isfinite(indices.values[forecast_row]).all(axis=1)[None, :]
    period_rows = np.arange(design.lag, len(panel.dates))
    unlike: dict[int, list[str]] = {}
    for name in style_names:
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

    scheme = None
    if args.merge_industry and not args.industries:
        raise SystemExit("  --merge-industry needs --industries: there is no scheme to merge in")
    if args.industries:
        print("\nIndustries")
        codes: dict[str, str | None] = {}
        if on_securities:
            # Each security's own latest record with a code, under its own tickers inside its
            # spans: a ticker's record can be another security's.
            codes = dict(extras["sic_codes"])
            missing = sorted(code for code, sic in codes.items() if not sic)
            print(f"  each security's SIC code from its own latest record with one: "
                  f"{len(codes) - len(missing)} of {len(codes)}"
                  + (f"; none for {', '.join(missing)}" if missing else ""))
        else:
            try:
                client = PolygonClient(
                    api_key=read_api_key(args.env), cache_dir=Path(args.cache)
                )
            except PolygonError as error:
                raise SystemExit(f"  industries need the reference cache: {error}") from None
            carried: list[str] = []
            for ticker in panel.securities:
                code, on = client.sic_code_any(ticker, panel.dates[-1])
                codes[ticker] = code
                if code and on != panel.dates[-1]:
                    carried.append(ticker)
            if carried:
                # A company acquired mid-window has no record at the end of it, but its
                # industry while it traded is exactly what the regression needs.
                print(f"  {len(carried)} classification(s) carried from an earlier date "
                      f"(delisted inside the window): {', '.join(sorted(carried)[:5])}")
        # Counted over the securities in at least one estimated period's regression, not
        # over the panel. A member that never enters a regression contributes nothing to its
        # factor, and counting them let a four-member Unclassified -- one member present in
        # most periods -- past a floor of five.
        universe = regression_universe(
            return_panel, indices, regression_weights=weights, membership=membership,
            periods=periods,
        )
        held = "" if membership is None else " while an index member"
        print(f"  {len(universe)} of {panel.n_securities} securities are in at least one "
              f"estimated period's regression{held}; every count below is over them")
        scheme = build_scheme(codes, min_members=args.min_industry, universe=universe)
        if args.merge_industry:
            merges: dict[str, str] = {}
            for spec in args.merge_industry:
                source, sep, target = spec.partition("=")
                if not sep or not source.strip() or not target.strip():
                    raise SystemExit(f"  --merge-industry takes A=B, got {spec!r}")
                merges[source.strip()] = target.strip()
            before = scheme.counts()
            try:
                scheme = merge_industries(scheme, merges)
            except ValueError as error:
                raise SystemExit(f"  --merge-industry: {error}") from None
            after = scheme.counts()
            for source, target in sorted(scheme.folded.items()):
                print(f"  {source} ({before[source]}) folded into {target}, which holds "
                      f"{after[target]}")
        print(f"  {scheme.describe()}")
        for name, count in sorted(scheme.counts().items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {name:<24} {count:>4}")
        for note in scheme.qualifications():
            print(f"  - {note}")
        indices = combine(
            indices,
            industry_exposures(
                scheme, panel.dates, panel.securities, frequency=Frequency.DAILY
            ),
        )

    print("\nModel")
    print(f"  regression weights: "
          f"{'equal' if args.equal_weight else 'square root of capitalisation'}")
    model = estimate_cross_sectional_model(
        return_panel,
        indices,
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
        covariance_estimator=NeweyWestCovariance(
            lags=args.newey_west_lags, half_life=args.half_life
        ),
        sampling_correction=args.sampling_correction,
    )
    notes = model.notes
    print(f"  {model.describe()}")
    # Barra's convention, and the descriptor screen's: everything that defines period t's
    # regression is fixed the day before, so today's cap -- which holds today's return -- can
    # make no stock look as though it outperformed today.
    print(f"  each period regresses day t's returns on the exposures, weights and membership "
          f"of {'day t' if notes.lag == 0 else f'{notes.lag} trading day(s) earlier'}; the "
          f"forecast exposures are {panel.dates[-1]}'s")
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
    # The mean is each factor's plain average return over the periods, annualised. Under
    # same-day exposures Size's was inflated: today's cap holds today's return.
    print(f"  {'':<22} {'':>11} {'':>12} {'':>12}   periods with a member present")
    print(f"  {'factor':<22} {'annual vol':>11} {'annual mean':>12} {'noise share':>12}   "
          f"{'>=1':>6} {'exactly 1':>10} {'none':>6}")
    volatilities = np.sqrt(np.diag(model.covariance.matrix))
    average = (
        notes.factor_returns.returns.mean(axis=0) * return_panel.frequency.periods_per_year
    )
    for i, fid in enumerate(model.factor_ids):
        share = notes.noise_share[i] if notes.noise_share else float("nan")
        print(f"  {model.factor_names[fid]:<22} {volatilities[i]:>10.2%} {average[i]:>+11.2%} "
              f"{share:>11.1%}   {notes.present_periods[i]:>6,} "
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
        print(f"  {'':<17} the characteristics' weight in a blended forecast, at the median "
              f"history of {typical} periods: {risk_model.model_weight(typical):.2%}")

    qualifications = model.qualifications()
    if qualifications:
        print("\nQualifications")
        for note in qualifications:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
