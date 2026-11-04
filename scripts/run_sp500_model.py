"""Build a factor model from a fetched S&P 500 panel, and say which descriptors earn a place.

    python scripts/run_sp500_model.py data/external/polygon/sp500_daily_*.npz

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
    regression_universe,
)
from praxis.midoffice.model.descriptors import (  # noqa: E402
    SecurityPanel,
    build_risk_indices,
    compute_descriptors,
    risk_index_members,
)
from praxis.midoffice.model.panel import Frequency, ReturnPanel  # noqa: E402
from praxis.midoffice.refdata.industries import (  # noqa: E402
    build_scheme,
    combine,
    industry_exposures,
)
from praxis.midoffice.refdata.polygon import (  # noqa: E402
    PolygonClient,
    PolygonError,
    read_api_key,
)


def load_panel(path: Path) -> tuple[SecurityPanel, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    dates = tuple(dt.date.fromisoformat(str(d)) for d in data["dates"])
    panel = SecurityPanel(
        dates=dates,
        securities=tuple(str(s) for s in data["securities"]),
        prices=data["prices"],
        unadjusted_prices=data["unadjusted_prices"],
        volumes=data["volumes"],
        shares_outstanding=data["shares_outstanding"],
        highs=data["highs"],
        lows=data["lows"],
        frequency=Frequency.DAILY,
    )
    return panel, data["market_returns"]


def screen(standardised: CharacteristicPanel, returns: np.ndarray) -> list[dict]:
    """One univariate cross-sectional regression per descriptor per period.

    Returns are shifted forward by one period against the exposures: a descriptor computed
    from data up to day t can only explain day t+1's return. Regressing it on day t's return
    would use the price that produced it to predict itself.
    """
    rows = []
    for k, name in enumerate(standardised.names):
        exposure = standardised.values[:-1, :, k]
        future = returns[1:, :]
        coefficients, r_squared = [], []
        for t in range(exposure.shape[0]):
            x, y = exposure[t], future[t]
            ok = np.isfinite(x) & np.isfinite(y)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", help="the .npz written by fetch_sp500_panel.py")
    parser.add_argument("--beta-window", type=float, default=36.0,
                        help="months of history for BTSG/HALPHA/SERDP (default 36; the "
                             "USE3 value of 60 needs more days than a 5-year panel has)")
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
    args = parser.parse_args(argv)

    path = Path(args.panel)
    if not path.exists():
        raise SystemExit(f"no such panel: {path}")
    panel, market = load_panel(path)
    returns = panel.returns()

    print(f"Panel\n  {len(panel.dates)} trading days, {panel.n_securities} securities")
    print(f"  {panel.dates[0]} .. {panel.dates[-1]}")
    usable = np.isfinite(returns).sum()
    print(f"  {usable:,} of {returns.size:,} return observations present "
          f"({usable / returns.size:.1%})")

    print("\nDescriptors")
    descriptors, skipped = compute_descriptors(
        panel,
        market_returns=market,
        beta_window_months=args.beta_window,
    )
    print(f"  computed {len(descriptors.names)}: {', '.join(descriptors.names)}")
    for name, reason in sorted(skipped.items()):
        print(f"  - {name}: {reason}")

    caps = panel.market_capitalisation()
    standardised = descriptors.standardise(caps)
    if standardised.winsorised:
        print(f"  {standardised.winsorised:,} raw values winsorised before standardising")

    print("\nWhich descriptors carry signal")
    print("  Ranked by the cross-sectional return variance each explains on its own.")
    print(f"  {'descriptor':<10} {'cover':>6} {'periods':>8} {'mean R2':>9} "
          f"{'factor vol':>11} {'t-stat':>8}")
    rows = screen(standardised, returns)
    scored = [r for r in rows if "mean_r2" in r]
    thin = [r for r in rows if "mean_r2" not in r]
    for row in sorted(scored, key=lambda r: -r["mean_r2"]):
        print(f"  {row['name']:<10} {row['coverage']:>5.0%} {row['periods']:>8,} "
              f"{row['mean_r2']:>8.2%} {row['factor_vol']:>10.2%} {row['t_stat']:>8.2f}")
    for row in thin:
        print(f"  {row['name']:<10} {'-':>6} {row['periods']:>8,}   too few usable periods")

    print("\nRisk indices")
    available = [
        name
        for name in ("Volatility", "Momentum", "Size", "Size Nonlinearity",
                     "Trading Activity")
        if any(m in standardised.names for m in risk_index_members(name))
    ]
    indices = build_risk_indices(standardised, cap_weights=caps, indices=available)
    for name in indices.names:
        members = [m for m in risk_index_members(name) if m in standardised.names]
        print(f"  {name:<20} {', '.join(members)}")

    # "Regression Weighting Scheme: Square root of capitalisation" appears on every model
    # sheet in the Equity Risk Model Reference Guide. Equal weighting lets the smallest,
    # noisiest names set the factor returns -- there are far more of them, and in a
    # cap-weighted world almost none of the money. Needed before the industries, because
    # which securities a regression sees depends on the weights.
    weights = None if args.equal_weight else np.sqrt(np.where(caps > 0.0, caps, np.nan))
    return_panel = ReturnPanel(panel.dates, panel.securities, returns, Frequency.DAILY)

    scheme = None
    if args.industries:
        print("\nIndustries")
        try:
            client = PolygonClient(
                api_key=read_api_key(args.env), cache_dir=Path(args.cache)
            )
        except PolygonError as error:
            raise SystemExit(f"  industries need the reference cache: {error}") from None
        codes: dict[str, str | None] = {}
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
        universe = regression_universe(return_panel, indices, regression_weights=weights)
        print(f"  {len(universe)} of {panel.n_securities} securities are in at least one "
              f"estimated period's regression; every count below is over them")
        scheme = build_scheme(codes, min_members=args.min_industry, universe=universe)
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
        # Industries sum to one per security, which IS the unit column. Fitting a market
        # factor alongside them makes the design exactly singular, not nearly so.
        fit_market=scheme is None,
        covariance_estimator=NeweyWestCovariance(
            lags=args.newey_west_lags, half_life=args.half_life
        ),
    )
    notes = model.notes
    print(f"  {model.describe()}")
    print(f"  mean cross-sectional R-squared {notes.mean_r_squared:.3f}")
    print(f"  {len(notes.rank_deficient_periods)} collinear period(s) of {notes.n_periods} "
          f"estimated; {sum(notes.empty_periods)} factor-period(s) left out for having no "
          f"member present")
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

    # noise_share cannot see a factor with one member: the regression fits that member
    # exactly, its residual is zero, and the sampling variance looks small. Other Industries
    # read 5.7 percent with one member present in every period. The exactly-1 column can.
    print(f"\n  Over the {notes.n_periods} estimated periods. noise share is blind to a "
          f"one-member factor; the exactly-1 column is not.")
    print(f"  {'':<22} {'':>11} {'':>12}   periods with a member present")
    print(f"  {'factor':<22} {'annual vol':>11} {'noise share':>12}   "
          f"{'>=1':>6} {'exactly 1':>10} {'none':>6}")
    volatilities = np.sqrt(np.diag(model.covariance.matrix))
    for i, fid in enumerate(model.factor_ids):
        share = notes.noise_share[i] if notes.noise_share else float("nan")
        print(f"  {model.factor_names[fid]:<22} {volatilities[i]:>10.2%} {share:>11.1%}   "
              f"{notes.present_periods[i]:>6,} {notes.single_member_periods[i]:>10,} "
              f"{notes.empty_periods[i]:>6,}")

    specific = np.array(
        [ev.specific.by_issuer[code] for code, ev in model.exposures.items()]
    )
    print(f"\n  specific risk   median {np.median(specific):.2%}   "
          f"p10 {np.percentile(specific, 10):.2%}   p90 {np.percentile(specific, 90):.2%}")

    qualifications = model.qualifications()
    if qualifications:
        print("\nQualifications")
        for note in qualifications:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
