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
from praxis.midoffice.refdata import sp500  # noqa: E402
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
    parser.add_argument("--membership", action="store_true",
                        help="hold each period's regression to that day's index members, from "
                             "the newest snapshot in --snapshots; without it every panel "
                             "security with data is regressed")
    parser.add_argument("--snapshots", default="data/external/sp500")
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
        try:
            membership, index_sizes = index_membership(
                snapshot, panel.dates, panel.securities
            )
        except sp500.CoverageError as error:
            raise SystemExit(f"  the run stops: {error}") from None
        print(f"  each period's regression holds only that day's members: "
              f"{int(membership[0].sum())} of the panel's {panel.n_securities} on "
              f"{panel.dates[0]}, {int(membership[-1].sum())} on {panel.dates[-1]}")
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
        universe = regression_universe(
            return_panel, indices, regression_weights=weights, membership=membership
        )
        held = "" if membership is None else " while an index member"
        print(f"  {len(universe)} of {panel.n_securities} securities are in at least one "
              f"estimated period's regression{held}; every count below is over them")
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
        membership=membership,
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

    # A restriction to members only removes, so the cross-sections shrink and sampling error
    # rises with them; the sizes are where that shows.
    rows = [("in the regression", np.array(notes.cross_section_sizes))]
    if membership is not None:
        at = {d: i for i, d in enumerate(panel.dates)}
        held = [at[d] for d in notes.estimated_dates]
        rows = [
            ("index members", np.array(index_sizes)[held]),
            ("of them in the panel", membership[held].sum(axis=1)),
            *rows,
        ]
    print(f"\n  Securities per estimated period, over {notes.n_periods}")
    print(f"  {'':<22} {'min':>6} {'median':>7} {'max':>6}   "
          f"{notes.estimated_dates[0]!s:>10}   {notes.estimated_dates[-1]!s:>10}")
    for label, counts in rows:
        print(f"  {label:<22} {counts.min():>6,} {np.median(counts):>7,.0f} "
              f"{counts.max():>6,}   {counts[0]:>10,}   {counts[-1]:>10,}")

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
    # Who the V fit learns from, and who sets the residual spread -- v_model in every blend
    # weight, so how far each security's own history is trusted. Not the same names: FRC was
    # most of the spread and 2 percent of V^2.
    risk_model = notes.specific_risk_model
    if risk_model is not None:
        def named(shares: tuple[tuple[str, float], ...]) -> str:
            return ", ".join(f"{code} {share:.1%}" for code, share in shares)

        print(f"  EQ 3-14's V fit   largest shares of V^2: "
              f"{named(risk_model.notes.v_squared_shares)}")
        print(f"  {'':<17} relative residual sd {risk_model.relative_residual_sd:.3f}, "
              f"largest shares of its square: "
              f"{named(risk_model.notes.residual_spread_shares)}")

    qualifications = model.qualifications()
    if qualifications:
        print("\nQualifications")
        for note in qualifications:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
