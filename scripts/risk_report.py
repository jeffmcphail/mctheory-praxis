"""The risk report (midoffice20): a sample book against the cap-weighted index, from the model
store and the risk core, with its checks.

    python scripts/risk_report.py --portfolio LOWVOL --date 2025-06-30 --model v1

The book (LOWVOL, MOM40 or EQ50, `build_sample_portfolios.py`) and the index are valued at the
date's close and measured by the engine's own entry point, `engine.compute_risk`, on the store's
forecast: the four dates resolved by `asof.AsOf` -- a date the store does not hold answers with
the latest before it, and the report says so -- the exposures and specific risk of the exposure
date, the covariance of the model date, the factor groups (industries, styles) as a
`risk.factors.FactorTree`.

It prints: the index's coverage -- its members without a forecast are left out, as the bias
test leaves them, and the report gives their share of the members' capitalisation on the date
(a member with no capitalisation that day counts at none); total risk, split into factor and
specific; tracking error against the index; the decomposition by factor group, total risk's and
tracking error's, the largest factors in each; the ten largest position contributions to total
risk and to tracking error; one-day VaR and expected shortfall at 95% and 99% -- parametric from
the covariance (`risk.tail`, normal), and historical from the factor returns: the book's factor
exposures as of the date applied to each of the last 252 days' estimated factor returns through
it (`--var-days` sets another window), so the historical figures are the factor part's, set
beside the parametric factor part's.

And its checks:
1. factor and specific variance sum to the total -- the engine's, against the variance computed
   again from the store's arrays as w'(X F X' + D)w;
2. the position contributions sum to total risk and to tracking error, and the factor groups'
   to each (`contributions.check_sum`, `Decomposition.check`);
3. on a test date, LOWVOL's forecast under v1 equals the bias run's stored forecast of the
   long-only minimum-variance portfolio (D19's, kept beside v1 in the store): the report's own
   date where it is one -- a date of the bias test whose universe is the test's own, every
   member with a return the next day -- or else the nearest before it.
Exit status 1 if a check fails; 2, before any of the report prints, if the store holds fewer
days of factor returns through the date than the historical VaR's window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from praxis.midoffice import store  # noqa: E402
from praxis.midoffice.asof import AsOf, DateCalendar  # noqa: E402
from praxis.midoffice.engine import compute_risk, positions_from  # noqa: E402
from praxis.midoffice.risk.basis import FactorCovariance  # noqa: E402
from praxis.midoffice.risk.decomposition import group_decomposition  # noqa: E402
from praxis.midoffice.risk.exposures import ExposureVector, SpecificRisk  # noqa: E402
from praxis.midoffice.risk.factors import FactorTree  # noqa: E402
from praxis.midoffice.risk.portfolio import Portfolio  # noqa: E402
from praxis.midoffice.risk.tail import TailSpec, tail_risk  # noqa: E402
from praxis.midoffice.value.valuation import PortfolioValue, PositionValue  # noqa: E402

BENCHMARK = "INDEX"
BOOKS = ("LOWVOL", "MOM40", "EQ50")
GROUPS = ("Industries", "Styles")
ROOT_NODE = "Factors"
CONFIDENCES = (0.95, 0.99)
TEST_PORTFOLIO = "minimum variance, long-only"
INDEX_PORTFOLIO = "the index, cap-weighted"
VAR_DAYS = 252
"""The historical VaR's window, in days of factor returns through the date (Jeff's mid-piece
ruling 2 on midoffice20): a year, the Basel minimum. Every forecast date has at least a year of
factor returns behind it -- the first, 2024-02-08, exactly 252 -- so every date's VaR rests on
the same length of history. `--var-days` sets another."""
CONTRIBUTION_TOLERANCE = 1e-10
TOLERANCE = 1e-12
"""Check 3's relative tolerance: the engine sums a book's exposures position by position and its
specific variance issuer by issuer, the bias test by matrix products -- the same numbers in
another order, so they agree to rounding (about 1e-15), not to the bit."""


class Measured:
    """One book against the index on one date: the engine's result and what the report reads."""

    def __init__(self, model: store.StoredModel, holdings: store.StoredHoldings, book: str,
                 request: dt.date | str) -> None:
        calendars = {
            "model": DateCalendar.of("model", model.dates),
            "exposure": DateCalendar.of("exposure", model.exposure_dates),
            "market": DateCalendar.of("market", holdings.dates),
            "security": DateCalendar.of("security", holdings.dates),
        }
        self.as_of = AsOf.resolve(calendars, model=request, exposure=request, market=request,
                                  security=request)
        self.book = book
        market_day = self.as_of.market.actual
        self.forecast = model.forecast(self.as_of.exposure.actual)
        self.covariance_matrix = (self.forecast.covariance
                                  if self.as_of.model.actual == self.as_of.exposure.actual
                                  else model.covariance(self.as_of.model.actual))
        self.held = holdings.holdings(book, market_day)
        self.index = holdings.holdings(BENCHMARK, market_day)
        self.factor_ids = tuple(range(1, len(model.factor_names) + 1))
        self.names = dict(zip(self.factor_ids, model.factor_names))
        groups = {g["name"]: g["group"] for g in model.manifest["factors"]}
        self.group_of = {fid: ("Industries" if groups[name] == "industry" else "Styles")
                         for fid, name in self.names.items()}
        covered = set(self.forecast.securities)
        price = dict(zip(self.held.securities, self.held.price))
        price.update(zip(self.index.securities, self.index.price))
        mine = dict(zip(self.held.securities, self.held.quantity))
        theirs = dict(zip(self.index.securities, self.index.quantity))
        codes = list(dict.fromkeys(self.held.securities + self.index.securities))
        self.value = PortfolioValue(
            name=book, benchmark_name=BENCHMARK, stated_nav=self.held.value,
            positions=tuple(PositionValue(security_code=c, clean_price=float(price[c]),
                                          quantity=float(mine.get(c, 0.0)),
                                          benchmark_quantity=float(theirs.get(c, 0.0)),
                                          covered=c in covered) for c in codes))
        self.exposures = {
            code: ExposureVector({fid: float(v) for fid, v in zip(self.factor_ids, row) if v},
                                 SpecificRisk.for_security(code, float(s)))
            for code, row, s in zip(self.forecast.securities, self.forecast.exposures,
                                    self.forecast.specific_risk)}
        self.tree = FactorTree.build(
            [(ROOT_NODE, g) for g in GROUPS],
            {g: [fid for fid, grp in self.group_of.items() if grp == g] for g in GROUPS},
            root=ROOT_NODE)
        self.result = compute_risk(
            self.as_of, self.value, self.exposures,
            FactorCovariance(self.factor_ids, self.covariance_matrix), tree=self.tree,
            factor_names=self.names,
            tails=tuple(TailSpec(confidence=c, horizon_days=1.0) for c in CONFIDENCES))
        positions, _ = positions_from(self.value, self.exposures)
        self.portfolio = Portfolio.of(name=book, positions=positions, benchmark_name=BENCHMARK)
        self.tickers = dict(zip(self.held.securities, self.held.tickers))
        self.tickers.update(zip(self.index.securities, self.index.tickers))
        self.periods = float(model.manifest["units"]["periods_per_year"])

    def risk_of_index(self) -> float:
        return self.result.risk.benchmark.total_risk

    def dense(self, vector: ExposureVector) -> np.ndarray:
        return np.array([vector.exposures.get(fid, 0.0) for fid in self.factor_ids])


def historical(measured: Measured, model: store.StoredModel, vector: ExposureVector,
               days: int) -> tuple[dict[float, tuple[float, float]], int, dt.date, dt.date]:
    """VaR and ES of the factor part at each confidence, as fractions of the book: its factor
    exposures as of the date on each of the last `days` days' factor returns through the model
    date. The k worst of the N days, k = ceil((1 - c) N): VaR the k-th worst loss, ES their mean.
    The window is fixed, so a date with fewer days behind it raises ValueError rather than
    resting on a shorter history."""
    dates, returns = model.factor_returns(through=measured.as_of.model.actual)
    if len(dates) < days:
        raise ValueError(f"the store holds {len(dates)} days of factor returns through "
                         f"{measured.as_of.model.actual}, fewer than the historical VaR's window "
                         f"of {days}")
    dates, returns = dates[-days:], returns[-days:]
    losses = np.sort(-(returns @ measured.dense(vector)))[::-1]
    out = {}
    for c in CONFIDENCES:
        k = math.ceil(round((1.0 - c) * losses.size, 9))
        out[c] = (float(losses[k - 1]), float(losses[:k].mean()))
    return out, losses.size, dates[0], dates[-1]


def test_date(model, holdings, request: dt.date) -> tuple[dt.date | None, str]:
    """Check 3's date: the report's where it is a bias-test date whose universe is the test's
    own (every member with a return the next day), else the nearest such date before it, else
    the first after."""
    reference = model.bias_reference()
    facts = holdings.all_facts()
    good = sorted(d for d in reference if facts.get(d, {}).get("next_day_complete"))
    if not good:
        return None, "the store keeps no bias run's forecasts beside v1"
    if request in good:
        return request, "the report's own date"
    before = [d for d in good if d < request]
    if before:
        why = ("the report's date is not one of the bias test's" if request not in reference
               else "on the report's date a member of the universe has no return the next day, "
                    "so the bias test's universe is not the book's")
        return before[-1], f"the nearest before the report's date: {why}"
    return good[0], "the first: the report's date is before the bias test's"


def pct(x: float, places: int = 2) -> str:
    return f"{x * 100:.{places}f}%"


def money(x: float) -> str:
    return f"${x / 1e3:,.0f}k"


def report(args) -> int:
    began = time.perf_counter()
    root = Path(args.store)
    model = store.open_model(args.model, root)
    holdings = store.StoredHoldings(root)
    request = args.date
    m = Measured(model, holdings, args.portfolio, request)
    try:
        hist, n_days, first, last = historical(m, model, m.portfolio.exposure, args.var_days)
    except ValueError as error:
        print(f"risk_report: {error}; --var-days sets a shorter one", file=sys.stderr)
        return 2
    r = m.result
    risk = r.risk
    value = m.held.value
    facts = holdings.facts(m.as_of.market.actual)
    print(f"{args.portfolio} against the cap-weighted index, model {args.model}, "
          f"as of {m.as_of.model.actual}")
    print(f"  dates: {m.as_of.describe()}"
          + ("" if m.as_of.is_exact else f" -- asked for {request}: "
             + "; ".join(m.as_of.qualifications())))
    print(f"  {args.portfolio}: {RULES.get(args.portfolio, '')}; ${value / 1e6:,.2f}m in "
          f"{len(m.held.securities)} holdings")
    print(f"  the index: {len(m.index.securities)} of the date's {facts['members']} members, "
          f"cap-weighted; {facts['left_out']} left out, with no "
          f"{holdings.manifest.get('model_id', 'v1')} forecast or no capitalisation: "
          f"{pct(facts['left_out_cap_share'], 3)} of the members' capitalisation")
    print(f"  model {args.model}: {model.manifest['description'].split(':')[0]}; the "
          f"covariance's regime multipliers as of the date "
          + ", ".join(f"{k} {v:.4f}" for k, v in model.multipliers(m.as_of.model.actual).items()))

    print("\nRisk, annualised             book      index     active")
    sides = (risk.portfolio, risk.benchmark, risk.active)
    for label, attr in (("total", "total_risk"), ("factor", "common_factor_risk"),
                        ("specific", "specific_risk")):
        print(f"  {label:<24} " + "  ".join(f"{pct(getattr(s, attr)):>8}" for s in sides))
    print(f"  tracking error {pct(r.tracking_error)}; beta to the index {risk.beta:.3f}, "
          f"correlation {risk.correlation:.3f}")

    print("\nBy factor group: contribution to risk (share of it)")
    print(f"  {'':<22} {'total risk':>20}   {'tracking error':>20}")
    total_groups = r.group_decompositions["portfolio"]
    active_groups = r.group_decompositions["active"]
    for row_t, row_a in zip(total_groups.rows, active_groups.rows):
        print(f"  {row_t.name:<22} {pct(row_t.contribution, 3):>9} "
              f"({pct(row_t.litterman_proportion, 1):>6})   {pct(row_a.contribution, 3):>9} "
              f"({pct(row_a.litterman_proportion, 1):>6})")
    for label, vector in (("total risk", m.portfolio.exposure),
                          ("tracking error", m.portfolio.active_exposure)):
        parts = []
        for group in GROUPS:
            inner = group_decomposition(vector, r.basis, m.tree, parent=group, names=m.names)
            top = sorted(inner.rows, key=lambda row: -abs(row.contribution))[:3]
            parts.append(f"{group.lower()} " + ", ".join(
                f"{row.name} {pct(row.contribution, 3)}" for row in top))
        print(f"  the largest in {label}: " + "; ".join(parts))

    for label, side, risk_number in (("total risk", "portfolio", r.total_risk),
                                     ("tracking error", "active", r.tracking_error)):
        rows = sorted(r.position_contributions[side], key=lambda c: -abs(c.total))[:10]
        weight_label = "weight" if side == "portfolio" else "active wt"
        print(f"\nThe ten largest contributions to {label} ({pct(risk_number)})")
        print(f"  {'':<7} {weight_label:>9} {'contribution':>13} {'share':>8} {'factor':>9} "
              f"{'specific':>9}")
        for c in rows:
            print(f"  {m.tickers.get(c.security_code, c.security_code):<7} "
                  f"{pct(c.weight):>9} {pct(c.total, 3):>13} {pct(c.percent, 1):>8} "
                  f"{pct(c.common_factor, 3):>9} {pct(c.specific, 3):>9}")

    print(f"\nOne-day VaR and expected shortfall, on ${value / 1e6:,.0f}m      "
          + "   ".join(f"{c:.0%} VaR    {c:.0%} ES" for c in CONFIDENCES))
    lines = [("parametric, total (normal)", {c: (t.value_at_risk, t.expected_shortfall)
                                             for c, t in zip(CONFIDENCES, r.tails["portfolio"])}),
             ("parametric, factor part", {
                 c: (tail_risk(risk.portfolio.common_factor_risk, s).value_at_risk,
                     tail_risk(risk.portfolio.common_factor_risk, s).expected_shortfall)
                 for c, s in zip(CONFIDENCES, (t.spec for t in r.tails["portfolio"]))}),
             ("historical, factor part", hist)]
    for label, by in lines:
        print(f"  {label:<38} " + "   ".join(f"{money(by[c][0] * value):>8} "
                                             f"{money(by[c][1] * value):>8}" for c in CONFIDENCES))
    print(f"  historical: the book's factor exposures as of the date on each of the last {n_days} "
          f"days' factor returns through it, {first} to {last} (--var-days); the specific part "
          f"is not in them")

    # The checks.
    failures = []
    print("\nChecks")
    x = {c: row for c, row in zip(m.forecast.securities, m.forecast.exposures)}
    s = dict(zip(m.forecast.securities, m.forecast.specific_risk))
    weights = {"portfolio": m.value.weights(), "benchmark": m.value.benchmark_weights()}
    weights["active"] = {c: weights["portfolio"].get(c, 0.0) - weights["benchmark"].get(c, 0.0)
                         for c in set(weights["portfolio"]) | set(weights["benchmark"])}
    worst = 0.0
    for side, decomposition in zip(("portfolio", "benchmark", "active"), sides):
        w = weights[side]
        exposure = sum((w[c] * x[c] for c in w if c in x and w[c]), np.zeros(len(m.factor_ids)))
        direct = float(exposure @ m.covariance_matrix @ exposure
                       + sum((w[c] * s[c]) ** 2 for c in w if c in s))
        parts = decomposition.common_factor_variance + decomposition.specific_variance
        worst = max(worst, abs(parts / direct - 1.0), abs(decomposition.total_variance / parts
                                                          - 1.0))
    ok = worst < TOLERANCE
    failures += [] if ok else ["1"]
    print(f"  1. factor and specific variance sum to the total, the book's, the index's and the "
          f"active book's, against w'(X F X' + D)w from the store's arrays: the largest relative "
          f"difference {worst:.1e} -- {'passes' if ok else 'FAILS'}")
    residuals = [sum(c.total for c in r.position_contributions["portfolio"]) - r.total_risk,
                 sum(c.total for c in r.position_contributions["active"]) - r.tracking_error,
                 r.group_decompositions["portfolio"].residual(),
                 r.group_decompositions["active"].residual()]
    ok = all(abs(v) <= CONTRIBUTION_TOLERANCE for v in residuals)
    failures += [] if ok else ["2"]
    print(f"  2. the position contributions sum to total risk (residual {residuals[0]:+.1e}) and "
          f"to tracking error ({residuals[1]:+.1e}); the factor groups and specific to each "
          f"({residuals[2]:+.1e}, {residuals[3]:+.1e}) -- "
          + ("passes" if ok else "FAILS") + f" (to {CONTRIBUTION_TOLERANCE:.0e}, the risk "
          f"core's `check_sum`)")
    day, why = test_date(model if args.model == "v1" else store.open_model("v1", root), holdings,
                         m.as_of.model.actual)
    if day is None:
        print(f"  3. not made: {why}")
        failures.append("3")
    else:
        v1 = model if args.model == "v1" else store.open_model("v1", root)
        lowvol = Measured(v1, holdings, "LOWVOL", day)
        daily = lowvol.result.total_risk / math.sqrt(lowvol.periods)
        stored = v1.bias_reference()[day][TEST_PORTFOLIO]
        difference = daily / stored - 1.0
        ok = abs(difference) < TOLERANCE
        failures += [] if ok else ["3"]
        print(f"  3. on {day} ({why}): LOWVOL's forecast under v1, {pct(daily, 6)} a day, "
              f"against the bias run's stored forecast of the {TEST_PORTFOLIO} portfolio, "
              f"{pct(stored, 6)}: relative difference {difference:+.1e} -- "
              + ("passes" if ok else "FAILS")
              + f" (to {TOLERANCE:.0e}: the same sums in another order)")
        index_daily = lowvol.risk_of_index() / math.sqrt(lowvol.periods)
        index_stored = v1.bias_reference()[day][INDEX_PORTFOLIO]
        print(f"     and beside it the index, {pct(index_daily, 6)} a day against the bias run's "
              f"{INDEX_PORTFOLIO!r}, {pct(index_stored, 6)}: "
              f"{index_daily / index_stored - 1.0:+.1e}")
    notes = r.qualifications()
    if notes:
        print("\nQualifications")
        for note in notes:
            print(f"  - {note}")
    print(f"\n{'All checks pass' if not failures else 'FAILED: check ' + ', '.join(failures)}; "
          f"{time.perf_counter() - began:.2f} s")
    return 1 if failures else 0


RULES = {
    "LOWVOL": "the long-only minimum-variance portfolio from v1's forecast as of the date",
    "MOM40": "the 40 members with the highest Momentum exposure, cap-weighted",
    "EQ50": "the 50 largest members, equally weighted",
}


def window(text: str) -> int:
    days = int(text)
    if days < 1:
        raise argparse.ArgumentTypeError(f"a window of at least one day, not {text}")
    return days


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--portfolio", required=True, choices=BOOKS)
    parser.add_argument("--date", default="latest",
                        help="the date, YYYY-MM-DD (the latest the store holds on or before it), "
                             "or latest")
    parser.add_argument("--model", default="v1", help="the model id in the store")
    parser.add_argument("--store", default=str(store.STORE_ROOT), help="the store's root")
    parser.add_argument("--var-days", type=window, default=VAR_DAYS,
                        help=f"the historical VaR's window: the last N days of factor returns "
                             f"through the date, {VAR_DAYS} by default (a year; every forecast "
                             f"date has at least that many behind it)")
    args = parser.parse_args(argv)
    if args.date != "latest":
        args.date = dt.date.fromisoformat(args.date)
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
