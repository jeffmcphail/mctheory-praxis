"""Build the sample portfolios (midoffice20): three $100m books and the cap-weighted index on
every date of v1's store, each by its rule, so every date has holdings.

    python scripts/build_sample_portfolios.py --bias-run claude/scratch/midoffice19/bias/D19.npz

Each date's universe is what was known at its close: the date's index members with a forecast
in v1's store and a positive capitalisation (the raw close times the shares outstanding, the
model's). From it:

- INDEX, the benchmark: the universe cap-weighted -- the bias test's cap-weighted index;
- LOWVOL: the long-only minimum-variance portfolio from v1's forecast as of the date
  (`bias.minimum_variance`, the bias test's own solver, on the store's forecast);
- MOM40: the 40 members with the highest Momentum exposure as of the date, cap-weighted;
- EQ50: the 50 largest members by capitalisation, equally weighted.

Each book is $100m at the date's close: a holding's quantity is its dollars over its raw close,
in fractional shares, so its weight is the rule's to rounding. The bias test leaves out of each
date's portfolios a member with no return the next day, which a book formed at the close cannot
know; where every member of the universe has one, the universe is the bias test's own, and
LOWVOL is its long-only minimum-variance portfolio (`dates.parquet`'s `next_day_complete`).

Writes `<store>/portfolios/`: `holdings.parquet` (portfolio, holdings_date, security,
security_key, ticker, quantity, price, market_value, weight; one row group a date), `dates.parquet`
(each date's universe, its capitalisation left out, LOWVOL's names and its solver's KKT
violation, `next_day_complete`) and `manifest.json`. Checks: every book's weights sum to one and
its value to $100m, every holding is in the universe; given D19, LOWVOL against its stored
long-only minimum-variance holdings on every date the universes agree, to the bit. Refuses to
replace the portfolios without `--overwrite`.
"""

from __future__ import annotations

import os

for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from praxis.midoffice import store  # noqa: E402
from praxis.midoffice.model.bias import RiskForecast, minimum_variance  # noqa: E402

DEFAULT_PANEL = "data/external/polygon/sp500_securities_2022-01-03_2026-11-03_released.npz"
BOOK_VALUE = 100e6
MOMENTUM_NAMES = 40
LARGEST = 50
BOOKS = ("INDEX", "LOWVOL", "MOM40", "EQ50")
RULES = {
    "INDEX": "the universe cap-weighted: the benchmark, the bias test's cap-weighted index",
    "LOWVOL": "the long-only minimum-variance portfolio from v1's forecast as of the date",
    "MOM40": f"the {MOMENTUM_NAMES} members with the highest Momentum exposure, cap-weighted",
    "EQ50": f"the {LARGEST} largest members by capitalisation, equally weighted",
}
LONG_ONLY = "minimum variance, long-only"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=str(store.STORE_ROOT), help="the store's root")
    parser.add_argument("--model", default="v1", help="the model the books are formed from")
    parser.add_argument("--panel", default=DEFAULT_PANEL,
                        help="the panel on securities: membership, prices, shares, returns")
    parser.add_argument("--bias-run", default=None,
                        help="D19's --bias-out file: LOWVOL checked against its stored holdings")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    began = time.monotonic()
    model = store.open_model(args.model, args.store)
    momentum = model.factor_names.index("Momentum")
    with np.load(args.panel, allow_pickle=False) as data:
        dates = [dt.date.fromisoformat(str(d)) for d in data["dates"]]
        labels = [str(s) for s in data["labels"]]
        keys = [str(s) for s in data["securities"]]
        tickers = data["tickers"]
        raw = data["unadjusted_prices"]
        caps = raw * data["shares_outstanding"]
        membership = data["membership"].astype(bool)
        prices = data["total_return_prices"]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.full_like(prices, np.nan)
        returns[1:] = np.where(prices[:-1] > 0.0, prices[1:] / prices[:-1] - 1.0, np.nan)
    at = {d: i for i, d in enumerate(dates)}
    column = {code: j for j, code in enumerate(labels)}
    n = len(labels)
    writer = store.HoldingsWriter(args.store, overwrite=args.overwrite)
    print(f"The sample portfolios from {args.model}'s store, {len(model.dates)} dates, "
          f"{model.dates[0]} to {model.dates[-1]}, ${BOOK_VALUE / 1e6:.0f}m each:")
    for book in BOOKS:
        print(f"  {book:<7} {RULES[book]}")
    lowvol: dict[dt.date, np.ndarray] = {}
    complete: dict[dt.date, bool] = {}
    problems: list[str] = []
    for count, day in enumerate(model.dates):
        t = at[day]
        forecast = model.forecast(day)
        cols = np.array([column[s] for s in forecast.securities])
        exposures = np.full((n, len(model.factor_names)), np.nan)
        specific = np.full(n, np.nan)
        exposures[cols], specific[cols] = forecast.exposures, forecast.specific_risk
        engine = RiskForecast(forecast.covariance, exposures, specific,
                              float(model.manifest["units"]["periods_per_year"]))
        cap = caps[t]
        universe = membership[t] & engine.has_forecast & np.isfinite(cap) & (cap > 0.0)
        idx = np.flatnonzero(universe)
        members = np.flatnonzero(membership[t])
        left_out = members[~universe[members]]
        member_cap = np.nansum(np.where(cap[members] > 0.0, cap[members], 0.0))
        weights = {book: np.zeros(n) for book in BOOKS}
        weights["INDEX"][idx] = cap[idx] / cap[idx].sum()
        solved = minimum_variance(engine, universe, long_only=True)
        weights["LOWVOL"] = solved.weights
        top = idx[np.argsort(-exposures[idx, momentum], kind="stable")[:MOMENTUM_NAMES]]
        weights["MOM40"][top] = cap[top] / cap[top].sum()
        largest = idx[np.argsort(cap[idx], kind="stable")[-LARGEST:]]
        weights["EQ50"][largest] = 1.0 / largest.size
        rows = []
        for book in BOOKS:
            w = weights[book]
            held = np.flatnonzero(w)
            if not np.isclose(w.sum(), 1.0, rtol=0.0, atol=1e-12) or (~universe[held]).any():
                problems.append(f"{day} {book}: weights sum to {w.sum():.15f}, or a holding is "
                                f"outside the universe")
            value = BOOK_VALUE * w[held]
            quantity = value / raw[t, held]
            for j, v, q in zip(held, value, quantity):
                rows.append((book, labels[j], keys[j], str(tickers[t, j]) or labels[j], float(q),
                             float(raw[t, j]), float(q * raw[t, j]), float(w[j])))
        next_complete = bool(t + 1 < len(dates) and np.isfinite(returns[t + 1, idx]).all())
        complete[day] = next_complete
        lowvol[day] = weights["LOWVOL"]
        writer.add(day, rows, {
            "universe": int(idx.size), "members": int(members.size),
            "left_out": int(left_out.size),
            "left_out_cap_share": float(np.nansum(np.where(cap[left_out] > 0.0, cap[left_out],
                                                           0.0)) / member_cap),
            "lowvol_held": int(solved.held), "lowvol_violation": float(solved.violation),
            "next_day_complete": next_complete})
        if (count + 1) % 100 == 0 or count + 1 == len(model.dates):
            print(f"  [portfolios] {count + 1} of {len(model.dates)} dates, "
                  f"{(time.monotonic() - began):.0f} s", file=sys.stderr, flush=True)
    checks: dict = {"weights_and_values": not problems}
    print("\nChecks\n  every book on every date: weights sum to one, $100m, every holding in "
          "the universe -- " + ("yes" if not problems else f"NO, {len(problems)}: "
                                 + "; ".join(problems[:3])))
    if args.bias_run:
        path = Path(args.bias_run)
        with np.load(path, allow_pickle=False) as data:
            run_dates = [dt.date.fromisoformat(str(d)) for d in data["dates"]]
            stored = data["min_variance_weights_long_only"]
            securities = [str(s) for s in data["securities"]]
        if securities != labels:
            problems.append("the bias run's securities are not the panel's")
        shared = [(i, d) for i, d in enumerate(run_dates) if d in complete]
        agree = [(i, d) for i, d in shared if complete[d]]
        equal = sum(np.array_equal(lowvol[d], stored[i]) for i, d in agree)
        worst = max(float(np.max(np.abs(lowvol[d] - stored[i]))) for i, d in agree)
        print(f"  LOWVOL against {path.as_posix()}'s stored long-only minimum-variance holdings: "
              f"equal to the bit on {equal} of the {len(agree)} dates whose universe is the bias "
              f"test's -- of the {len(shared)} it shares with the store, "
              f"{len(shared) - len(agree)} have a member with no return the next day, which the "
              f"bias test leaves out; the largest difference {worst:.1e}")
        checks["lowvol_against_bias_run"] = {"path": path.as_posix(), "dates": len(agree),
                                             "equal": int(equal), "largest_difference": worst}
        if equal != len(agree):
            problems.append(f"LOWVOL differs from the bias run's on {len(agree) - equal} dates")
    model.close()
    writer.finish({"model_id": args.model, "panel": Path(args.panel).as_posix(),
                   "portfolios": list(BOOKS), "rules": RULES, "book_value": BOOK_VALUE,
                   "universe": "the date's index members with a forecast in the model's store "
                               "and a positive capitalisation",
                   "quantities": "fractional shares: the holding's dollars over its raw close"})
    if problems:
        print("\nThe build stops, the portfolios left unmoved: " + "; ".join(problems[:5]))
        return 1
    place = writer.commit({"checks": checks, "built_utc": dt.datetime.now(dt.timezone.utc)
                           .isoformat(timespec="seconds")})
    books = store.StoredHoldings(args.store)
    held = [books.facts(d)["lowvol_held"] for d in model.dates]
    books.close()
    print(f"  LOWVOL holds {min(held)} to {max(held)} names (median {np.median(held):.0f})")
    print(f"  {place.as_posix()}, {(time.monotonic() - began) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
