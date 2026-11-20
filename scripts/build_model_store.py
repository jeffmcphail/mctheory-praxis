"""Build the model store (midoffice20): a model's forecast as of every date its fit covers, in
Parquet, each date looked up in milliseconds (`praxis.midoffice.store`).

    python scripts/build_model_store.py --model v1 --model v1-blocks \\
        --bias-run claude/scratch/midoffice19/bias/D19.npz \\
        --blocks-run claude/scratch/midoffice19/bias/C19.npz

v1 is D19's model (ruling 1 of midoffice20's brief): the run's defaults,
`run_sp500_model.py --industries --membership`. The build runs it -- the run's own code, through
`main`'s `after_fit` hook, its report printed first -- and writes, for each date from the first
whose covariance has a year of estimated periods (the bias test's first forecast) to the last,
the model's forecast as of that date's close:

- the factor covariance as the engine uses it (`bias.RiskForecast`'s, through `nearest_psd`)
  times lambda_F^2, the volatility regime adjustment's factor multiplier as of the date, read
  from the full fit's factor returns as a plain run reads it (`regime.factor_history`);
- the exposures as of the date, of every security with a forecast;
- the specific risk: the specific risk model fitted on the residuals through the date, blended
  with each security's history through it.

The full fit forecasts specific risk as of its last date alone -- its specific risk model is
fitted once, on every residual through that date -- so each date's forecast is the model
refitted as of the date, the bias test's own refits (`cross_sectional_models_as_of`): the store
holds exactly the forecasts the bias statistics tested, and at the last date the full fit's own.
The factor returns are the full fit's, which are every refit's through its date.

`v1-blocks` is the two-block setting (midoffice19's C19): v1's exposures and specific risk, its
covariance estimated in the country basis and scaled there by lambda_C, the country's own
multiplier, and lambda_R, the other factors', before it is mapped back -- the leading candidate
for a v2 (ruling 1 of midoffice20's brief). Each model is its own build under its own model_id;
given together, they share one pass of refits, the covariance the only thing that differs.

The checks, printed, and kept in each model's manifest; the store is moved into place only if
they pass:
- the last date's refit is the full fit, to the bit;
- the guard (item 2 of the brief): as of the bias test's guard's five dates, the store's
  covariance, specific risk and exposures, read back from its files, equal the model refitted as
  of the date from the panel cut at it, the industry scheme held -- the bias guard's own rebuild
  (`run_sp500_model.model_inputs` on `cut_panel`) -- to the bit;
- given the stored bias runs, every date's covariance (and v1-blocks' multipliers) against
  theirs, to the bit; v1 then keeps D19's own forecasts of the long-only minimum-variance
  portfolio and the cap-weighted index beside it (`bias_reference.parquet`), for the risk
  report's check;
- a date's lookup, timed.

BLAS on one thread, as every stored run was made. Refuses to replace a model without
`--overwrite`. `--first N` builds the first N dates alone, a smoke (into a store of its own).
"""

from __future__ import annotations

import os

for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import argparse  # noqa: E402
import bisect  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_sp500_model as run  # noqa: E402  (puts src on the path)

from praxis.midoffice import store  # noqa: E402
from praxis.midoffice.model.bias import RiskForecast  # noqa: E402
from praxis.midoffice.model.country import (  # noqa: E402
    block_scaled,
    country_covariance,
    country_map,
    country_weights,
    to_country_basis,
)
from praxis.midoffice.model.crosssectional import (  # noqa: E402
    cross_sectional_models_as_of,
    estimate_cross_sectional_model,
)
from praxis.midoffice.model.regime import (  # noqa: E402
    RegimeHistory,
    factor_history,
    forecast_volatilities,
    regime_multiplier,
)

RUN_ARGS = ("--industries", "--membership")
"""v1's run: the defaults (ruling 1 of midoffice20's brief, D19's)."""

MODELS = {
    "v1": ("v1, D19's model (ruling 1 of midoffice20's brief), the run's defaults: the industry "
           "basis with every member on the market, USE3's risk indices with USE4's market model "
           "and Size Nonlinearity, total returns, the two-half-life factor covariance "
           "(volatilities 90 days, correlations 504), the one-day horizon, the volatility regime "
           "adjustment's factor side on 21 days (its specific side off), specific risk blended "
           "from covered histories on a 63-day half-life"),
    "v1-blocks": ("the two-block setting (midoffice19's C19, 'blocks'): v1's exposures and "
                  "specific risk, its factor covariance estimated in the country basis and "
                  "scaled there by two regime multipliers -- lambda_C from the country's own "
                  "squared bias statistic, lambda_R from the other factors' -- before it is "
                  "mapped back; the leading candidate for a v2, out by the rule on the "
                  "optimiser's test (ruling 1 of midoffice20's brief)"),
}

GUARD = 5
"""The bias test's guard's date count (`--bias-guard`'s default), spread over its forecast dates."""

REFERENCE = ("minimum variance, long-only", "the index, cap-weighted")
"""The stored bias run's forecasts v1 keeps beside it for the risk report's check."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()
    return {"head": git("rev-parse", "HEAD"),
            "dirty": [line[3:] for line in git("status", "--porcelain").splitlines()]}


def multiplier_as_of(history: RegimeHistory, day: dt.date, half_life: float | None) -> float:
    """lambda_F as of `day`: the history's statistics through it (`regime_multiplier`), so a day
    the model did not estimate carries the day before's, as the bias test's forecasts do."""
    k = bisect.bisect_right(history.dates, day)
    return regime_multiplier(history.factor_squared[:k], half_life, history.min_days)


class Blocks:
    """The two-block setting, a date at a time, as the bias test forms it (midoffice19's
    `bias_section`, its `blocks` setting): the basis's covariance from each refit's own factor
    returns, the country's and the other factors' histories from the basis's forecast the day
    before, the basis scaled by the two multipliers as of the date and mapped back."""

    def __init__(self, fitted_run, factor_names, half_life) -> None:
        inputs, options = fitted_run.inputs, fitted_run.options
        held = inputs.built_scheme
        self.columns = [k for k, name in enumerate(factor_names) if name in held.industries]
        self.weights = country_weights(
            inputs.return_panel, inputs.indices, options.get("capitalisations", inputs.caps),
            self.columns, regression_weights=options.get("regression_weights"),
            membership=options.get("membership"), lag=options.get("lag", 1))
        self.t_map = country_map(len(factor_names), self.columns)
        self.estimator = options["covariance_estimator"]
        self.half_life = half_life
        self.country, self.rest = RegimeHistory(), RegimeHistory()
        self.previous = None

    def step(self, t, day, fitted, periods):
        basis, _ = country_covariance(fitted, self.estimator, self.weights, self.columns)
        own = fitted.notes.factor_returns
        if self.previous is not None and own.dates[-1] == day:
            if self.previous[0] != t - 1:
                raise SystemExit(f"  the build stops: the blocks' day {day} follows a forecast "
                                 f"as of row {self.previous[0]}, not the day before")
            day_return = to_country_basis(own.returns[-1:], self.columns,
                                          self.weights[day][None])[0]
            vols = forecast_volatilities(self.previous[1], periods)
            self.country.add_factor(day, vols[:1], day_return[:1])
            self.rest.add_factor(day, vols[1:], day_return[1:])
        self.previous = (t, basis)
        pair = (self.country.multiplier("factor", self.half_life),
                self.rest.multiplier("factor", self.half_life))
        return pair, self.t_map @ block_scaled(basis, *pair) @ self.t_map.T

    def mapped(self, fitted, pair):
        """The blocks' covariance from a fit at given multipliers -- the guard's rebuild."""
        basis, _ = country_covariance(fitted, self.estimator, self.weights, self.columns)
        return self.t_map @ block_scaled(basis, *pair) @ self.t_map.T


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", choices=sorted(MODELS),
                        help="a model to build, repeatable; v1 by default")
    parser.add_argument("--out", default=str(store.STORE_ROOT),
                        help=f"the store's root (default {store.STORE_ROOT.as_posix()})")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a model already in the store")
    parser.add_argument("--bias-run", default=None,
                        help="D19's --bias-out file: v1's covariance checked against it on every "
                             "date, and its forecasts kept beside v1 for the report's check")
    parser.add_argument("--blocks-run", default=None,
                        help="C19's --bias-out file: v1-blocks' covariance and multipliers "
                             "checked against its blocks setting on every date")
    parser.add_argument("--first", type=int, default=None,
                        help="a smoke: the first N dates alone")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    models = list(dict.fromkeys(args.model or ["v1"]))
    out = Path(args.out)
    for model_id in models:
        if (out / model_id).exists() and not args.overwrite:
            raise SystemExit(f"  {out / model_id} exists; --overwrite to replace it")
    threads = {name: os.environ.get(name) for name in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    began = time.monotonic()
    started = dt.datetime.now(dt.timezone.utc)

    # v1's full fit, by the run itself: its report first, then the objects it fitted.
    captured: dict = {}
    print(f"The model store: building {', '.join(models)} into {out.as_posix()}\n")
    print(f"v1's full fit: run_sp500_model.py {' '.join(RUN_ARGS)}\n", flush=True)
    code = run.main(list(RUN_ARGS), after_fit=lambda fitted: captured.setdefault("run", fitted))
    if code != 0 or "run" not in captured:
        raise SystemExit("  the build stops: the run did not fit")
    fitted_run = captured["run"]
    fit_seconds = time.monotonic() - began
    panel, model, inputs, options = (fitted_run.panel, fitted_run.model, fitted_run.inputs,
                                     fitted_run.options)
    dates, securities = panel.dates, panel.securities
    keys = fitted_run.extras["keys"]
    panel_path = Path(fitted_run.args.panel)
    with np.load(panel_path, allow_pickle=False) as data:
        tickers = data["tickers"]
    factor_names = tuple(model.factor_names[fid] for fid in model.factor_ids)
    industries = set(inputs.scheme.industries)
    half_lives = (fitted_run.args.regime_factor_half_life,
                  fitted_run.args.regime_specific_half_life)
    if half_lives[1] is not None:
        raise SystemExit("  the build stops: the regime adjustment's specific side needs every "
                         "refit's specific statistic, which this build does not read")
    notes = model.notes
    periods = notes.frequency.periods_per_year if notes.annualised else 1.0

    # The dates: the bias test's forecast dates and the last, the run's own.
    rows = run.forecast_rows(model, dates)
    picks = sorted(set(np.linspace(0, len(rows) - 1, GUARD).round().astype(int)))
    built_rows = rows + [len(dates) - 1]
    if args.first:
        built_rows = built_rows[: args.first]
        picks = sorted(set(np.linspace(0, len(built_rows) - 1, min(GUARD, len(built_rows)))
                           .round().astype(int)))
    guard_rows = [built_rows[i] for i in picks]
    built_dates = [dates[t] for t in built_rows]
    history = factor_history(model, options["covariance_estimator"],
                             [dates[t] for t in rows + [len(dates) - 1]])
    lambdas = [multiplier_as_of(history, day, half_lives[0]) for day in built_dates]
    blocks = Blocks(fitted_run, factor_names, half_lives[0]) if "v1-blocks" in models else None

    print(f"\nThe model store: {', '.join(models)}, {len(built_dates)} dates, {built_dates[0]} to "
          f"{built_dates[-1]} -- the bias test's forecast dates and the run's own last -- each "
          f"the model refitted as of the date (the bias test's refits); lambda_F as of each from "
          f"the full fit's history ({len(history)} days)", flush=True)
    writers = {m: store.ModelWriter(out, m, factor_names, overwrite=args.overwrite)
               for m in models}
    refits = cross_sectional_models_as_of(inputs.return_panel, inputs.indices, built_dates,
                                          **options)
    refit_began = time.monotonic()
    pairs: dict[dt.date, tuple[float, float]] = {}
    sizes: list[int] = []
    last = None
    for count, (t, lam, (day, fitted)) in enumerate(zip(built_rows, lambdas, refits)):
        forecast = RiskForecast.from_model(fitted, securities)
        idx = np.flatnonzero(forecast.has_forecast)
        held = [securities[j] for j in idx]
        held_keys = [keys[j] for j in idx]
        held_tickers = [str(tickers[t, j]) or securities[j] for j in idx]
        exposures, specific = forecast.exposures[idx], forecast.specific[idx]
        sizes.append(idx.size)
        if "v1" in writers:
            writers["v1"].add(day, forecast.regime_adjusted(lam, 1.0).covariance, held, held_keys,
                              held_tickers, exposures, specific, {"lambda_factor": lam})
        if blocks is not None:
            pair, mapped = blocks.step(t, day, fitted, periods)
            pairs[day] = pair
            covariance = RiskForecast.from_model(fitted, securities, covariance=mapped).covariance
            writers["v1-blocks"].add(day, covariance, held, held_keys, held_tickers, exposures,
                                     specific, {"lambda_country": pair[0],
                                                "lambda_rest": pair[1]})
        last = fitted
        if (count + 1) % 50 == 0 or count + 1 == len(built_rows):
            elapsed = time.monotonic() - refit_began
            print(f"  [store] {count + 1} of {len(built_rows)} dates, {elapsed / 60:.1f} min",
                  file=sys.stderr, flush=True)
    refit_minutes = (time.monotonic() - refit_began) / 60
    factor_dates = tuple(notes.factor_returns.dates)
    factor_matrix = notes.factor_returns.returns
    manifest = {
        "built": {"started_utc": started.isoformat(timespec="seconds"),
                  "command": "python scripts/build_model_store.py "
                             + " ".join(argv if argv is not None else sys.argv[1:]),
                  "run": "python scripts/run_sp500_model.py " + " ".join(RUN_ARGS),
                  "git": git_state(), "python": platform.python_version(),
                  "numpy": np.__version__, "blas_threads": threads,
                  "full_fit_seconds": round(fit_seconds, 1),
                  "refit_minutes": round(refit_minutes, 2)},
        "panel": {"path": panel_path.as_posix(), "sha256": sha256(panel_path),
                  "snapshot": fitted_run.extras.get("snapshot")},
        "factors": [{"name": name, "group": "industry" if name in industries else "style"}
                    for name in factor_names],
        "units": {"covariance": "annualised variance, the engine's: through nearest_psd at the "
                                "risk config's floor, times the regime multipliers",
                  "specific_risk": "annualised volatility",
                  "factor_returns": "daily, as estimated",
                  "periods_per_year": periods},
        "securities_per_date": {"min": int(min(sizes)), "median": float(np.median(sizes)),
                                "max": int(max(sizes))},
        "regime": {"factor_half_life": half_lives[0], "specific_half_life": half_lives[1],
                   "min_days": history.min_days,
                   "source": "the full fit's factor returns (regime.factor_history), as a plain "
                             "run reads them"},
    }
    for model_id, writer in writers.items():
        writer.finish(factor_dates, factor_matrix,
                      {"description": MODELS[model_id], **manifest})
    print(f"  written: {len(built_dates)} dates, {min(sizes)} to {max(sizes)} securities a date "
          f"(median {np.median(sizes):.0f}), {refit_minutes:.1f} min of refits")

    # The checks.
    checks: dict[str, dict] = {m: {} for m in models}
    failures: list[str] = []
    partial = out / store.PARTIAL
    stored = {m: store.StoredModel(partial, m) for m in models}

    # 1. The last date's refit is the full fit.
    if not args.first:
        differ = run._models_differ(last, model)
        line = ("the last date's refit is the full fit to the bit -- covariance, exposures, "
                "specific risk, factor returns" if not differ
                else f"the last date's refit differs from the full fit: {', '.join(differ)}")
        print(f"\nChecks\n  {line}")
        if differ:
            failures.append(line)
        lam_last = lambdas[-1]
        print(f"  the last date, {dates[-1]}: lambda_F {lam_last:.4f}, the run's own")
        for m in models:
            checks[m]["last_date_is_the_full_fit"] = not differ
    else:
        print("\nChecks (a smoke of the first dates: the last date's check is not made)")

    # 2. The guard: the model refitted as of each guard date from the panel cut at it.
    held_scheme = inputs.built_scheme
    guard_lines = []
    for t in guard_rows:
        day = dates[t]
        cut = run.cut_panel(panel, t + 1)
        members = None if fitted_run.membership is None else fitted_run.membership[: t + 1]
        rebuilt = run.model_inputs(cut, fitted_run.market[: t + 1], members, fitted_run.args,
                                   codes=fitted_run.codes, merges=fitted_run.merges,
                                   scheme=held_scheme)
        direct = estimate_cross_sectional_model(
            rebuilt.return_panel, rebuilt.indices,
            **dict(options, capitalisations=rebuilt.caps, regression_weights=rebuilt.weights,
                   membership=members, periods=rebuilt.periods))
        forecast = RiskForecast.from_model(direct, securities)
        idx = np.flatnonzero(forecast.has_forecast)
        lam = lambdas[built_rows.index(t)]
        expected = {"v1": forecast.regime_adjusted(lam, 1.0).covariance}
        if blocks is not None:
            expected["v1-blocks"] = RiskForecast.from_model(
                direct, securities, covariance=blocks.mapped(direct, pairs[day])).covariance
        for m in models:
            got = stored[m].forecast(day)
            same = {
                "covariance": np.array_equal(got.covariance, expected[m]),
                "specific risk": (got.securities == tuple(securities[j] for j in idx)
                                  and np.array_equal(got.specific_risk, forecast.specific[idx])),
                "exposures": np.array_equal(got.exposures, forecast.exposures[idx]),
            }
            bad = [name for name, ok in same.items() if not ok]
            checks[m].setdefault("guard", {})[str(day)] = not bad
            if bad:
                failures.append(f"{m} {day}: {', '.join(bad)} not the refit's")
        guard_lines.append(str(day))
    ok = not any("not the refit's" in f for f in failures)
    print(f"  guard: as of {', '.join(guard_lines)}, the store's covariance, specific risk and "
          f"exposures, read back from its files, "
          + ("equal" if ok else "do NOT all equal")
          + " the model refitted as of the date from the panel cut at it, the scheme held, to "
            "the bit, for " + ", ".join(models))

    # 3. Against the stored bias runs.
    if args.bias_run and "v1" in models:
        path = Path(args.bias_run)
        with np.load(path, allow_pickle=False) as data:
            run_dates = [dt.date.fromisoformat(str(d)) for d in data["dates"]]
            covariances = data["factor_covariance"]
            names = [str(n) for n in data["names"]]
            forecasts = data["forecast"]
        shared = [(i, d) for i, d in enumerate(run_dates) if d in set(stored["v1"].dates)]
        equal = sum(np.array_equal(stored["v1"].covariance(d), covariances[i]) for i, d in shared)
        print(f"  against {path.as_posix()}: v1's covariance equals the bias run's on {equal} of "
              f"{len(shared)} dates, to the bit")
        checks["v1"]["bias_run"] = {"path": path.as_posix(), "sha256": sha256(path),
                                    "dates": len(shared), "covariance_equal": int(equal)}
        if equal != len(shared):
            failures.append(f"v1's covariance differs from {path.name}'s on "
                            f"{len(shared) - equal} dates")
        rows_kept = [(d, name, float(forecasts[names.index(name), i]))
                     for i, d in shared for name in REFERENCE]
        writers["v1"].reference(rows_kept)
        print(f"  kept beside v1: the bias run's forecasts of {' and '.join(REFERENCE)} on "
              f"{len(shared)} dates (bias_reference.parquet), for the risk report's check")
    if args.blocks_run and blocks is not None:
        path = Path(args.blocks_run)
        with np.load(path, allow_pickle=False) as data:
            run_dates = [dt.date.fromisoformat(str(d)) for d in data["dates"]]
            covariances = data["factor_covariance_blocks"]
            multipliers = data["blocks_multipliers"]
        shared = [(i, d) for i, d in enumerate(run_dates) if d in pairs]
        equal = sum(np.array_equal(stored["v1-blocks"].covariance(d), covariances[i])
                    for i, d in shared)
        same_pairs = sum(tuple(multipliers[i]) == pairs[d] for i, d in shared)
        print(f"  against {path.as_posix()}: v1-blocks' covariance equals its blocks setting's on "
              f"{equal} of {len(shared)} dates, and its two multipliers on {same_pairs}, to the "
              f"bit")
        checks["v1-blocks"]["blocks_run"] = {"path": path.as_posix(), "sha256": sha256(path),
                                             "dates": len(shared),
                                             "covariance_equal": int(equal),
                                             "multipliers_equal": int(same_pairs)}
        if equal != len(shared) or same_pairs != len(shared):
            failures.append(f"v1-blocks differs from {path.name}'s blocks setting")

    # 4. A date's lookup, timed, on the files as written.
    rng = np.random.default_rng(20)
    for m in models:
        opened = time.perf_counter()
        fresh = store.StoredModel(partial, m)
        open_ms = (time.perf_counter() - opened) * 1e3
        days = [fresh.dates[i] for i in rng.integers(0, len(fresh.dates), 100)]
        times = []
        for day in days:
            began_one = time.perf_counter()
            fresh.forecast(day)
            times.append((time.perf_counter() - began_one) * 1e3)
        print(f"  {m}: opened in {open_ms:.0f} ms; a date's forecast -- covariance, exposures, "
              f"specific risk -- in {np.median(times):.1f} ms at the median of 100 random dates "
              f"(at most {max(times):.1f})")
        checks[m]["lookup_ms"] = {"open": round(open_ms, 1),
                                  "median": round(float(np.median(times)), 2),
                                  "max": round(float(max(times)), 2)}
        fresh.close()
    for reader in stored.values():
        reader.close()

    if failures:
        print("\nThe build stops, the store left unmoved in "
              f"{partial.as_posix()}: " + "; ".join(failures))
        return 1
    for m, writer in writers.items():
        place = writer.commit({"checks": checks[m], "guard_dates": guard_lines,
                               "finished_utc": dt.datetime.now(dt.timezone.utc)
                               .isoformat(timespec="seconds")})
        files = sorted(place.glob("*"))
        total = sum(p.stat().st_size for p in files) / 2**20
        print(f"  {m}: {place.as_posix()}, {len(files)} files, {total:.1f} MB")
    print(f"\nDone in {(time.monotonic() - began) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
