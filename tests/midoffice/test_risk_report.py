"""The sample portfolios and the risk report (midoffice20), end to end on a small synthetic store:
the books follow their rules, and the report prints with its three checks passing -- check 3
against a reference made the bias test's way -- the index's coverage and the historical VaR's
fixed window with it."""

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from praxis.midoffice import store
from praxis.midoffice.model.bias import RiskForecast, minimum_variance

ROOT = Path(__file__).resolve().parents[2]
FACTORS = ("SIC28", "SIC60", "Momentum")
GROUPS = ("industry", "industry", "style")
PANEL_DATES = [date(2025, 6, d) for d in (24, 25, 26, 27, 30)]
STORE_DATES = PANEL_DATES[:3]
N = 60
GONE = N - 3
"""A member with no forecast in the store on any date: the index leaves it out."""


def _weekdays(last: date, count: int) -> list[date]:
    days, day = [], last
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    return days[::-1]


FACTOR_DATES = _weekdays(STORE_DATES[-1], 300)
"""The factor returns' dates: 300 weekdays through the last store date, so a year's window fits
every store date."""


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    rng = np.random.default_rng(3)
    labels = [f"S{j:02d}" for j in range(N)]
    keys = [f"BBG{j:09d}" for j in range(N)]
    industry = rng.integers(0, 2, N)
    momentum = rng.normal(size=(len(PANEL_DATES), N))
    specific = rng.uniform(0.15, 0.45, N)
    vols = np.array([0.20, 0.25, 0.05])
    corr = np.array([[1.0, 0.6, 0.1], [0.6, 1.0, -0.2], [0.1, -0.2, 1.0]])
    covariance = corr * np.outer(vols, vols)
    shares = rng.uniform(1e8, 5e9, N)
    raw = rng.uniform(20.0, 400.0, size=(len(PANEL_DATES), N))
    membership = np.ones((len(PANEL_DATES), N), dtype=bool)
    membership[:, -2:] = False     # two non-members: in no book
    stored = np.flatnonzero(np.arange(N) != GONE)
    writer = store.ModelWriter(root, "v1", FACTORS)
    reference = []
    for t, day in enumerate(STORE_DATES):
        exposures = np.column_stack([industry == 0, industry == 1, momentum[t]]).astype(float)
        writer.add(day, covariance, [labels[j] for j in stored], [keys[j] for j in stored],
                   [labels[j] for j in stored], exposures[stored], specific[stored],
                   {"lambda_factor": 1.0})
        exposures[GONE] = np.nan
        forecast = RiskForecast(covariance, exposures,
                                np.where(np.arange(N) == GONE, np.nan, specific), 252.0)
        caps = raw[t] * shares
        eligible = membership[t] & forecast.has_forecast & (caps > 0.0)
        lowvol = minimum_variance(forecast, eligible, long_only=True).weights
        index = np.where(eligible, caps, 0.0) / caps[eligible].sum()
        daily = np.sqrt(forecast.variances(np.vstack([lowvol, index])) / 252.0)
        reference += [(day, "minimum variance, long-only", daily[0]),
                      (day, "the index, cap-weighted", daily[1])]
    writer.finish(FACTOR_DATES, rng.normal(scale=0.01, size=(len(FACTOR_DATES), 3)), {
        "description": "a synthetic model: three factors", "units": {"periods_per_year": 252.0},
        "factors": [{"name": n, "group": g} for n, g in zip(FACTORS, GROUPS)]})
    writer.reference(reference)
    writer.commit()
    panel = root / "panel.npz"
    np.savez(panel, dates=np.array([d.isoformat() for d in PANEL_DATES]),
             labels=np.array(labels), securities=np.array(keys),
             tickers=np.array([labels] * len(PANEL_DATES)), unadjusted_prices=raw,
             shares_outstanding=np.tile(shares, (len(PANEL_DATES), 1)), membership=membership,
             total_return_prices=raw)
    code = _script("build_sample_portfolios").main(["--store", str(root), "--panel", str(panel)])
    assert code == 0
    return root, momentum, raw * shares


def test_the_books_follow_their_rules(built):
    root, momentum, caps = built
    books = store.StoredHoldings(root)
    assert books.dates == tuple(STORE_DATES)
    for t, day in enumerate(STORE_DATES):
        universe = np.arange(N - 3)     # the members, less the one without a forecast
        for name in ("INDEX", "LOWVOL", "MOM40", "EQ50"):
            held = books.holdings(name, day)
            assert held.value == pytest.approx(100e6, rel=1e-12)
            assert held.weight.sum() == pytest.approx(1.0, abs=1e-12)
            assert np.allclose(held.quantity * held.price, held.market_value, rtol=1e-15)
            assert f"S{GONE:02d}" not in held.securities
        mom = books.holdings("MOM40", day)
        top = universe[np.argsort(-momentum[t, universe], kind="stable")[:40]]
        assert set(mom.securities) == {f"S{j:02d}" for j in top}
        eq = books.holdings("EQ50", day)
        assert len(eq.securities) == 50 and np.allclose(eq.weight, 1 / 50)
        largest = universe[np.argsort(caps[t, universe], kind="stable")[-50:]]
        assert set(eq.securities) == {f"S{j:02d}" for j in largest}
        assert len(books.holdings("INDEX", day).securities) == N - 3
        facts = books.facts(day)
        assert (facts["members"], facts["universe"], facts["left_out"]) == (N - 2, N - 3, 1)
        assert facts["left_out_cap_share"] == pytest.approx(
            caps[t, GONE] / caps[t, : N - 2].sum(), rel=1e-12)
    # The last panel date is not in the store; each store date has a next day.
    assert all(books.facts(d)["next_day_complete"] for d in STORE_DATES)
    books.close()


@pytest.mark.parametrize("portfolio", ["LOWVOL", "MOM40", "EQ50"])
def test_the_report_prints_with_its_checks_passing(built, portfolio, capsys):
    root, _, caps = built
    code = _script("risk_report").main(["--portfolio", portfolio, "--date", "2025-06-25",
                                        "--store", str(root)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "All checks pass" in out
    assert "3. on 2025-06-25 (the report's own date)" in out
    for heading in ("Risk, annualised", "By factor group", "ten largest contributions to total",
                    "ten largest contributions to tracking error", "One-day VaR"):
        assert heading in out
    share = caps[1, GONE] / caps[1, : N - 2].sum()
    assert (f"the index: {N - 3} of the date's {N - 2} members, cap-weighted; 1 left out, with "
            f"no v1 forecast or no capitalisation: {share * 100:.3f}% of the members' "
            f"capitalisation") in out
    window = [d for d in FACTOR_DATES if d <= date(2025, 6, 25)][-252:]
    assert f"on each of the last 252 days' factor returns through it, {window[0]} to " \
           f"2025-06-25" in out


def test_the_historical_var_is_the_worst_days_of_the_last_year(built):
    root, _, _ = built
    report = _script("risk_report")
    model, books = store.open_model("v1", root), store.StoredHoldings(root)
    day = date(2025, 6, 25)
    m = report.Measured(model, books, "LOWVOL", day)
    by, n, first, last = report.historical(m, model, m.portfolio.exposure, report.VAR_DAYS)
    window = [d for d in FACTOR_DATES if d <= day][-252:]
    assert (report.VAR_DAYS, n, first, last) == (252, 252, window[0], day)
    _, returns = model.factor_returns(through=day)
    losses = np.sort(-(returns[-252:] @ m.dense(m.portfolio.exposure)))[::-1]
    for c, k in ((0.95, 13), (0.99, 3)):        # k = ceil((1 - c) 252)
        assert by[c] == (losses[k - 1], losses[:k].mean())
    model.close()
    books.close()


def test_a_window_longer_than_the_history_is_refused(built, capsys):
    root, _, _ = built
    code = _script("risk_report").main(["--portfolio", "LOWVOL", "--date", "2025-06-25",
                                        "--store", str(root), "--var-days", "300"])
    captured = capsys.readouterr()
    assert code == 2 and captured.out == ""
    assert ("the store holds 299 days of factor returns through 2025-06-25, fewer than the "
            "historical VaR's window of 300") in captured.err


def test_a_date_the_store_lacks_resolves_to_the_one_before(built, capsys):
    root, _, _ = built
    code = _script("risk_report").main(["--portfolio", "LOWVOL", "--date", "2025-06-28",
                                        "--store", str(root)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "as of 2025-06-26" in out and "asked for 2025-06-28" in out
