"""The model store (midoffice20): what a build writes is what a lookup reads, to the bit."""

from datetime import date

import numpy as np
import pyarrow.parquet as pq
import pytest

from praxis.midoffice import store

FACTORS = ("SIC28", "SIC60", "Momentum")


def _write(root, model_id="v1", overwrite=False, commit=True):
    rng = np.random.default_rng(7)
    writer = store.ModelWriter(root, model_id, FACTORS, overwrite=overwrite)
    days = (date(2025, 6, 27), date(2025, 6, 30))
    written = {}
    for i, day in enumerate(days):
        a = rng.normal(size=(3, 3))
        covariance = a @ a.T
        securities = ("AAA", "BBB", "CCC")[: 2 + i]
        exposures = rng.normal(size=(len(securities), 3))
        specific = rng.uniform(0.1, 0.5, size=len(securities))
        writer.add(day, covariance, securities, [s + "-key" for s in securities],
                   [s.lower() for s in securities], exposures, specific,
                   {"lambda_factor": 1.0 + i / 10})
        written[day] = (covariance, securities, exposures, specific)
    returns = rng.normal(scale=0.01, size=(5, 3))
    return_dates = [date(2025, 6, d) for d in (24, 25, 26, 27, 30)]
    writer.finish(return_dates, returns, {"description": "a test"})
    writer.reference([(days[1], "minimum variance, long-only", 0.0051)])
    if commit:
        writer.commit({"checks": {"ok": True}})
    return writer, written, return_dates, returns


def test_round_trip_is_exact(tmp_path):
    _, written, return_dates, returns = _write(tmp_path)
    model = store.open_model("v1", tmp_path)
    assert model.dates == tuple(sorted(written))
    assert model.factor_names == FACTORS
    for day, (covariance, securities, exposures, specific) in written.items():
        forecast = model.forecast(day)
        assert np.array_equal(forecast.covariance, covariance)
        assert forecast.securities == securities
        assert forecast.security_keys == tuple(s + "-key" for s in securities)
        assert forecast.tickers == tuple(s.lower() for s in securities)
        assert np.array_equal(forecast.exposures, exposures)
        assert np.array_equal(forecast.specific_risk, specific)
        assert np.array_equal(forecast.exposure_of("Momentum"), exposures[:, 2])
    dates, matrix = model.factor_returns(through=date(2025, 6, 26))
    assert dates == tuple(return_dates[:3]) and np.array_equal(matrix, returns[:3])
    assert model.multipliers(date(2025, 6, 30)) == {"lambda_factor": 1.1}
    assert model.bias_reference() == {date(2025, 6, 30): {"minimum variance, long-only": 0.0051}}
    assert model.manifest["checks"] == {"ok": True}
    assert model.manifest["model_id"] == "v1"


def test_one_row_group_a_date(tmp_path):
    _write(tmp_path)
    for name in (store.COVARIANCE, store.EXPOSURES, store.SPECIFIC):
        meta = pq.ParquetFile(tmp_path / "v1" / f"{name}.parquet").metadata
        assert meta.num_row_groups == 2


def test_refuses_to_replace_a_model_without_overwrite(tmp_path):
    _write(tmp_path)
    with pytest.raises(FileExistsError):
        store.ModelWriter(tmp_path, "v1", FACTORS)
    _write(tmp_path, overwrite=True)
    assert store.list_models(tmp_path) == ["v1"]


def test_a_partial_build_is_not_a_model(tmp_path):
    writer, *_ = _write(tmp_path, model_id="v1-blocks", commit=False)
    assert store.list_models(tmp_path) == []
    with pytest.raises(KeyError):
        store.open_model("v1-blocks", tmp_path)
    # The checks read the finished files before the commit, and close them so it can move them.
    with store.StoredModel(writer.partial.parent, "v1-blocks") as partial:
        assert partial.dates == (date(2025, 6, 27), date(2025, 6, 30))
    assert writer.commit().name == "v1-blocks"
    assert store.list_models(tmp_path) == ["v1-blocks"]
    store.open_model("v1-blocks", tmp_path).close()


def test_written_in_date_order_and_finite(tmp_path):
    writer = store.ModelWriter(tmp_path, "v1", FACTORS)
    eye = np.eye(3)
    writer.add(date(2025, 6, 30), eye, ("AAA",), ("k",), ("a",), np.zeros((1, 3)),
               np.ones(1), {})
    with pytest.raises(ValueError):
        writer.add(date(2025, 6, 27), eye, ("AAA",), ("k",), ("a",), np.zeros((1, 3)),
                   np.ones(1), {})
    with pytest.raises(ValueError):
        writer.add(date(2025, 7, 1), eye, ("AAA",), ("k",), ("a",),
                   np.full((1, 3), np.nan), np.ones(1), {})
    with pytest.raises(ValueError):
        store.ModelWriter(tmp_path, "v1/x", FACTORS)
