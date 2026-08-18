"""Tests for engines.chan_cpo.data_loader.

The focus is the crossed-bar cost floor. Kibot builds each minute's bid and
ask OHLC quartets independently, so a small fraction of delivered bars are
crossed (bid above ask). A raw ask-minus-bid on those bars is negative, and a
cost model that consumes it gets *paid* to trade -- the exact class of sign
error that survives review because the affected bar count is tiny.
"""

import numpy as np
import pandas as pd
import pytest
from engines.chan_cpo.data_loader import (
    DEFAULT_BIDASK_SCHEMA,
    DEFAULT_OHLCV_SCHEMA,
    KibotSchemaError,
    floor_spread_cost,
    load_kibot,
)

# One normal bar, one locked bar (bid == ask), one CROSSED bar (bid > ask).
BIDASK_LINES = [
    "09/28/2009,09:30,96.89,96.95,96.88,96.90,96.95,97.00,96.94,96.96",
    "09/28/2009,09:31,96.90,96.92,96.90,96.91,96.90,96.92,96.90,96.91",
    "09/28/2009,09:32,97.10,97.20,97.05,97.15,96.90,96.95,96.85,96.90",
]
OHLCV_LINES = [
    "01/02/2009,09:30,85.52,85.60,85.50,85.55,1500",
    "01/02/2009,09:31,85.60,85.65,85.58,85.62,1000",
]


@pytest.fixture
def bidask_file(tmp_path):
    path = tmp_path / "TST_1m_bidask_adj.txt"
    path.write_text("\n".join(BIDASK_LINES) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def ohlcv_file(tmp_path):
    path = tmp_path / "TST.1m_adj.txt"
    path.write_text("\n".join(OHLCV_LINES) + "\n", encoding="utf-8")
    return path


class TestFloorSpreadCost:
    def test_crossed_spread_floors_to_zero(self):
        """A crossed quote must never yield a negative cost."""
        assert floor_spread_cost(-0.25) == 0.0

    def test_normal_spread_passes_through(self):
        assert floor_spread_cost(0.05) == pytest.approx(0.05)

    def test_locked_spread_stays_zero(self):
        assert floor_spread_cost(0.0) == 0.0

    def test_series_input_is_floored_elementwise(self):
        s = pd.Series([-0.30, 0.0, 0.02], index=pd.RangeIndex(3))
        out = floor_spread_cost(s)
        assert isinstance(out, pd.Series)
        assert (out >= 0).all()
        assert out.tolist() == pytest.approx([0.0, 0.0, 0.02])

    def test_ndarray_input_is_floored_elementwise(self):
        out = floor_spread_cost(np.array([-1.0, 0.5]))
        assert out.tolist() == pytest.approx([0.0, 0.5])


class TestLoadedCostColumns:
    def test_crossed_bar_produces_non_negative_cost(self, bidask_file):
        """The load-level contract: cost columns are >= 0 on every bar."""
        df = load_kibot(bidask_file, DEFAULT_BIDASK_SCHEMA)
        crossed = df["bid_close"] > df["ask_close"]
        assert crossed.sum() == 1, "fixture must contain exactly one crossed bar"
        assert (df["cost_spread"] >= 0).all()
        assert (df["cost_spread_bps"] >= 0).all()
        assert df.loc[crossed, "cost_spread"].iloc[0] == 0.0
        assert df.loc[crossed, "cost_spread_bps"].iloc[0] == 0.0

    def test_signed_spread_stays_negative_on_a_crossed_bar(self, bidask_file):
        """The diagnostic column must keep the sign, or the validator's
        crossed-bar checks would have nothing to detect."""
        df = load_kibot(bidask_file, DEFAULT_BIDASK_SCHEMA)
        crossed = df["bid_close"] > df["ask_close"]
        assert df.loc[crossed, "spread"].iloc[0] < 0
        assert df.loc[crossed, "spread_bps"].iloc[0] < 0

    def test_uncrossed_bars_agree_between_signed_and_floored(self, bidask_file):
        df = load_kibot(bidask_file, DEFAULT_BIDASK_SCHEMA)
        ok = df["spread"] >= 0
        assert (df.loc[ok, "cost_spread"] == df.loc[ok, "spread"]).all()
        assert (df.loc[ok, "cost_spread_bps"] == df.loc[ok, "spread_bps"]).all()


class TestSchemaMapping:
    def test_bidask_maps_to_two_ohlc_quartets(self, bidask_file):
        df = load_kibot(bidask_file, DEFAULT_BIDASK_SCHEMA)
        first = df.iloc[0]
        assert first["bid_open"] == 96.89 and first["bid_close"] == 96.90
        assert first["ask_open"] == 96.95 and first["ask_close"] == 96.96
        assert "volume" not in df.columns, "bid/ask layout carries no volume"

    def test_timestamp_parsed_from_us_eastern_wall_clock(self, bidask_file):
        df = load_kibot(bidask_file, DEFAULT_BIDASK_SCHEMA)
        assert df.index[0] == pd.Timestamp("2009-09-28 09:30:00")
        assert df.index.tz is None, "kept tz-naive ET on purpose"

    def test_ohlcv_layout_has_volume_and_no_quotes(self, ohlcv_file):
        df = load_kibot(ohlcv_file, DEFAULT_OHLCV_SCHEMA)
        assert df.iloc[0]["volume"] == 1500
        assert "spread" not in df.columns and "cost_spread" not in df.columns

    def test_wrong_schema_is_fatal_not_silently_mapped(self, bidask_file):
        with pytest.raises(KibotSchemaError, match="delivered 10 columns"):
            load_kibot(bidask_file, DEFAULT_OHLCV_SCHEMA)
