"""
Tests for Simple Data Feed (Phase 1.10).

Covers:
- Synthetic price generation (GBM)
- OHLCV structure validation
- Multi-ticker synthetic data
- PriceData.from_config extraction
- CLI --synthetic mode
- End-to-end: synthetic data → backtest → metrics
- yfinance mock test (verifies wiring without network)
"""

from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from praxis.config import ModelConfig
from praxis.data import fetch_prices, generate_synthetic_prices, PriceData
from praxis.logger.core import PraxisLogger
from praxis.registry import FunctionRegistry
from praxis.registry.defaults import register_defaults
from praxis.runner import PraxisRunner, run_cli


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_singletons():
    FunctionRegistry.reset()
    PraxisLogger.reset()
    yield
    FunctionRegistry.reset()
    PraxisLogger.reset()


@pytest.fixture
def sma_config():
    return ModelConfig.from_dict({
        "model": {"name": "test_sma", "type": "SingleAssetModel"},
        "signal": {"method": "sma_crossover", "fast_period": 10, "slow_period": 50},
        "sizing": {"method": "fixed_fraction", "fraction": 1.0},
    })


# ═══════════════════════════════════════════════════════════════════
#  Synthetic Price Generation
# ═══════════════════════════════════════════════════════════════════

class TestSyntheticPrices:
    def test_default_generation(self):
        df = generate_synthetic_prices()
        assert len(df) == 252
        assert "date" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_custom_bars(self):
        df = generate_synthetic_prices(n_bars=100)
        assert len(df) == 100

    def test_deterministic_with_seed(self):
        a = generate_synthetic_prices(seed=42)
        b = generate_synthetic_prices(seed=42)
        assert a["close"].to_list() == b["close"].to_list()

    def test_different_seeds_differ(self):
        a = generate_synthetic_prices(seed=1)
        b = generate_synthetic_prices(seed=2)
        assert a["close"].to_list() != b["close"].to_list()

    def test_ohlcv_relationships(self):
        """High >= max(open, close), Low <= min(open, close)."""
        df = generate_synthetic_prices(n_bars=100, seed=42)
        for i in range(len(df)):
            row = df.row(i, named=True)
            assert row["high"] >= max(row["open"], row["close"])
            assert row["low"] <= min(row["open"], row["close"])

    def test_dates_are_weekdays(self):
        df = generate_synthetic_prices(n_bars=100, seed=42)
        dates = df["date"].to_list()
        for d in dates:
            assert d.weekday() < 5  # Mon-Fri only

    def test_initial_price(self):
        df = generate_synthetic_prices(n_bars=10, initial_price=50.0, seed=42)
        # First open should be the initial price
        assert df["open"][0] == 50.0

    def test_drift_positive(self):
        """Strong positive drift should trend upward."""
        df = generate_synthetic_prices(
            n_bars=252, drift=0.005, volatility=0.01, seed=42
        )
        assert df["close"][-1] > df["close"][0]

    def test_drift_negative(self):
        """Strong negative drift should trend downward."""
        df = generate_synthetic_prices(
            n_bars=252, drift=-0.005, volatility=0.01, seed=42
        )
        assert df["close"][-1] < df["close"][0]

    def test_ticker_column(self):
        df = generate_synthetic_prices(n_bars=10, ticker="AAPL")
        assert "ticker" in df.columns
        assert df["ticker"][0] == "AAPL"

    def test_no_ticker_by_default(self):
        df = generate_synthetic_prices(n_bars=10)
        assert "ticker" not in df.columns

    def test_volume_positive(self):
        df = generate_synthetic_prices(n_bars=100, seed=42)
        assert (df["volume"] > 0).all()


# ═══════════════════════════════════════════════════════════════════
#  yfinance Mock Tests
# ═══════════════════════════════════════════════════════════════════

class TestYFinanceMock:
    """Test yfinance wiring with mocked network calls."""

    def _make_mock_df(self, n=100):
        """Create a pandas DataFrame mimicking yfinance output."""
        dates = pd.date_range("2023-01-03", periods=n, freq="B")
        return pd.DataFrame({
            "Open": np.random.randn(n) * 1 + 150,
            "High": np.random.randn(n) * 1 + 152,
            "Low": np.random.randn(n) * 1 + 148,
            "Close": np.random.randn(n) * 1 + 150,
            "Volume": np.random.randint(1e6, 1e7, n),
        }, index=dates)

    @patch("praxis.data.yf")
    def test_single_ticker(self, mock_yf):
        mock_yf.download.return_value = self._make_mock_df()
        df = fetch_prices("AAPL", start="2023-01-01", end="2023-06-01")

        assert len(df) == 100
        assert "close" in df.columns
        mock_yf.download.assert_called_once()

    @patch("praxis.data.yf")
    def test_multiple_tickers(self, mock_yf):
        mock_yf.download.return_value = self._make_mock_df(50)
        df = fetch_prices(
            ["AAPL", "MSFT"], start="2023-01-01", end="2023-06-01"
        )

        assert "ticker" in df.columns
        assert mock_yf.download.call_count == 2

    @patch("praxis.data.yf")
    def test_empty_response_raises(self, mock_yf):
        mock_yf.download.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No data"):
            fetch_prices("INVALID", start="2023-01-01")

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            fetch_prices("AAPL", provider="bloomberg")


# ═══════════════════════════════════════════════════════════════════
#  PriceData.from_config
# ═══════════════════════════════════════════════════════════════════

class TestPriceDataFromConfig:
    def test_no_tickers_raises(self):
        config = ModelConfig.from_dict({
            "model": {"name": "test", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover"},
        })
        with pytest.raises(ValueError, match="No tickers"):
            PriceData.from_config(config)

    @patch("praxis.data.yf")
    def test_tickers_from_config(self, mock_yf):
        mock_df = pd.DataFrame({
            "Open": [150.0], "High": [152.0],
            "Low": [148.0], "Close": [150.0], "Volume": [1000000],
        }, index=pd.DatetimeIndex(["2023-01-03"]))
        mock_yf.download.return_value = mock_df

        config = ModelConfig.from_dict({
            "model": {"name": "test", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover"},
            "construction": {
                "universe": {"method": "static", "instruments": ["AAPL"]},
            },
        })
        df = PriceData.from_config(config)
        assert len(df) > 0

    @patch("praxis.data.yf")
    def test_tickers_override(self, mock_yf):
        mock_df = pd.DataFrame({
            "Open": [150.0], "High": [152.0],
            "Low": [148.0], "Close": [150.0], "Volume": [1000000],
        }, index=pd.DatetimeIndex(["2023-01-03"]))
        mock_yf.download.return_value = mock_df

        config = ModelConfig.from_dict({
            "model": {"name": "test", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover"},
        })
        df = PriceData.from_config(config, tickers=["MSFT"])
        assert len(df) > 0
        mock_yf.download.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
#  CLI --synthetic Mode
# ═══════════════════════════════════════════════════════════════════

class TestCLISynthetic:
    def test_synthetic_flag(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("""
model:
  name: cli_synthetic
  type: SingleAssetModel
signal:
  method: sma_crossover
  fast_period: 10
  slow_period: 50
""")
        result = run_cli([str(yaml_file), "--synthetic", "200"])
        assert result == 0

    def test_synthetic_default_bars(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("""
model:
  name: cli_default_synth
  type: SingleAssetModel
signal:
  method: sma_crossover
""")
        result = run_cli([str(yaml_file), "--synthetic"])
        assert result == 0


# ═══════════════════════════════════════════════════════════════════
#  End-to-End: Synthetic → Backtest → Metrics
# ═══════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_synthetic_to_backtest(self, sma_config):
        register_defaults()
        prices = generate_synthetic_prices(n_bars=252, seed=42)
        runner = PraxisRunner()
        result = runner.run_config(sma_config, prices)

        assert result.success is True
        assert result.metrics["total_trades"] > 0
        assert isinstance(result.metrics["sharpe_ratio"], float)

    def test_milestone1_sma_crossover(self, tmp_path):
        """
        Milestone 1 criterion:
        `praxis run sma_crossover.yaml` produces BacktestResult
        with Sharpe ratio, total return, max drawdown.
        """
        yaml_file = tmp_path / "sma_crossover.yaml"
        yaml_file.write_text("""
model:
  name: sma_crossover
  type: SingleAssetModel
  version: v1.0
signal:
  method: sma_crossover
  fast_period: 10
  slow_period: 50
sizing:
  method: fixed_fraction
  fraction: 1.0
backtest:
  engine: vectorized
""")
        exit_code = run_cli([str(yaml_file), "--synthetic", "252"])
        assert exit_code == 0
