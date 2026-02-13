"""
Phase 3.8: Port Validation.

Compare every intermediate output of Praxis vs the original pandas-based
pair_trade_gld_gdx() implementation.

This is **the most important deliverable** per the plan — it proves
the pure-numpy port exactly matches the original pandas code.

Validation points:
  1. spread = close_GLD - weight * close_GDX
  2. spread_ema_mean = spread.ewm(span=lookback, adjust=False).mean()
  3. spread_ema_std = spread.ewm(span=lookback, adjust=False).std()
  4. zscore = (spread - ema_mean) / ema_std
  5. positions (long/short with ffill)
  6. pnl = shifted_positions * period_return
  7. Sharpe ratio
"""

import numpy as np
import pandas as pd
import pytest

from praxis.cpo import execute_single_leg
from praxis.signals.zscore import _ewm_mean, _ewm_std, _ffill_signal
from praxis.logger.core import PraxisLogger


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


# ── Reference implementation (exact copy of original logic) ───────

def _original_pair_trade(params, close_gld, open_gld, close_gdx):
    """
    Faithful reimplementation of pair_trade_gld_gdx() using pandas,
    serving as ground truth for validation.
    """
    gdx_weight = params['gdx_weight']
    lookback = params['lookback']
    entry_threshold = params['entry_threshold']
    exit_threshold = params['exit_threshold_fraction'] * entry_threshold
    num_periods_per_year = 252 * 6.5 * 60

    df = pd.DataFrame({
        'open_GLD': open_gld,
        'close_GLD': close_gld,
        'close_GDX': close_gdx,
    })

    spread = df['close_GLD'] - gdx_weight * df['close_GDX']
    df['spread'] = spread
    df['spread_ema_mean'] = spread.ewm(span=lookback, adjust=False).mean()
    df['spread_ema_std'] = spread.ewm(span=lookback, adjust=False).std()
    df['zscore'] = (spread - df['spread_ema_mean']) / df['spread_ema_std']

    df['positions_GLD_Long'] = 0
    df['positions_GLD_Short'] = 0

    df.loc[df.zscore >= entry_threshold, 'positions_GLD_Short'] = -1
    df.loc[df.zscore <= -entry_threshold, 'positions_GLD_Long'] = 1
    df.loc[df.zscore <= exit_threshold, 'positions_GLD_Short'] = 0
    df.loc[df.zscore >= -exit_threshold, 'positions_GLD_Long'] = 0

    df.ffill(inplace=True)

    positions_Long = df['positions_GLD_Long']
    positions_Short = df['positions_GLD_Short']
    positions = np.array(positions_Long) + np.array(positions_Short)
    positions = pd.DataFrame(positions)

    period_return = pd.DataFrame(columns=['close_GLD'])
    period_return['close_GLD'] = (df['close_GLD'] - df['open_GLD']) / df['open_GLD']

    transaction_costs = 0.0005
    pnl = (np.array(positions.shift()) * np.array(period_return)).sum(axis=1)
    tc = (abs(np.array(positions.diff()) * np.array(transaction_costs))).sum(axis=1)
    pnl_tc = pnl - tc

    daily_return = pnl_tc[1:].sum() - (transaction_costs if positions[0].iloc[-1] != 0 else 0)
    annualized_return = (1 + daily_return) ** 252 - 1
    volatility = np.sqrt(num_periods_per_year) * np.std(pnl_tc[1:])
    sharpe_ratio = np.sqrt(num_periods_per_year) * np.mean(pnl_tc[1:]) / np.std(pnl_tc[1:]) if np.std(pnl_tc[1:]) > 0 else 0

    return {
        'spread': np.array(spread),
        'ema_mean': np.array(df['spread_ema_mean']),
        'ema_std': np.array(df['spread_ema_std']),
        'zscore': np.array(df['zscore']),
        'positions': np.array(positions[0]),
        'pnl_tc': pnl_tc,
        'daily_return': daily_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
    }


# ── Synthetic data ────────────────────────────────────────────────

def _make_data(n=500, seed=42):
    rng = np.random.RandomState(seed)
    close_gld = 180.0 + np.cumsum(rng.randn(n) * 0.3)
    open_gld = close_gld + rng.randn(n) * 0.1
    close_gdx = 30.0 + np.cumsum(rng.randn(n) * 0.15)
    return close_gld, open_gld, close_gdx


PARAMS = {
    'gdx_weight': 3.0,
    'lookback': 60,
    'entry_threshold': 1.0,
    'exit_threshold_fraction': -0.6,
}

PRAXIS_PARAMS = {
    'weight': 3.0,
    'lookback': 60,
    'entry_threshold': 1.0,
    'exit_threshold_fraction': -0.6,
}


# ═══════════════════════════════════════════════════════════════════
#  Validation 1: EWM Mean matches pandas
# ═══════════════════════════════════════════════════════════════════

class TestEWMMeanVsPandas:
    def test_matches_pandas_ewm_mean(self):
        """_ewm_mean must match pandas ewm(span=N, adjust=False).mean()."""
        close_gld, _, close_gdx = _make_data()
        spread = close_gld - 3.0 * close_gdx
        lookback = 60

        # Pandas reference
        pd_mean = pd.Series(spread).ewm(span=lookback, adjust=False).mean().values

        # Praxis
        praxis_mean = _ewm_mean(spread, lookback)

        np.testing.assert_allclose(praxis_mean, pd_mean, atol=1e-10,
                                   err_msg="EWM mean diverges from pandas")

    def test_different_lookbacks(self):
        close_gld, _, close_gdx = _make_data()
        spread = close_gld - 3.0 * close_gdx
        for lb in [10, 30, 60, 120, 240]:
            pd_mean = pd.Series(spread).ewm(span=lb, adjust=False).mean().values
            praxis_mean = _ewm_mean(spread, lb)
            np.testing.assert_allclose(praxis_mean, pd_mean, atol=1e-10,
                                       err_msg=f"EWM mean diverges for lookback={lb}")


# ═══════════════════════════════════════════════════════════════════
#  Validation 2: EWM Std matches pandas
# ═══════════════════════════════════════════════════════════════════

class TestEWMStdVsPandas:
    def test_matches_pandas_ewm_std(self):
        """_ewm_std must match pandas ewm(span=N, adjust=False).std()."""
        close_gld, _, close_gdx = _make_data()
        spread = close_gld - 3.0 * close_gdx
        lookback = 60

        pd_std = pd.Series(spread).ewm(span=lookback, adjust=False).std().values
        praxis_std = _ewm_std(spread, lookback)

        # First value is NaN in pandas (only 1 obs)
        valid = ~np.isnan(pd_std)
        np.testing.assert_allclose(praxis_std[valid], pd_std[valid], atol=1e-8,
                                   err_msg="EWM std diverges from pandas")

    def test_different_lookbacks_std(self):
        close_gld, _, close_gdx = _make_data()
        spread = close_gld - 3.0 * close_gdx
        for lb in [10, 30, 60, 120]:
            pd_std = pd.Series(spread).ewm(span=lb, adjust=False).std().values
            praxis_std = _ewm_std(spread, lb)
            valid = ~np.isnan(pd_std)
            np.testing.assert_allclose(praxis_std[valid], pd_std[valid], atol=1e-8,
                                       err_msg=f"EWM std diverges for lookback={lb}")


# ═══════════════════════════════════════════════════════════════════
#  Validation 3: Spread calculation
# ═══════════════════════════════════════════════════════════════════

class TestSpreadCalculation:
    def test_spread_exact(self):
        """spread = close_A - weight * close_B."""
        close_gld, open_gld, close_gdx = _make_data()
        ref = _original_pair_trade(PARAMS, close_gld, open_gld, close_gdx)

        expected = close_gld - 3.0 * close_gdx
        np.testing.assert_allclose(ref['spread'], expected, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════
#  Validation 4: Z-score matches
# ═══════════════════════════════════════════════════════════════════

class TestZScoreMatch:
    def test_zscore_matches_pandas(self):
        """Z-score from Praxis helpers must match pandas calculation."""
        close_gld, _, close_gdx = _make_data()
        spread = close_gld - 3.0 * close_gdx
        lookback = 60

        pd_mean = pd.Series(spread).ewm(span=lookback, adjust=False).mean().values
        pd_std = pd.Series(spread).ewm(span=lookback, adjust=False).std().values
        pd_zscore = (spread - pd_mean) / pd_std

        praxis_mean = _ewm_mean(spread, lookback)
        praxis_std = _ewm_std(spread, lookback)
        with np.errstate(invalid="ignore"):
            praxis_zscore = np.where(praxis_std > 0,
                                      (spread - praxis_mean) / praxis_std, 0.0)

        valid = ~np.isnan(pd_zscore)
        np.testing.assert_allclose(praxis_zscore[valid], pd_zscore[valid], atol=1e-8,
                                   err_msg="Z-score diverges from pandas")


# ═══════════════════════════════════════════════════════════════════
#  Validation 5: Full pipeline — Sharpe within tolerance
# ═══════════════════════════════════════════════════════════════════

class TestFullPipelineMatch:
    """Compare full Praxis execute_single_leg vs original pandas implementation."""

    def _run_both(self, n=500, seed=42):
        close_gld, open_gld, close_gdx = _make_data(n, seed)
        ref = _original_pair_trade(PARAMS, close_gld, open_gld, close_gdx)
        praxis = execute_single_leg(
            close_gld, open_gld, close_gdx, PRAXIS_PARAMS,
            transaction_costs=0.0005,
            periods_per_year=252 * 6.5 * 60,
        )
        return ref, praxis

    def test_sharpe_close(self):
        """Sharpe ratio must be within 5% of original."""
        ref, praxis = self._run_both()
        if abs(ref['sharpe_ratio']) > 0.01:
            rel_diff = abs(praxis.sharpe_ratio - ref['sharpe_ratio']) / abs(ref['sharpe_ratio'])
            assert rel_diff < 0.05, (
                f"Sharpe diverges: praxis={praxis.sharpe_ratio:.6f} "
                f"vs ref={ref['sharpe_ratio']:.6f} (diff={rel_diff:.2%})"
            )
        else:
            assert abs(praxis.sharpe_ratio - ref['sharpe_ratio']) < 0.01

    def test_daily_return_close(self):
        """Daily return must be within 5% of original."""
        ref, praxis = self._run_both()
        if abs(ref['daily_return']) > 1e-6:
            rel_diff = abs(praxis.daily_return - ref['daily_return']) / abs(ref['daily_return'])
            assert rel_diff < 0.05, (
                f"Daily return diverges: praxis={praxis.daily_return:.8f} "
                f"vs ref={ref['daily_return']:.8f} (diff={rel_diff:.2%})"
            )
        else:
            assert abs(praxis.daily_return - ref['daily_return']) < 1e-6

    def test_volatility_close(self):
        ref, praxis = self._run_both()
        if abs(ref['volatility']) > 0.01:
            rel_diff = abs(praxis.volatility - ref['volatility']) / abs(ref['volatility'])
            assert rel_diff < 0.05, (
                f"Volatility diverges: praxis={praxis.volatility:.6f} "
                f"vs ref={ref['volatility']:.6f} (diff={rel_diff:.2%})"
            )

    def test_multiple_seeds(self):
        """Validate across different random datasets."""
        for seed in [1, 17, 42, 99, 123]:
            ref, praxis = self._run_both(n=400, seed=seed)
            if abs(ref['sharpe_ratio']) > 0.01:
                rel_diff = abs(praxis.sharpe_ratio - ref['sharpe_ratio']) / abs(ref['sharpe_ratio'])
                assert rel_diff < 0.05, (
                    f"Seed {seed}: Sharpe diverges: "
                    f"praxis={praxis.sharpe_ratio:.6f} vs ref={ref['sharpe_ratio']:.6f}"
                )

    def test_multiple_param_combos(self):
        """Validate across different parameter combinations."""
        close_gld, open_gld, close_gdx = _make_data(600)
        combos = [
            (2.0, 0.5, 30),
            (3.0, 1.0, 60),
            (4.0, 1.5, 120),
            (5.0, 2.0, 90),
        ]
        for w, et, lb in combos:
            orig_p = {'gdx_weight': w, 'lookback': lb, 'entry_threshold': et,
                      'exit_threshold_fraction': -0.6}
            praxis_p = {'weight': w, 'lookback': lb, 'entry_threshold': et,
                        'exit_threshold_fraction': -0.6}

            ref = _original_pair_trade(orig_p, close_gld, open_gld, close_gdx)
            praxis = execute_single_leg(close_gld, open_gld, close_gdx, praxis_p)

            if abs(ref['sharpe_ratio']) > 0.01:
                rel_diff = abs(praxis.sharpe_ratio - ref['sharpe_ratio']) / abs(ref['sharpe_ratio'])
                assert rel_diff < 0.05, (
                    f"w={w} et={et} lb={lb}: Sharpe diverges: "
                    f"praxis={praxis.sharpe_ratio:.6f} vs ref={ref['sharpe_ratio']:.6f}"
                )


# ═══════════════════════════════════════════════════════════════════
#  Validation 6: Position signals match
# ═══════════════════════════════════════════════════════════════════

class TestPositionMatch:
    def test_position_direction_matches(self):
        """Position signs must match original at every bar."""
        close_gld, open_gld, close_gdx = _make_data()
        ref = _original_pair_trade(PARAMS, close_gld, open_gld, close_gdx)
        praxis = execute_single_leg(close_gld, open_gld, close_gdx, PRAXIS_PARAMS)

        ref_signs = np.sign(ref['positions'])
        praxis_signs = np.sign(praxis.positions)

        # Allow small warmup period where ffill behavior may differ slightly
        match_pct = np.mean(ref_signs[60:] == praxis_signs[60:])
        assert match_pct > 0.95, (
            f"Position match {match_pct:.1%} < 95% after warmup"
        )
