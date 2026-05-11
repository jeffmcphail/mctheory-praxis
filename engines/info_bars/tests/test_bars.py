"""Unit tests for engines/info_bars/bars.py.

Synthetic trades, no DB required.
"""

from __future__ import annotations

import pytest

from engines.info_bars.bars import (
    DollarBars,
    VolumeBars,
    VolumeImbalanceBars,
    VolumeRunBars,
    Trade,
    build_for,
)


def mk(trade_id, ts, price, amount, side):
    return Trade(
        trade_id=trade_id,
        timestamp_ms=ts,
        price=price,
        amount=amount,
        quote_amount=price * amount,
        side=side,
    )


# ---------------------------------------------------------------- DollarBars


def test_dollar_bars_three_buys_close_at_third():
    """3 trades of $400k each at $1M threshold -> single closed bar
    containing all 3 trades (cum = $1.2M crosses on trade 3)."""
    builder = DollarBars(asset="BTC", threshold_value=1_000_000)
    # 0.005 BTC * $80,000,000 = $400k per trade. Use simpler $400k via
    # price=80000, amount=5 -> $400k
    trades = [
        mk(1, 1000, 80_000, 5.0, "buy"),
        mk(2, 1100, 80_001, 5.0, "buy"),
        mk(3, 1200, 80_002, 5.0, "buy"),
    ]
    closed = builder.push_all(trades)
    assert len(closed) == 1
    bar = closed[0]
    assert bar.bar_type == "dollar"
    assert bar.tick_count == 3
    assert bar.start_timestamp == 1000
    assert bar.end_timestamp == 1200
    assert bar.open == 80_000
    assert bar.close == 80_002
    assert bar.high == 80_002
    assert bar.low == 80_000
    assert bar.base_volume == pytest.approx(15.0)
    assert bar.quote_volume == pytest.approx(80_000 * 5 + 80_001 * 5 + 80_002 * 5)
    assert bar.buy_quote == pytest.approx(bar.quote_volume)
    assert bar.sell_quote == 0.0
    assert bar.imbalance_quote == pytest.approx(bar.buy_quote)


def test_dollar_bars_overshoot_stays_with_triggering_bar():
    """Overshoot stays with the bar that crossed the threshold;
    next bar opens with the next trade."""
    builder = DollarBars(asset="BTC", threshold_value=1_000_000)
    trades = [
        mk(1, 1000, 80_000, 5.0, "buy"),    # $400k
        mk(2, 1100, 80_000, 8.0, "buy"),    # $640k -> bar closes at $1.04M
        mk(3, 1200, 80_000, 5.0, "buy"),    # $400k -> open bar 2
        mk(4, 1300, 80_000, 8.0, "buy"),    # $640k -> bar 2 closes at $1.04M
    ]
    closed = builder.push_all(trades)
    assert len(closed) == 2
    assert closed[0].tick_count == 2
    assert closed[0].quote_volume == pytest.approx(80_000 * (5 + 8))
    assert closed[1].tick_count == 2
    assert closed[1].quote_volume == pytest.approx(80_000 * (5 + 8))


def test_dollar_bars_partial_state_not_persisted():
    """Trades below threshold leave a partial bar; flush_partial
    inspects it but the caller does not persist it."""
    builder = DollarBars(asset="BTC", threshold_value=1_000_000)
    trades = [mk(1, 1000, 80_000, 5.0, "buy")]
    closed = builder.push_all(trades)
    assert closed == []
    partial = builder.flush_partial()
    assert partial is not None
    assert partial.tick_count == 1
    assert partial.quote_volume == pytest.approx(400_000)


# ---------------------------------------------------------------- VolumeBars


def test_volume_bars_close_on_base_amount_threshold():
    """3 trades of 4 BTC each at threshold 10 BTC -> closes at trade 3."""
    builder = VolumeBars(asset="BTC", threshold_value=10.0)
    trades = [
        mk(1, 1000, 80_000, 4.0, "buy"),
        mk(2, 1100, 80_001, 4.0, "sell"),
        mk(3, 1200, 80_002, 4.0, "buy"),
    ]
    closed = builder.push_all(trades)
    assert len(closed) == 1
    assert closed[0].bar_type == "volume"
    assert closed[0].tick_count == 3
    assert closed[0].base_volume == pytest.approx(12.0)


# ---------------------------------------------------------------- VolumeImbalanceBars


def test_vib_closes_when_signed_quote_crosses_positive_threshold():
    """Pure buy aggression: cumulative +signed quote crosses
    +threshold -> bar closes with positive imbalance_quote."""
    builder = VolumeImbalanceBars(asset="BTC", threshold_value=500_000)
    trades = [
        mk(1, 1000, 80_000, 3.0, "buy"),   # +$240k
        mk(2, 1100, 80_000, 4.0, "buy"),   # +$320k cum +$560k -> close
    ]
    closed = builder.push_all(trades)
    assert len(closed) == 1
    bar = closed[0]
    assert bar.bar_type == "vib"
    assert bar.imbalance_quote > 0  # positive direction preserved
    assert bar.imbalance_quote == pytest.approx(80_000 * 7)
    assert bar.sell_quote == 0.0


def test_vib_closes_when_signed_quote_crosses_negative_threshold():
    """Pure sell aggression: cumulative -signed quote crosses
    -threshold -> bar closes with negative imbalance_quote."""
    builder = VolumeImbalanceBars(asset="BTC", threshold_value=500_000)
    trades = [
        mk(1, 1000, 80_000, 3.0, "sell"),  # -$240k
        mk(2, 1100, 80_000, 4.0, "sell"),  # -$320k cum -$560k -> close
    ]
    closed = builder.push_all(trades)
    assert len(closed) == 1
    bar = closed[0]
    assert bar.imbalance_quote < 0
    assert bar.imbalance_quote == pytest.approx(-80_000 * 7)
    assert bar.buy_quote == 0.0


def test_vib_buy_sell_offset_below_threshold():
    """If buys and sells offset, |theta| stays below threshold even
    with high notional traded -> no bar closes."""
    builder = VolumeImbalanceBars(asset="BTC", threshold_value=500_000)
    trades = [
        mk(1, 1000, 80_000, 5.0, "buy"),    # +$400k
        mk(2, 1100, 80_000, 5.0, "sell"),   # -$400k cum 0
        mk(3, 1200, 80_000, 5.0, "buy"),    # +$400k cum +$400k (< 500k)
    ]
    closed = builder.push_all(trades)
    assert closed == []


# ---------------------------------------------------------------- VolumeRunBars


def test_vrb_closes_when_max_side_run_exceeds_threshold():
    """Pure buy aggression: cum buy_quote >= threshold -> closes."""
    builder = VolumeRunBars(asset="BTC", threshold_value=500_000)
    trades = [
        mk(1, 1000, 80_000, 4.0, "buy"),   # buy_run = $320k
        mk(2, 1100, 80_000, 3.0, "buy"),   # buy_run = $560k -> close
    ]
    closed = builder.push_all(trades)
    assert len(closed) == 1
    bar = closed[0]
    assert bar.bar_type == "vrb"
    assert bar.buy_quote == pytest.approx(80_000 * 7)
    assert bar.sell_quote == 0.0


def test_vrb_closes_on_either_side_run():
    """Cum sell_quote reaches threshold first -> closes even with
    smaller buy activity in between."""
    builder = VolumeRunBars(asset="BTC", threshold_value=500_000)
    trades = [
        mk(1, 1000, 80_000, 1.0, "buy"),   # buy_run = $80k
        mk(2, 1100, 80_000, 4.0, "sell"),  # sell_run = $320k
        mk(3, 1200, 80_000, 3.0, "sell"),  # sell_run = $560k -> close
    ]
    closed = builder.push_all(trades)
    assert len(closed) == 1
    bar = closed[0]
    assert bar.sell_quote == pytest.approx(80_000 * 7)
    assert bar.buy_quote == pytest.approx(80_000 * 1)


# ---------------------------------------------------------------- shared


def test_idempotency_same_trades_same_bars():
    """Feeding the same trade list to two fresh builders produces
    identical closed bars."""
    trades = [
        mk(1, 1000, 80_000, 4.0, "buy"),
        mk(2, 1100, 80_000, 4.0, "buy"),
        mk(3, 1200, 80_000, 5.0, "sell"),
        mk(4, 1300, 80_000, 4.0, "buy"),
    ]
    a = DollarBars(asset="BTC", threshold_value=500_000)
    b = DollarBars(asset="BTC", threshold_value=500_000)
    bars_a = a.push_all(trades)
    bars_b = b.push_all(trades)
    assert len(bars_a) == len(bars_b)
    for x, y in zip(bars_a, bars_b):
        assert x == y


def test_ohlc_correctness_under_mixed_prices():
    """OHLC reflects actual first / max / min / last seen prices."""
    builder = DollarBars(asset="BTC", threshold_value=10_000_000)  # never closes
    trades = [
        mk(1, 1000, 80_000, 0.1, "buy"),
        mk(2, 1100, 80_500, 0.1, "buy"),    # high
        mk(3, 1200, 79_500, 0.1, "sell"),   # low
        mk(4, 1300, 80_100, 0.1, "buy"),    # close
    ]
    builder.push_all(trades)
    partial = builder.flush_partial()
    assert partial.open == 80_000
    assert partial.high == 80_500
    assert partial.low == 79_500
    assert partial.close == 80_100


def test_build_for_factory():
    assert build_for("dollar", "BTC", 1.0).bar_type == "dollar"
    assert build_for("volume", "BTC", 1.0).bar_type == "volume"
    assert build_for("vib", "BTC", 1.0).bar_type == "vib"
    assert build_for("vrb", "BTC", 1.0).bar_type == "vrb"
    with pytest.raises(ValueError):
        build_for("nope", "BTC", 1.0)


def test_iso_datetime_format_has_plus_zero():
    """Rule 35: ISO datetime strings end in +00:00 (UTC)."""
    builder = DollarBars(asset="BTC", threshold_value=500)
    # $800 > $500 threshold -> bar closes; check closed-bar format
    trades = [mk(1, 1_700_000_000_000, 80_000, 0.01, "buy")]
    closed = builder.push_all(trades)
    assert len(closed) == 1
    assert closed[0].start_datetime.endswith("+00:00")
    assert closed[0].end_datetime.endswith("+00:00")
