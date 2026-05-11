"""Info Bars v0.1 -- Lopez de Prado AFML Ch. 2 information-driven bars.

Public surface:

- bars.DollarBars, VolumeBars, VolumeImbalanceBars, VolumeRunBars
- bars.build_for(bar_type, asset, threshold_value) factory
- writer.backfill_slice, writer.live_update

Aggressor-direction convention (verified against
engines/crypto_data_collector.py):
    side='buy'  => taker bought (BUY-side aggression,  sign +1)
    side='sell' => taker sold   (SELL-side aggression, sign -1)

The trade-table column for milliseconds is `timestamp` (not
`timestamp_ms`); ccxt's fetch_trades emits epoch-ms.
"""

from engines.info_bars.bars import (  # noqa: F401
    BarBuilder,
    ClosedBar,
    DollarBars,
    VolumeBars,
    VolumeImbalanceBars,
    VolumeRunBars,
    build_for,
)
