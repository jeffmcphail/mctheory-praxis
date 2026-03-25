"""
gui/mcb_studio/backend/batch_engine.py
=========================================
Runs multiple backtest configs sequentially and streams
a result frame after each one completes.

Used for cross-year and cross-asset comparison.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import AsyncGenerator

import pandas as pd

from data_feed import fetch_ohlcv
from praxis.indicators.market_cipher_b import MarketCipherB
from engines.backtest_engine import run_backtest_streaming, StatsTracker
from engines.mcb_strategies import get_strategy


@dataclass
class BatchRunConfig:
    label: str          # display label, e.g. "BTC 2022"
    symbol: str
    interval: str
    start: str
    end: str
    strategy: str
    params: dict
    tc_pct: float = 0.1


async def run_batch(
    runs: list[BatchRunConfig],
) -> AsyncGenerator[dict, None]:
    """
    Runs each config to completion, yields a result frame after each.
    Frame types:
        "batch_start"   — emitted once before any runs
        "run_start"     — emitted before each individual run
        "run_progress"  — progress % during a run (throttled to ~10 updates)
        "run_result"    — final stats after each run completes
        "batch_done"    — emitted once all runs complete
    """
    yield {"type": "batch_start", "total_runs": len(runs)}

    all_results = []

    for run_idx, run in enumerate(runs):
        yield {
            "type":    "run_start",
            "run_idx": run_idx,
            "label":   run.label,
            "symbol":  run.symbol,
            "start":   run.start,
            "end":     run.end,
            "strategy": run.strategy,
        }

        try:
            # Fetch data
            df = await asyncio.get_event_loop().run_in_executor(
                None, lambda: fetch_ohlcv(run.symbol, run.interval, run.start, run.end)
            )

            # Calculate MCb
            mcb = MarketCipherB()
            df = await asyncio.get_event_loop().run_in_executor(
                None, lambda: mcb.calculate(df)
            )

            # Run backtest — collect all frames (no delay)
            strategy = get_strategy(run.strategy, run.params)
            final_stats = None
            trades = []
            total_bars = len(df)
            last_pct_reported = -1

            async for frame in run_backtest_streaming(df, strategy, replay_delay_ms=0, tc_pct=run.tc_pct):
                if frame["type"] == "bar":
                    pct = int(frame["i"] / total_bars * 100)
                    # Throttle: only emit every 10%
                    if pct // 10 > last_pct_reported // 10:
                        last_pct_reported = pct
                        yield {
                            "type":    "run_progress",
                            "run_idx": run_idx,
                            "pct":     pct,
                        }
                elif frame["type"] == "final_stats":
                    final_stats = frame["stats"]
                    trades = frame.get("trades", [])

            result = {
                "type":     "run_result",
                "run_idx":  run_idx,
                "label":    run.label,
                "symbol":   run.symbol,
                "interval": run.interval,
                "start":    run.start,
                "end":      run.end,
                "strategy": run.strategy,
                "tc_pct":   run.tc_pct,
                "stats":    final_stats or {},
                "n_trades": len(trades),
                "error":    None,
            }

        except Exception as e:
            result = {
                "type":     "run_result",
                "run_idx":  run_idx,
                "label":    run.label,
                "symbol":   run.symbol,
                "interval": run.interval,
                "start":    run.start,
                "end":      run.end,
                "strategy": run.strategy,
                "tc_pct":   run.tc_pct,
                "stats":    {},
                "n_trades": 0,
                "error":    str(e),
            }

        all_results.append(result)
        yield result
        await asyncio.sleep(0.01)

    yield {
        "type":    "batch_done",
        "results": all_results,
    }
