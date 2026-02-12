"""
BacktestResult storage (Phase 1.8).

Bridges BacktestOutput → fact_backtest_run in DuckDB.
Generates universal keys, serializes config as JSON params snapshot,
and inserts the results STRUCT.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

import duckdb

from praxis.backtest import BacktestOutput, BacktestMetrics
from praxis.config import ModelConfig
from praxis.datastore.keys import EntityKeys
from praxis.logger.core import PraxisLogger


class BacktestResultStore:
    """
    Stores backtest results in fact_backtest_run.

    Usage:
        store = BacktestResultStore(conn)
        run_id = store.save(config, bt_output)
        result = store.load(run_id)
        results = store.list_runs(model_name="sma_crossover")
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn
        self._log = PraxisLogger.instance()

    def save(
        self,
        config: ModelConfig,
        output: BacktestOutput,
        *,
        model_def_base_id: Optional[int] = None,
        model_def_hist_id: Optional[datetime] = None,
        run_mode: str = "backtest",
        platform_mode: str = "persistent",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        created_by: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Save a backtest result to fact_backtest_run.

        Args:
            config: ModelConfig used for this run.
            output: BacktestOutput from VectorizedEngine.
            model_def_base_id: FK to fact_model_definition (optional).
            model_def_hist_id: Exact model version used (optional).
            run_mode: 'backtest', 'paper', 'live'.
            platform_mode: 'persistent' or 'ephemeral'.
            start_date: Backtest start date (ISO string or auto-derived).
            end_date: Backtest end date (ISO string or auto-derived).

        Returns:
            Dict with run keys: {run_hist_id, run_base_id, run_bpk}.
        """
        now = datetime.now(timezone.utc)
        run_bpk = f"{config.model.name}|{now.isoformat()}"
        keys = EntityKeys.create(run_bpk)

        # Params snapshot: full config as JSON
        params_json = json.dumps(
            config.to_dict(), default=str, sort_keys=True
        )

        # Engine type
        engine = "vectorized"
        if config.backtest and config.backtest.engine:
            engine = config.backtest.engine

        # Derive dates if not provided
        if start_date is None:
            start_date = "1970-01-01"
        if end_date is None:
            end_date = now.strftime("%Y-%m-%d")

        m = output.metrics

        self._log.info(
            f"Saving backtest result: {config.model.name} "
            f"(return={m.total_return:.2%}, sharpe={m.sharpe_ratio:.2f})",
            tags={"datastore", "trade_cycle"},
            model_name=config.model.name,
            run_bpk=run_bpk,
        )

        self._conn.execute("""
            INSERT INTO fact_backtest_run (
                run_hist_id, run_base_id, run_bpk,
                model_def_base_id, model_def_hist_id,
                run_timestamp, run_mode, platform_mode,
                start_date, end_date,
                results,
                params,
                duration_seconds, bar_count, engine,
                created_by, notes
            ) VALUES (
                $1, $2, $3,
                $4, $5,
                $6, $7, $8,
                $9::DATE, $10::DATE,
                {
                    total_return: $11,
                    annualized_return: $12,
                    sharpe_ratio: $13,
                    sortino_ratio: $14,
                    max_drawdown: $15,
                    max_drawdown_duration_days: $16,
                    win_rate: $17,
                    profit_factor: $18,
                    total_trades: $19,
                    avg_trade_return: $20,
                    avg_holding_days: $21,
                    calmar_ratio: $22,
                    volatility: $23
                },
                $24::JSON,
                $25, $26, $27,
                $28, $29
            )
        """, [
            keys.hist_id, keys.base_id, keys.bpk,
            model_def_base_id or 0, model_def_hist_id,
            now, run_mode, platform_mode,
            start_date, end_date,
            m.total_return, m.annualized_return,
            m.sharpe_ratio, m.sortino_ratio,
            m.max_drawdown, m.max_drawdown_duration_days,
            m.win_rate, m.profit_factor,
            m.total_trades, m.avg_trade_return,
            m.avg_holding_days, m.calmar_ratio, m.volatility,
            params_json,
            output.duration_seconds, output.bar_count, engine,
            created_by, notes,
        ])

        self._log.debug(
            f"Saved run {run_bpk}",
            tags={"datastore"},
        )

        return {
            "run_hist_id": keys.hist_id,
            "run_base_id": keys.base_id,
            "run_bpk": keys.bpk,
        }

    def load(self, run_bpk: str) -> Optional[dict[str, Any]]:
        """
        Load a backtest result by BPK.

        Returns dict with all fields or None if not found.
        """
        result = self._conn.execute("""
            SELECT
                run_hist_id, run_base_id, run_bpk,
                model_def_base_id, run_timestamp,
                run_mode, platform_mode,
                start_date, end_date,
                results,
                params,
                duration_seconds, bar_count, engine,
                created_by, notes
            FROM fact_backtest_run
            WHERE run_bpk = $1
        """, [run_bpk]).fetchone()

        if result is None:
            return None

        cols = [
            "run_hist_id", "run_base_id", "run_bpk",
            "model_def_base_id", "run_timestamp",
            "run_mode", "platform_mode",
            "start_date", "end_date",
            "results", "params",
            "duration_seconds", "bar_count", "engine",
            "created_by", "notes",
        ]
        return dict(zip(cols, result))

    def list_runs(
        self,
        model_name: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        List backtest runs, optionally filtered by model name.

        Returns list of summary dicts with key metrics.
        """
        if model_name:
            rows = self._conn.execute("""
                SELECT
                    run_bpk, run_timestamp,
                    results.total_return,
                    results.sharpe_ratio,
                    results.max_drawdown,
                    results.total_trades,
                    engine, bar_count
                FROM fact_backtest_run
                WHERE run_bpk LIKE $1 || '|%'
                ORDER BY run_timestamp DESC
                LIMIT $2
            """, [model_name, limit]).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT
                    run_bpk, run_timestamp,
                    results.total_return,
                    results.sharpe_ratio,
                    results.max_drawdown,
                    results.total_trades,
                    engine, bar_count
                FROM fact_backtest_run
                ORDER BY run_timestamp DESC
                LIMIT $1
            """, [limit]).fetchall()

        cols = [
            "run_bpk", "run_timestamp",
            "total_return", "sharpe_ratio", "max_drawdown",
            "total_trades", "engine", "bar_count",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def count(self) -> int:
        """Total number of backtest runs stored."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM fact_backtest_run"
        ).fetchone()[0]
