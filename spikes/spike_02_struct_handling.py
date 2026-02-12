"""
Spike 2: DuckDB STRUCT Handling
================================
McTheory Praxis — Phase 0

ASSUMPTION: DuckDB's STRUCT type handles the nested model definition schema
(Spec §6.1-6.10) — especially deeply nested STRUCTs with arrays, optional
fields, and JSON escape hatches.

TEST:
    1. Create fact_model_definition with full STRUCT schemas from §6.2
    2. Insert Chan CPO config and SMA config
    3. Query into nested STRUCT fields
    4. Test NULL handling, array-of-STRUCT, STRUCT update patterns

PASS CRITERIA:
    - All queries return correct results
    - Insert/query time < 100ms for single model

FAIL THRESHOLD: STRUCT nesting limit hit, NULL handling broken, or query
syntax too cumbersome.

FALLBACK: Flatten STRUCTs to JSON with typed Pydantic validation at the
application layer.
"""

import duckdb
import time
import json
from datetime import datetime
from dataclasses import dataclass


PASS_THRESHOLD_MS = 100


@dataclass
class TestResult:
    name: str
    passed: bool
    time_ms: float = 0.0
    detail: str = ""

    def __str__(self):
        icon = "✅" if self.passed else "❌"
        time_str = f" ({self.time_ms:.1f}ms)" if self.time_ms > 0 else ""
        detail_str = f" — {self.detail}" if self.detail else ""
        return f"  {icon} {self.name}{time_str}{detail_str}"


def create_schema(con: duckdb.DuckDBPyConnection) -> TestResult:
    """Test 1: Create the full fact_model_definition table with all STRUCT schemas."""
    
    start = time.perf_counter()
    
    # Full schema per spec §6.1-6.10
    # Using the actual STRUCT definitions from the spec
    con.execute("""
        CREATE TABLE fact_model_definition (
            -- Universal keys (§2.2)
            model_def_hist_id TIMESTAMP PRIMARY KEY,
            model_def_base_id BIGINT NOT NULL,
            model_def_bpk VARCHAR NOT NULL,
            
            -- Identity
            model_name VARCHAR NOT NULL,
            model_type VARCHAR NOT NULL,
            model_version VARCHAR,
            
            -- §6.2 Construction Params (the stress test — deeply nested)
            construction_params STRUCT(
                universe STRUCT(
                    method VARCHAR,
                    instruments VARCHAR[],
                    index_name VARCHAR,
                    filters STRUCT(
                        min_price DOUBLE, max_price DOUBLE, min_adv DOUBLE,
                        min_market_cap DOUBLE, exclude_types VARCHAR[],
                        sectors VARCHAR[], exclude_sectors VARCHAR[]
                    ),
                    refresh_frequency VARCHAR
                ),
                candidate_generation STRUCT(
                    method VARCHAR,
                    n_per_basket INTEGER, max_candidates INTEGER,
                    regression_method VARCHAR, residual_threshold DOUBLE,
                    clustering_method VARCHAR, n_clusters INTEGER, distance_metric VARCHAR,
                    factor_exposures VARCHAR[], exposure_threshold DOUBLE,
                    embedding_model VARCHAR, similarity_threshold DOUBLE
                ),
                validation STRUCT(
                    method VARCHAR,
                    tests VARCHAR[],
                    significance_level DOUBLE,
                    empirical STRUCT(
                        enabled BOOLEAN, n_simulations INTEGER, simulation_method VARCHAR,
                        surface_dimensions VARCHAR[], surface_percentiles DOUBLE[],
                        cache_key VARCHAR, cache_ttl_days INTEGER
                    ),
                    walk_forward STRUCT(
                        train_window VARCHAR, test_window VARCHAR, 
                        step VARCHAR, min_train_samples INTEGER
                    )
                ),
                selection STRUCT(
                    method VARCHAR, rank_by VARCHAR, top_k INTEGER,
                    min_threshold DOUBLE, diversification_penalty DOUBLE
                ),
                calibration STRUCT(
                    method VARCHAR, objective VARCHAR,
                    constraints STRUCT(
                        long_only BOOLEAN, max_weight DOUBLE, min_weight DOUBLE,
                        max_turnover DOUBLE, sum_to_one BOOLEAN
                    ),
                    estimation_window INTEGER, shrinkage VARCHAR,
                    halflife_method VARCHAR, recalibrate_frequency VARCHAR
                ),
                features STRUCT(
                    enabled BOOLEAN,
                    feature_list STRUCT(
                        name VARCHAR, method VARCHAR,
                        params STRUCT(windows INTEGER[], normalize BOOLEAN)
                    )[],
                    transforms VARCHAR[], lookback INTEGER
                ),
                ml_training STRUCT(
                    enabled BOOLEAN, algorithm VARCHAR,
                    target STRUCT(method VARCHAR, params STRUCT(
                        horizon INTEGER, threshold DOUBLE, pt DOUBLE, sl DOUBLE
                    )),
                    cv STRUCT(method VARCHAR, n_splits INTEGER, 
                             embargo_days INTEGER, purge_days INTEGER),
                    hyperopt STRUCT(enabled BOOLEAN, method VARCHAR, 
                                   n_trials INTEGER, metric VARCHAR),
                    sample_weights STRUCT(method VARCHAR, decay_halflife INTEGER)
                ),
                custom_function VARCHAR,
                custom_params JSON
            ),
            
            -- §6.3 Signal Params
            signal_params STRUCT(
                method VARCHAR,
                fast_period INTEGER, slow_period INTEGER, indicator VARCHAR,
                threshold_long DOUBLE, threshold_short DOUBLE, threshold_exit DOUBLE,
                lookback INTEGER, zscore_window INTEGER, half_life_method VARCHAR,
                model_ref VARCHAR, prediction_threshold DOUBLE,
                composite_method VARCHAR, component_signals VARCHAR[], 
                component_weights DOUBLE[],
                meta_label_enabled BOOLEAN, primary_signal_ref VARCHAR,
                custom_function VARCHAR, custom_params JSON
            ),
            
            -- §6.4 Entry Params
            entry_params STRUCT(
                method VARCHAR,
                long_threshold DOUBLE, short_threshold DOUBLE,
                confirmation_periods INTEGER, confirmation_pct DOUBLE,
                entry_window_start VARCHAR, entry_window_end VARCHAR, 
                entry_days VARCHAR[],
                regime_field VARCHAR,
                regime_conditions STRUCT(
                    high_vol STRUCT(condition VARCHAR, threshold_long DOUBLE, 
                                   threshold_short DOUBLE),
                    low_vol STRUCT(condition VARCHAR, threshold_long DOUBLE, 
                                  threshold_short DOUBLE)
                ),
                entry_order_type VARCHAR, limit_offset_bps DOUBLE,
                custom_function VARCHAR, custom_params JSON
            ),
            
            -- §6.5 Exit Params
            exit_params STRUCT(
                method VARCHAR,
                profit_target_pct DOUBLE, profit_target_atr_mult DOUBLE,
                stop_loss_pct DOUBLE, stop_loss_atr_mult DOUBLE, 
                stop_loss_zscore DOUBLE,
                trailing_enabled BOOLEAN, trailing_activation_pct DOUBLE,
                trailing_distance_pct DOUBLE,
                max_holding_bars INTEGER, max_holding_days INTEGER,
                decay_exit_enabled BOOLEAN,
                mean_reversion_level DOUBLE, mean_reversion_partial BOOLEAN,
                invalidation_enabled BOOLEAN, invalidation_test VARCHAR, 
                invalidation_threshold DOUBLE,
                triple_barrier STRUCT(pt_level DOUBLE, sl_level DOUBLE, 
                                     max_days INTEGER, vertical_first BOOLEAN),
                custom_function VARCHAR, custom_params JSON
            ),
            
            -- §6.6 Sizing Params
            sizing_params STRUCT(
                method VARCHAR,
                fraction DOUBLE,
                target_vol DOUBLE, vol_lookback INTEGER, vol_method VARCHAR, 
                vol_floor DOUBLE, vol_cap DOUBLE,
                kelly_fraction DOUBLE, kelly_distribution VARCHAR, 
                kelly_df DOUBLE, kelly_lookback INTEGER,
                risk_contribution_target VARCHAR,
                custom_risk_weights STRUCT(asset VARCHAR, weight DOUBLE)[],
                max_position_pct DOUBLE, max_sector_pct DOUBLE, 
                max_correlated_pct DOUBLE, max_drawdown_cut DOUBLE,
                regime_sizing_enabled BOOLEAN,
                regime_sizing STRUCT(
                    high_vol STRUCT(condition VARCHAR, method VARCHAR, 
                                   target_vol DOUBLE, max_position_pct DOUBLE),
                    low_vol STRUCT(condition VARCHAR, method VARCHAR, 
                                  target_vol DOUBLE, max_position_pct DOUBLE)
                ),
                forecast_scalar DOUBLE, forecast_cap DOUBLE, forecast_floor DOUBLE,
                custom_function VARCHAR, custom_params JSON
            ),
            
            -- §6.7 Single-Leg Params
            single_leg_params STRUCT(
                enabled BOOLEAN,
                target_selection STRUCT(
                    method VARCHAR, fixed_target VARCHAR,
                    liquidity_metric VARCHAR, lag_metric VARCHAR
                ),
                signal_basket STRUCT(
                    type VARCHAR, constituents VARCHAR[], weights DOUBLE[],
                    weight_method VARCHAR, reestimate_frequency VARCHAR
                ),
                lag_analysis STRUCT(
                    method VARCHAR, max_lag INTEGER, min_significance DOUBLE, 
                    stability_window INTEGER
                )
            ),
            
            -- §6.8 CPO Params
            cpo_params STRUCT(
                enabled BOOLEAN,
                search_method VARCHAR,
                parameter_grid STRUCT(
                    param_name VARCHAR, values DOUBLE[],
                    range DOUBLE[], scale VARCHAR
                )[],
                max_evaluations INTEGER,
                features STRUCT(
                    indicators VARCHAR[], apply_to VARCHAR[], 
                    lookback_windows INTEGER[]
                ),
                model STRUCT(
                    algorithm VARCHAR, target VARCHAR,
                    training STRUCT(method VARCHAR, min_train_days INTEGER)
                ),
                prediction STRUCT(frequency VARCHAR, selection VARCHAR),
                fallback STRUCT(
                    enabled BOOLEAN, monitoring_window_days INTEGER,
                    underperformance_threshold DOUBLE, action VARCHAR
                ),
                baseline STRUCT(method VARCHAR, recalculate_frequency VARCHAR)
            ),
            
            -- §6.9 Backtest Params
            backtest_params STRUCT(
                engine VARCHAR,
                reconciliation_tolerance DOUBLE,
                costs STRUCT(
                    commission_per_share DOUBLE, commission_min DOUBLE,
                    commission_pct DOUBLE, sec_fee_pct DOUBLE, 
                    exchange_fee_per_contract DOUBLE
                ),
                slippage STRUCT(method VARCHAR, fixed_bps DOUBLE, 
                               volume_participation DOUBLE, market_impact_coef DOUBLE),
                fills STRUCT(method VARCHAR, partial_fills BOOLEAN, fill_ratio DOUBLE),
                data STRUCT(survivorship_bias_free BOOLEAN, point_in_time BOOLEAN, 
                           corporate_actions VARCHAR),
                validation STRUCT(walk_forward BOOLEAN, n_splits INTEGER, 
                                 train_pct DOUBLE, embargo_days INTEGER)
            ),
            
            -- §6.10 Risk Params
            risk_params STRUCT(
                position_limits STRUCT(max_notional DOUBLE, max_quantity INTEGER, 
                                      max_pct_adv DOUBLE, max_concentration DOUBLE),
                greeks STRUCT(delta_limit DOUBLE, gamma_limit DOUBLE, 
                             vega_limit DOUBLE, theta_limit DOUBLE),
                var STRUCT(method VARCHAR, confidence DOUBLE, 
                          horizon_days INTEGER, lookback_days INTEGER),
                stress_tests STRUCT(scenarios VARCHAR[], 
                                   custom_shocks STRUCT(factor VARCHAR, shock_pct DOUBLE)[]),
                drawdown STRUCT(max_drawdown_pct DOUBLE, drawdown_action VARCHAR, 
                               recovery_threshold DOUBLE),
                correlation STRUCT(max_portfolio_correlation DOUBLE, 
                                  correlation_lookback INTEGER, regime_adjustment BOOLEAN)
            ),
            
            -- Workflow params
            workflow_params STRUCT(
                enabled BOOLEAN,
                steps STRUCT(
                    id VARCHAR,
                    function_name VARCHAR,
                    params JSON,
                    depends_on VARCHAR[],
                    condition VARCHAR,
                    for_each VARCHAR,
                    as_var VARCHAR,
                    parallel BOOLEAN
                )[]
            ),
            
            -- Metadata
            created_by VARCHAR,
            config_hash VARCHAR,
            source_yaml TEXT
        )
    """)
    
    elapsed = (time.perf_counter() - start) * 1000
    return TestResult("Create full schema (all STRUCTs)", True, elapsed)


def insert_chan_cpo(con: duckdb.DuckDBPyConnection) -> TestResult:
    """Test 2: Insert a Chan CPO model config with complex nested data."""
    
    start = time.perf_counter()
    
    con.execute("""
        INSERT INTO fact_model_definition VALUES (
            -- Keys
            TIMESTAMP '2026-02-12 10:00:00',   -- hist_id
            1234567890,                         -- base_id
            'chan_cpo_gld_gdx|v1.0',           -- bpk
            
            -- Identity
            'chan_cpo_gld_gdx',                 -- model_name
            'CPOModel',                         -- model_type
            'v1.0',                             -- model_version
            
            -- construction_params
            {
                universe: {
                    method: 'static',
                    instruments: ['GLD', 'GDX'],
                    index_name: NULL,
                    filters: NULL,
                    refresh_frequency: NULL
                },
                candidate_generation: NULL,
                validation: NULL,
                selection: NULL,
                calibration: NULL,
                features: {
                    enabled: true,
                    feature_list: [
                        {name: 'zscore', method: 'rolling_zscore', 
                         params: {windows: [20, 60], normalize: true}},
                        {name: 'mfi', method: 'money_flow_index', 
                         params: {windows: [14], normalize: false}},
                        {name: 'force_index', method: 'force_index', 
                         params: {windows: [13], normalize: true}}
                    ],
                    transforms: ['log_return', 'standardize'],
                    lookback: 252
                },
                ml_training: NULL,
                custom_function: NULL,
                custom_params: NULL
            },
            
            -- signal_params
            {
                method: 'zscore_spread',
                fast_period: NULL, slow_period: NULL, indicator: NULL,
                threshold_long: -2.0, threshold_short: 2.0, threshold_exit: 0.0,
                lookback: 60, zscore_window: 60, half_life_method: 'ols',
                model_ref: NULL, prediction_threshold: NULL,
                composite_method: NULL, component_signals: NULL,
                component_weights: NULL,
                meta_label_enabled: false, primary_signal_ref: NULL,
                custom_function: NULL, custom_params: NULL
            },
            
            -- entry_params
            {
                method: 'threshold',
                long_threshold: -2.0, short_threshold: 2.0,
                confirmation_periods: 0, confirmation_pct: NULL,
                entry_window_start: NULL, entry_window_end: NULL, entry_days: NULL,
                regime_field: NULL, regime_conditions: NULL,
                entry_order_type: 'market', limit_offset_bps: NULL,
                custom_function: NULL, custom_params: NULL
            },
            
            -- exit_params
            {
                method: 'mean_reversion',
                profit_target_pct: NULL, profit_target_atr_mult: NULL,
                stop_loss_pct: 0.05, stop_loss_atr_mult: NULL, stop_loss_zscore: NULL,
                trailing_enabled: false, trailing_activation_pct: NULL,
                trailing_distance_pct: NULL,
                max_holding_bars: NULL, max_holding_days: 30,
                decay_exit_enabled: false,
                mean_reversion_level: 0.0, mean_reversion_partial: false,
                invalidation_enabled: false, invalidation_test: NULL, 
                invalidation_threshold: NULL,
                triple_barrier: NULL,
                custom_function: NULL, custom_params: NULL
            },
            
            -- sizing_params
            {
                method: 'volatility_target',
                fraction: NULL,
                target_vol: 0.15, vol_lookback: 60, vol_method: 'ewm', 
                vol_floor: 0.05, vol_cap: 0.40,
                kelly_fraction: NULL, kelly_distribution: NULL, 
                kelly_df: NULL, kelly_lookback: NULL,
                risk_contribution_target: NULL, custom_risk_weights: NULL,
                max_position_pct: 0.20, max_sector_pct: NULL, 
                max_correlated_pct: NULL, max_drawdown_cut: 0.10,
                regime_sizing_enabled: false, regime_sizing: NULL,
                forecast_scalar: NULL, forecast_cap: NULL, forecast_floor: NULL,
                custom_function: NULL, custom_params: NULL
            },
            
            -- single_leg_params
            NULL,
            
            -- cpo_params (THE key feature for this model)
            {
                enabled: true,
                search_method: 'grid',
                parameter_grid: [
                    {param_name: 'gdx_weight', values: [0.3, 0.4, 0.5, 0.6, 0.7], 
                     range: NULL, scale: 'linear'},
                    {param_name: 'lookback', values: [20, 40, 60, 80, 100], 
                     range: NULL, scale: 'linear'},
                    {param_name: 'entry_threshold', values: [-1.5, -2.0, -2.5, -3.0], 
                     range: NULL, scale: 'linear'}
                ],
                max_evaluations: 500,
                features: {
                    indicators: ['zscore', 'mfi', 'force_index', 'donchian', 
                                 'atr', 'awesome_oscillator', 'adx'],
                    apply_to: ['spread', 'gld', 'gdx'],
                    lookback_windows: [20, 60, 120]
                },
                model: {
                    algorithm: 'random_forest',
                    target: 'best_parameter_set',
                    training: {method: 'time_series_split', min_train_days: 504}
                },
                prediction: {frequency: 'daily', selection: 'highest_probability'},
                fallback: {
                    enabled: true, monitoring_window_days: 60,
                    underperformance_threshold: -0.05, action: 'revert_to_static'
                },
                baseline: {method: 'equal_weight', recalculate_frequency: 'quarterly'}
            },
            
            -- backtest_params
            {
                engine: 'vectorized',
                reconciliation_tolerance: 0.001,
                costs: {
                    commission_per_share: 0.005, commission_min: 1.0,
                    commission_pct: NULL, sec_fee_pct: 0.0000278, 
                    exchange_fee_per_contract: NULL
                },
                slippage: {method: 'fixed', fixed_bps: 5.0, 
                          volume_participation: NULL, market_impact_coef: NULL},
                fills: {method: 'next_bar_open', partial_fills: false, fill_ratio: 1.0},
                data: {survivorship_bias_free: true, point_in_time: true, 
                      corporate_actions: 'adjusted'},
                validation: {walk_forward: true, n_splits: 5, 
                            train_pct: 0.6, embargo_days: 10}
            },
            
            -- risk_params
            {
                position_limits: {max_notional: 100000.0, max_quantity: NULL, 
                                 max_pct_adv: 0.01, max_concentration: 0.25},
                greeks: NULL,
                var: {method: 'historical', confidence: 0.99, 
                     horizon_days: 1, lookback_days: 252},
                stress_tests: {
                    scenarios: ['2008_crisis', 'covid_crash', 'rates_shock'],
                    custom_shocks: [
                        {factor: 'gold_price', shock_pct: -0.15},
                        {factor: 'gold_miners', shock_pct: -0.30}
                    ]
                },
                drawdown: {max_drawdown_pct: 0.15, drawdown_action: 'reduce_50pct', 
                          recovery_threshold: 0.05},
                correlation: {max_portfolio_correlation: 0.80, 
                             correlation_lookback: 120, regime_adjustment: true}
            },
            
            -- workflow_params
            NULL,
            
            -- Metadata
            'jeff',
            'sha256_placeholder_abc123',
            'model_name: chan_cpo_gld_gdx\nmodel_type: CPOModel\n...'
        )
    """)
    
    elapsed = (time.perf_counter() - start) * 1000
    return TestResult("Insert Chan CPO config", True, elapsed)


def insert_sma_model(con: duckdb.DuckDBPyConnection) -> TestResult:
    """Test 3: Insert a simple SMA model — most fields NULL."""
    
    start = time.perf_counter()
    
    con.execute("""
        INSERT INTO fact_model_definition VALUES (
            TIMESTAMP '2026-02-12 11:00:00',
            9876543210,
            'sma_crossover_spy|v1.0',
            'sma_crossover_spy',
            'SingleAssetModel',
            'v1.0',
            
            -- construction (simple static universe)
            {
                universe: {method: 'static', instruments: ['SPY'], 
                          index_name: NULL, filters: NULL, refresh_frequency: NULL},
                candidate_generation: NULL, validation: NULL, selection: NULL,
                calibration: NULL, features: NULL, ml_training: NULL,
                custom_function: NULL, custom_params: NULL
            },
            
            -- signal
            {
                method: 'sma_crossover',
                fast_period: 10, slow_period: 50, indicator: 'close',
                threshold_long: NULL, threshold_short: NULL, threshold_exit: NULL,
                lookback: NULL, zscore_window: NULL, half_life_method: NULL,
                model_ref: NULL, prediction_threshold: NULL,
                composite_method: NULL, component_signals: NULL, component_weights: NULL,
                meta_label_enabled: false, primary_signal_ref: NULL,
                custom_function: NULL, custom_params: NULL
            },
            
            -- entry
            {
                method: 'crossover',
                long_threshold: NULL, short_threshold: NULL,
                confirmation_periods: 0, confirmation_pct: NULL,
                entry_window_start: NULL, entry_window_end: NULL, entry_days: NULL,
                regime_field: NULL, regime_conditions: NULL,
                entry_order_type: 'market', limit_offset_bps: NULL,
                custom_function: NULL, custom_params: NULL
            },
            
            -- exit (simple stop loss)
            {
                method: 'crossover_reversal',
                profit_target_pct: NULL, profit_target_atr_mult: NULL,
                stop_loss_pct: 0.02, stop_loss_atr_mult: NULL, stop_loss_zscore: NULL,
                trailing_enabled: false, trailing_activation_pct: NULL,
                trailing_distance_pct: NULL,
                max_holding_bars: NULL, max_holding_days: NULL,
                decay_exit_enabled: false,
                mean_reversion_level: NULL, mean_reversion_partial: NULL,
                invalidation_enabled: false, invalidation_test: NULL,
                invalidation_threshold: NULL, triple_barrier: NULL,
                custom_function: NULL, custom_params: NULL
            },
            
            -- sizing (fixed fraction)
            {
                method: 'fixed_fraction', fraction: 1.0,
                target_vol: NULL, vol_lookback: NULL, vol_method: NULL,
                vol_floor: NULL, vol_cap: NULL,
                kelly_fraction: NULL, kelly_distribution: NULL,
                kelly_df: NULL, kelly_lookback: NULL,
                risk_contribution_target: NULL, custom_risk_weights: NULL,
                max_position_pct: 1.0, max_sector_pct: NULL,
                max_correlated_pct: NULL, max_drawdown_cut: NULL,
                regime_sizing_enabled: false, regime_sizing: NULL,
                forecast_scalar: NULL, forecast_cap: NULL, forecast_floor: NULL,
                custom_function: NULL, custom_params: NULL
            },
            
            NULL,  -- single_leg_params
            NULL,  -- cpo_params (not a CPO model)
            
            -- backtest
            {
                engine: 'vectorized',
                reconciliation_tolerance: 0.001,
                costs: {commission_per_share: 0.005, commission_min: 1.0,
                       commission_pct: NULL, sec_fee_pct: 0.0000278,
                       exchange_fee_per_contract: NULL},
                slippage: {method: 'fixed', fixed_bps: 5.0,
                          volume_participation: NULL, market_impact_coef: NULL},
                fills: {method: 'next_bar_open', partial_fills: false, fill_ratio: 1.0},
                data: {survivorship_bias_free: true, point_in_time: true,
                      corporate_actions: 'adjusted'},
                validation: {walk_forward: false, n_splits: NULL,
                            train_pct: NULL, embargo_days: NULL}
            },
            
            NULL,  -- risk_params
            NULL,  -- workflow_params
            
            'jeff',
            'sha256_placeholder_xyz789',
            'model_name: sma_crossover_spy\n...'
        )
    """)
    
    elapsed = (time.perf_counter() - start) * 1000
    return TestResult("Insert SMA config (mostly NULLs)", True, elapsed)


def run_struct_queries(con: duckdb.DuckDBPyConnection) -> list[TestResult]:
    """Tests 4-12: Query into nested STRUCT fields."""
    
    results = []
    
    # Test 4: Access top-level STRUCT field
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name, signal_params.method 
        FROM fact_model_definition 
        WHERE signal_params.method = 'zscore_spread'
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "Query top-level STRUCT field (signal_params.method)",
        len(r) == 1 and r[0][0] == 'chan_cpo_gld_gdx',
        elapsed,
        f"Got: {r}"
    ))
    
    # Test 5: Access deeply nested STRUCT (3 levels deep)
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name, sizing_params.target_vol 
        FROM fact_model_definition 
        WHERE sizing_params.target_vol > 0.10
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "Query nested STRUCT field (sizing_params.target_vol > 0.10)",
        len(r) == 1 and r[0][0] == 'chan_cpo_gld_gdx',
        elapsed,
        f"Got: {r}"
    ))
    
    # Test 6: Access array inside STRUCT (instruments list)
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name, construction_params.universe.instruments
        FROM fact_model_definition 
        WHERE list_contains(construction_params.universe.instruments, 'GLD')
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "Query array in STRUCT (list_contains instruments 'GLD')",
        len(r) == 1 and r[0][0] == 'chan_cpo_gld_gdx',
        elapsed,
        f"Got: {r}"
    ))
    
    # Test 7: Array-of-STRUCT access (parameter_grid) — THE critical test
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name, cpo_params.parameter_grid[1].param_name
        FROM fact_model_definition 
        WHERE cpo_params.enabled = true
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "Array-of-STRUCT: cpo_params.parameter_grid[1].param_name",
        len(r) == 1 and r[0][1] == 'gdx_weight',
        elapsed,
        f"Got: {r}"
    ))
    
    # Test 8: Access specific element deep in array-of-STRUCT
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name, cpo_params.parameter_grid[2].param_name,
               cpo_params.parameter_grid[2].values
        FROM fact_model_definition 
        WHERE cpo_params.enabled = true
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "Array-of-STRUCT: parameter_grid[2] (lookback values)",
        len(r) == 1 and r[0][1] == 'lookback' and r[0][2] == [20.0, 40.0, 60.0, 80.0, 100.0],
        elapsed,
        f"Got: {r}"
    ))
    
    # Test 9: Deeply nested STRUCT (4 levels: risk_params.stress_tests.custom_shocks[1].factor)
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name, 
               risk_params.stress_tests.custom_shocks[1].factor,
               risk_params.stress_tests.custom_shocks[1].shock_pct
        FROM fact_model_definition 
        WHERE model_name = 'chan_cpo_gld_gdx'
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "4-level nest: risk.stress_tests.custom_shocks[1].factor",
        len(r) == 1 and r[0][1] == 'gold_price' and r[0][2] == -0.15,
        elapsed,
        f"Got: {r}"
    ))
    
    # Test 10: NULL handling — query model where cpo_params IS NULL
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name FROM fact_model_definition 
        WHERE cpo_params IS NULL
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "NULL STRUCT handling (cpo_params IS NULL)",
        len(r) == 1 and r[0][0] == 'sma_crossover_spy',
        elapsed,
        f"Got: {r}"
    ))
    
    # Test 11: NULL nested field — construction_params.candidate_generation IS NULL
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name 
        FROM fact_model_definition 
        WHERE construction_params.candidate_generation IS NULL
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "NULL nested STRUCT (construction.candidate_generation IS NULL)",
        len(r) == 2,  # Both models have NULL candidate_generation
        elapsed,
        f"Got {len(r)} rows (both models)"
    ))
    
    # Test 12: Feature list array-of-STRUCT with nested params
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name,
               construction_params.features.feature_list[1].name,
               construction_params.features.feature_list[1].params.windows
        FROM fact_model_definition 
        WHERE construction_params.features.enabled = true
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "Array-of-STRUCT with nested params (features.feature_list[1].params.windows)",
        len(r) == 1 and r[0][1] == 'zscore' and r[0][2] == [20, 60],
        elapsed,
        f"Got: {r}"
    ))
    
    # Test 13: Query combining multiple STRUCT paths
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name,
               signal_params.method,
               sizing_params.method,
               backtest_params.engine,
               construction_params.universe.method
        FROM fact_model_definition 
        WHERE backtest_params.engine = 'vectorized'
          AND construction_params.universe.method = 'static'
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "Multi-STRUCT path query (signal + sizing + backtest + construction)",
        len(r) == 2,  # Both models match
        elapsed,
        f"Got {len(r)} rows"
    ))
    
    return results


def test_struct_update(con: duckdb.DuckDBPyConnection) -> list[TestResult]:
    """Test STRUCT update patterns (new version with modified fields)."""
    
    results = []
    
    # Test 14: Insert new version of Chan CPO with modified parameters
    start = time.perf_counter()
    con.execute("""
        INSERT INTO fact_model_definition 
        SELECT 
            TIMESTAMP '2026-02-12 14:00:00' AS model_def_hist_id,
            model_def_base_id,
            'chan_cpo_gld_gdx|v1.1' AS model_def_bpk,
            model_name,
            model_type,
            'v1.1' AS model_version,
            construction_params,
            -- Modify signal_params: change lookback from 60 to 90
            {
                method: signal_params.method,
                fast_period: signal_params.fast_period,
                slow_period: signal_params.slow_period,
                indicator: signal_params.indicator,
                threshold_long: signal_params.threshold_long,
                threshold_short: signal_params.threshold_short,
                threshold_exit: signal_params.threshold_exit,
                lookback: 90,
                zscore_window: 90,
                half_life_method: signal_params.half_life_method,
                model_ref: signal_params.model_ref,
                prediction_threshold: signal_params.prediction_threshold,
                composite_method: signal_params.composite_method,
                component_signals: signal_params.component_signals,
                component_weights: signal_params.component_weights,
                meta_label_enabled: signal_params.meta_label_enabled,
                primary_signal_ref: signal_params.primary_signal_ref,
                custom_function: signal_params.custom_function,
                custom_params: signal_params.custom_params
            } AS signal_params,
            entry_params,
            exit_params,
            sizing_params,
            single_leg_params,
            cpo_params,
            backtest_params,
            risk_params,
            workflow_params,
            created_by,
            'sha256_new_version',
            source_yaml
        FROM fact_model_definition
        WHERE model_def_hist_id = TIMESTAMP '2026-02-12 10:00:00'
    """)
    elapsed = (time.perf_counter() - start) * 1000
    
    # Verify the new version exists with updated lookback
    r = con.execute("""
        SELECT model_version, signal_params.lookback
        FROM fact_model_definition 
        WHERE model_name = 'chan_cpo_gld_gdx'
        ORDER BY model_def_hist_id
    """).fetchall()
    
    results.append(TestResult(
        "Version update (INSERT-AS-SELECT with modified STRUCT field)",
        len(r) == 2 and r[0][1] == 60 and r[1][1] == 90,
        elapsed,
        f"v1.0 lookback={r[0][1]}, v1.1 lookback={r[1][1]}"
    ))
    
    return results


def test_json_escape_hatch(con: duckdb.DuckDBPyConnection) -> list[TestResult]:
    """Test JSON fields for arbitrary/unstructured data alongside STRUCTs."""
    
    results = []
    
    # Test 15: Insert a model version with custom_params JSON (our actual pattern)
    custom_data = {
        "proprietary_signal": {
            "alpha_decay": 0.95,
            "regime_model": "hmm_3state",
            "features": ["vol_of_vol", "skew", "term_structure"]
        }
    }
    custom_json = json.dumps(custom_data)
    
    start = time.perf_counter()
    
    # Use inline JSON (this is how it would come from YAML → JSON serialization)
    con.execute(f"""
        INSERT INTO fact_model_definition VALUES (
            TIMESTAMP '2026-02-12 15:00:00',
            1234567890,
            'chan_cpo_gld_gdx|v1.2',
            'chan_cpo_gld_gdx',
            'CPOModel',
            'v1.2',
            {{
                universe: {{method: 'static', instruments: ['GLD', 'GDX'], 
                          index_name: NULL, filters: NULL, refresh_frequency: NULL}},
                candidate_generation: NULL, validation: NULL, selection: NULL,
                calibration: NULL, features: NULL, ml_training: NULL,
                custom_function: 'my_custom_constructor',
                custom_params: '{custom_json}'::JSON
            }},
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            'jeff', 'sha256_json_test', NULL
        )
    """)
    
    # Query the JSON back
    r = con.execute("""
        SELECT construction_params.custom_function,
               construction_params.custom_params::VARCHAR
        FROM fact_model_definition 
        WHERE model_def_hist_id = TIMESTAMP '2026-02-12 15:00:00'
    """).fetchall()
    
    elapsed = (time.perf_counter() - start) * 1000
    parsed = json.loads(r[0][1]) if r else None
    
    results.append(TestResult(
        "JSON escape hatch (custom_params INSERT + round-trip)",
        parsed is not None and parsed["proprietary_signal"]["alpha_decay"] == 0.95,
        elapsed,
        f"JSON round-trip: alpha_decay={parsed['proprietary_signal']['alpha_decay'] if parsed else 'FAIL'}"
    ))
    
    return results


def test_syntax_ergonomics(con: duckdb.DuckDBPyConnection) -> list[TestResult]:
    """Assess whether the STRUCT query syntax is too cumbersome for daily use."""
    
    results = []
    
    # Test 16: "Show me all models where sizing uses vol target > 0.10"
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name, sizing_params.target_vol
        FROM fact_model_definition
        WHERE sizing_params.method = 'volatility_target'
          AND sizing_params.target_vol > 0.10
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "Ergonomic: 'models with vol target > 0.10'",
        len(r) >= 1,
        elapsed,
        f"Natural SQL syntax works: {r}"
    ))
    
    # Test 17: "What's the CPO parameter grid for GLD/GDX?"
    start = time.perf_counter()
    r = con.execute("""
        SELECT 
            u.param_name,
            u.vals
        FROM (
            SELECT UNNEST(cpo_params.parameter_grid) AS grid_entry
            FROM fact_model_definition
            WHERE model_name = 'chan_cpo_gld_gdx'
              AND cpo_params.enabled = true
              AND model_def_hist_id = TIMESTAMP '2026-02-12 10:00:00'
        ) t,
        LATERAL (SELECT t.grid_entry.param_name AS param_name, 
                        t.grid_entry.values AS vals) u
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "Ergonomic: UNNEST array-of-STRUCT (parameter grid)",
        len(r) == 3 and r[0][0] == 'gdx_weight',
        elapsed,
        f"UNNEST works: {[row[0] for row in r]}"
    ))
    
    # Test 18: "Find models with walk-forward validation enabled"
    start = time.perf_counter()
    r = con.execute("""
        SELECT model_name, backtest_params.validation.walk_forward
        FROM fact_model_definition
        WHERE backtest_params.validation.walk_forward = true
    """).fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    results.append(TestResult(
        "Ergonomic: nested boolean query (walk_forward = true)",
        len(r) >= 1,
        elapsed,
        f"Deep boolean access works: {r}"
    ))
    
    return results


def main():
    print("McTheory Praxis — Spike 2: DuckDB STRUCT Handling")
    print("=" * 70)
    print()
    
    con = duckdb.connect(":memory:")
    all_results: list[TestResult] = []
    overall_pass = True
    
    # 1. Schema creation
    print("1. Schema creation...")
    r = create_schema(con)
    all_results.append(r)
    print(r)
    
    # 2. Insert Chan CPO
    print("\n2. Insert complex model (Chan CPO)...")
    try:
        r = insert_chan_cpo(con)
        all_results.append(r)
        print(r)
    except Exception as e:
        r = TestResult("Insert Chan CPO config", False, detail=str(e))
        all_results.append(r)
        print(r)
        overall_pass = False
    
    # 3. Insert SMA (simple model, mostly NULLs)
    print("\n3. Insert simple model (SMA)...")
    try:
        r = insert_sma_model(con)
        all_results.append(r)
        print(r)
    except Exception as e:
        r = TestResult("Insert SMA config", False, detail=str(e))
        all_results.append(r)
        print(r)
        overall_pass = False
    
    # 4-13. STRUCT queries
    print("\n4-13. STRUCT field queries...")
    try:
        for r in run_struct_queries(con):
            all_results.append(r)
            print(r)
            if not r.passed:
                overall_pass = False
    except Exception as e:
        r = TestResult("STRUCT queries", False, detail=str(e))
        all_results.append(r)
        print(r)
        overall_pass = False
    
    # 14. STRUCT update (versioning)
    print("\n14. STRUCT version update...")
    try:
        for r in test_struct_update(con):
            all_results.append(r)
            print(r)
            if not r.passed:
                overall_pass = False
    except Exception as e:
        r = TestResult("STRUCT update", False, detail=str(e))
        all_results.append(r)
        print(r)
        overall_pass = False
    
    # 15. JSON escape hatch
    print("\n15. JSON escape hatch...")
    try:
        for r in test_json_escape_hatch(con):
            all_results.append(r)
            print(r)
            if not r.passed:
                overall_pass = False
    except Exception as e:
        r = TestResult("JSON escape hatch", False, detail=str(e))
        all_results.append(r)
        print(r)
        overall_pass = False
    
    # 16-18. Syntax ergonomics
    print("\n16-18. Syntax ergonomics...")
    try:
        for r in test_syntax_ergonomics(con):
            all_results.append(r)
            print(r)
            if not r.passed:
                overall_pass = False
    except Exception as e:
        r = TestResult("Syntax ergonomics", False, detail=str(e))
        all_results.append(r)
        print(r)
        overall_pass = False
    
    # Performance summary
    print("\n" + "=" * 70)
    max_time = max(r.time_ms for r in all_results if r.time_ms > 0)
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    
    print(f"Tests: {passed}/{total} passed")
    print(f"Max single-operation time: {max_time:.1f}ms (threshold: {PASS_THRESHOLD_MS}ms)")
    print(f"Performance: {'PASS' if max_time < PASS_THRESHOLD_MS else 'FAIL'}")
    
    if overall_pass and max_time < PASS_THRESHOLD_MS:
        print("\nSPIKE 2 VERDICT: ✅ PASS")
        print("DuckDB STRUCT handling is viable for the model definition schema.")
        print("Decision D2: PROCEED with full STRUCTs (Option A).")
        print("\nKey findings:")
        print("  - 4+ levels of STRUCT nesting: works")
        print("  - Array-of-STRUCT with indexed access: works")
        print("  - UNNEST for array-of-STRUCT iteration: works")
        print("  - NULL handling at all levels: works")
        print("  - JSON escape hatch for arbitrary data: works")
        print("  - INSERT-AS-SELECT for version updates: works (verbose but functional)")
    else:
        print("\nSPIKE 2 VERDICT: ❌ FAIL")
        print("Review failures above. Consider JSON fallback (Option B/C).")
    
    print("=" * 70)
    
    con.close()
    return overall_pass


if __name__ == "__main__":
    main()
