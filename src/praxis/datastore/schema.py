"""
DuckDB table and view DDL definitions.

All tables follow the universal key pattern (§2.2):
    {entity}_hist_id TIMESTAMP PRIMARY KEY   -- creation time = PK
    {entity}_base_id BIGINT NOT NULL         -- xxHash64 of bpk
    {entity}_bpk VARCHAR NOT NULL            -- human-readable business key

Views follow the naming convention (§2.4):
    vew_  → Current state (latest version per base_id)
    vt2_  → Point-in-time with start_date/end_date derived from hist_id sequence
    rpt_  → Fact report views (pre-joined)

Applications NEVER query dim_*/fact_* tables directly. Always use views.
"""

# ═══════════════════════════════════════════════════════════════════
#  dim_security — §3
#  Minimal for Phase 1: just enough for SMA crossover on a ticker
# ═══════════════════════════════════════════════════════════════════

DIM_SECURITY = """
CREATE TABLE IF NOT EXISTS dim_security (
    -- Universal keys (§2.2)
    security_hist_id TIMESTAMP PRIMARY KEY,
    security_base_id BIGINT NOT NULL,
    security_bpk VARCHAR NOT NULL,

    -- Identity
    sec_type VARCHAR NOT NULL,

    -- Identifiers as columns (star schema, §3.2)
    isin VARCHAR,
    cusip VARCHAR,
    figi VARCHAR,
    sedol VARCHAR,
    ticker VARCHAR,
    ric VARCHAR,
    bloomberg_id VARCHAR,
    permid VARCHAR,
    cik VARCHAR,
    lei VARCHAR,
    contract_address VARCHAR,
    symbol VARCHAR,
    occ_symbol VARCHAR,
    exchange_code VARCHAR,

    -- Descriptive
    name VARCHAR,
    currency VARCHAR,
    exchange VARCHAR,
    sector VARCHAR,
    industry VARCHAR,
    country VARCHAR,

    -- Status
    status VARCHAR DEFAULT 'ACTIVE',

    -- Metadata
    created_by VARCHAR
);
"""

# ═══════════════════════════════════════════════════════════════════
#  dim_security_identifier_audit — §3.4
#  Tracks which source provided which identifier, when.
#  NOT used in operational queries — admin/audit only.
# ═══════════════════════════════════════════════════════════════════

DIM_SECURITY_IDENTIFIER_AUDIT = """
CREATE TABLE IF NOT EXISTS dim_security_identifier_audit (
    audit_hist_id TIMESTAMP PRIMARY KEY,
    audit_base_id BIGINT NOT NULL,
    audit_bpk VARCHAR NOT NULL,

    security_base_id BIGINT NOT NULL,
    secid_type VARCHAR NOT NULL,
    secid_value VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    confidence VARCHAR DEFAULT 'HIGH'
);
"""

# ═══════════════════════════════════════════════════════════════════
#  dim_security_exchange — §3.5
#  Exchange-specific attributes (ticker, SEDOL, RIC vary by exchange)
# ═══════════════════════════════════════════════════════════════════

DIM_SECURITY_EXCHANGE = """
CREATE TABLE IF NOT EXISTS dim_security_exchange (
    sec_exch_hist_id TIMESTAMP PRIMARY KEY,
    sec_exch_base_id BIGINT NOT NULL,
    sec_exch_bpk VARCHAR NOT NULL,

    security_base_id BIGINT NOT NULL,
    exchange_code VARCHAR NOT NULL,

    local_ticker VARCHAR,
    local_sedol VARCHAR,
    local_ric VARCHAR,
    is_primary_exchange BOOLEAN DEFAULT FALSE,

    lot_size INTEGER,
    tick_size DOUBLE,
    trading_hours VARCHAR,
    settlement_cycle VARCHAR
);
"""

# ═══════════════════════════════════════════════════════════════════
#  conflict_queue — §3.6
#  Holds identifier conflicts for manual or auto resolution.
# ═══════════════════════════════════════════════════════════════════

CONFLICT_QUEUE = """
CREATE TABLE IF NOT EXISTS conflict_queue (
    conflict_id BIGINT PRIMARY KEY,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    security_base_id BIGINT,
    source VARCHAR NOT NULL,
    batch_id VARCHAR NOT NULL,

    conflict_type VARCHAR NOT NULL,
    conflict_detail JSON,

    resolution_status VARCHAR DEFAULT 'OPEN',
    resolution_action VARCHAR,
    resolved_by VARCHAR,
    resolved_timestamp TIMESTAMP
);
"""

# ═══════════════════════════════════════════════════════════════════
#  fact_model_definition — §6.1
#  Full STRUCT schema (validated by Spike 2)
#  Phase 1 uses only signal_params, entry_params, exit_params,
#  sizing_params, backtest_params for SMA. Rest are NULL.
# ═══════════════════════════════════════════════════════════════════

FACT_MODEL_DEFINITION = """
CREATE TABLE IF NOT EXISTS fact_model_definition (
    -- Universal keys
    model_def_hist_id TIMESTAMP PRIMARY KEY,
    model_def_base_id BIGINT NOT NULL,
    model_def_bpk VARCHAR NOT NULL,

    -- Identity
    model_name VARCHAR NOT NULL,
    model_type VARCHAR NOT NULL,
    model_version VARCHAR,

    -- §6.2 Construction params
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
        data_source STRUCT(
            provider VARCHAR,
            resolution VARCHAR,
            fields VARCHAR[],
            adjustments VARCHAR,
            lookback_days INTEGER,
            api_params JSON
        ),
        pair_selection STRUCT(
            method VARCHAR,
            cointegration STRUCT(
                test VARCHAR, p_value_threshold DOUBLE, lookback_days INTEGER,
                min_half_life INTEGER, max_half_life INTEGER
            ),
            correlation STRUCT(
                min_correlation DOUBLE, method VARCHAR, lookback_days INTEGER
            ),
            filters STRUCT(
                same_sector BOOLEAN, same_country BOOLEAN, min_adv_ratio DOUBLE,
                max_market_cap_ratio DOUBLE
            ),
            max_pairs INTEGER,
            rebalance_frequency VARCHAR
        ),
        spread STRUCT(
            method VARCHAR,
            hedge_ratio STRUCT(method VARCHAR, lookback INTEGER, regularization DOUBLE),
            normalization VARCHAR,
            mean_lookback INTEGER,
            std_lookback INTEGER
        )
    ),

    -- §6.3 Signal params
    signal_params STRUCT(
        method VARCHAR,
        params JSON,
        lookback INTEGER,
        threshold DOUBLE,
        confirmation STRUCT(method VARCHAR, periods INTEGER, threshold DOUBLE),
        filters STRUCT(
            volume_filter STRUCT(min_adv_ratio DOUBLE),
            volatility_filter STRUCT(max_volatility DOUBLE, lookback INTEGER),
            trend_filter STRUCT(method VARCHAR, lookback INTEGER),
            time_filter STRUCT(allowed_days VARCHAR[], start_time VARCHAR, end_time VARCHAR)
        ),
        composite STRUCT(
            method VARCHAR,
            signals STRUCT(
                name VARCHAR, method VARCHAR, weight DOUBLE, params JSON
            )[]
        )
    ),

    -- §6.4 Entry params
    entry_params STRUCT(
        method VARCHAR,
        long_threshold DOUBLE,
        short_threshold DOUBLE,
        order_type VARCHAR,
        limit_offset_pct DOUBLE,
        time_in_force VARCHAR,
        max_entry_attempts INTEGER,
        scale_in STRUCT(
            enabled BOOLEAN, max_entries INTEGER, scale_factor DOUBLE, min_interval VARCHAR
        )
    ),

    -- §6.5 Exit params
    exit_params STRUCT(
        method VARCHAR,
        take_profit STRUCT(method VARCHAR, target DOUBLE, "trailing" BOOLEAN, trail_pct DOUBLE),
        stop_loss STRUCT(method VARCHAR, level DOUBLE, "trailing" BOOLEAN, trail_pct DOUBLE),
        time_exit STRUCT(max_holding_days INTEGER, max_calendar_days INTEGER),
        signal_exit STRUCT(method VARCHAR, threshold DOUBLE)
    ),

    -- §6.6 Sizing params
    sizing_params STRUCT(
        method VARCHAR,
        fixed_fraction DOUBLE,
        max_position_pct DOUBLE,
        kelly STRUCT(
            fraction DOUBLE, max_leverage DOUBLE, lookback INTEGER, shrinkage DOUBLE
        ),
        volatility STRUCT(
            target_vol DOUBLE, vol_lookback INTEGER, vol_method VARCHAR, max_leverage DOUBLE
        ),
        risk_parity STRUCT(
            risk_measure VARCHAR, lookback INTEGER, rebalance_frequency VARCHAR
        )
    ),

    -- §6.7 Single-leg params (SMA, EMA, momentum, etc.)
    single_leg_params STRUCT(
        indicators STRUCT(
            name VARCHAR, method VARCHAR, params JSON
        )[]
    ),

    -- §6.8 CPO params
    cpo_params STRUCT(
        enabled BOOLEAN,
        objective VARCHAR,
        features STRUCT(
            market_features VARCHAR[],
            technical_features VARCHAR[],
            custom_features STRUCT(name VARCHAR, function VARCHAR, params JSON)[]
        ),
        model STRUCT(
            type VARCHAR, hidden_layers INTEGER[], dropout DOUBLE,
            learning_rate DOUBLE, epochs INTEGER, batch_size INTEGER,
            early_stopping STRUCT(patience INTEGER, min_delta DOUBLE),
            regularization STRUCT(l1 DOUBLE, l2 DOUBLE)
        ),
        training STRUCT(
            train_window_days INTEGER, retrain_frequency VARCHAR,
            walk_forward BOOLEAN, n_splits INTEGER, embargo_days INTEGER
        ),
        parameter_grid STRUCT(
            param_name VARCHAR, "values" DOUBLE[]
        )[]
    ),

    -- §6.9 Backtest params
    backtest_params STRUCT(
        engine VARCHAR,
        reconciliation_tolerance DOUBLE,
        costs STRUCT(
            commission_per_share DOUBLE, commission_min DOUBLE,
            commission_pct DOUBLE, sec_fee_pct DOUBLE, exchange_fee_per_contract DOUBLE
        ),
        slippage STRUCT(method VARCHAR, fixed_bps DOUBLE, volume_participation DOUBLE, market_impact_coef DOUBLE),
        fills STRUCT(method VARCHAR, partial_fills BOOLEAN, fill_ratio DOUBLE),
        data STRUCT(survivorship_bias_free BOOLEAN, point_in_time BOOLEAN, corporate_actions VARCHAR),
        validation STRUCT(walk_forward BOOLEAN, n_splits INTEGER, train_pct DOUBLE, embargo_days INTEGER)
    ),

    -- §6.10 Risk params
    risk_params STRUCT(
        position_limits STRUCT(max_notional DOUBLE, max_quantity INTEGER, max_pct_adv DOUBLE, max_concentration DOUBLE),
        greeks STRUCT(delta_limit DOUBLE, gamma_limit DOUBLE, vega_limit DOUBLE, theta_limit DOUBLE),
        var STRUCT(method VARCHAR, confidence DOUBLE, horizon_days INTEGER, lookback_days INTEGER),
        stress_tests STRUCT(scenarios VARCHAR[], custom_shocks STRUCT(factor VARCHAR, shock_pct DOUBLE)[]),
        drawdown STRUCT(max_drawdown_pct DOUBLE, drawdown_action VARCHAR, recovery_threshold DOUBLE),
        correlation STRUCT(max_portfolio_correlation DOUBLE, correlation_lookback INTEGER, regime_adjustment BOOLEAN)
    ),

    -- Workflow params
    workflow_params STRUCT(
        enabled BOOLEAN,
        steps STRUCT(
            id VARCHAR,
            function VARCHAR,
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
);
"""


# ═══════════════════════════════════════════════════════════════════
#  fact_backtest_run — derived from §2.4 rpt_backtest_summary, §6.9
#  Stores one row per backtest execution.
# ═══════════════════════════════════════════════════════════════════

FACT_BACKTEST_RUN = """
CREATE TABLE IF NOT EXISTS fact_backtest_run (
    -- Universal keys
    run_hist_id TIMESTAMP PRIMARY KEY,
    run_base_id BIGINT NOT NULL,
    run_bpk VARCHAR NOT NULL,              -- 'model_name|run_timestamp_iso'

    -- Foreign keys
    model_def_base_id BIGINT NOT NULL,     -- FK to fact_model_definition
    model_def_hist_id TIMESTAMP,           -- Exact version used for this run

    -- Run metadata
    run_timestamp TIMESTAMP NOT NULL,      -- For AS-WAS replay (§2.4)
    run_mode VARCHAR NOT NULL,             -- 'backtest', 'paper', 'live'
    platform_mode VARCHAR,                 -- 'persistent', 'ephemeral'

    -- Date range
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,

    -- Results
    results STRUCT(
        total_return DOUBLE,
        annualized_return DOUBLE,
        sharpe_ratio DOUBLE,
        sortino_ratio DOUBLE,
        max_drawdown DOUBLE,
        max_drawdown_duration_days INTEGER,
        win_rate DOUBLE,
        profit_factor DOUBLE,
        total_trades INTEGER,
        avg_trade_return DOUBLE,
        avg_holding_days DOUBLE,
        calmar_ratio DOUBLE,
        volatility DOUBLE
    ),

    -- Parameters snapshot (captures exact config used)
    params JSON,

    -- Execution metadata
    duration_seconds DOUBLE,
    bar_count INTEGER,
    engine VARCHAR,                       -- 'vectorized', 'event_driven'

    -- Metadata
    created_by VARCHAR,
    notes VARCHAR
);
"""


# ═══════════════════════════════════════════════════════════════════
#  fact_log — §18.8
#  Database adapter writes here. No GIN index in DuckDB (use list functions).
# ═══════════════════════════════════════════════════════════════════

FACT_LOG = """
CREATE TABLE IF NOT EXISTS fact_log (
    log_id BIGINT PRIMARY KEY,
    log_timestamp TIMESTAMP NOT NULL,
    level INTEGER NOT NULL,
    level_name VARCHAR NOT NULL,
    message TEXT NOT NULL,
    tags VARCHAR[] NOT NULL DEFAULT [],

    -- Context
    context JSON,

    -- Who triggered this
    source VARCHAR,
    session_id VARCHAR
);
"""

# DuckDB doesn't have GIN indexes — use standard index on timestamp
FACT_LOG_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_log_timestamp ON fact_log (log_timestamp DESC);
"""


# ═══════════════════════════════════════════════════════════════════
#  VIEWS — §2.4
# ═══════════════════════════════════════════════════════════════════

# ── vew_ views: current state (latest version per base_id) ────────

VEW_SECURITY = """
CREATE OR REPLACE VIEW vew_security AS
SELECT * EXCLUDE (rn) FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY security_base_id ORDER BY security_hist_id DESC) AS rn
    FROM dim_security
) WHERE rn = 1;
"""

VEW_MODEL_DEFINITION = """
CREATE OR REPLACE VIEW vew_model_definition AS
SELECT * EXCLUDE (rn) FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY model_def_base_id ORDER BY model_def_hist_id DESC) AS rn
    FROM fact_model_definition
) WHERE rn = 1;
"""


# ── vt2_ views: point-in-time with start_date/end_date ───────────

VT2_SECURITY = """
CREATE OR REPLACE VIEW vt2_security AS
WITH bucketed AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY security_base_id, CAST(security_hist_id AS DATE)
               ORDER BY security_hist_id DESC
           ) AS rn_within_day
    FROM dim_security
),
latest_per_day AS (
    SELECT *,
           CAST(security_hist_id AS DATE) AS start_date,
           COALESCE(
               LEAD(CAST(security_hist_id AS DATE)) OVER (
                   PARTITION BY security_base_id ORDER BY security_hist_id
               ) - INTERVAL 1 DAY,
               DATE '9999-12-31'
           )::DATE AS end_date
    FROM bucketed
    WHERE rn_within_day = 1
)
SELECT * EXCLUDE (rn_within_day) FROM latest_per_day;
"""

VT2_MODEL_DEFINITION = """
CREATE OR REPLACE VIEW vt2_model_definition AS
WITH bucketed AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY model_def_base_id, CAST(model_def_hist_id AS DATE)
               ORDER BY model_def_hist_id DESC
           ) AS rn_within_day
    FROM fact_model_definition
),
latest_per_day AS (
    SELECT *,
           CAST(model_def_hist_id AS DATE) AS start_date,
           COALESCE(
               LEAD(CAST(model_def_hist_id AS DATE)) OVER (
                   PARTITION BY model_def_base_id ORDER BY model_def_hist_id
               ) - INTERVAL 1 DAY,
               DATE '9999-12-31'
           )::DATE AS end_date
    FROM bucketed
    WHERE rn_within_day = 1
)
SELECT * EXCLUDE (rn_within_day) FROM latest_per_day;
"""


# ── rpt_ views: fact report views (pre-joined) ───────────────────

RPT_BACKTEST_SUMMARY = """
CREATE OR REPLACE VIEW rpt_backtest_summary AS
SELECT
    bt.run_hist_id,
    bt.run_bpk,
    bt.run_timestamp,
    md.model_name,
    md.model_type,
    md.model_version,
    bt.run_mode,
    bt.start_date,
    bt.end_date,
    bt.results.total_return AS total_return,
    bt.results.annualized_return AS annualized_return,
    bt.results.sharpe_ratio AS sharpe_ratio,
    bt.results.sortino_ratio AS sortino_ratio,
    bt.results.max_drawdown AS max_drawdown,
    bt.results.total_trades AS total_trades,
    bt.results.win_rate AS win_rate,
    bt.results.profit_factor AS profit_factor,
    bt.duration_seconds,
    bt.engine,
    bt.notes
FROM fact_backtest_run bt
JOIN vew_model_definition md ON bt.model_def_base_id = md.model_def_base_id;
"""


# ═══════════════════════════════════════════════════════════════════
#  ldr_yfinance — §2.6 Loader working table
#  Wiped after each batch. Safety check: never wipe until all
#  records have terminal process_status.
# ═══════════════════════════════════════════════════════════════════

LDR_YFINANCE = """
CREATE TABLE IF NOT EXISTS ldr_yfinance (
    load_id BIGINT PRIMARY KEY,
    load_timestamp TIMESTAMP NOT NULL,
    batch_id VARCHAR NOT NULL,

    -- Source fields
    symbol VARCHAR NOT NULL,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adj_close DOUBLE,
    volume BIGINT,

    -- Processing outcome
    security_base_id BIGINT,
    process_status VARCHAR,
    reject_reason VARCHAR
);
"""

LDR_YFINANCE_HIST = """
CREATE TABLE IF NOT EXISTS ldr_yfinance_hist (
    load_id BIGINT NOT NULL,
    load_timestamp TIMESTAMP NOT NULL,
    batch_id VARCHAR NOT NULL,

    symbol VARCHAR NOT NULL,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adj_close DOUBLE,
    volume BIGINT,

    security_base_id BIGINT,
    process_status VARCHAR,
    reject_reason VARCHAR,

    archived_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# ═══════════════════════════════════════════════════════════════════
#  fact_price_daily — §2.11
#  OHLCV daily prices, FK to dim_security.
# ═══════════════════════════════════════════════════════════════════

FACT_PRICE_DAILY = """
CREATE TABLE IF NOT EXISTS fact_price_daily (
    price_hist_id TIMESTAMP PRIMARY KEY,
    price_base_id BIGINT NOT NULL,
    price_bpk VARCHAR NOT NULL,

    security_base_id BIGINT NOT NULL,
    trade_date DATE NOT NULL,

    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adj_close DOUBLE,
    volume BIGINT,

    source VARCHAR,
    batch_id VARCHAR,
    created_by VARCHAR
);
"""

# ═══════════════════════════════════════════════════════════════════
#  Ordered lists for initialization
# ═══════════════════════════════════════════════════════════════════

ALL_TABLES = [
    ("dim_security", DIM_SECURITY),
    ("dim_security_identifier_audit", DIM_SECURITY_IDENTIFIER_AUDIT),
    ("dim_security_exchange", DIM_SECURITY_EXCHANGE),
    ("conflict_queue", CONFLICT_QUEUE),
    ("fact_model_definition", FACT_MODEL_DEFINITION),
    ("fact_backtest_run", FACT_BACKTEST_RUN),
    ("fact_log", FACT_LOG),
    ("ldr_yfinance", LDR_YFINANCE),
    ("ldr_yfinance_hist", LDR_YFINANCE_HIST),
    ("fact_price_daily", FACT_PRICE_DAILY),
]

ALL_INDEXES = [
    ("idx_log_timestamp", FACT_LOG_INDEXES),
]

ALL_VIEWS = [
    ("vew_security", VEW_SECURITY),
    ("vew_model_definition", VEW_MODEL_DEFINITION),
    ("vt2_security", VT2_SECURITY),
    ("vt2_model_definition", VT2_MODEL_DEFINITION),
    ("rpt_backtest_summary", RPT_BACKTEST_SUMMARY),
]
