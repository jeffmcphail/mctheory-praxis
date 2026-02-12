"""
Tests for Pydantic config schemas (Phase 1.4).

Covers:
- ModelConfig validation (SMA minimal, stat arb complex)
- Two-User Test (§1.3): high schooler SMA + quant CPO
- YAML parsing and validation
- Config hashing (identical configs → same hash)
- Error cases (missing required fields)
- Individual param sections
- LoggerConfig
- PlatformConfig
- BPK generation
"""

import pytest
import yaml

from praxis.config import (
    ModelConfig,
    ModelIdentity,
    ModelType,
    PlatformMode,
    SignalParams,
    EntryParams,
    ExitParams,
    SizingParams,
    BacktestParams,
    ConstructionParams,
    UniverseConfig,
    DataSourceConfig,
    CPOParams,
    RiskParams,
    WorkflowParams,
    WorkflowStep,
    LoggerConfig,
    LogAdapterConfig,
    LogRoutingConfig,
    PlatformConfig,
    CostsConfig,
    SlippageConfig,
    TripleBarrierConfig,
)


# ═══════════════════════════════════════════════════════════════════
#  The Two-User Test (§1.3)
# ═══════════════════════════════════════════════════════════════════

class TestTwoUserTest:
    """
    Filter 1: High schooler writes 15-line YAML for SMA crossover.
    Filter 2: Quant PM configures Burgess stat arb with CPO.
    """

    SMA_YAML = """
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
"""

    def test_high_schooler_sma(self):
        """Filter 1: Simple SMA crossover validates."""
        config = ModelConfig.from_yaml_string(self.SMA_YAML)
        assert config.model.name == "sma_crossover"
        assert config.model.type == ModelType.SINGLE_ASSET
        assert config.signal.method == "sma_crossover"
        assert config.signal.fast_period == 10
        assert config.signal.slow_period == 50
        assert config.sizing.method == "fixed_fraction"
        assert config.sizing.fraction == 1.0

    QUANT_YAML = """
model:
  name: burgess_stat_arb
  type: PairModel
  version: v2.1
construction:
  universe:
    method: index_constituents
    index_name: SPX
  data_source:
    provider: polygon
    resolution: daily
    lookback_days: 756
  pair_selection:
    method: cointegration
    cointegration:
      test: johansen
      p_value_threshold: 0.05
      lookback_days: 504
    max_pairs: 20
signal:
  method: zscore
  zscore_window: 60
  lookback: 252
entry:
  method: threshold
  long_threshold: -2.0
  short_threshold: 2.0
exit:
  method: threshold
  mean_reversion_level: 0.0
  stop_loss_pct: 0.05
  max_holding_days: 30
sizing:
  method: volatility_target
  target_vol: 0.10
  vol_lookback: 60
cpo:
  enabled: true
  search_method: bayesian
  max_evaluations: 500
backtest:
  engine: vectorized
  costs:
    commission_per_share: 0.005
  slippage:
    method: fixed
    fixed_bps: 5.0
"""

    def test_quant_stat_arb(self):
        """Filter 2: Complex stat arb with CPO validates."""
        config = ModelConfig.from_yaml_string(self.QUANT_YAML)
        assert config.model.name == "burgess_stat_arb"
        assert config.model.type == ModelType.PAIR
        assert config.construction.universe.method == "index_constituents"
        assert config.construction.pair_selection.cointegration.test == "johansen"
        assert config.signal.method == "zscore"
        assert config.entry.long_threshold == -2.0
        assert config.exit.stop_loss_pct == 0.05
        assert config.sizing.target_vol == 0.10
        assert config.cpo.enabled is True
        assert config.cpo.search_method == "bayesian"
        assert config.backtest.costs.commission_per_share == 0.005


# ═══════════════════════════════════════════════════════════════════
#  ModelConfig Basics
# ═══════════════════════════════════════════════════════════════════

class TestModelConfig:
    def test_minimal_from_dict(self):
        config = ModelConfig.from_dict({
            "model": {"name": "test", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover"},
        })
        assert config.model.name == "test"

    def test_missing_model_raises(self):
        with pytest.raises(Exception):
            ModelConfig.from_dict({"signal": {"method": "sma"}})

    def test_single_asset_requires_signal(self):
        """SingleAssetModel without signal section should fail."""
        with pytest.raises(ValueError, match="requires 'signal'"):
            ModelConfig.from_dict({
                "model": {"name": "test", "type": "SingleAssetModel"},
            })

    def test_pair_model_without_signal_ok(self):
        """Non-SingleAsset types can omit signal (construction may be primary)."""
        config = ModelConfig.from_dict({
            "model": {"name": "test", "type": "PairModel"},
        })
        assert config.signal is None

    def test_bpk_with_version(self):
        config = ModelConfig.from_dict({
            "model": {"name": "sma", "type": "SingleAssetModel", "version": "v2.0"},
            "signal": {"method": "sma_crossover"},
        })
        assert config.bpk == "sma|v2.0"

    def test_bpk_without_version(self):
        config = ModelConfig.from_dict({
            "model": {"name": "sma", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover"},
        })
        assert config.bpk == "sma|v0"

    def test_config_hash_deterministic(self):
        data = {
            "model": {"name": "test", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover", "fast_period": 10},
        }
        c1 = ModelConfig.from_dict(data)
        c2 = ModelConfig.from_dict(data)
        assert c1.config_hash == c2.config_hash

    def test_config_hash_changes_with_params(self):
        base = {
            "model": {"name": "test", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover", "fast_period": 10},
        }
        modified = {
            "model": {"name": "test", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover", "fast_period": 20},
        }
        c1 = ModelConfig.from_dict(base)
        c2 = ModelConfig.from_dict(modified)
        assert c1.config_hash != c2.config_hash

    def test_to_dict_excludes_none(self):
        config = ModelConfig.from_dict({
            "model": {"name": "test", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover"},
        })
        d = config.to_dict()
        # None fields should not appear
        assert "construction" not in d
        assert "cpo" not in d

    def test_to_dict_includes_all(self):
        config = ModelConfig.from_dict({
            "model": {"name": "test", "type": "SingleAssetModel"},
            "signal": {"method": "sma_crossover"},
        })
        d = config.to_dict(exclude_none=False)
        assert "construction" in d
        assert d["construction"] is None


# ═══════════════════════════════════════════════════════════════════
#  YAML Loading
# ═══════════════════════════════════════════════════════════════════

class TestYamlLoading:
    def test_from_yaml_string(self):
        yaml_str = """
model:
  name: ema_test
  type: SingleAssetModel
signal:
  method: ema_crossover
  fast_period: 12
  slow_period: 26
"""
        config = ModelConfig.from_yaml_string(yaml_str)
        assert config.model.name == "ema_test"
        assert config.signal.fast_period == 12
        assert config.source_yaml == yaml_str

    def test_from_yaml_file(self, tmp_path):
        yaml_file = tmp_path / "test_model.yaml"
        yaml_file.write_text("""
model:
  name: file_test
  type: SingleAssetModel
signal:
  method: sma_crossover
  fast_period: 5
  slow_period: 20
""")
        config = ModelConfig.from_yaml(yaml_file)
        assert config.model.name == "file_test"
        assert config.signal.slow_period == 20

    def test_invalid_yaml_raises(self):
        with pytest.raises(Exception):
            ModelConfig.from_yaml_string("not: {valid: [yaml: oops")


# ═══════════════════════════════════════════════════════════════════
#  Signal Params
# ═══════════════════════════════════════════════════════════════════

class TestSignalParams:
    def test_sma_params(self):
        p = SignalParams(method="sma_crossover", fast_period=10, slow_period=50)
        assert p.method == "sma_crossover"
        assert p.fast_period == 10

    def test_zscore_params(self):
        p = SignalParams(method="zscore", zscore_window=60, lookback=252)
        assert p.zscore_window == 60

    def test_composite_params(self):
        p = SignalParams(
            method="composite",
            composite_method="weighted",
            component_signals=["sma", "rsi"],
            component_weights=[0.6, 0.4],
        )
        assert len(p.component_signals) == 2

    def test_custom_function(self):
        p = SignalParams(
            method="custom_inline",
            custom_function="my_module.my_signal",
            custom_params={"lookback": 30},
        )
        assert p.custom_function == "my_module.my_signal"

    def test_method_required(self):
        with pytest.raises(Exception):
            SignalParams()


# ═══════════════════════════════════════════════════════════════════
#  Entry Params
# ═══════════════════════════════════════════════════════════════════

class TestEntryParams:
    def test_threshold_entry(self):
        p = EntryParams(method="threshold", long_threshold=-2.0, short_threshold=2.0)
        assert p.long_threshold == -2.0

    def test_regime_conditions(self):
        p = EntryParams(
            method="threshold",
            regime_conditions={
                "high_vol": {"condition": "vix > 25", "threshold_long": -2.5},
                "low_vol": {"condition": "vix < 15", "threshold_long": -1.5},
            },
        )
        assert p.regime_conditions.high_vol.threshold_long == -2.5


# ═══════════════════════════════════════════════════════════════════
#  Exit Params
# ═══════════════════════════════════════════════════════════════════

class TestExitParams:
    def test_basic_exit(self):
        p = ExitParams(
            method="threshold",
            profit_target_pct=0.05,
            stop_loss_pct=0.03,
            max_holding_days=20,
        )
        assert p.profit_target_pct == 0.05
        assert p.max_holding_days == 20

    def test_triple_barrier(self):
        p = ExitParams(
            method="triple_barrier",
            triple_barrier=TripleBarrierConfig(
                pt_level=0.05, sl_level=0.03, max_days=10
            ),
        )
        assert p.triple_barrier.pt_level == 0.05

    def test_trailing_stop(self):
        p = ExitParams(
            method="trailing",
            trailing_enabled=True,
            trailing_activation_pct=0.02,
            trailing_distance_pct=0.01,
        )
        assert p.trailing_enabled is True


# ═══════════════════════════════════════════════════════════════════
#  Sizing Params
# ═══════════════════════════════════════════════════════════════════

class TestSizingParams:
    def test_fixed_fraction(self):
        p = SizingParams(method="fixed_fraction", fraction=1.0)
        assert p.fraction == 1.0

    def test_volatility_target(self):
        p = SizingParams(
            method="volatility_target",
            target_vol=0.10,
            vol_lookback=60,
            max_position_pct=0.25,
        )
        assert p.target_vol == 0.10

    def test_kelly(self):
        p = SizingParams(
            method="kelly",
            kelly_fraction=0.5,
            kelly_lookback=252,
        )
        assert p.kelly_fraction == 0.5


# ═══════════════════════════════════════════════════════════════════
#  Backtest Params
# ═══════════════════════════════════════════════════════════════════

class TestBacktestParams:
    def test_defaults(self):
        p = BacktestParams()
        assert p.engine == "vectorized"

    def test_with_costs(self):
        p = BacktestParams(
            costs=CostsConfig(commission_per_share=0.005, commission_min=1.0),
            slippage=SlippageConfig(method="fixed", fixed_bps=5.0),
        )
        assert p.costs.commission_per_share == 0.005
        assert p.slippage.fixed_bps == 5.0

    def test_with_validation(self):
        p = BacktestParams(
            validation={"walk_forward": True, "n_splits": 5, "embargo_days": 5}
        )
        assert p.validation.walk_forward is True


# ═══════════════════════════════════════════════════════════════════
#  Construction Params
# ═══════════════════════════════════════════════════════════════════

class TestConstructionParams:
    def test_static_universe(self):
        p = ConstructionParams(
            universe=UniverseConfig(
                method="static",
                instruments=["AAPL", "MSFT", "GOOG"],
            ),
        )
        assert len(p.universe.instruments) == 3

    def test_index_universe(self):
        p = ConstructionParams(
            universe=UniverseConfig(
                method="index_constituents",
                index_name="SPX",
            ),
            data_source=DataSourceConfig(
                provider="polygon",
                lookback_days=756,
            ),
        )
        assert p.universe.index_name == "SPX"
        assert p.data_source.lookback_days == 756


# ═══════════════════════════════════════════════════════════════════
#  Logger Config
# ═══════════════════════════════════════════════════════════════════

class TestLoggerConfig:
    def test_default_level(self):
        cfg = LoggerConfig()
        assert cfg.current_level == 20

    def test_full_config(self):
        cfg = LoggerConfig(
            current_level="INFO",
            adapters={
                "terminal": LogAdapterConfig(type="terminal", min_level=20, color=True),
                "logfile": LogAdapterConfig(type="file", min_level=10, path="logs/praxis.log"),
                "database": LogAdapterConfig(
                    type="database", min_level=20,
                    buffer_size=1000, backtest_throttle=30,
                ),
            },
            tag_levels={"trade_cycle": 20, "cpo_cycle": 10},
            routing=LogRoutingConfig(
                critical_override=True,
                tag_routes={"trade_cycle": ["terminal", "database"]},
            ),
        )
        assert cfg.adapters["terminal"].color is True
        assert cfg.tag_levels["trade_cycle"] == 20
        assert cfg.routing.tag_routes["trade_cycle"] == ["terminal", "database"]

    def test_from_yaml(self):
        yaml_str = """
current_level: 20
adapters:
  terminal:
    type: terminal
    min_level: 20
    color: true
tag_levels:
  trade_cycle: 20
routing:
  critical_override: true
  tag_routes:
    trade_cycle: [terminal, database]
"""
        data = yaml.safe_load(yaml_str)
        cfg = LoggerConfig.model_validate(data)
        assert cfg.adapters["terminal"].type == "terminal"


# ═══════════════════════════════════════════════════════════════════
#  Platform Config
# ═══════════════════════════════════════════════════════════════════

class TestPlatformConfig:
    def test_default_ephemeral(self):
        cfg = PlatformConfig()
        assert cfg.mode == PlatformMode.EPHEMERAL

    def test_full_mode(self):
        cfg = PlatformConfig(mode="full", db_path="data/praxis.duckdb")
        assert cfg.mode == PlatformMode.FULL


# ═══════════════════════════════════════════════════════════════════
#  Workflow Params
# ═══════════════════════════════════════════════════════════════════

class TestWorkflowParams:
    def test_basic_workflow(self):
        wf = WorkflowParams(
            enabled=True,
            steps=[
                WorkflowStep(id="step1", function="load_data", depends_on=[]),
                WorkflowStep(id="step2", function="compute_signal", depends_on=["step1"]),
            ],
        )
        assert len(wf.steps) == 2
        assert wf.steps[1].depends_on == ["step1"]


# ═══════════════════════════════════════════════════════════════════
#  ModelConfig with Logger
# ═══════════════════════════════════════════════════════════════════

class TestModelConfigWithLogger:
    def test_sma_with_logger(self):
        yaml_str = """
model:
  name: sma_with_logging
  type: SingleAssetModel
signal:
  method: sma_crossover
  fast_period: 10
  slow_period: 50
sizing:
  method: fixed_fraction
  fraction: 1.0
logger:
  current_level: 10
  adapters:
    terminal:
      type: terminal
      min_level: 20
    logfile:
      type: file
      min_level: 10
      path: logs/sma.log
  tag_levels:
    trade_cycle: 20
"""
        config = ModelConfig.from_yaml_string(yaml_str)
        assert config.logger.current_level == 10
        assert config.logger.adapters["logfile"].path == "logs/sma.log"
        assert config.logger.tag_levels["trade_cycle"] == 20


# ═══════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════

class TestEnums:
    def test_model_type_values(self):
        assert ModelType.SINGLE_ASSET == "SingleAssetModel"
        assert ModelType.PAIR == "PairModel"
        assert ModelType.CPO == "CPOModel"

    def test_model_type_from_string(self):
        config = ModelConfig.from_dict({
            "model": {"name": "test", "type": "PairModel"},
        })
        assert config.model.type == ModelType.PAIR

    def test_invalid_model_type(self):
        with pytest.raises(Exception):
            ModelConfig.from_dict({
                "model": {"name": "test", "type": "InvalidModel"},
                "signal": {"method": "sma"},
            })

    def test_backtest_engine_enum(self):
        from praxis.config import BacktestEngine
        assert BacktestEngine.VECTORIZED == "vectorized"


# ═══════════════════════════════════════════════════════════════════
#  CPO Params
# ═══════════════════════════════════════════════════════════════════

class TestCPOParams:
    def test_basic_cpo(self):
        p = CPOParams(
            enabled=True,
            search_method="bayesian",
            max_evaluations=500,
        )
        assert p.enabled is True
        assert p.search_method == "bayesian"

    def test_cpo_disabled_by_default(self):
        p = CPOParams()
        assert p.enabled is False
