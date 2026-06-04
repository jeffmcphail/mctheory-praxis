"""
engines/funding_replay.py
=========================
Cycle 54: the SHARED Engine-7 replay engine -- ONE replay engine, two callers.

Lifted from scripts/cycle53_backtest_replay.py (Cycle 53 D7+D8) so there is
exactly one code path that:
  - regenerates per-(asset, day) OOS predictions reproducing atlas Exp 13,
  - synthesizes a self-contained harness DB (funding inputs + paper tables),
  - drives the PRODUCTION FundingExecutor (entry + exit lifecycle) against it
    with an injected clock.

The Cycle 53 script imports these for its decomposition/reporting; the GUI
(gui/funding_studio) imports them to run paper_replay sessions. The replay
WINDOW and the injected CLOCK are PARAMETERS the caller supplies -- the
defaults are Cycle 53's values, so the Cycle 53 script's behavior is
byte-unchanged. That parameterization is what makes this GUI-ready rather
than a Cycle-53-shaped lift; the per-window logic itself is identical to the
verified Cycle 53 original.

No execution-venue interaction; the only side effects are local SQLite writes
to caller-supplied harness DBs (and the paper tables inside them, written by
the production executor).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import joblib
from engines.funding_rate_strategy import FundingRateStrategy
from engines.cpo_core import predict_model
from engines.funding_executor import FundingExecutor

REPO = Path(__file__).resolve().parent.parent

# ── Cycle 53 defaults (used as parameter defaults; callers may override) ─────
ASSETS         = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]
OOS_START      = "2025-01-01"
OOS_END        = "2026-03-26"
WARMUP_START   = "2024-09-03"
WARMUP_END     = "2025-01-01"
CACHE_DIR      = str(REPO / "data" / "funding_cache")
MODEL_PATH     = str(REPO / "outputs" / "funding_carry_repro" / "cpo" /
                     "phase3_models_funding.joblib")
GATES          = [0.50, 0.70]
NOTIONAL_USD   = 500.0
TC_ROUND_TRIP  = 0.0008   # 4 bps one-way x2; matches executor + atlas

# Default injected replay clock: after the latest possible OOS exit
# (last entry 2026-03-25 + 14d hold = 2026-04-08).
REPLAY_NOW     = datetime(2026, 4, 30, tzinfo=timezone.utc)
MS_PER_DAY     = 86_400_000

# Relaxed risk config (criterion 4). max_notional_per_asset stays at $500.
RELAXED = {
    "max_concurrent_positions_per_asset": 1_000_000_000,
    "max_total_notional_usd":             1e15,
    "max_notional_per_asset_usd":         NOTIONAL_USD,
    "max_signal_age_seconds":             10**15,
    "max_daily_loss_usd":                 1e15,
    "max_daily_loss_pct":                 1e9,
}

# Production table DDLs (mirrors crypto_data.db; harness is self-contained).
# paper_trades + paper_position_exits carry session_id NOT NULL (Cycle 54).
DDL = {
    "funding_rates": """
        CREATE TABLE funding_rates (
            asset TEXT NOT NULL, venue TEXT NOT NULL,
            timestamp INTEGER NOT NULL, datetime TEXT NOT NULL,
            funding_rate REAL,
            PRIMARY KEY (asset, venue, timestamp))""",
    "funding_signals": """
        CREATE TABLE funding_signals (
            asset TEXT NOT NULL, timestamp INTEGER NOT NULL, datetime TEXT NOT NULL,
            p_profitable REAL NOT NULL, above_gate INTEGER NOT NULL,
            above_gate_050 INTEGER NOT NULL, gate_threshold REAL NOT NULL,
            best_config_id TEXT, hold_days INTEGER, min_funding_ann REAL,
            expected_return REAL, ann_rate REAL, basis_pct REAL, pct_positive REAL,
            min_pct_positive REAL, base_rate REAL, features_json TEXT,
            monitor_version TEXT NOT NULL,
            PRIMARY KEY (asset, timestamp))""",
    "funding_alerts": """
        CREATE TABLE funding_alerts (
            asset TEXT NOT NULL, timestamp INTEGER NOT NULL, datetime TEXT NOT NULL,
            alerted_at TEXT NOT NULL, p_profitable REAL NOT NULL,
            gate_threshold REAL NOT NULL, monitor_version TEXT NOT NULL,
            PRIMARY KEY (asset, timestamp))""",
    "paper_trades": """
        CREATE TABLE paper_trades (
            asset TEXT NOT NULL, signal_timestamp INTEGER NOT NULL,
            signal_datetime TEXT NOT NULL, funding_alert_alerted_at TEXT NOT NULL,
            decided_at TEXT NOT NULL, decision TEXT NOT NULL, skip_reason TEXT,
            intended_direction TEXT, intended_size_usd REAL,
            p_profitable REAL NOT NULL, gate_threshold REAL NOT NULL,
            risk_checks_json TEXT NOT NULL, executor_version TEXT NOT NULL,
            hold_days INTEGER, session_id TEXT NOT NULL,
            PRIMARY KEY (asset, signal_timestamp))""",
    "paper_position_exits": """
        CREATE TABLE paper_position_exits (
            asset TEXT NOT NULL, signal_timestamp INTEGER NOT NULL,
            entry_decided_at TEXT NOT NULL, exit_decided_at TEXT NOT NULL,
            exit_timestamp INTEGER NOT NULL, exit_datetime TEXT NOT NULL,
            hold_days INTEGER NOT NULL, funding_events_count INTEGER NOT NULL,
            funding_payments_usd REAL NOT NULL, tc_entry_usd REAL NOT NULL,
            tc_exit_usd REAL NOT NULL, net_pnl_usd REAL NOT NULL,
            notional_usd REAL NOT NULL, direction TEXT NOT NULL,
            executor_version TEXT NOT NULL, session_id TEXT NOT NULL,
            PRIMARY KEY (asset, signal_timestamp))""",
}


def midnight_ms(day: pd.Timestamp) -> int:
    """UTC-midnight epoch-ms for a normalized trading day."""
    d = pd.Timestamp(day)
    if d.tzinfo is None:
        d = d.tz_localize("UTC")
    return int(d.normalize().timestamp() * 1000)


# ── D7a: regenerate per-(asset, day) predictions ────────────────────────────

def regenerate_predictions(assets: list | None = None,
                           oos_start: str = OOS_START,
                           oos_end: str = OOS_END,
                           warmup_start: str = WARMUP_START,
                           warmup_end: str = WARMUP_END,
                           cache_dir: str = CACHE_DIR,
                           model_path: str = MODEL_PATH) -> tuple[dict, dict]:
    """Return (preds, funding_series).

    preds[asset] -> list of per-day dicts (only days with finite features):
        ts_ms, datetime, p_profitable, config_id, hold_days,
        min_funding_ann, min_pct_positive, atlas_net, atlas_gross,
        funding_term, basis_term, n_payments
    funding_series[asset] -> pd.Series of 8h funding rates (for harness load).

    The replay WINDOW (oos_start/oos_end + warmup) and the assets/cache/model
    are PARAMETERS; defaults reproduce Cycle 53's OOS run exactly. The per-day
    logic is identical to the Cycle 53 original.
    """
    assets = ASSETS if assets is None else assets
    print("=" * 72)
    print(" D7a  Regenerating per-(asset, day) OOS predictions")
    print("=" * 72)
    strat = FundingRateStrategy(assets=assets, cache_dir=cache_dir,
                                feature_mode="funding")
    models = joblib.load(model_path)
    grid = strat.get_param_grid()
    oos = strat.fetch_oos_data(strat.get_models(), oos_start, oos_end)
    warmup = strat.fetch_warmup_daily(strat.get_models(), warmup_start, warmup_end)
    # warmup is fetched for parity with phase4's data assembly; daily-price
    # warmup does not feed the intraday funding-feature window, so predictions
    # are driven purely by the OOS hourly caches (verified to reproduce atlas).
    _ = warmup

    days = strat.get_trading_days(oos)
    print(f"  trading days: {len(days)}  ({days[0].date()} .. {days[-1].date()})")

    preds: dict[str, list] = {}
    funding_series: dict[str, pd.Series] = {}
    for m in strat.get_models():
        asset = m.asset
        tm = models.get(m.model_id, {})
        perp = oos.get("perp", {}).get(asset)
        if tm.get("model") is None or perp is None:
            print(f"  {asset}: no model/data -> skipped")
            continue
        fseries = perp["funding"].sort_index()
        funding_series[asset] = fseries

        rows = []
        for day in days:
            ds = day.strftime("%Y-%m-%d")
            fv = strat.compute_features(m, ds, oos)
            if fv is None:
                continue
            cfg, p, _er = predict_model(tm, fv, grid)
            res = strat.run_single_day(m, cfg, day, oos)
            # Config-gate inputs (mirror funding_monitor: feat[1]=ann_rate,
            # feat[4]=pct_positive) and the argmax config's hard thresholds.
            ann_rate = float(fv[1])
            pct_pos = float(fv[4])
            config_ok = (ann_rate >= cfg.min_funding_ann_pct
                         and pct_pos >= cfg.min_pct_positive)
            ts0 = midnight_ms(day)
            target = ts0 + cfg.hold_days * MS_PER_DAY
            # Funding term over the executor's window (exclusive entry / inclusive exit).
            mask = (fseries.index > pd.Timestamp(ts0, unit="ms", tz="UTC")) & \
                   (fseries.index <= pd.Timestamp(target, unit="ms", tz="UTC"))
            hold_fr = fseries[mask]
            funding_term = float(hold_fr.sum()) if len(hold_fr) else 0.0
            atlas_gross = float(res["gross_return"])
            atlas_net = float(res["daily_return"])
            # basis term = gross - funding (gross = spot_ret + perp_ret + funding,
            # pre-clip; clip only bites at |ret|>0.99 which never happens for carry).
            basis_term = atlas_gross - funding_term
            rows.append({
                "ts_ms": ts0,
                "datetime": pd.Timestamp(ts0, unit="ms", tz="UTC")
                              .strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "p_profitable": float(p),
                "config_id": cfg.config_id,
                "hold_days": int(cfg.hold_days),
                "min_funding_ann": float(cfg.min_funding_ann_pct),
                "min_pct_positive": float(cfg.min_pct_positive),
                "atlas_net": atlas_net,
                "atlas_gross": atlas_gross,
                "funding_term": funding_term,
                "basis_term": basis_term,
                "n_payments": int(len(hold_fr)),
                "ann_rate": ann_rate,
                "pct_positive": pct_pos,
                "min_funding_ann": float(cfg.min_funding_ann_pct),
                "min_pct_positive": float(cfg.min_pct_positive),
                "config_ok": config_ok,
            })
        preds[asset] = rows
        n70 = sum(1 for r in rows if r["p_profitable"] > 0.70)
        n50 = sum(1 for r in rows if r["p_profitable"] > 0.50)
        print(f"  {asset:<5} feat_days={len(rows):>4}  P>0.50={n50:>4}  P>0.70={n70:>4}")
    return preds, funding_series


# ── D7b: harness DB synthesis + executor run ─────────────────────────────────

def build_harness(db_path: Path, preds: dict, funding_series: dict,
                  gate: float, monitor_gate: bool = False) -> int:
    """Create a self-contained harness DB and populate it for one gate stream.

    monitor_gate mirrors the D8a fix: when True, an alert is emitted only when
    the argmax config's hard thresholds are met (config_ok), so funding_alerts
    is atlas-faithful. When False (pre-fix baseline), every P>gate day alerts.
    Returns the number of alerts emitted.
    funding_signals always carries the four config-gate fields so the executor's
    D8b 11th check can JOIN-resolve them regardless of monitor_gate.
    """
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for ddl in DDL.values():
        cur.execute(ddl)

    # funding_rates from the identical cache series (criterion-1 exactness).
    for asset, fseries in funding_series.items():
        for ts, rate in fseries.items():
            ts_ms = int(ts.timestamp() * 1000)
            cur.execute(
                "INSERT OR IGNORE INTO funding_rates "
                "(asset, venue, timestamp, datetime, funding_rate) "
                "VALUES (?, 'binance', ?, ?, ?)",
                (asset, ts_ms, ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                 float(rate) if rate is not None and np.isfinite(rate) else None),
            )

    # signals (always; carry config-gate fields) + alerts (gated) for P>gate days.
    n_alerts = 0
    for asset, rows in preds.items():
        for r in rows:
            if r["p_profitable"] <= gate:
                continue
            above050 = 1 if r["p_profitable"] > 0.50 else 0
            cur.execute(
                "INSERT OR IGNORE INTO funding_signals "
                "(asset, timestamp, datetime, p_profitable, above_gate, "
                " above_gate_050, gate_threshold, best_config_id, hold_days, "
                " min_funding_ann, expected_return, ann_rate, basis_pct, "
                " pct_positive, min_pct_positive, base_rate, features_json, "
                " monitor_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (asset, r["ts_ms"], r["datetime"], r["p_profitable"], 1,
                 above050, gate, r["config_id"], r["hold_days"],
                 r["min_funding_ann"], 0.0, r["ann_rate"], 0.0,
                 r["pct_positive"], r["min_pct_positive"], 0.0, None,
                 "cycle53-replay"),
            )
            # D8a mirror: emit an alert only if config_ok (when monitor_gate).
            if monitor_gate and not r["config_ok"]:
                continue
            cur.execute(
                "INSERT OR IGNORE INTO funding_alerts "
                "(asset, timestamp, datetime, alerted_at, p_profitable, "
                " gate_threshold, monitor_version) VALUES (?,?,?,?,?,?,?)",
                (asset, r["ts_ms"], r["datetime"], r["datetime"],
                 r["p_profitable"], gate, "cycle53-replay"),
            )
            n_alerts += 1
    conn.commit()
    conn.close()
    tag = "monitor-gate ON" if monitor_gate else "pre-fix (all P>gate)"
    print(f"  harness {db_path.name}: {n_alerts} alerts (gate {gate:.2f}, {tag})")
    return n_alerts


def run_executor(db_path: Path, enforce_config_gate: bool,
                 session_id: str = "cycle53-replay",
                 now: datetime = REPLAY_NOW,
                 defaults_override: dict | None = None) -> dict:
    """Drive the PRODUCTION executor against a harness with an injected clock.

    `now` is the injected terminal instant (Cycle 53 default = REPLAY_NOW;
    the GUI passes its replay window's end). `defaults_override` defaults to
    the relaxed backtest caps; pass a dict to override.
    """
    ex = FundingExecutor(db_path=db_path,
                         defaults_override=defaults_override or RELAXED,
                         now_func=lambda: now,
                         session_id=session_id,
                         enforce_config_gate=enforce_config_gate)
    return ex.run_once()
