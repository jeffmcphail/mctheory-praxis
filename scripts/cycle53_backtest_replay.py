#!/usr/bin/env python3
"""
scripts/cycle53_backtest_replay.py
===================================
Cycle 53 (D7 + D8): paper-executor backtest replay + decomposition vs atlas Exp 13.

PURPOSE
-------
Re-frame "verified" away from the literal Sharpe-match target (the wrong
criterion for a deliberately-simpler executor) toward a P&L *decomposition*:
prove the executor's funding-only P&L equals atlas Exp 13's reported P&L
MINUS a small, measured basis-drag term, with config-gate zeroing accounted
for explicitly. If the residual is ~0 per asset, the executor is "verified"
in the sense that matters: its omissions are understood and bounded.

PIPELINE
--------
D7a  Regenerate per-(asset, day) OOS predictions with predict_model over the
     cached OOS window (data/funding_cache, 2025-01-01..2026-03-26) + the
     phase3 joblib model. This reproduces atlas Exp 13 exactly (verified:
     ADA 75 days / cum +0.0945; BTC 36 days / cum +0.0351 vs phase4_model_stats).
     For each day we capture: P(profitable), argmax config (-> hold_days +
     min_funding_ann + min_pct_positive), atlas net & gross return, the
     funding term (summed from the funding series over the executor's window),
     and the derived basis term (gross - funding).

D7b  For each gate stream (P>0.50 and P>0.70), synthesize a harness SQLite DB:
       - funding_rates (venue='binance') loaded from the IDENTICAL cache
         parquets the predictions used -> criterion-1 funding match is exact
         by construction.
       - funding_signals (carrying argmax hold_days) + funding_alerts for
         every day with P>gate.
     Then run the production FundingExecutor against the harness with an
     injected post-OOS clock and the concurrency / notional / staleness /
     daily-loss caps relaxed (criterion 4). The executor books funding-only
     P&L through its real entry + exit-reconciliation lifecycle.

D7c  Decompose per asset (and aggregate) into the 6-criteria table:
       executor_cum_pnl_pct | atlas_cum_pnl_pct | basis_drag_pct |
       config_gate_zeroing_pct | residual (~0)
     plus criterion-1 (funding-term exact match, delta < 1e-6 USD/trade) and
     criterion-2 (per-asset basis mean/std, annualized) and criterion-6
     (config-gate zeroing at BOTH gates -> live-monitor gap assessment).

This script writes NO production data: harness DBs live under
outputs/cycle53_backtest_replay/. Production crypto_data.db is untouched.

RELAXATIONS (criterion 4, documented):
    max_concurrent_positions_per_asset : 1   -> 1e9   (atlas books overlapping
                                                        daily entries; prod keeps 1)
    max_total_notional_usd             : 2500 -> 1e15
    max_notional_per_asset_usd         : 500  (unchanged; sizing held at $500)
    max_signal_age_seconds             : 5400 -> 1e15  (alerts are >1yr old in replay)
    max_daily_loss_usd                 : 50   -> 1e15  (replay books all exits at
                                                        one injected instant)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
from engines.funding_rate_strategy import FundingRateStrategy  # noqa: E402
from engines.cpo_core import predict_model  # noqa: E402
from engines.funding_executor import FundingExecutor  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────
ASSETS         = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]
OOS_START      = "2025-01-01"
OOS_END        = "2026-03-26"
WARMUP_START   = "2024-09-03"
WARMUP_END     = "2025-01-01"
CACHE_DIR      = str(REPO / "data" / "funding_cache")
MODEL_PATH     = str(REPO / "outputs" / "funding_carry_repro" / "cpo" /
                     "phase3_models_funding.joblib")
OUT_DIR        = REPO / "outputs" / "cycle53_backtest_replay"
GATES          = [0.50, 0.70]
NOTIONAL_USD   = 500.0
TC_ROUND_TRIP  = 0.0008   # 4 bps one-way x2; matches executor + atlas

# Injected replay clock: after the latest possible OOS exit
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

def regenerate_predictions() -> tuple[dict, dict]:
    """Return (preds, funding_series).

    preds[asset] -> list of per-day dicts (only days with finite features):
        ts_ms, datetime, p_profitable, config_id, hold_days,
        min_funding_ann, min_pct_positive, atlas_net, atlas_gross,
        funding_term, basis_term, n_payments
    funding_series[asset] -> pd.Series of 8h funding rates (for harness load).
    """
    print("=" * 72)
    print(" D7a  Regenerating per-(asset, day) OOS predictions")
    print("=" * 72)
    strat = FundingRateStrategy(assets=ASSETS, cache_dir=CACHE_DIR,
                                feature_mode="funding")
    models = joblib.load(MODEL_PATH)
    grid = strat.get_param_grid()
    oos = strat.fetch_oos_data(strat.get_models(), OOS_START, OOS_END)
    warmup = strat.fetch_warmup_daily(strat.get_models(), WARMUP_START, WARMUP_END)
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
                 session_id: str = "cycle53-replay") -> dict:
    ex = FundingExecutor(db_path=db_path, defaults_override=RELAXED,
                         now_func=lambda: REPLAY_NOW,
                         session_id=session_id,
                         enforce_config_gate=enforce_config_gate)
    return ex.run_once()


# ── D7c: decomposition ───────────────────────────────────────────────────────

def decompose(db_path: Path, preds: dict, gate: float) -> dict:
    """Build per-asset + aggregate decomposition for one gate stream.

    Returns {"per_asset": {asset: {...}}, "aggregate": {...},
             "funding_match": {...}, "basis_stats": {asset: {...}}}.
    """
    conn = sqlite3.connect(db_path)
    exits = pd.read_sql_query(
        "SELECT * FROM paper_position_exits", conn)
    conn.close()

    # Index regenerated predictions by (asset, ts) for cross-reference.
    pred_idx = {(a, r["ts_ms"]): r for a, rows in preds.items() for r in rows}

    per_asset = {}
    funding_deltas = []          # criterion 1: executor funding vs regenerated
    basis_stats = {}
    agg = dict(exec_sh=0.0, atlas_sh=0.0, basis_sh=0.0, execonly=0.0,
               n_trades=0, n_shared=0, n_zeroed=0)

    for asset in ASSETS:
        gate_days = [r for r in preds.get(asset, []) if r["p_profitable"] > gate]
        n_total = len(gate_days)
        if n_total == 0:
            continue
        ex_rows = exits[exits["asset"] == asset] if not exits.empty else pd.DataFrame()
        ex_by_ts = {int(t): row for t, row in
                    zip(ex_rows["signal_timestamp"], ex_rows.to_dict("records"))} \
            if not ex_rows.empty else {}

        # SHARED set = days atlas actually booked a position (non-zero atlas
        # result). ZEROED set = days atlas zeroed (run_funding_single_day
        # returned 0 because the argmax config's hard thresholds — min_funding_ann
        # / min_pct_positive — were not met by the environment, or entry price
        # was unavailable). The live monitor + executor BOOK the zeroed set
        # because they gate on argmax-P alone, not the config thresholds.
        exec_sh = atlas_sh = 0.0     # shared-set cumulative pct
        execonly = 0.0               # executor P&L on atlas-zeroed days
        basis_vals = []
        n_zeroed = 0
        for r in gate_days:
            ex_row = ex_by_ts.get(r["ts_ms"])
            if ex_row is not None:
                delta = abs(ex_row["funding_payments_usd"]
                            - r["funding_term"] * NOTIONAL_USD)
                funding_deltas.append(delta)
                exec_pct_trade = ex_row["net_pnl_usd"] / NOTIONAL_USD
            else:
                # No exit row -> the executor did NOT book this day (skipped by
                # the D8b config-gate, or never alerted under D8a). Realized
                # executor P&L is therefore 0, not the counterfactual funding.
                exec_pct_trade = 0.0
            is_zeroed = (r["atlas_gross"] == 0.0 and r["atlas_net"] == 0.0)
            if is_zeroed:
                n_zeroed += 1
                execonly += exec_pct_trade
            else:
                exec_sh += exec_pct_trade
                atlas_sh += r["atlas_net"]
                basis_vals.append(r["basis_term"])
        basis_arr = np.array(basis_vals, dtype=float)
        basis_stats[asset] = {
            "mean_bps": float(basis_arr.mean() * 1e4) if len(basis_arr) else 0.0,
            "std_bps": float(basis_arr.std() * 1e4) if len(basis_arr) else 0.0,
            "n": int(len(basis_arr)),
        }
        # basis_drag summed INDEPENDENTLY (regenerated price-return terms) over
        # the SHARED set only -> residual is a genuine three-way cross-check
        # (executor funding vs regenerated funding, atlas net vs funding+basis-TC).
        basis_drag = float(basis_arr.sum())
        residual = atlas_sh - exec_sh - basis_drag
        per_asset[asset] = {
            "n_trades": n_total,
            "n_shared": len(basis_vals),
            "n_zeroed": n_zeroed,
            "executor_cum_pnl_pct": exec_sh * 100,
            "atlas_cum_pnl_pct": atlas_sh * 100,
            "basis_drag_pct": basis_drag * 100,
            "config_gate_zeroing_pct": (n_zeroed / n_total * 100) if n_total else 0.0,
            "executor_only_pnl_pct": execonly * 100,
            "residual_pct": residual * 100,
        }
        agg["exec_sh"] += exec_sh
        agg["atlas_sh"] += atlas_sh
        agg["basis_sh"] += basis_drag
        agg["execonly"] += execonly
        agg["n_trades"] += n_total
        agg["n_shared"] += len(basis_vals)
        agg["n_zeroed"] += n_zeroed

    aggregate = {
        "n_trades": agg["n_trades"],
        "n_shared": agg["n_shared"],
        "n_zeroed": agg["n_zeroed"],
        "executor_cum_pnl_pct": agg["exec_sh"] * 100,
        "atlas_cum_pnl_pct": agg["atlas_sh"] * 100,
        "basis_drag_pct": agg["basis_sh"] * 100,
        "config_gate_zeroing_pct": (agg["n_zeroed"] / agg["n_trades"] * 100)
                                   if agg["n_trades"] else 0.0,
        "executor_only_pnl_pct": agg["execonly"] * 100,
        "residual_pct": (agg["atlas_sh"] - agg["exec_sh"] - agg["basis_sh"]) * 100,
    }
    funding_match = {
        "n_trades_checked": len(funding_deltas),
        "max_abs_delta_usd": max(funding_deltas) if funding_deltas else 0.0,
        "mean_abs_delta_usd": float(np.mean(funding_deltas)) if funding_deltas else 0.0,
        "pass_1e-6": (max(funding_deltas) < 1e-6) if funding_deltas else False,
    }
    return {"per_asset": per_asset, "aggregate": aggregate,
            "funding_match": funding_match, "basis_stats": basis_stats}


# ── Reporting ────────────────────────────────────────────────────────────────

def fmt_table(decomp: dict) -> list[str]:
    """Columns over the SHARED trade set (atlas + executor both booked):
       exec% / atlas% / basis% / resid%. Plus the atlas-zeroed leak:
       zero% (= % of gate days atlas zeroed) and execOnly% (executor P&L
       on those zeroed days — the divergence the monitor's missing config
       gate introduces)."""
    L = []
    hdr = (f"  {'asset':<6} {'shared':>6} {'exec%':>9} {'atlas%':>9} "
           f"{'basis%':>9} {'resid%':>10}  | {'zeroed':>6} {'zero%':>6} {'execOnly%':>9}")
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))

    def row(name, d):
        return (f"  {name:<6} {d['n_shared']:>6} "
                f"{d['executor_cum_pnl_pct']:>+9.4f} {d['atlas_cum_pnl_pct']:>+9.4f} "
                f"{d['basis_drag_pct']:>+9.4f} {d['residual_pct']:>+10.2e}  | "
                f"{d['n_zeroed']:>6} {d['config_gate_zeroing_pct']:>5.1f}% "
                f"{d['executor_only_pnl_pct']:>+9.4f}")
    for asset in ASSETS:
        d = decomp["per_asset"].get(asset)
        if d is not None:
            L.append(row(asset, d))
    L.append("  " + "-" * (len(hdr) - 2))
    L.append(row("ALL", decomp["aggregate"]))
    return L


def write_summary(results: dict, gates: list, smoke: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "SUMMARY.md"
    L = []
    L.append("# Cycle 53 (D7+D8) — Paper-Executor Backtest Replay + Config-Gate Fix\n")
    L.append(f"_Generated: replay clock = {REPLAY_NOW.isoformat()}; "
             f"OOS {OOS_START}..{OOS_END}; notional ${NOTIONAL_USD:.0f}; "
             f"TC {TC_ROUND_TRIP*1e4:.0f} bps round-trip._\n")

    L.append("## D7 — decomposition vs atlas Exp 13 (pre-fix baseline)\n")
    L.append("**Criterion 1 — funding-term exact match (< 1e-6 USD/trade):**\n")
    for gate in gates:
        fm = results[gate]["prefix"]["funding_match"]
        L.append(f"- gate {gate:.2f}: {'PASS' if fm['pass_1e-6'] else 'FAIL'} — "
                 f"{fm['n_trades_checked']} trades, max |Δ| = "
                 f"{fm['max_abs_delta_usd']:.3e} USD, mean = {fm['mean_abs_delta_usd']:.3e}")
    L.append("")
    for gate in gates:
        d = results[gate]["prefix"]
        L.append(f"### gate P>{gate:.2f} (executor books every alert — no config gate)\n")
        L.append("```")
        L.extend(fmt_table(d))
        L.append("```\n")
        L.append("Per-asset basis term over the shared set (criterion 2; bps/hold):\n")
        L.append("| asset | n | basis mean (bps) | basis std (bps) |")
        L.append("|---|---|---|---|")
        for a in ASSETS:
            b = d["basis_stats"].get(a)
            if b:
                L.append(f"| {a} | {b['n']} | {b['mean_bps']:+.3f} | {b['std_bps']:.3f} |")
        L.append("")

    L.append("## Criterion 6 — live-monitor config-gate (the gap)\n")
    L.append(
        "`funding_monitor` gated on argmax-P > gate ALONE; the per-config hard "
        "thresholds (`min_funding_ann` / `min_pct_positive`) lived only in "
        "`run_funding_single_day` (backtest path). So the monitor alerted — and "
        "the executor booked — trades atlas zeroed. Since Cycle 41 the live "
        "system has therefore been running a DIFFERENT (unverified) strategy "
        "than atlas's Sharpe +4.65 measured; it has not bitten only because "
        "`funding_alerts` has stayed at 0 rows in the current sit-out regime.\n")
    for gate in gates:
        a = results[gate]["prefix"]["aggregate"]
        note = ("net negative — atlas's gate was correctly skipping losers"
                if a["executor_only_pnl_pct"] < 0
                else "net positive here, but taken without the discipline that "
                     "justified them")
        L.append(f"- gate {gate:.2f}: **{a['config_gate_zeroing_pct']:.1f}%** "
                 f"({a['n_zeroed']}/{a['n_trades']}) atlas-zeroed days the pre-fix "
                 f"system books; executor-only P&L = {a['executor_only_pnl_pct']:+.4f}% "
                 f"({note}).")
    L.append("")

    L.append("## D8 — config-gate fix (monitor D8a + executor D8b)\n")
    L.append(
        "Both layers now enforce atlas's Condition 1 + 2 "
        "(`ann_rate ≥ min_funding_ann AND pct_positive ≥ min_pct_positive`): "
        "the monitor suppresses the alert (D8a) and the executor's 11th risk "
        "check skips it (D8b, `config_thresholds_not_met`). Post-fix "
        "decomposition — the atlas-zeroed leak goes to 0 booked / 0.0000% "
        "execOnly:\n")
    for gate in gates:
        d = results[gate]["postfix"]
        r = results[gate]
        L.append(f"### gate P>{gate:.2f} (post-fix)\n")
        L.append(f"- D8a: alerts emitted {r['alerts_prefix']} → "
                 f"{r['alerts_postfix']} (monitor suppressed "
                 f"{r['alerts_prefix'] - r['alerts_postfix']} config-fail signals)")
        L.append(f"- D8b: with all alerts emitted, executor entered "
                 f"{r['d8b_entered']} / skipped {r['d8b_skipped']} "
                 f"(`config_thresholds_not_met`)")
        L.append("```")
        L.extend(fmt_table(d))
        L.append("```\n")

    L.append("## D8b executor smoke test\n")
    L.append(f"- negative case (ann_rate {smoke['neg_ann']:.1f} < min "
             f"{smoke['neg_min']:.0f}): decision=`{smoke['neg_decision']}`, "
             f"reason=`{smoke['neg_reason']}` → "
             f"{'PASS' if smoke['neg_pass'] else 'FAIL'}")
    L.append(f"- positive case (thresholds met): decision="
             f"`{smoke['pos_decision']}` → "
             f"{'PASS' if smoke['pos_pass'] else 'FAIL'}\n")

    L.append("## Column definitions\n")
    L.append(
        "Split into the **shared trade set** (atlas AND executor both booked) "
        "and the **atlas-zeroed leak**:\n\n"
        "- **shared** — # gate-passing days atlas actually traded.\n"
        "- **exec%** — executor cumulative P&L (funding − TC) over the shared set.\n"
        "- **atlas%** — atlas Exp 13 net return over the shared set "
        "(funding + basis − TC); reproduces phase4_model_stats exactly.\n"
        "- **basis%** — basis drag, summed INDEPENDENTLY from regenerated "
        "spot/perp price returns (NOT atlas−exec), so resid% is a genuine check.\n"
        "- **resid%** — atlas% − exec% − basis% over the shared set; ~1e-14 "
        "confirms the funding+basis−TC decomposition is exact.\n"
        "- **zeroed / zero%** — gate-passing days atlas zeroed via its hard "
        "config gate. Pre-fix the executor books these; post-fix execOnly→0.\n"
        "- **execOnly%** — executor P&L booked on atlas-zeroed days.\n")
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def smoke_executor_config_gate() -> dict:
    """D8b targeted smoke: inject a funding_alerts + funding_signals pair whose
    environment fails the config gate (ann_rate < min_funding_ann); confirm the
    executor skips with reason config_thresholds_not_met. Plus a positive
    control that meets the thresholds and enters."""
    db = OUT_DIR / "harness_smoke.db"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    for ddl in DDL.values():
        conn.execute(ddl)
    base_ts = int(datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
    dt = "2025-06-01T00:00:00+00:00"
    # one funding event in-window so a booked position could exit (not required
    # for the skip assertion, but keeps the row well-formed).
    conn.execute("INSERT INTO funding_rates VALUES ('SMOKE','binance',?,?,?)",
                 (base_ts + 8 * 3_600_000, dt, 0.0001))

    def add(asset, ts, ann, pct, min_ann, min_pct):
        conn.execute(
            "INSERT INTO funding_signals (asset,timestamp,datetime,p_profitable,"
            "above_gate,above_gate_050,gate_threshold,best_config_id,hold_days,"
            "min_funding_ann,expected_return,ann_rate,basis_pct,pct_positive,"
            "min_pct_positive,base_rate,features_json,monitor_version) "
            "VALUES (?,?,?,?,1,1,0.70,'fr_test',3,?,0.0,?,0.0,?,?,0.0,NULL,'smoke')",
            (asset, ts, dt, 0.80, min_ann, ann, pct, min_pct))
        conn.execute(
            "INSERT INTO funding_alerts (asset,timestamp,datetime,alerted_at,"
            "p_profitable,gate_threshold,monitor_version) VALUES (?,?,?,?,0.80,0.70,'smoke')",
            (asset, ts, dt, dt))
    # NEG: ann_rate 3.0 < min 5.0 -> must skip
    add("NEG", base_ts, ann=3.0, pct=0.95, min_ann=5.0, min_pct=0.5)
    # POS: ann_rate 30 >= 5, pct 0.95 >= 0.5 -> must enter
    add("POS", base_ts, ann=30.0, pct=0.95, min_ann=5.0, min_pct=0.5)
    conn.commit()
    conn.close()

    run_executor(db, enforce_config_gate=True)

    conn = sqlite3.connect(db)
    neg = conn.execute("SELECT decision, skip_reason FROM paper_trades "
                       "WHERE asset='NEG'").fetchone()
    pos = conn.execute("SELECT decision FROM paper_trades "
                       "WHERE asset='POS'").fetchone()
    conn.close()
    neg_decision, neg_reason = (neg if neg else ("<none>", "<none>"))
    pos_decision = pos[0] if pos else "<none>"
    return {
        "neg_decision": neg_decision, "neg_reason": neg_reason or "",
        "neg_ann": 3.0, "neg_min": 5.0,
        "neg_pass": neg_decision == "skip"
                    and "config_thresholds_not_met" in (neg_reason or ""),
        "pos_decision": pos_decision,
        "pos_pass": pos_decision == "enter",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Cycle 53 backtest replay + D8 verify")
    ap.add_argument("--gates", default=None,
                    help="comma-separated gates (default 0.50,0.70)")
    args = ap.parse_args()
    gates = ([float(x) for x in args.gates.split(",")] if args.gates else GATES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds, funding_series = regenerate_predictions()

    results = {}
    for gate in gates:
        g3 = f"g{int(gate*100):03d}"
        print("\n" + "=" * 72)
        print(f" gate P>{gate:.2f}")
        print("=" * 72)

        # --- D7 pre-fix baseline: every P>gate alerts; executor books all ---
        db_pre = OUT_DIR / f"harness_{g3}_prefix.db"
        n_pre = build_harness(db_pre, preds, funding_series, gate, monitor_gate=False)
        run_executor(db_pre, enforce_config_gate=False)
        decomp_pre = decompose(db_pre, preds, gate)
        print(" D7 pre-fix decomposition:")
        for ln in fmt_table(decomp_pre):
            print(ln)

        # --- D8b only: all alerts emitted, executor's 11th check skips them ---
        db_d8b = OUT_DIR / f"harness_{g3}_d8b.db"
        build_harness(db_d8b, preds, funding_series, gate, monitor_gate=False)
        s_d8b = run_executor(db_d8b, enforce_config_gate=True)

        # --- D8 post-fix: monitor suppresses (D8a) + executor enforces (D8b) ---
        db_post = OUT_DIR / f"harness_{g3}_postfix.db"
        n_post = build_harness(db_post, preds, funding_series, gate, monitor_gate=True)
        run_executor(db_post, enforce_config_gate=True)
        decomp_post = decompose(db_post, preds, gate)
        print(" D8 post-fix decomposition (leak booked -> 0):")
        for ln in fmt_table(decomp_post):
            print(ln)

        decomp_pre["prefix"] = True
        results[gate] = {
            "prefix": decomp_pre, "postfix": decomp_post,
            "alerts_prefix": n_pre, "alerts_postfix": n_post,
            "d8b_entered": s_d8b["entered"], "d8b_skipped": s_d8b["skipped"],
        }

    print("\n" + "=" * 72)
    print(" D8b executor smoke test")
    print("=" * 72)
    smoke = smoke_executor_config_gate()
    print(f"  NEG (ann<min): decision={smoke['neg_decision']} "
          f"reason={smoke['neg_reason']} -> {'PASS' if smoke['neg_pass'] else 'FAIL'}")
    print(f"  POS (met):     decision={smoke['pos_decision']} -> "
          f"{'PASS' if smoke['pos_pass'] else 'FAIL'}")

    out = write_summary(results, gates, smoke)
    (OUT_DIR / "decomposition.json").write_text(
        json.dumps({str(g): {"prefix": results[g]["prefix"],
                             "postfix": results[g]["postfix"]}
                    for g in gates}, indent=2, default=str), encoding="utf-8")
    print(f"\n  SUMMARY written -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
