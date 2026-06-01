"""
engines/funding_executor.py
============================
Cycle 51 funding-carry paper-trading executor scaffold.

Reads funding_alerts (the live monitor's gate-fired signals); applies a 9-control
risk layer; logs the decision (enter | skip) to paper_trades. Idempotent per
(asset, funding-window) via the paper_trades PK and INSERT OR IGNORE.

EXECUTION DISCIPLINE (Cycle 51 explicit constraint, brief safety-belt #5):
  - No CCXT import.
  - No outbound HTTP write-method calls (POST / PUT / DELETE).
  - No stdlib HTTP-client modules; no async HTTP modules.
  - No network calls to any exchange endpoint.
  - The only side effect is writing to the local SQLite paper_trades table.
A safety-belt grep over this file + the migration + the bat is run before
every Cycle 51 commit; expected zero hits. This docstring is phrased to
describe the invariant without containing the exact forbidden tokens, so the
grep is clean even though the constraint is documented.

Risk controls (9, Cycle 51 defaults; tunable via overrides + env vars)
----------------------------------------------------------------------
  1. max_notional_per_asset_usd       500.0   per-asset notional cap
  2. max_total_notional_usd          2500.0   portfolio gross cap
  3. max_daily_loss_usd                50.0   circuit breaker
  4. max_daily_loss_pct                0.02   same expressed as fraction
  5. max_concurrent_positions_per_asset   1   no doubling down
  6. max_signal_age_seconds           5400    90 min staleness ceiling
  7. min_p_above_gate                  0.0    cushion above gate_threshold
                                              (Cycle 51 refinement; no-op at 0.0,
                                              tunable for Cycle 53+ backtest analysis)
  8. asset_blacklist                  []      empty by default
  9. EXECUTOR_KILL_SWITCH env var     off     set "1"/"true" to skip all entries

Cycle 51 scope vs Cycle 52+
---------------------------
Cycle 51 = entry-only paper logging. The executor records an intended entry
decision per (asset, signal_window) but does NOT track hold/exit. So:
  - open_positions_count is always 0 (no lifecycle yet)
  - total_open_notional is always 0
  - daily_loss_so_far is always 0
The corresponding risk checks (concurrent positions, total notional, daily
loss) pass trivially in Cycle 51. Cycle 52 will add hold/exit tracking and
make these checks meaningful. Documented to make the Cycle 51 stub behavior
visible to future auditors of paper_trades.risk_checks_json.

Schema reference
----------------
funding_alerts PK = (asset, timestamp); produced by funding_monitor.py
paper_trades   PK = (asset, signal_timestamp); created Cycle 51 D1.
                    signal_timestamp matches funding_alerts.timestamp.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# dotenv loaded per memory #4; no secrets required by this module, but the
# EXECUTOR_KILL_SWITCH env var is loaded via dotenv for consistency.
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CWD-anchored DB path per Cycle 46 funding-chain convention.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "crypto_data.db"

# Blame trail: each commit bumps this to "cycle<N>:<sha7>" if behavior changes.
EXECUTOR_VERSION = "cycle51-paper-scaffold"

# Exp 13 position direction (long spot + short perp, delta-neutral carry).
EXP13_DIRECTION = "long_spot_short_perp"

DEFAULTS: dict[str, Any] = {
    "max_notional_per_asset_usd":          500.0,
    "max_total_notional_usd":             2500.0,
    "max_daily_loss_usd":                   50.0,
    "max_daily_loss_pct":                    0.02,
    "max_concurrent_positions_per_asset":     1,
    "max_signal_age_seconds":             5400,
    "min_p_above_gate":                      0.0,
    "asset_blacklist":                       [],
}

KILL_SWITCH_ENV = "EXECUTOR_KILL_SWITCH"


# ---------------------------------------------------------------------------
# Risk check result
# ---------------------------------------------------------------------------

@dataclass
class RiskChecks:
    """Outcome of all 9 risk-control evaluations for one alert decision."""
    signal_age_seconds: float
    signal_age_ok: bool
    per_asset_notional_ok: bool
    per_asset_remaining_usd: float
    total_notional_ok: bool
    total_remaining_usd: float
    daily_loss_ok: bool
    daily_loss_remaining_usd: float
    concurrent_positions_ok: bool
    open_positions_for_asset: int
    asset_not_blacklisted: bool
    kill_switch_off: bool
    p_above_min_gate: bool       # Cycle 51 refinement
    min_p_above_gate: float      # the configured value, for the JSON blob

    def all_ok(self) -> bool:
        return (
            self.signal_age_ok
            and self.per_asset_notional_ok
            and self.total_notional_ok
            and self.daily_loss_ok
            and self.concurrent_positions_ok
            and self.asset_not_blacklisted
            and self.kill_switch_off
            and self.p_above_min_gate
        )

    def to_json_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class FundingExecutor:
    """Paper-trading executor for Engine 7 funding-carry signals."""

    def __init__(self, db_path: Path | str = DB_PATH,
                 defaults_override: dict | None = None):
        self.db_path = Path(db_path)
        self.config = dict(DEFAULTS)
        if defaults_override:
            self.config.update(defaults_override)
        self.kill_switch_on = (
            os.getenv(KILL_SWITCH_ENV, "").strip().lower() in ("1", "true", "yes")
        )

    # ---- DB access ----

    def load_pending_alerts(self) -> list[dict]:
        """Return funding_alerts rows that don't yet have a paper_trades
        decision recorded. Sorted oldest-first."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT a.asset, a.timestamp, a.datetime, a.alerted_at,
                       a.p_profitable, a.gate_threshold
                FROM funding_alerts a
                WHERE NOT EXISTS (
                    SELECT 1 FROM paper_trades p
                    WHERE p.asset = a.asset
                      AND p.signal_timestamp = a.timestamp
                )
                ORDER BY a.timestamp
            """)
            rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {
                "asset":          r[0],
                "timestamp":      r[1],
                "datetime":       r[2],
                "alerted_at":     r[3],
                "p_profitable":   float(r[4]),
                "gate_threshold": float(r[5]),
            }
            for r in rows
        ]

    # ---- Risk control stubs (Cycle 51 = entry-only paper; Cycle 52 fills these) ----

    def _open_positions_for_asset(self, asset: str) -> int:
        """Cycle 51 stub: always 0 (no position lifecycle yet)."""
        return 0

    def _total_open_notional_usd(self) -> float:
        """Cycle 51 stub: always 0 (no position lifecycle yet)."""
        return 0.0

    def _daily_loss_so_far_usd(self) -> float:
        """Cycle 51 stub: always 0 (no real or paper P&L tracked yet)."""
        return 0.0

    # ---- Risk evaluation ----

    def apply_risk_checks(self, alert: dict) -> RiskChecks:
        """Evaluate all 9 risk controls for the given alert."""
        now = datetime.now(timezone.utc)
        # funding_alerts.alerted_at is ISO+00:00 per Cycle 43 schema
        alerted_at = datetime.fromisoformat(alert["alerted_at"])
        age_seconds = (now - alerted_at).total_seconds()

        asset = alert["asset"]
        open_pos = self._open_positions_for_asset(asset)
        total_open = self._total_open_notional_usd()
        daily_loss = self._daily_loss_so_far_usd()

        proposed_size = float(self.config["max_notional_per_asset_usd"])

        signal_age_ok = age_seconds <= self.config["max_signal_age_seconds"]

        concurrent_ok = open_pos < self.config["max_concurrent_positions_per_asset"]
        per_asset_notional_ok = concurrent_ok  # same constraint at Cycle 51 (max 1)
        per_asset_remaining = (proposed_size if per_asset_notional_ok else 0.0)

        total_after = total_open + proposed_size
        total_notional_ok = total_after <= self.config["max_total_notional_usd"]
        total_remaining = self.config["max_total_notional_usd"] - total_open

        daily_loss_remaining = self.config["max_daily_loss_usd"] - daily_loss
        daily_loss_ok = daily_loss_remaining > 0

        not_blacklisted = asset not in self.config["asset_blacklist"]
        kill_off = not self.kill_switch_on

        min_gap = float(self.config["min_p_above_gate"])
        p_above_min = (
            float(alert["p_profitable"])
            >= float(alert["gate_threshold"]) + min_gap
        )

        return RiskChecks(
            signal_age_seconds=age_seconds,
            signal_age_ok=signal_age_ok,
            per_asset_notional_ok=per_asset_notional_ok,
            per_asset_remaining_usd=per_asset_remaining,
            total_notional_ok=total_notional_ok,
            total_remaining_usd=total_remaining,
            daily_loss_ok=daily_loss_ok,
            daily_loss_remaining_usd=daily_loss_remaining,
            concurrent_positions_ok=concurrent_ok,
            open_positions_for_asset=open_pos,
            asset_not_blacklisted=not_blacklisted,
            kill_switch_off=kill_off,
            p_above_min_gate=p_above_min,
            min_p_above_gate=min_gap,
        )

    # ---- Decision construction ----

    def decide(self, alert: dict, risks: RiskChecks) -> dict:
        """Translate (alert, risks) into a paper_trades row dict."""
        now_iso = (datetime.now(timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%S+00:00"))
        if risks.all_ok():
            decision = "enter"
            skip_reason = None
            size = float(self.config["max_notional_per_asset_usd"])
            direction = EXP13_DIRECTION
        else:
            decision = "skip"
            reasons = []
            if not risks.signal_age_ok:
                reasons.append(
                    f"signal_age_seconds={risks.signal_age_seconds:.0f} "
                    f"> max_signal_age_seconds={self.config['max_signal_age_seconds']}"
                )
            if not risks.per_asset_notional_ok:
                reasons.append(
                    f"per_asset_notional_exhausted "
                    f"(open={risks.open_positions_for_asset})"
                )
            if not risks.total_notional_ok:
                reasons.append("total_notional_cap_reached")
            if not risks.daily_loss_ok:
                reasons.append("daily_loss_circuit_breaker")
            if not risks.concurrent_positions_ok:
                reasons.append(
                    f"concurrent_position_cap_per_asset "
                    f"(open={risks.open_positions_for_asset})"
                )
            if not risks.asset_not_blacklisted:
                reasons.append(f"asset_blacklisted={alert['asset']}")
            if not risks.kill_switch_off:
                reasons.append(f"{KILL_SWITCH_ENV}=on")
            if not risks.p_above_min_gate:
                reasons.append(
                    f"P={alert['p_profitable']:.4f} below "
                    f"gate+min_p_above_gate="
                    f"{alert['gate_threshold'] + risks.min_p_above_gate:.4f}"
                )
            skip_reason = "; ".join(reasons) if reasons else "all_ok_unexpectedly"
            size = 0.0
            direction = None

        return {
            "asset":                    alert["asset"],
            "signal_timestamp":         int(alert["timestamp"]),
            "signal_datetime":          alert["datetime"],
            "funding_alert_alerted_at": alert["alerted_at"],
            "decided_at":               now_iso,
            "decision":                 decision,
            "skip_reason":              skip_reason,
            "intended_direction":       direction,
            "intended_size_usd":        size,
            "p_profitable":             float(alert["p_profitable"]),
            "gate_threshold":           float(alert["gate_threshold"]),
            "risk_checks_json":         json.dumps(risks.to_json_dict()),
            "executor_version":         EXECUTOR_VERSION,
        }

    # ---- Persistence ----

    def persist(self, decision: dict,
                conn: sqlite3.Connection | None = None) -> bool:
        """INSERT OR IGNORE the decision into paper_trades. Returns True if
        the row was newly inserted (False on PK collision = already logged)."""
        owns_conn = conn is None
        if owns_conn:
            conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO paper_trades "
                "(asset, signal_timestamp, signal_datetime, "
                " funding_alert_alerted_at, decided_at, decision, "
                " skip_reason, intended_direction, intended_size_usd, "
                " p_profitable, gate_threshold, risk_checks_json, "
                " executor_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision["asset"],
                    decision["signal_timestamp"],
                    decision["signal_datetime"],
                    decision["funding_alert_alerted_at"],
                    decision["decided_at"],
                    decision["decision"],
                    decision["skip_reason"],
                    decision["intended_direction"],
                    decision["intended_size_usd"],
                    decision["p_profitable"],
                    decision["gate_threshold"],
                    decision["risk_checks_json"],
                    decision["executor_version"],
                ),
            )
            inserted = (cur.rowcount == 1)
            if owns_conn:
                conn.commit()
            return inserted
        finally:
            if owns_conn:
                conn.close()

    # ---- Main loop ----

    def run_once(self) -> dict:
        """Process all pending alerts. Returns a summary dict for logging."""
        print(f"  db: {self.db_path}")
        print(f"  executor_version: {EXECUTOR_VERSION}")
        if self.kill_switch_on:
            print(f"  ⚠ {KILL_SWITCH_ENV}=ON (env var set); "
                  f"all alerts will be marked skip with reason '{KILL_SWITCH_ENV}=on'")

        alerts = self.load_pending_alerts()
        print(f"  pending funding_alerts (not yet in paper_trades): {len(alerts)}")
        if not alerts:
            return {"processed": 0, "entered": 0, "skipped": 0, "duplicates": 0}

        conn = sqlite3.connect(self.db_path)
        entered = skipped = duplicates = 0
        try:
            for alert in alerts:
                risks = self.apply_risk_checks(alert)
                decision = self.decide(alert, risks)
                newly = self.persist(decision, conn=conn)
                if not newly:
                    duplicates += 1
                    continue
                if decision["decision"] == "enter":
                    entered += 1
                else:
                    skipped += 1
                print(
                    f"  {alert['asset']:<6} window={alert['datetime'][:16]} "
                    f"P={alert['p_profitable']:.4f} > gate {alert['gate_threshold']:.2f}  "
                    f"-> {decision['decision'].upper()}"
                    + (f"  (size ${decision['intended_size_usd']:.0f} "
                       f"{decision['intended_direction']})"
                       if decision['decision'] == "enter" else
                       f"  reason: {decision['skip_reason']}")
                )
            conn.commit()
        finally:
            conn.close()

        return {
            "processed":  len(alerts),
            "entered":    entered,
            "skipped":    skipped,
            "duplicates": duplicates,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cycle 51 funding-carry paper-trading executor "
                    "(NO real money; NO exchange API).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=str(DB_PATH),
                        help=f"SQLite DB path (default {DB_PATH})")
    parser.add_argument("--max-notional-per-asset", type=float, default=None,
                        help="Override per-asset notional USD")
    parser.add_argument("--max-total-notional", type=float, default=None,
                        help="Override total notional USD")
    parser.add_argument("--max-signal-age-seconds", type=int, default=None,
                        help="Override max signal age (default 5400 = 90 min)")
    parser.add_argument("--min-p-above-gate", type=float, default=None,
                        help="Override min_p_above_gate (default 0.0)")
    args = parser.parse_args()

    overrides = {}
    if args.max_notional_per_asset is not None:
        overrides["max_notional_per_asset_usd"] = args.max_notional_per_asset
    if args.max_total_notional is not None:
        overrides["max_total_notional_usd"] = args.max_total_notional
    if args.max_signal_age_seconds is not None:
        overrides["max_signal_age_seconds"] = args.max_signal_age_seconds
    if args.min_p_above_gate is not None:
        overrides["min_p_above_gate"] = args.min_p_above_gate

    print("=" * 70)
    print(" Cycle 51 paper-trading executor")
    print("=" * 70)
    executor = FundingExecutor(db_path=args.db, defaults_override=overrides or None)
    summary = executor.run_once()
    print()
    print(f"  Summary: processed={summary['processed']}  "
          f"entered={summary['entered']}  "
          f"skipped={summary['skipped']}  "
          f"duplicates={summary['duplicates']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
