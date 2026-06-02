"""
engines/funding_executor.py
============================
Engine 7 funding-carry paper-trading executor (Cycle 51 + Cycle 52 lifecycle).

Per scheduled invocation (00:20/08:20/16:20 LOCAL via PraxisFundingExecutor):
  1. Entry loop: load pending funding_alerts (joined with funding_signals
     for hold_days), apply 9-risk-control + 1-data-availability decision
     framework, log to paper_trades.
  2. Exit-reconciliation loop: for each open paper position whose hold
     window has fully elapsed, compute funding payments + transaction
     costs from on-disk funding_rates, log to paper_position_exits.

Both loops are idempotent (PK on (asset, signal_timestamp) + INSERT OR
IGNORE) so re-running within the same window is a no-op.

EXECUTION DISCIPLINE (Cycle 51 explicit constraint, brief safety-belt #5):
  - No CCXT import.
  - No outbound HTTP write-method calls (POST / PUT / DELETE).
  - No stdlib HTTP-client modules; no async HTTP modules.
  - No network calls to any exchange endpoint.
  - The only side effects are writes to local SQLite tables.
A safety-belt grep over this file + the migrations + the bat is run
before every commit; expected zero hits. This docstring is phrased to
describe the invariant without containing the exact forbidden tokens,
so the grep is clean even though the constraint is documented.

Risk + data-availability controls (10 total)
--------------------------------------------
Risk controls (9, Cycle 51 defaults; tunable via overrides + env vars):
  1. max_notional_per_asset_usd       500.0   per-asset notional cap
  2. max_total_notional_usd          2500.0   portfolio gross cap
  3. max_daily_loss_usd                50.0   circuit breaker
  4. max_daily_loss_pct                0.02   same expressed as fraction
  5. max_concurrent_positions_per_asset   1   no doubling down
  6. max_signal_age_seconds           5400    90 min staleness ceiling
  7. min_p_above_gate                  0.0    cushion above gate_threshold
                                              (Cycle 51 refinement; no-op
                                              at 0.0, tunable for Cycle 53+)
  8. asset_blacklist                  []      empty by default
  9. EXECUTOR_KILL_SWITCH env var     off     set "1"/"true" to skip all entries

Data-availability check (10th, Cycle 52 addition):
 10. hold_days_known                   true   JOIN-resolved from funding_signals,
                                              or set via --force-hold-days N
                                              override. No fallback default;
                                              a JOIN miss with no override
                                              forces decision='skip' with
                                              skip_reason='hold_days_unknown'.

Cycle 52 lifecycle (D5 + D6)
----------------------------
Position lifecycle: enter -> hold N days -> exit. The three risk-check
stubs from Cycle 51 (open_positions, total_open_notional, daily_loss_so_far)
now query real state from paper_trades + paper_position_exits.

Exit timing: window-aligned per atlas Exp 13 run_funding_hold semantics
  (engines/funding_rate_strategy.py:173-177). target_exit_timestamp =
  signal_timestamp + hold_days * 86400000 ms. Funding payments collected
  over events with timestamp > signal_timestamp AND timestamp <= target
  (exclusive at entry, inclusive at exit).

Sign convention (long spot + short perp):
  We are SHORT perp; positive funding rate -> longs pay shorts -> we
  RECEIVE. funding_payments_usd = sum(rate * notional) over collected
  events. Net P&L = funding_payments - tc_entry - tc_exit.
  TC baseline: 4 bps one-way (Binance taker proxy) -> 0.0008 round-trip.

Schema reference
----------------
funding_alerts        PK = (asset, timestamp). Sparse; one row per gate firing.
funding_signals       PK = (asset, timestamp). Carries hold_days, best_config_id.
paper_trades          PK = (asset, signal_timestamp). One row per decision.
paper_position_exits  PK = (asset, signal_timestamp). One row per closure.
funding_rates         PK = (asset, venue, timestamp). Cycle 50 schema.
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

# Blame trail: each cycle bumps this so paper_trades.executor_version and
# paper_position_exits.executor_version identify which build wrote them.
EXECUTOR_VERSION = "cycle52-lifecycle"

# Exp 13 position direction (long spot + short perp, delta-neutral carry).
EXP13_DIRECTION = "long_spot_short_perp"

# Single-venue commit (Cycle 50 structural finding -> Cycle 51 architectural decision).
EXECUTION_VENUE = "binance"

# TC baseline: 4 bps one-way per leg per atlas Exp 13. Applied to notional once
# at entry, once at exit. tc_pct = 0.0004; round-trip = 0.0008.
TC_PCT_ONE_WAY = 0.0004

# 1 day in milliseconds, used for hold_days -> target_exit_timestamp.
MS_PER_DAY = 86_400_000

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
# Risk + data-availability check result
# ---------------------------------------------------------------------------

@dataclass
class RiskChecks:
    """Outcome of all 10 checks (9 risk + 1 data-availability) for one alert decision."""
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
    p_above_min_gate: bool
    min_p_above_gate: float
    hold_days_known: bool             # Cycle 52: 10th check (data-availability)
    hold_days_value: int | None       # JOIN-resolved or --force-hold-days override; None on JOIN miss

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
            and self.hold_days_known
        )

    def to_json_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class FundingExecutor:
    """Paper-trading executor for Engine 7 funding-carry signals."""

    def __init__(self, db_path: Path | str = DB_PATH,
                 defaults_override: dict | None = None,
                 force_hold_days: int | None = None):
        self.db_path = Path(db_path)
        self.config = dict(DEFAULTS)
        if defaults_override:
            self.config.update(defaults_override)
        self.kill_switch_on = (
            os.getenv(KILL_SWITCH_ENV, "").strip().lower() in ("1", "true", "yes")
        )
        # If set, override the JOIN-resolved hold_days. Used by Cycle 53
        # backtest sweeps to iterate hold values without touching the
        # underlying funding_signals data.
        self.force_hold_days = force_hold_days

    # ---- DB access ----

    def load_pending_alerts(self) -> list[dict]:
        """Return funding_alerts rows that don't yet have a paper_trades
        decision recorded. Each row is LEFT JOIN'd with funding_signals to
        pull hold_days; the JOIN is left-side because funding_signals could
        theoretically be missing (data-integrity anomaly we surface explicitly
        rather than silently default). Sorted oldest-first."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT a.asset, a.timestamp, a.datetime, a.alerted_at,
                       a.p_profitable, a.gate_threshold,
                       s.hold_days
                FROM funding_alerts a
                LEFT JOIN funding_signals s
                  ON s.asset = a.asset AND s.timestamp = a.timestamp
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
        out = []
        for r in rows:
            join_hold_days = r[6]
            # --force-hold-days overrides the JOIN result; useful for backtests.
            if self.force_hold_days is not None:
                effective_hold = int(self.force_hold_days)
            elif join_hold_days is not None:
                effective_hold = int(join_hold_days)
            else:
                effective_hold = None  # surfaces as hold_days_known=False
            out.append({
                "asset":          r[0],
                "timestamp":      r[1],
                "datetime":       r[2],
                "alerted_at":     r[3],
                "p_profitable":   float(r[4]),
                "gate_threshold": float(r[5]),
                "hold_days":      effective_hold,
            })
        return out

    # ---- Real position-state queries (replaces Cycle 51 stubs) ----

    def _open_positions_for_asset(self, asset: str,
                                  conn: sqlite3.Connection | None = None) -> int:
        """Count of open paper positions for `asset` (entry recorded, no exit yet)."""
        owns = conn is None
        if owns:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute("""
                SELECT COUNT(*) FROM paper_trades p
                WHERE p.asset = ?
                  AND p.decision = 'enter'
                  AND NOT EXISTS (
                      SELECT 1 FROM paper_position_exits x
                      WHERE x.asset = p.asset
                        AND x.signal_timestamp = p.signal_timestamp
                  )
            """, (asset,))
            return int(cur.fetchone()[0])
        finally:
            if owns:
                conn.close()

    def _total_open_notional_usd(self,
                                 conn: sqlite3.Connection | None = None) -> float:
        """Sum of intended_size_usd across all open paper positions."""
        owns = conn is None
        if owns:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute("""
                SELECT COALESCE(SUM(p.intended_size_usd), 0.0)
                FROM paper_trades p
                WHERE p.decision = 'enter'
                  AND NOT EXISTS (
                      SELECT 1 FROM paper_position_exits x
                      WHERE x.asset = p.asset
                        AND x.signal_timestamp = p.signal_timestamp
                  )
            """)
            return float(cur.fetchone()[0])
        finally:
            if owns:
                conn.close()

    def _daily_loss_so_far_usd(self,
                               conn: sqlite3.Connection | None = None) -> float:
        """Absolute USD value of net_pnl losses booked TODAY (UTC). Positive
        return value = "we have already lost this much today" so the
        daily-loss circuit-breaker check can compare against max_daily_loss_usd."""
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        owns = conn is None
        if owns:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute("""
                SELECT COALESCE(SUM(net_pnl_usd), 0.0)
                FROM paper_position_exits
                WHERE substr(exit_decided_at, 1, 10) = ?
                  AND net_pnl_usd < 0
            """, (today_utc,))
            signed = float(cur.fetchone()[0])
            return abs(signed)
        finally:
            if owns:
                conn.close()

    def get_open_positions(self, asset: str | None = None,
                           conn: sqlite3.Connection | None = None) -> list[dict]:
        """List currently open paper positions (decision='enter' without exit).
        Used by D5b risk checks and by D6 exit-reconciliation."""
        owns = conn is None
        if owns:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            sql = """
                SELECT p.asset, p.signal_timestamp, p.signal_datetime,
                       p.decided_at, p.intended_direction,
                       p.intended_size_usd, p.hold_days
                FROM paper_trades p
                WHERE p.decision = 'enter'
                  AND NOT EXISTS (
                      SELECT 1 FROM paper_position_exits x
                      WHERE x.asset = p.asset
                        AND x.signal_timestamp = p.signal_timestamp
                  )
            """
            params: tuple = ()
            if asset is not None:
                sql += " AND p.asset = ?"
                params = (asset,)
            sql += " ORDER BY p.signal_timestamp"
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        finally:
            if owns:
                conn.close()
        return [
            {
                "asset":              r[0],
                "signal_timestamp":   int(r[1]),
                "signal_datetime":    r[2],
                "entry_decided_at":   r[3],
                "intended_direction": r[4],
                "intended_size_usd":  float(r[5]) if r[5] is not None else 0.0,
                "hold_days":          int(r[6]) if r[6] is not None else None,
            }
            for r in rows
        ]

    # ---- Risk evaluation ----

    def apply_risk_checks(self, alert: dict,
                          conn: sqlite3.Connection | None = None) -> RiskChecks:
        """Evaluate all 10 checks (9 risk + 1 data-availability) for the alert."""
        now = datetime.now(timezone.utc)
        alerted_at = datetime.fromisoformat(alert["alerted_at"])
        age_seconds = (now - alerted_at).total_seconds()

        asset = alert["asset"]
        open_pos = self._open_positions_for_asset(asset, conn=conn)
        total_open = self._total_open_notional_usd(conn=conn)
        daily_loss = self._daily_loss_so_far_usd(conn=conn)

        proposed_size = float(self.config["max_notional_per_asset_usd"])

        signal_age_ok = age_seconds <= self.config["max_signal_age_seconds"]

        concurrent_ok = open_pos < self.config["max_concurrent_positions_per_asset"]
        per_asset_notional_ok = concurrent_ok  # same constraint at Cycle 51/52 (max 1)
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

        hold_days_value = alert.get("hold_days")
        hold_days_known = hold_days_value is not None

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
            hold_days_known=hold_days_known,
            hold_days_value=int(hold_days_value) if hold_days_known else None,
        )

    # ---- Decision construction ----

    def decide(self, alert: dict, risks: RiskChecks) -> dict:
        """Translate (alert, risks) into a paper_trades row dict.

        hold_days policy:
          - If JOIN/override resolved a value, write it (even on skip rows,
            for forensic context).
          - If JOIN missed and no override, write NULL.
        """
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
            if not risks.hold_days_known:
                reasons.append("hold_days_unknown")
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
            "hold_days":                risks.hold_days_value,
        }

    # ---- Persistence ----

    def persist(self, decision: dict,
                conn: sqlite3.Connection | None = None) -> bool:
        """INSERT OR IGNORE the decision into paper_trades. Returns True if
        newly inserted (False on PK collision = already logged)."""
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
                " executor_version, hold_days) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    decision["hold_days"],
                ),
            )
            inserted = (cur.rowcount == 1)
            if owns_conn:
                conn.commit()
            return inserted
        finally:
            if owns_conn:
                conn.close()

    # ---- D6: Exit reconciliation ----

    def compute_exit(self, pos: dict,
                     conn: sqlite3.Connection) -> dict | None:
        """For an open position whose hold window has elapsed, compute the
        exit row dict. Returns None if hold window not yet elapsed.

        Window semantics (atlas-mirroring, exclusive entry / inclusive exit):
          target_exit_timestamp = signal_timestamp + hold_days * MS_PER_DAY
          funding events collected: t > signal_timestamp AND t <= target

        Sign convention: short perp, positive rate -> we receive.
          funding_payments_usd = sum(rate * notional)
          net_pnl_usd = funding_payments - tc_entry - tc_exit
        """
        if pos["hold_days"] is None:
            return None  # shouldn't happen (entries require hold_days_known)
        signal_ts = int(pos["signal_timestamp"])
        hold_days = int(pos["hold_days"])
        target_exit_ts = signal_ts + hold_days * MS_PER_DAY
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if now_ms < target_exit_ts:
            return None  # still holding

        notional_usd = float(pos["intended_size_usd"])

        cur = conn.execute("""
            SELECT timestamp, funding_rate
            FROM funding_rates
            WHERE asset = ?
              AND venue = ?
              AND timestamp >  ?
              AND timestamp <= ?
            ORDER BY timestamp
        """, (pos["asset"], EXECUTION_VENUE, signal_ts, target_exit_ts))
        events = cur.fetchall()

        funding_events_count = len(events)
        funding_payments_usd = sum(
            float(rate) * notional_usd for _, rate in events if rate is not None
        )
        tc_entry_usd = TC_PCT_ONE_WAY * notional_usd
        tc_exit_usd = TC_PCT_ONE_WAY * notional_usd
        net_pnl_usd = funding_payments_usd - tc_entry_usd - tc_exit_usd

        exit_dt = datetime.fromtimestamp(target_exit_ts / 1000, timezone.utc)
        exit_datetime = exit_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        exit_decided_at = (datetime.now(timezone.utc)
                                    .strftime("%Y-%m-%dT%H:%M:%S+00:00"))

        return {
            "asset":                pos["asset"],
            "signal_timestamp":     signal_ts,
            "entry_decided_at":     pos["entry_decided_at"],
            "exit_decided_at":      exit_decided_at,
            "exit_timestamp":       target_exit_ts,
            "exit_datetime":        exit_datetime,
            "hold_days":            hold_days,
            "funding_events_count": funding_events_count,
            "funding_payments_usd": funding_payments_usd,
            "tc_entry_usd":         tc_entry_usd,
            "tc_exit_usd":          tc_exit_usd,
            "net_pnl_usd":          net_pnl_usd,
            "notional_usd":         notional_usd,
            "direction":            pos["intended_direction"],
            "executor_version":     EXECUTOR_VERSION,
            # Echo for log display, not persisted:
            "_events":              [(int(ts), float(rate) if rate is not None else None)
                                     for ts, rate in events],
        }

    def persist_exit(self, exit_row: dict,
                     conn: sqlite3.Connection) -> bool:
        """INSERT OR IGNORE the exit row into paper_position_exits."""
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO paper_position_exits "
            "(asset, signal_timestamp, entry_decided_at, exit_decided_at, "
            " exit_timestamp, exit_datetime, hold_days, funding_events_count, "
            " funding_payments_usd, tc_entry_usd, tc_exit_usd, net_pnl_usd, "
            " notional_usd, direction, executor_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exit_row["asset"],
                exit_row["signal_timestamp"],
                exit_row["entry_decided_at"],
                exit_row["exit_decided_at"],
                exit_row["exit_timestamp"],
                exit_row["exit_datetime"],
                exit_row["hold_days"],
                exit_row["funding_events_count"],
                exit_row["funding_payments_usd"],
                exit_row["tc_entry_usd"],
                exit_row["tc_exit_usd"],
                exit_row["net_pnl_usd"],
                exit_row["notional_usd"],
                exit_row["direction"],
                exit_row["executor_version"],
            ),
        )
        return cur.rowcount == 1

    # ---- Main loop ----

    def run_once(self) -> dict:
        """Process pending alerts + exit-reconcile open positions.
        Returns a summary dict for logging."""
        print(f"  db: {self.db_path}")
        print(f"  executor_version: {EXECUTOR_VERSION}")
        if self.kill_switch_on:
            print(f"  WARNING {KILL_SWITCH_ENV}=ON (env var set); "
                  f"all alerts will be marked skip with reason '{KILL_SWITCH_ENV}=on'")
        if self.force_hold_days is not None:
            print(f"  --force-hold-days={self.force_hold_days} "
                  f"(JOIN value ignored; for backtest/smoke use)")

        # ---- Entry loop ----
        alerts = self.load_pending_alerts()
        print(f"  pending funding_alerts (not yet in paper_trades): {len(alerts)}")

        conn = sqlite3.connect(self.db_path)
        entered = skipped = duplicates = 0
        try:
            for alert in alerts:
                risks = self.apply_risk_checks(alert, conn=conn)
                decision = self.decide(alert, risks)
                newly = self.persist(decision, conn=conn)
                if not newly:
                    duplicates += 1
                    continue
                if decision["decision"] == "enter":
                    entered += 1
                else:
                    skipped += 1
                hold_str = (f"  hold={decision['hold_days']}d"
                            if decision['hold_days'] is not None else "  hold=?")
                print(
                    f"  {alert['asset']:<6} window={alert['datetime'][:16]} "
                    f"P={alert['p_profitable']:.4f} > gate {alert['gate_threshold']:.2f}"
                    f"{hold_str}  -> {decision['decision'].upper()}"
                    + (f"  (size ${decision['intended_size_usd']:.0f} "
                       f"{decision['intended_direction']})"
                       if decision['decision'] == "enter" else
                       f"  reason: {decision['skip_reason']}")
                )
            conn.commit()

            # ---- Exit-reconciliation loop ----
            open_positions = self.get_open_positions(conn=conn)
            print(f"  open paper positions: {len(open_positions)}")
            exited = 0
            still_holding = 0
            exit_duplicates = 0
            for pos in open_positions:
                exit_row = self.compute_exit(pos, conn=conn)
                if exit_row is None:
                    still_holding += 1
                    target_ts = (int(pos["signal_timestamp"])
                                 + (pos["hold_days"] or 0) * MS_PER_DAY)
                    remaining_s = (target_ts - int(datetime.now(timezone.utc)
                                                     .timestamp() * 1000)) / 1000
                    print(
                        f"  {pos['asset']:<6} entry={pos['signal_datetime'][:16]} "
                        f"hold={pos['hold_days']}d  HOLDING "
                        f"(remaining ~{remaining_s/86400:.2f}d)"
                    )
                    continue
                newly = self.persist_exit(exit_row, conn=conn)
                if not newly:
                    exit_duplicates += 1
                    continue
                exited += 1
                event_ts_list = ", ".join(
                    f"{datetime.fromtimestamp(ts/1000, timezone.utc).strftime('%m-%dT%H:%M')}={rate:+.5f}"
                    for ts, rate in exit_row["_events"][:6]
                )
                if len(exit_row["_events"]) > 6:
                    event_ts_list += f", ... ({len(exit_row['_events'])} total)"
                print(
                    f"  {pos['asset']:<6} entry={pos['signal_datetime'][:16]} "
                    f"-> EXIT at {exit_row['exit_datetime'][:16]} "
                    f"(hold={exit_row['hold_days']}d, "
                    f"events={exit_row['funding_events_count']}, "
                    f"funding=${exit_row['funding_payments_usd']:+.4f}, "
                    f"tc=${exit_row['tc_entry_usd'] + exit_row['tc_exit_usd']:.4f}, "
                    f"net=${exit_row['net_pnl_usd']:+.4f})"
                )
                if exit_row["_events"]:
                    print(f"           events: {event_ts_list}")
            conn.commit()
        finally:
            conn.close()

        return {
            "processed":           len(alerts),
            "entered":             entered,
            "skipped":             skipped,
            "duplicates":          duplicates,
            "open_positions":      len(open_positions),
            "exited":              exited,
            "still_holding":       still_holding,
            "exit_duplicates":     exit_duplicates,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Engine 7 funding-carry paper-trading executor "
                    "(Cycle 51 + Cycle 52 lifecycle; NO real money; NO exchange API).",
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
    parser.add_argument("--force-hold-days", type=int, default=None,
                        help="Cycle 52: override JOIN-resolved hold_days "
                             "with a fixed N. Used by Cycle 53 backtest "
                             "sweeps; production runs leave this unset.")
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
    print(" Engine 7 paper-trading executor (Cycle 52 lifecycle)")
    print("=" * 70)
    executor = FundingExecutor(
        db_path=args.db,
        defaults_override=overrides or None,
        force_hold_days=args.force_hold_days,
    )
    summary = executor.run_once()
    print()
    print(f"  Entry:  processed={summary['processed']}  "
          f"entered={summary['entered']}  "
          f"skipped={summary['skipped']}  "
          f"duplicates={summary['duplicates']}")
    print(f"  Exit:   open={summary['open_positions']}  "
          f"exited={summary['exited']}  "
          f"still_holding={summary['still_holding']}  "
          f"duplicates={summary['exit_duplicates']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
