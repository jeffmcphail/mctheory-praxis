"""
engines/forced_trade/oi_audit.py

T4 -- the open-interest gap: how bad is it?

THREE SEPARATE QUESTIONS, ANSWERED SEPARATELY
---------------------------------------------
1. Is OI actually absent from crypto_data.db?  (schema scan -- no assumptions)
2. What does `compute_funding_regime` DO about it today -- error, or silently
   degrade?  (static reading of the classifier plus a live A/B call)
3. Can OI be collected from venues already polled, at what cadence, with what
   history?  (live CCXT probe -- capability flags AND an actual retention walk,
   because a `has` flag is a claim and a 400 response is a fact)

WHY (2) MATTERS MORE THAN (1)
-----------------------------
An absent table is visible. A classifier that quietly narrows its own state
space is not. `compute_funding_regime` reaches states +/-2 only when
`oi_change_7d` clears +/-0.10; with `oi_values=None` that variable is
initialised to 0.0 and never updated, so the two extreme states become
UNREACHABLE and class F silently degrades from a five-state axis to a
three-state one. Nothing raises, nothing logs, and `RegimeState.missing` does
not list F. `degradation_probe` demonstrates this by calling the function
directly rather than arguing it from the source.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from engines.forced_trade.common import DEFAULT_DB, read_only_db

logger = logging.getLogger("forced_trade.oi")

# Names that would indicate OI storage if anyone had added it.
OI_TABLE_HINTS = ("open_interest", "openinterest", "oi_history", "oi_snapshots")
OI_COLUMN_HINTS = ("open_interest", "openinterest", "oi_usd", "oi_amount",
                   "oi_value", "sum_open_interest")


def scan_schema_for_oi(db_path=DEFAULT_DB) -> dict:
    """Every table and column in the DB, checked for anything OI-shaped."""
    with read_only_db(db_path) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "ORDER BY name")]
        table_hits, column_hits, all_cols = [], [], {}
        for t in tables:
            if any(h in t.lower() for h in OI_TABLE_HINTS):
                table_hits.append(t)
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info('{t}')")]
            all_cols[t] = cols
            for c in cols:
                if any(h in c.lower() for h in OI_COLUMN_HINTS):
                    column_hits.append(f"{t}.{c}")
    logger.info("schema scan: %d tables, oi_tables=%s oi_columns=%s",
                len(tables), table_hits or "NONE", column_hits or "NONE")
    return {"n_tables": len(tables), "tables": tables,
            "oi_table_hits": table_hits, "oi_column_hits": column_hits,
            "columns_by_table": all_cols,
            "oi_present": bool(table_hits or column_hits)}


def degradation_probe() -> dict:
    """Call compute_funding_regime WITH and WITHOUT OI and compare reachability.

    Sweeps funding across the classifier decision boundaries and records which
    states are attainable in each case. The finding is the SET DIFFERENCE.
    """
    from engines.regime_engine import compute_funding_regime, REGIME_STATE_RANGES

    # Funding values (8h fractional) spanning well past every threshold:
    # fr_ann = fr * 3 * 365 * 100, so +/-0.0005 is about +/-55% annualised.
    fr_grid = np.linspace(-0.0010, 0.0010, 81)
    oi_grid = (-0.30, -0.15, 0.0, 0.15, 0.30)

    def states_for(with_oi: bool) -> set:
        seen = set()
        for fr in fr_grid:
            series = np.full(30, fr, dtype=float)
            if not with_oi:
                s, _ = compute_funding_regime(series, None)
                seen.add(s)
            else:
                for target in oi_grid:
                    # 22 points back is the classifier 7-day reference
                    oi = np.full(30, 1.0, dtype=float)
                    oi[-1] = 1.0 * (1.0 + target)
                    s, _ = compute_funding_regime(series, oi)
                    seen.add(s)
        return seen

    without = states_for(False)
    with_oi = states_for(True)
    declared = set(REGIME_STATE_RANGES["F"])

    # Does it raise, or degrade quietly?
    raised = None
    try:
        compute_funding_regime(np.full(30, 0.001), None)
        raised = False
    except Exception as e:  # noqa: BLE001
        raised = repr(e)

    out = {
        "declared_states": sorted(declared),
        "reachable_with_oi": sorted(with_oi),
        "reachable_without_oi": sorted(without),
        "lost_states": sorted(declared - without),
        "errors_when_oi_missing": raised,
        "silently_degrades": (raised is False) and bool(declared - without),
    }
    logger.info("class F reachability: declared=%s with_oi=%s without_oi=%s "
                "LOST=%s errors=%s", out["declared_states"],
                out["reachable_with_oi"], out["reachable_without_oi"],
                out["lost_states"], out["errors_when_oi_missing"])
    return out


def caller_audit() -> dict:
    """Do the production callers actually pass OI? Read the source, not memory."""
    import inspect
    import re as _re
    from pathlib import Path
    from engines.forced_trade.common import REPO

    targets = ["engines/cpo_training.py", "engines/funding_rate_strategy.py"]
    findings = []
    for rel in targets:
        p = Path(REPO) / rel
        if not p.exists():
            findings.append({"file": rel, "status": "MISSING"})
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        calls = [m.start() for m in _re.finditer(r"\.compute\(", src)]
        for c in calls:
            snippet = src[c:c + 400]
            if "ohlcv_hourly" in snippet or "funding_rates" in snippet:
                findings.append({
                    "file": rel,
                    "line": src[:c].count("\n") + 1,
                    "passes_oi": "oi_series" in snippet,
                })
    return {"callsites": findings,
            "any_passes_oi": any(f.get("passes_oi") for f in findings)}


def probe_venue_oi(venues=("binance", "bybit"), symbol="BTC/USDT:USDT",
                   timeframes=("5m", "1h", "1d"),
                   retention_probe_days=(30, 90, 200, 400, 730)) -> dict:
    """Live CCXT probe: capability, cadence, and ACTUAL retention depth.

    Requires network. The retention walk is the part that matters: both venues
    advertise fetchOpenInterestHistory, and both silently cap how far back it
    goes -- which decides whether a historical backfill is possible at all.
    """
    try:
        import ccxt
    except ImportError:
        return {"error": "ccxt not installed"}

    out = {"ccxt_version": ccxt.__version__, "venues": {}}
    for vid in venues:
        v = {"symbol": symbol}
        try:
            ex = getattr(ccxt, vid)({"enableRateLimit": True,
                                     "options": {"defaultType": "swap"}})
        except Exception as e:  # noqa: BLE001
            out["venues"][vid] = {"error": f"construct failed: {e}"}
            continue

        v["has"] = {k: ex.has.get(k) for k in
                    ("fetchOpenInterest", "fetchOpenInterestHistory")}
        try:
            oi = ex.fetch_open_interest(symbol)
            v["spot_reading"] = {
                "openInterestAmount": oi.get("openInterestAmount"),
                "datetime": oi.get("datetime"),
            }
        except Exception as e:  # noqa: BLE001
            v["spot_reading"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}

        v["timeframes"] = {}
        for tf in timeframes:
            try:
                h = ex.fetch_open_interest_history(symbol, tf, limit=1000)
                v["timeframes"][tf] = {
                    "n": len(h),
                    "first": h[0].get("datetime") if h else None,
                    "last": h[-1].get("datetime") if h else None,
                }
            except Exception as e:  # noqa: BLE001
                v["timeframes"][tf] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            time.sleep(0.35)

        v["retention_1d"] = {}
        now = ex.milliseconds()
        for d in retention_probe_days:
            since = now - d * 86400 * 1000
            try:
                h = ex.fetch_open_interest_history(symbol, "1d", since=since, limit=1000)
                v["retention_1d"][f"since_-{d}d"] = {
                    "n": len(h), "first": h[0].get("datetime") if h else None}
            except Exception as e:  # noqa: BLE001
                v["retention_1d"][f"since_-{d}d"] = {
                    "error": f"{type(e).__name__}: {str(e)[:120]}"}
            time.sleep(0.35)

        try:
            mk = ex.load_markets()
            v["n_active_linear_swaps"] = sum(
                1 for m in mk.values()
                if m.get("swap") and m.get("linear") and m.get("active"))
        except Exception as e:  # noqa: BLE001
            v["n_active_linear_swaps"] = f"error: {str(e)[:80]}"

        out["venues"][vid] = v
    return out
