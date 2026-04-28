"""
dashboards/data_collector.py — Praxis Data Collection Dashboard

Central command for all data collection jobs. Monitor, start, stop,
configure, and inspect data pipelines.

Eventually absorbed by a DataCollector agent that coordinates with
Research and Trader agents for automated data lifecycle management.

Usage:
    streamlit run dashboards/data_collector.py
"""
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import streamlit as st

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

PRAXIS_DIR = Path(__file__).parent.parent
DATA_DIR = PRAXIS_DIR / "data"
LIVE_DB = DATA_DIR / "live_collector.db"
SPIKE_DB = DATA_DIR / "spike_scanner.db"
LOG_DIR = PRAXIS_DIR / "logs"
TRAINING_DIR = DATA_DIR / "training"

# Job definitions — extensible registry for future agent integration
JOB_REGISTRY = {
    "live_collector": {
        "name": "Live Forward Collector",
        "description": "Samples CLOB midpoints for top markets every 60s",
        "module": "engines.live_collector",
        "command": "start",
        "db": LIVE_DB,
        "icon": "📡",
        "default_args": {"top": 50, "interval": 60},
        "task_name": "PraxisLiveCollector",
    },
    "btc_momentum": {
        "name": "BTC Momentum Collector",
        "description": "Collects BTC 5-min prediction market prices",
        "module": "engines.btc_momentum",
        "command": "collect",
        "db": DATA_DIR / "btc_momentum.db",
        "icon": "₿",
        "default_args": {"minutes": 0},
        "task_name": None,
    },
    "spike_scanner": {
        "name": "Historical Spike Scanner",
        "description": "Collects historical price data from resolved markets",
        "module": "engines.spike_scanner",
        "command": "collect",
        "db": SPIKE_DB,
        "icon": "📊",
        "default_args": {"max_markets": 500, "days": 90},
        "task_name": None,
    },
    "event_classifier": {
        "name": "Event Classifier",
        "description": "LLM-based market classification via DeepSeek",
        "module": "engines.event_classifier",
        "command": "reclassify",
        "db": SPIKE_DB,
        "icon": "🏷️",
        "default_args": {},
        "task_name": None,
    },
    "feature_builder": {
        "name": "Feature Engineering",
        "description": "Builds Stage 1/Stage 2 training datasets from collected data",
        "module": "engines.spike_features",
        "command": "build",
        "db": None,
        "icon": "⚙️",
        "default_args": {"threshold": 5.0},
        "task_name": None,
    },
}


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def safe_db_query(db_path, query, params=()):
    """Execute a query safely, returning empty list on error."""
    if not db_path or not Path(db_path).exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def safe_db_scalar(db_path, query, params=(), default=0):
    """Execute a scalar query safely."""
    if not db_path or not Path(db_path).exists():
        return default
    try:
        conn = sqlite3.connect(str(db_path))
        result = conn.execute(query, params).fetchone()
        conn.close()
        return result[0] if result else default
    except Exception:
        return default


def check_task_status(task_name):
    """Check if a Windows Scheduled Task is running."""
    if not task_name:
        return "not_configured"
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", task_name, "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            status = parts[2].strip('"') if len(parts) > 2 else "Unknown"
            return status.lower()
        return "not_found"
    except Exception:
        return "unknown"


def check_process_running(module_name):
    """Check if a Python module is currently running."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5
        )
        # This is a rough check — just see if python is running
        return "python" in result.stdout.lower()
    except Exception:
        return False


def get_live_collector_stats():
    """Get detailed stats for the live collector."""
    stats = {
        "total_snapshots": safe_db_scalar(LIVE_DB, "SELECT COUNT(*) FROM price_snapshots"),
        "active_markets": safe_db_scalar(LIVE_DB, "SELECT COUNT(*) FROM tracked_markets WHERE active=1"),
        "total_markets": safe_db_scalar(LIVE_DB, "SELECT COUNT(*) FROM tracked_markets"),
        "spike_alerts": safe_db_scalar(LIVE_DB, "SELECT COUNT(*) FROM spike_alerts"),
    }

    # Time range
    first_ts = safe_db_scalar(LIVE_DB, "SELECT MIN(timestamp) FROM price_snapshots")
    last_ts = safe_db_scalar(LIVE_DB, "SELECT MAX(timestamp) FROM price_snapshots")

    if first_ts and last_ts and first_ts > 0:
        stats["first_sample"] = datetime.fromtimestamp(first_ts, tz=timezone.utc)
        stats["last_sample"] = datetime.fromtimestamp(last_ts, tz=timezone.utc)
        stats["duration"] = stats["last_sample"] - stats["first_sample"]
        stats["samples_per_hour"] = stats["total_snapshots"] / max(
            stats["duration"].total_seconds() / 3600, 0.01)
    else:
        stats["first_sample"] = None
        stats["last_sample"] = None
        stats["duration"] = timedelta(0)
        stats["samples_per_hour"] = 0

    # Type distribution
    stats["by_type"] = safe_db_query(LIVE_DB, """
        SELECT event_type, COUNT(DISTINCT tm.slug) as markets, COUNT(ps.id) as snapshots
        FROM tracked_markets tm
        LEFT JOIN price_snapshots ps ON tm.slug = ps.slug
        WHERE tm.active=1
        GROUP BY event_type
        ORDER BY snapshots DESC
    """)

    # Recent collection log
    stats["recent_log"] = safe_db_query(LIVE_DB, """
        SELECT timestamp, markets_tracked, samples_taken, errors, duration_ms
        FROM collection_log ORDER BY id DESC LIMIT 20
    """)

    return stats


def get_spike_scanner_stats():
    """Get stats for the historical spike scanner."""
    stats = {
        "total_markets": safe_db_scalar(SPIKE_DB, "SELECT COUNT(*) FROM markets"),
        "with_prices": safe_db_scalar(SPIKE_DB,
            "SELECT COUNT(*) FROM markets WHERE price_history_fetched=1"),
        "total_spikes": safe_db_scalar(SPIKE_DB, "SELECT COUNT(*) FROM spikes"),
        "price_points": safe_db_scalar(SPIKE_DB, "SELECT COUNT(*) FROM price_history"),
    }

    # Classification stats
    stats["classified"] = safe_db_scalar(SPIKE_DB, "SELECT COUNT(*) FROM taxonomy")
    stats["corrected"] = safe_db_scalar(SPIKE_DB,
        "SELECT COUNT(*) FROM taxonomy WHERE corrected_to IS NOT NULL")

    stats["by_type"] = safe_db_query(SPIKE_DB, """
        SELECT COALESCE(t.corrected_to, t.classified_as, 'unclassified') as etype,
               COUNT(DISTINCT m.slug) as markets,
               COUNT(DISTINCT s.id) as spikes
        FROM markets m
        LEFT JOIN taxonomy t ON m.slug = t.slug
        LEFT JOIN spikes s ON m.slug = s.slug
        GROUP BY etype
        ORDER BY markets DESC
    """)

    return stats


def get_training_stats():
    """Get stats for training data."""
    stats = {}

    s1_path = TRAINING_DIR / "stage1_training.csv"
    s2_path = TRAINING_DIR / "stage2_training.csv"

    for path, name in [(s1_path, "stage1"), (s2_path, "stage2")]:
        if path.exists():
            with open(path) as f:
                lines = f.readlines()
            stats[f"{name}_rows"] = len(lines) - 1  # Exclude header
            stats[f"{name}_cols"] = len(lines[0].split(",")) if lines else 0
            stats[f"{name}_size"] = path.stat().st_size
        else:
            stats[f"{name}_rows"] = 0

    return stats


# ═══════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Praxis Data Collector",
        page_icon="📡",
        layout="wide",
    )

    st.title("📡 Praxis Data Collection Dashboard")
    st.caption("Central command for all data pipelines • Future DataCollector agent interface")

    # ── Sidebar: Job Controls ──
    with st.sidebar:
        st.header("🎛️ Job Controls")

        st.subheader("Live Collector")
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.number_input("Top N markets", 10, 200, 50, step=10)
        with col2:
            interval = st.number_input("Interval (s)", 30, 300, 60, step=10)

        if st.button("🚀 Start Live Collector", use_container_width=True):
            cmd = f"start cmd /k python -m engines.live_collector start --top {top_n} --interval {interval}"
            os.system(cmd)
            st.success(f"Started in new terminal (top {top_n}, every {interval}s)")

        st.divider()

        st.subheader("Historical Collection")
        max_markets = st.number_input("Max markets", 100, 5000, 1000, step=100)
        days_back = st.number_input("Days lookback", 30, 365, 90, step=30)

        if st.button("📊 Run Spike Scanner", use_container_width=True):
            cmd = f"start cmd /k python -m engines.spike_scanner collect --max-markets {max_markets} --days {days_back}"
            os.system(cmd)
            st.success("Spike scanner started in new terminal")

        st.divider()

        st.subheader("Processing")
        if st.button("🏷️ Classify Events (LLM)", use_container_width=True):
            cmd = "start cmd /k python -m engines.event_classifier reclassify"
            os.system(cmd)
            st.success("Classifier started in new terminal")

        if st.button("⚙️ Build Training Data", use_container_width=True):
            cmd = "start cmd /k python -m engines.spike_features build"
            os.system(cmd)
            st.success("Feature builder started in new terminal")

        st.divider()
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()

    # ── Main: Status Overview ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📡 Live Collector",
        "📊 Historical Data",
        "⚙️ Training Data",
        "📋 Logs",
        "🤖 Agent Config",
    ])

    # ═══════════════ TAB 1: Live Collector ═══════════════
    with tab1:
        stats = get_live_collector_stats()

        # Status banner
        task_status = check_task_status("PraxisLiveCollector")
        if stats["total_snapshots"] > 0:
            # Check if data is recent (within last 5 min)
            if stats["last_sample"]:
                age = datetime.now(timezone.utc) - stats["last_sample"]
                if age < timedelta(minutes=5):
                    st.success(f"🟢 **COLLECTING** — Last sample {age.seconds}s ago")
                else:
                    st.warning(f"🟡 **STALE** — Last sample {age} ago. Collector may have stopped.")
            else:
                st.info("🔵 **DATA EXISTS** but no active collection detected")
        else:
            st.error("🔴 **NO DATA** — Start the live collector")

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Snapshots", f"{stats['total_snapshots']:,}")
        c2.metric("Active Markets", stats["active_markets"])
        c3.metric("Spike Alerts", stats["spike_alerts"])
        c4.metric("Samples/Hour", f"{stats['samples_per_hour']:,.0f}")

        if stats["duration"]:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Collection Duration", str(stats["duration"]).split(".")[0])
            with c2:
                if stats["first_sample"]:
                    st.metric("Started",
                              stats["first_sample"].strftime("%Y-%m-%d %H:%M UTC"))

        # Type distribution
        if stats["by_type"]:
            st.subheader("Markets by Event Type")
            type_data = []
            for row in stats["by_type"]:
                type_data.append({
                    "Type": row["event_type"] or "unknown",
                    "Markets": row["markets"],
                    "Snapshots": row["snapshots"],
                })
            st.dataframe(type_data, use_container_width=True, hide_index=True)

        # Recent spike alerts
        alerts = safe_db_query(LIVE_DB, """
            SELECT detected_at, question, event_type, move_pct,
                   price_before, price_now
            FROM spike_alerts ORDER BY detected_at DESC LIMIT 20
        """)
        if alerts:
            st.subheader("🚨 Recent Spike Alerts")
            alert_data = []
            for a in alerts:
                alert_data.append({
                    "Time": a["detected_at"][:19],
                    "Market": a["question"][:50],
                    "Type": a["event_type"],
                    "Move": f"{a['move_pct']:+.1f}%",
                    "Before": f"{a['price_before']:.3f}",
                    "After": f"{a['price_now']:.3f}",
                })
            st.dataframe(alert_data, use_container_width=True, hide_index=True)

        # Data quality check
        st.subheader("Data Quality")
        if stats["total_snapshots"] > 0:
            expected = stats["active_markets"] * (
                stats["duration"].total_seconds() / 60) if stats["duration"] else 0
            actual = stats["total_snapshots"]
            completeness = actual / expected * 100 if expected > 0 else 0
            st.progress(min(completeness / 100, 1.0),
                        text=f"Completeness: {completeness:.1f}% "
                             f"({actual:,} of {expected:,.0f} expected)")

    # ═══════════════ TAB 2: Historical Data ═══════════════
    with tab2:
        stats = get_spike_scanner_stats()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Markets", stats["total_markets"])
        c2.metric("With Prices", stats["with_prices"])
        c3.metric("Spikes Detected", stats["total_spikes"])
        c4.metric("Price Points", f"{stats['price_points']:,}")

        c1, c2 = st.columns(2)
        c1.metric("Classified", stats["classified"])
        c2.metric("Human Corrected", stats["corrected"])

        # Type breakdown
        if stats["by_type"]:
            st.subheader("Markets & Spikes by Event Type")
            type_data = []
            for row in stats["by_type"]:
                type_data.append({
                    "Type": row["etype"],
                    "Markets": row["markets"],
                    "Spikes": row["spikes"],
                    "Spike Rate": f"{row['spikes']/max(row['markets'],1)*100:.0f}%"
                })
            st.dataframe(type_data, use_container_width=True, hide_index=True)

        # Reversion analysis
        reversion_data = safe_db_query(SPIKE_DB, """
            SELECT COALESCE(t.corrected_to, t.classified_as) as etype,
                   COUNT(*) as n,
                   AVG(ABS(s.spike_pct)) as avg_move,
                   AVG(s.reversion_pct) as avg_revert
            FROM spikes s
            LEFT JOIN taxonomy t ON s.slug = t.slug
            GROUP BY etype
            HAVING n >= 3
            ORDER BY avg_revert DESC
        """)
        if reversion_data:
            st.subheader("Reversion Patterns (Tradeability Signal)")
            rev_rows = []
            for r in reversion_data:
                tradeable = "✅ Tradeable" if (r["avg_revert"] or 0) > 50 else (
                    "⚠️ Marginal" if (r["avg_revert"] or 0) > 20 else "❌ Info-driven")
                rev_rows.append({
                    "Type": r["etype"] or "?",
                    "Spikes": r["n"],
                    "Avg Move": f"{r['avg_move']:.1f}%",
                    "Avg Reversion": f"{r['avg_revert']:.1f}%",
                    "Assessment": tradeable,
                })
            st.dataframe(rev_rows, use_container_width=True, hide_index=True)

    # ═══════════════ TAB 3: Training Data ═══════════════
    with tab3:
        stats = get_training_stats()

        if stats.get("stage1_rows", 0) > 0:
            st.success(f"Training data available")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Stage 1 (Classifier)", f"{stats['stage1_rows']} rows")
                st.caption(f"{stats.get('stage1_cols', 0)} features")
            with c2:
                st.metric("Stage 2 (Peak Proximity)", f"{stats.get('stage2_rows', 0)} rows")

            st.warning(
                "⚠️ **Data quality issue:** Most price features are zeros due to hourly/daily "
                "resolution in historical CLOB data. Need live collector data (1-min resolution) "
                "for meaningful features. Let the live collector run for 1-2 weeks."
            )

            # Readiness checklist
            st.subheader("Model Training Readiness")
            live_stats = get_live_collector_stats()
            live_hours = live_stats["duration"].total_seconds() / 3600 if live_stats["duration"] else 0

            checks = [
                ("Live collector running", live_stats["total_snapshots"] > 0, "Start live collector"),
                ("1+ week of 1-min data", live_hours >= 168, f"{live_hours:.0f}h collected (need 168h)"),
                ("Spike captured in live data", live_stats["spike_alerts"] > 0, "Waiting for real spikes"),
                ("100+ Stage 1 samples", stats["stage1_rows"] >= 100, f"{stats['stage1_rows']} samples"),
                ("Feature quality verified", False, "Rebuild features with live data"),
            ]

            for label, done, note in checks:
                icon = "✅" if done else "⬜"
                st.write(f"{icon} **{label}** — {note}")

        else:
            st.info("No training data yet. Run: `python -m engines.spike_features build`")

    # ═══════════════ TAB 4: Logs ═══════════════
    with tab4:
        st.subheader("Log Viewer")

        log_file = LOG_DIR / "live_collector.log"
        if log_file.exists():
            tail_lines = st.slider("Lines to show", 10, 500, 50)
            with open(log_file) as f:
                lines = f.readlines()
            st.code("".join(lines[-tail_lines:]), language="text")
        else:
            st.info("No log file yet. Logs appear when the collector runs via Task Scheduler.")

        # Collection log from DB
        st.subheader("Collection History (from DB)")
        log_rows = safe_db_query(LIVE_DB, """
            SELECT timestamp, markets_tracked, samples_taken, errors, duration_ms
            FROM collection_log ORDER BY id DESC LIMIT 50
        """)
        if log_rows:
            log_data = [{
                "Time": r["timestamp"][:19],
                "Markets": r["markets_tracked"],
                "Samples": r["samples_taken"],
                "Errors": r["errors"],
                "Duration (ms)": f"{r['duration_ms']:.0f}",
            } for r in log_rows]
            st.dataframe(log_data, use_container_width=True, hide_index=True)
        else:
            st.info("No collection history in database.")

    # ═══════════════ TAB 5: Agent Config ═══════════════
    with tab5:
        st.subheader("🤖 Future: DataCollector Agent Interface")

        st.info(
            "This tab will eventually be the configuration interface for the "
            "DataCollector agent in AI Agent Factory. The agent will:\n\n"
            "1. **Auto-discover** new markets and data sources\n"
            "2. **Schedule** collection jobs based on Research agent requests\n"
            "3. **Monitor** data quality and completeness\n"
            "4. **Alert** Trader agent when spikes are detected\n"
            "5. **Coordinate** with Feature Engineering for model retraining\n"
            "6. **Self-optimize** collection parameters (top N, interval, thresholds)"
        )

        st.subheader("Job Registry")
        for job_id, job in JOB_REGISTRY.items():
            with st.expander(f"{job['icon']} {job['name']}", expanded=False):
                st.write(f"**Module:** `{job['module']}`")
                st.write(f"**Command:** `{job['command']}`")
                st.write(f"**Description:** {job['description']}")
                if job["task_name"]:
                    status = check_task_status(job["task_name"])
                    st.write(f"**Task Scheduler:** {status}")
                st.json(job["default_args"])

        st.subheader("Agent-Ready Data Schema")
        st.code("""
# Future DataCollector agent API (AI Agent Factory integration)
{
    "agent_type": "data_collector",
    "capabilities": [
        "market_discovery",      # Find new Polymarket markets
        "price_collection",      # CLOB midpoint sampling
        "spike_detection",       # Real-time spike alerts
        "event_classification",  # LLM-based market typing
        "feature_engineering",   # Training data generation
        "quality_monitoring",    # Data completeness checks
    ],
    "interfaces": {
        "research_agent": "request_data(market, resolution, duration)",
        "trader_agent": "on_spike_detected(market, move_pct, event_type)",
        "evolution_engine": "get_training_data(stage, min_quality)",
    },
    "schedule": {
        "live_collection": "continuous",
        "market_refresh": "every_15m",
        "classification": "on_new_market",
        "feature_rebuild": "on_demand",
    }
}
        """, language="json")


if __name__ == "__main__":
    main()
