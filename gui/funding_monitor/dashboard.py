"""
gui/funding_monitor/dashboard.py
==================================
Streamlit dashboard for the live funding rate carry monitor.

Launch:
    streamlit run gui/funding_monitor/dashboard.py

Features:
    - Real-time signal table (auto-refresh every 60s)
    - Per-asset funding rate history chart
    - Basis chart (perp - spot spread)
    - P(profitable) gauge for each asset
    - Signal log (last 24h)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

st.set_page_config(
    page_title="Funding Rate Monitor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background: #0f1520; }
    .signal-active { 
        background: rgba(38,166,154,0.15); 
        border: 1px solid #26a69a; 
        border-radius: 6px; 
        padding: 12px 16px;
        margin: 4px 0;
    }
    .signal-inactive { 
        background: rgba(26,32,53,0.8); 
        border: 1px solid #2a3248;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .metric-label { color: #5d6b8a; font-size: 11px; text-transform: uppercase; }
    .metric-value { color: #d1d4dc; font-size: 22px; font-weight: 700; }
    .positive { color: #26a69a !important; }
    .negative { color: #ef5350 !important; }
    .neutral  { color: #9ba8bf !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_ASSETS  = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]
DEFAULT_MODELS  = "output/funding_rate/cpo/phase3_models.joblib"
DEFAULT_GATE    = 0.70
DEFAULT_CACHE   = "data/funding_cache"

# ── Session state ────────────────────────────────────────────────────────────

if "signals"       not in st.session_state: st.session_state.signals       = []
if "last_refresh"  not in st.session_state: st.session_state.last_refresh  = None
if "signal_log"    not in st.session_state: st.session_state.signal_log    = []
if "raw_data"      not in st.session_state: st.session_state.raw_data      = {}
if "loading"       not in st.session_state: st.session_state.loading       = False

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    gate = st.slider("Gate threshold (P >)", 0.50, 0.95, DEFAULT_GATE, 0.05)

    assets_input = st.multiselect(
        "Assets",
        ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "BNB"],
        default=DEFAULT_ASSETS,
    )

    models_path = st.text_input("Models path", DEFAULT_MODELS)
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=True)

    refresh_btn = st.button("🔄 Refresh Now", use_container_width=True)

# ── Header ───────────────────────────────────────────────────────────────────

col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
with col_h1:
    st.title("₿ Funding Rate Carry Monitor")
with col_h2:
    now = datetime.now(timezone.utc)
    st.metric("UTC Time", now.strftime("%H:%M"))
with col_h3:
    from scripts.funding_monitor import next_funding_window
    nw = next_funding_window(now)
    mins = int((nw - now).total_seconds() / 60)
    st.metric("Next funding", f"in {mins}m")

st.divider()

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_signals(assets, models_path, gate, _ts):
    """Load live signals (cached for 5 min)."""
    try:
        from scripts.funding_monitor import (
            fetch_live_data, compute_live_features, run_inference
        )
        import argparse

        class FakeArgs:
            cache_dir = DEFAULT_CACHE

        data = fetch_live_data(assets, cache_dir=DEFAULT_CACHE)
        features = compute_live_features(data, assets)
        signals  = run_inference(features, models_path, assets, gate=gate)
        return signals, data, None
    except Exception as e:
        return [], {}, str(e)


def maybe_refresh():
    should = (
        refresh_btn or
        st.session_state.last_refresh is None or
        (auto_refresh and
         (datetime.now(timezone.utc) - st.session_state.last_refresh).seconds > 60)
    )
    if should and not st.session_state.loading:
        st.session_state.loading = True
        with st.spinner("Fetching live data from Binance..."):
            ts = datetime.now(timezone.utc).isoformat()
            signals, data, err = load_signals(
                tuple(assets_input), models_path, gate, ts
            )
        st.session_state.signals      = signals
        st.session_state.raw_data     = data
        st.session_state.last_refresh = datetime.now(timezone.utc)
        st.session_state.loading      = False

        # Log active signals
        active = [s for s in signals if s["above_gate"]]
        if active:
            st.session_state.signal_log.insert(0, {
                "time":    datetime.now(timezone.utc).strftime("%H:%M UTC"),
                "assets":  ", ".join(s["asset"] for s in active),
                "max_p":   max(s["p_profitable"] for s in active),
            })
            # Keep last 48 entries
            st.session_state.signal_log = st.session_state.signal_log[:48]

        if err:
            st.error(f"Error: {err}")

maybe_refresh()

signals   = st.session_state.signals
last_ref  = st.session_state.last_refresh

if last_ref:
    st.caption(f"Last refreshed: {last_ref.strftime('%H:%M:%S UTC')}")

# ── Signal summary cards ──────────────────────────────────────────────────────

active   = [s for s in signals if s["above_gate"]]
inactive = [s for s in signals if not s["above_gate"]]

if active:
    st.success(f"✅ {len(active)} asset(s) above gate — carry trade recommended")
else:
    st.info("⏸ No assets above gate — staying flat")

# ── Signal table ──────────────────────────────────────────────────────────────

st.subheader("Signal Table")

if signals:
    df = pd.DataFrame(signals)
    df["P(profit)"]  = df["p_profitable"].map("{:.3f}".format)
    df["Ann Rate"]   = df["ann_rate"].map("{:+.1f}%".format)
    df["Basis"]      = df["basis_pct"].map("{:+.3f}%".format)
    df["Pct+"]       = df["pct_positive"].map("{:.0%}".format)
    df["Hold"]       = df["hold_days"].map("{}d".format)
    df["ExpReturn"]  = df["exp_return"].map("{:+.4f}".format)
    df["Signal"]     = df["above_gate"].map(lambda x: "🟢 ACTIVE" if x else "⚪ flat")

    display_df = df[["asset","Signal","P(profit)","Ann Rate","Basis","Pct+","Hold","ExpReturn"]]
    display_df.columns = ["Asset","Signal","P(profit)","Ann Rate%","Basis%","Pct+","Hold","ExpReturn"]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Signal": st.column_config.TextColumn(width="small"),
            "P(profit)": st.column_config.NumberColumn(format="%.3f"),
        }
    )

# ── Charts ────────────────────────────────────────────────────────────────────

data = st.session_state.raw_data
if data.get("spot") and data.get("perp"):
    st.subheader("Funding Rate History (last 30 days)")

    # Build funding rate chart data
    fr_rows = []
    for asset in assets_input:
        perp_d = data["perp"].get(asset)
        if perp_d and "funding" in perp_d:
            fr = perp_d["funding"]
            # Annualize: 8h rate × 3 × 365
            fr_ann = fr * 3 * 365 * 100
            fr_30d = fr_ann[fr_ann.index >= (datetime.now(timezone.utc) - timedelta(days=30))]
            for ts, val in fr_30d.items():
                fr_rows.append({"ts": ts, "asset": asset, "ann_rate": float(val)})

    if fr_rows:
        fr_df = pd.DataFrame(fr_rows)
        fr_pivot = fr_df.pivot_table(index="ts", columns="asset", values="ann_rate")
        st.line_chart(fr_pivot, use_container_width=True)

    st.subheader("Basis (Perp - Spot) %")

    basis_rows = []
    for asset in assets_input:
        spot_d = data["spot"].get(asset)
        perp_d = data["perp"].get(asset)
        if spot_d is None or perp_d is None:
            continue
        merged = pd.concat([
            spot_d["close"].rename("spot"),
            perp_d["perp"]["close"].rename("perp"),
        ], axis=1).dropna()
        basis = (merged["perp"] - merged["spot"]) / merged["spot"] * 100
        basis_30d = basis[basis.index >= (datetime.now(timezone.utc) - timedelta(days=30))]
        for ts, val in basis_30d.items():
            basis_rows.append({"ts": ts, "asset": asset, "basis": float(val)})

    if basis_rows:
        b_df    = pd.DataFrame(basis_rows)
        b_pivot = b_df.pivot_table(index="ts", columns="asset", values="basis")
        st.line_chart(b_pivot, use_container_width=True)

# ── Signal log ────────────────────────────────────────────────────────────────

if st.session_state.signal_log:
    st.subheader("Signal Log (last 48 events)")
    log_df = pd.DataFrame(st.session_state.signal_log)
    st.dataframe(log_df, use_container_width=True, hide_index=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    st.markdown(
        '<meta http-equiv="refresh" content="60">',
        unsafe_allow_html=True,
    )

# ── Instructions ──────────────────────────────────────────────────────────────

with st.expander("ℹ️ How to use"):
    st.markdown("""
    **Setup:**
    1. Run Phase 2+3 to train the RF: `python scripts/run_cpo.py --strategy funding_rate ... phase2 phase3`
    2. Launch this dashboard: `streamlit run gui/funding_monitor/dashboard.py`

    **Signal interpretation:**
    - 🟢 ACTIVE = P(profitable) > gate threshold → enter carry trade
    - ⚪ flat = below gate → do not enter, conditions unfavorable

    **Trade execution when signal is active:**
    - Long spot (e.g. buy BTC/USDT on spot)
    - Short perp (e.g. sell BTC/USDT:USDT perp)
    - Hold for the RF-selected duration (3, 7, or 14 days)
    - Target: ~4 bps TC per leg (use Binance maker orders)

    **Gate tuning:**
    - P > 0.70 (default): ~50% of trading days active, Sharpe ~4-5
    - P > 0.80 (aggressive): ~15% of days active, Sharpe ~10+, 90% win rate

    **Webhook alerts:**
    Run the CLI monitor with `--webhook <url>` to get Slack/Discord alerts
    when signals cross the gate: `python scripts/funding_monitor.py --loop --webhook <url>`
    """)
