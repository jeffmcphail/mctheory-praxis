"""Extract headline + per-day gross exposure across all 4 cap settings."""
from pathlib import Path
import re
import pandas as pd
import numpy as np

CAP_DIRS = [
    ("2.0", "outputs/exp10_revival/cap_2"),
    ("1.0", "outputs/exp10_revival/cap_1"),
    ("0.5", "outputs/exp10_revival/cap_0.5"),
    ("0.25", "outputs/exp10_revival/cap_0.25"),
]


def parse_log(log_path: Path) -> dict:
    txt = log_path.read_text(encoding="utf-16-le", errors="replace")
    # Strip null bytes that come from UTF-16 single-byte runs
    txt = txt.replace("\x00", "")
    out = {}
    patterns = {
        "trading_days": r"Trading days:\s+(\d+)",
        "cum_return": r"Cumulative ret:\s+([+\-\d.]+)",
        "ann_return": r"Ann\. return:\s+([+\-\d.]+)",
        "ann_vol": r"Ann\. volatility:\s+([+\-\d.]+)",
        "sharpe": r"Sharpe ratio:\s+([+\-\d.]+)",
        "max_dd": r"Max drawdown:\s+([+\-\d.]+)",
        "win_days": r"Win days:\s+(\d+)/(\d+)",
        "avg_models": r"Avg models/day:\s+([\d.]+)",
        "max_models": r"Max models/day:\s+(\d+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, txt)
        if m:
            out[k] = m.group(0).split(":", 1)[1].strip()
    return out


def parquet_stats(pq_path: Path) -> dict:
    df = pd.read_parquet(pq_path)
    out = {"rows": len(df), "cols": list(df.columns)}
    if "gross_exposure" in df.columns:
        out["gross_exposure_max"] = df["gross_exposure"].max()
        out["gross_exposure_mean"] = df["gross_exposure"].mean()
        out["gross_exposure_median"] = df["gross_exposure"].median()
    if "n_models_active" in df.columns:
        out["n_models_active_max"] = int(df["n_models_active"].max())
        out["n_models_active_mean"] = round(df["n_models_active"].mean(), 2)
    if "portfolio_return" in df.columns:
        rets = df["portfolio_return"].values
        cum = np.cumsum(rets)
        out["recomputed_cum_pct"] = round(cum[-1] * 100, 4)
        daily_mean = np.mean(rets)
        daily_std = np.std(rets) + 1e-12
        out["recomputed_sharpe"] = round(daily_mean / daily_std * np.sqrt(252), 4)
        out["recomputed_max_dd_pct"] = round(np.min(cum - np.maximum.accumulate(cum)) * 100, 4)
        out["n_pos_days_pct"] = round((rets > 0).mean() * 100, 2)
    return out


print(f"{'cap':>5} | {'cum':>10} | {'sharpe':>8} | {'max_dd':>10} | {'gross_max':>10} | {'gross_mean':>10} | {'mod_max':>7} | {'mod_avg':>7}")
print("-" * 100)
results = []
for cap, d in CAP_DIRS:
    dp = Path(d)
    log_stats = parse_log(dp / "phase4.log")
    pq_stats = parquet_stats(dp / "phase4_portfolio.parquet")
    row = {
        "cap": cap,
        "cum": log_stats.get("cum_return", "?"),
        "sharpe": log_stats.get("sharpe", "?"),
        "max_dd": log_stats.get("max_dd", "?"),
        "gross_max": pq_stats.get("gross_exposure_max", "?"),
        "gross_mean": pq_stats.get("gross_exposure_mean", "?"),
        "mod_max": pq_stats.get("n_models_active_max", "?"),
        "mod_avg": pq_stats.get("n_models_active_mean", "?"),
        "recomp_cum_pct": pq_stats.get("recomputed_cum_pct", "?"),
        "recomp_sharpe": pq_stats.get("recomputed_sharpe", "?"),
        "win_pct": pq_stats.get("n_pos_days_pct", "?"),
    }
    results.append(row)
    gm_str = f"{row['gross_max']:.4f}" if isinstance(row['gross_max'], float) else str(row['gross_max'])
    gn_str = f"{row['gross_mean']:.4f}" if isinstance(row['gross_mean'], float) else str(row['gross_mean'])
    print(f"{cap:>5} | {row['cum']:>10} | {row['sharpe']:>8} | {row['max_dd']:>10} | {gm_str:>10} | {gn_str:>10} | {str(row['mod_max']):>7} | {str(row['mod_avg']):>7}")

print()
print("Recomputed-from-parquet (sanity check vs log values):")
for r in results:
    print(f"  cap={r['cap']}: cum={r['recomp_cum_pct']}% sharpe={r['recomp_sharpe']} win_days={r['win_pct']}%")
