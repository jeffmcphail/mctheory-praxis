import pyarrow.parquet as pq

t = pq.read_table("outputs/exp10_revival/cpo/phase2_returns.parquet").to_pandas()
print("rows:", len(t))
print("unique model_id count:", t["model_id"].nunique())
print("unique model_ids:", sorted(t["model_id"].unique()))
print("config_id range:", t["config_id"].min(), "-", t["config_id"].max())
print("unique config_id count:", t["config_id"].nunique())
print("date range:", t["date"].min(), "-", t["date"].max())
print("unique dates:", t["date"].nunique())
print("non-zero daily_return frac:", round((t["daily_return"] != 0).mean(), 4))
print()
print("per-asset row counts:")
for asset in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE"]:
    sub = t[t["model_id"].str.startswith(asset + "_")]
    n_models = sub["model_id"].nunique()
    cmin = sub["config_id"].min() if len(sub) else None
    cmax = sub["config_id"].max() if len(sub) else None
    print(f"  {asset}: rows={len(sub):>10,d}  models={n_models}  configs={cmin}-{cmax}")

# Features parquet
f = pq.read_table("outputs/exp10_revival/cpo/phase2_features_funding.parquet").to_pandas()
print()
print("features rows:", len(f))
print("features cols:", list(f.columns))
print("features non-null frac per col:")
for c in f.columns:
    nn = f[c].notna().mean()
    print(f"  {c}: {round(nn, 4)}")
