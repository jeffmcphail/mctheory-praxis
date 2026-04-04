"""
scripts/download_dvol.py
Download Deribit DVOL historical volatility index for BTC and ETH.
Run from praxis root: python scripts/download_dvol.py
"""
import urllib.request, json
import pandas as pd
from pathlib import Path

Path("data/vol_cache").mkdir(parents=True, exist_ok=True)

for currency in ["BTC", "ETH"]:
    url = f"https://www.deribit.com/api/v2/public/get_historical_volatility?currency={currency}"
    print(f"Fetching {currency} DVOL...", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    result = data["result"]
    df = pd.DataFrame(result, columns=["timestamp_ms", "dvol"])
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("date")[["dvol"]]
    df["dvol"] = df["dvol"] / 100.0   # convert percent to decimal fraction
    out = f"data/vol_cache/dvol_{currency}_live.parquet"
    df.to_parquet(out)
    print(f"{len(df)} days | latest={df['dvol'].iloc[-1]*100:.1f}% | earliest={df.index[0].date()}")
    print(f"  Saved: {out}")

print("\nDone. Re-run phase2 to use real DVOL data.")
