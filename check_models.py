"""Quick diagnostic: check all model files."""
import joblib
import os

model_dir = "output/funding_rate/cpo/"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]

for fname in sorted(model_files):
    path = os.path.join(model_dir, fname)
    m = joblib.load(path)
    print(f"\n{fname}:")
    if isinstance(m, dict):
        for k, v in m.items():
            if isinstance(v, dict):
                model = v.get("model")
                error = v.get("error", "none")
                mtype = type(model).__name__
                nfeat = getattr(model, "n_features_in_", "?") if model else "N/A"
                br = v.get("base_rate", "?")
                print(f"  {k}: model={mtype}, n_features={nfeat}, base_rate={br}, error={error}")
            else:
                print(f"  {k}: {type(v).__name__}")
    else:
        print(f"  type={type(m).__name__}")
