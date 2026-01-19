import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

INPUT_CSV = "data/spotify.csv"
OUT_X = "data/X.npy"
OUT_META = "data/meta.parquet"
OUT_SCALER = "artifacts/scaler.pkl"
OUT_FEATURES = "artifacts/features.json"

# Vector features used for similarity search
FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "key", "mode", "time_signature"
]

# Metadata kept for UI display + lookup
META_COLS = [
    "id", "name", "artists", "release_date", "popularity", "duration_ms", "explicit"
]

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # Basic sanity checks
    missing = [c for c in FEATURE_COLS + META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Keep only needed columns (meta + features)
    df = df[META_COLS + FEATURE_COLS].copy()

    # Convert types safely
    # explicit may be True/False; keep as bool in meta
    df["explicit"] = df["explicit"].astype(bool)

    # Ensure numeric for features
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Handle missing feature values: fill with median per column
    for c in FEATURE_COLS:
        med = df[c].median()
        df[c] = df[c].fillna(med)

    # Build vectors
    X_raw = df[FEATURE_COLS].values.astype(np.float32)

    # Normalize features to [0,1] so tempo/loudness don't dominate
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    # Save scaler + features list
    joblib.dump(scaler, OUT_SCALER)
    with open(OUT_FEATURES, "w", encoding="utf-8") as f:
        json.dump({"feature_cols": FEATURE_COLS}, f, indent=2)

    # Save vectors
    np.save(OUT_X, X)

    # Save metadata (include row_index for lookup)
    meta = df[META_COLS].copy()
    meta.insert(0, "row_index", np.arange(len(meta), dtype=np.int32))

    # Optional: store original (unscaled) features for explainability in UI
    for c in FEATURE_COLS:
        meta[c] = X_raw[:, FEATURE_COLS.index(c)]

    meta.to_parquet(OUT_META, index=False)

    print(" Preprocessing complete.")
    print(f"Saved: {OUT_X}  shape={X.shape} dtype={X.dtype}")
    print(f"Saved: {OUT_META} rows={len(meta)} cols={len(meta.columns)}")
    print(f"Saved: {OUT_SCALER}")
    print(f"Saved: {OUT_FEATURES}")

if __name__ == "__main__":
    main()
