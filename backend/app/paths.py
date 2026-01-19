from pathlib import Path

# /backend/app/paths.py
APP_DIR = Path(__file__).resolve().parent      # .../backend/app
ROOT_DIR = APP_DIR.parent                      # .../backend

DATA_DIR = ROOT_DIR / "data"
ART_DIR  = ROOT_DIR / "artifacts"

X_PATH        = DATA_DIR / "X.npy"
META_PATH     = DATA_DIR / "meta.parquet"

FEATURES_PATH = ART_DIR / "features.json"
SCALER_PATH   = ART_DIR / "scaler.pkl"

HNSW_PATH      = ART_DIR / "hnsw.index"
PQ_PATH        = ART_DIR / "faiss_pq.index"
IVFPQ_PATH     = ART_DIR / "faiss_ivfpq.index"
