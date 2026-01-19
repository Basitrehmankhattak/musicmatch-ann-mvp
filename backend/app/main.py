import time
from typing import List, Literal

import numpy as np
import pandas as pd
import joblib
import hnswlib
import faiss

from fastapi import FastAPI, Query
from pydantic import BaseModel

from app.bootstrap import ensure_assets
from app.paths import (
    ROOT_DIR,
    DATA_DIR,
    ART_DIR,
    X_PATH,
    META_PATH,
    FEATURES_PATH,
    SCALER_PATH,
    HNSW_PATH,
    PQ_PATH,
    IVFPQ_PATH,
)

Mode = Literal["hnsw", "pq", "ivfpq"]

app = FastAPI(title="MusicMatch ANN API", version="1.0.0")

# Global loaded resources (loaded once)
X: np.ndarray | None = None
meta: pd.DataFrame | None = None
name_norm: pd.Series | None = None

hnsw_index: hnswlib.Index | None = None
pq_index: faiss.Index | None = None
ivfpq_index: faiss.Index | None = None

features: List[str] | None = None
scaler = None


class SearchRequest(BaseModel):
    song_index: int
    k: int = 10
    mode: Mode = "hnsw"
    ef_search: int = 50   # only used for HNSW
    nprobe: int = 10      # only used for IVF-PQ


class SongHit(BaseModel):
    row_index: int
    id: str
    name: str
    artists: str
    release_date: str
    popularity: int
    duration_ms: int
    explicit: bool
    distance: float


class SearchResponse(BaseModel):
    query: SongHit
    results: List[SongHit]
    latency_ms: float
    mode: Mode


@app.on_event("startup")
def load_resources():
    """
    Startup sequence:
    1) Ensure assets exist (download+extract on Railway if missing)
    2) Load vectors, metadata, scaler, and ANN indexes
    """
    print("🚀 Startup: ensuring assets...")
    ensure_assets()
    print("✅ Startup: assets ensured. Loading resources...")

    # Debug prints (helpful on Railway). Keep for now; remove after stable.
    print("ROOT_DIR:", ROOT_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("ART_DIR:", ART_DIR)
    print("X_PATH:", X_PATH, "exists:", X_PATH.exists())
    print("META_PATH:", META_PATH, "exists:", META_PATH.exists())
    print("HNSW_PATH:", HNSW_PATH, "exists:", HNSW_PATH.exists())
    print("PQ_PATH:", PQ_PATH, "exists:", PQ_PATH.exists())
    print("IVFPQ_PATH:", IVFPQ_PATH, "exists:", IVFPQ_PATH.exists())

    global X, meta, name_norm, hnsw_index, pq_index, ivfpq_index, features, scaler

    # Load vectors and metadata
    X = np.load(X_PATH).astype(np.float32)
    meta = pd.read_parquet(META_PATH)

    # Normalize name for search
    name_norm = meta["name"].fillna("").astype(str).str.lower()

    # Load scaler (kept for future, not mandatory for query-by-index)
    scaler = joblib.load(SCALER_PATH)

    # Feature list for future explainability
    try:
        import json
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            features = json.load(f).get("feature_cols", [])
    except Exception:
        features = []

    # Load HNSW
    dim = int(X.shape[1])
    hnsw_index = hnswlib.Index(space="l2", dim=dim)
    hnsw_index.load_index(str(HNSW_PATH), max_elements=int(X.shape[0]))

    # Load FAISS
    pq_index = faiss.read_index(str(PQ_PATH))
    ivfpq_index = faiss.read_index(str(IVFPQ_PATH))

    app.state.ready = True
    print("✅ Startup complete. API ready.")


@app.get("/health")
def health():
    return {"status": "ok", "ready": getattr(app.state, "ready", False)}


@app.get("/songs")
def songs(q: str = Query(..., min_length=1), limit: int = 20):
    """
    Simple autocomplete: returns up to `limit` songs whose name contains query substring.
    """
    if meta is None or name_norm is None:
        return {"items": []}

    qn = q.strip().lower()
    if not qn:
        return {"items": []}

    mask = name_norm.str.contains(qn, na=False)
    idx = meta.loc[mask, "row_index"].head(limit).tolist()
    items = meta.loc[meta["row_index"].isin(idx), ["row_index", "name", "artists", "release_date"]].to_dict("records")
    return {"items": items}


def _row_to_hit(row: pd.Series, distance: float) -> SongHit:
    return SongHit(
        row_index=int(row["row_index"]),
        id=str(row["id"]),
        name=str(row["name"]),
        artists=str(row["artists"]),
        release_date=str(row["release_date"]),
        popularity=int(row["popularity"]),
        duration_ms=int(row["duration_ms"]),
        explicit=bool(row["explicit"]),
        distance=float(distance),
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if X is None or meta is None:
        raise ValueError("Server not ready yet")

    if req.song_index < 0 or req.song_index >= X.shape[0]:
        raise ValueError("song_index out of range")

    k = max(1, min(int(req.k), 50))
    query_vec = X[req.song_index].reshape(1, -1)

    t0 = time.perf_counter()

    if req.mode == "hnsw":
        if hnsw_index is None:
            raise ValueError("HNSW index not loaded")
        hnsw_index.set_ef(max(10, int(req.ef_search)))
        labels, dists = hnsw_index.knn_query(query_vec, k=k + 1)
        labels = labels[0].tolist()
        dists = dists[0].tolist()

    elif req.mode == "pq":
        if pq_index is None:
            raise ValueError("FAISS PQ index not loaded")
        dists, labels = pq_index.search(query_vec, k + 1)
        labels = labels[0].tolist()
        dists = dists[0].tolist()

    else:  # ivfpq
        if ivfpq_index is None:
            raise ValueError("FAISS IVFPQ index not loaded")
        ivfpq_index.nprobe = max(1, int(req.nprobe))
        dists, labels = ivfpq_index.search(query_vec, k + 1)
        labels = labels[0].tolist()
        dists = dists[0].tolist()

    # Remove the query itself if present
    filtered = [(lab, dist) for lab, dist in zip(labels, dists) if lab != req.song_index]
    filtered = filtered[:k]

    latency_ms = (time.perf_counter() - t0) * 1000

    # Query hit
    qrow = meta.iloc[req.song_index]
    query_hit = _row_to_hit(qrow, distance=0.0)

    # Result hits
    hits: List[SongHit] = []
    for lab, dist in filtered:
        row = meta.iloc[int(lab)]
        hits.append(_row_to_hit(row, distance=float(dist)))

    return SearchResponse(
        query=query_hit,
        results=hits,
        latency_ms=float(latency_ms),
        mode=req.mode,
    )
@app.get("/")
def root():
    return {"message": "MusicMatch ANN API is running"}
