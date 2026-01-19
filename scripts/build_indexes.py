import os
import time
import numpy as np

import hnswlib
import faiss

X_PATH = "data/X.npy"
OUT_HNSW = "artifacts/hnsw.index"
OUT_PQ = "artifacts/faiss_pq.index"
OUT_IVFPQ = "artifacts/faiss_ivfpq.index"

def build_hnsw(X: np.ndarray, M=16, ef_construction=200):
    dim = X.shape[1]
    p = hnswlib.Index(space="l2", dim=dim)
    p.init_index(max_elements=X.shape[0], ef_construction=ef_construction, M=M)
    p.add_items(X, np.arange(X.shape[0]))
    # default runtime quality/speed; can be changed at query time
    p.set_ef(50)
    return p

def build_faiss_pq(X: np.ndarray, m=4, nbits=8):
    # PQ requires dim % m == 0
    d = X.shape[1]
    assert d % m == 0, f"PQ requires dim % m == 0, got d={d}, m={m}"
    quantizer = faiss.IndexFlatL2(d)  # for training only
    index_pq = faiss.IndexPQ(d, m, nbits)
    index_pq.train(X)
    index_pq.add(X)
    return index_pq

def build_faiss_ivfpq(X: np.ndarray, nlist=100, m=4, nbits=8):
    d = X.shape[1]
    assert d % m == 0, f"IVFPQ requires dim % m == 0, got d={d}, m={m}"
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    index.train(X)
    index.add(X)
    # controls search speed/accuracy (higher = better recall, slower)
    index.nprobe = 10
    return index

def quick_test(index, X: np.ndarray, k=5, name="index"):
    # test a few queries for sanity + latency
    q_idx = np.random.choice(X.shape[0], size=5, replace=False)
    queries = X[q_idx]

    t0 = time.perf_counter()
    if isinstance(index, hnswlib.Index):
        labels, dists = index.knn_query(queries, k=k)
    else:
        dists, labels = index.search(queries, k)
    dt = (time.perf_counter() - t0) * 1000
    print(f"[{name}] sanity: labels shape={labels.shape}, latency_ms_for_5q={dt:.2f}")

def main():
    os.makedirs("artifacts", exist_ok=True)
    print(f"Loading vectors: {X_PATH}")
    X = np.load(X_PATH).astype(np.float32)
    print("X shape:", X.shape)

    # HNSW
    print("\nBuilding HNSW...")
    t0 = time.perf_counter()
    hnsw = build_hnsw(X, M=16, ef_construction=200)
    hnsw.save_index(OUT_HNSW)
    print(f"Saved {OUT_HNSW}  build_time_s={(time.perf_counter()-t0):.2f}")
    quick_test(hnsw, X, name="HNSW")

    # FAISS PQ
    print("\nBuilding FAISS PQ...")
    t0 = time.perf_counter()
    pq = build_faiss_pq(X, m=4, nbits=8)  # good default for 12D
    faiss.write_index(pq, OUT_PQ)
    print(f"Saved {OUT_PQ}  build_time_s={(time.perf_counter()-t0):.2f}")
    quick_test(pq, X, name="PQ")

    # FAISS IVF-PQ
    print("\nBuilding FAISS IVF-PQ...")
    t0 = time.perf_counter()
    ivfpq = build_faiss_ivfpq(X, nlist=100, m=4, nbits=8)
    faiss.write_index(ivfpq, OUT_IVFPQ)
    print(f"Saved {OUT_IVFPQ}  build_time_s={(time.perf_counter()-t0):.2f}")
    quick_test(ivfpq, X, name="IVF-PQ")

    print("\n All indexes built and saved.")

if __name__ == "__main__":
    main()
