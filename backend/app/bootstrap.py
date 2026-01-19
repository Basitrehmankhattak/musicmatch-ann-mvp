import os
import tarfile
import shutil
from pathlib import Path
import urllib.request

# In Railway container: this file is /app/app/bootstrap.py so parents[1] = /app
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ART_DIR = ROOT_DIR / "artifacts"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

ASSET_FILE = ROOT_DIR / "musicmatch_assets.tar.gz"
EXTRACT_DIR = ROOT_DIR / "deploy_assets"  # created by your tar

def ensure_assets():
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"ART_DIR: {ART_DIR}")
    asset_url = os.environ.get("ASSET_URL")

    # If already present, skip
    if (DATA_DIR / "X.npy").exists() and (DATA_DIR / "meta.parquet").exists() and (ART_DIR / "hnsw.index").exists():
        print("Assets already present. Skipping download.")
        return

    if not asset_url:
        raise RuntimeError("ASSET_URL environment variable not set")

    print(f"Downloading assets from {asset_url} ...")
    urllib.request.urlretrieve(asset_url, ASSET_FILE)

    print("Extracting assets...")
    with tarfile.open(ASSET_FILE, "r:gz") as tar:
        tar.extractall(ROOT_DIR)

    # If files extracted into deploy_assets/, move them into expected locations
    if EXTRACT_DIR.exists():
        print("Listing extracted files:", [p.name for p in EXTRACT_DIR.iterdir()])
        print("Moving extracted files into /data and /artifacts ...")
        for f in EXTRACT_DIR.iterdir():
            if f.name in ("X.npy", "meta.parquet"):
                shutil.copy2(f, DATA_DIR / f.name)
            elif f.name.endswith(".index") or f.name in ("scaler.pkl", "features.json"):
                shutil.copy2(f, ART_DIR / f.name)

        print("Cleaning up extracted folder...")
        # optional cleanup
        # shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    # Final check
    missing = []
    for need in [DATA_DIR/"X.npy", DATA_DIR/"meta.parquet", ART_DIR/"hnsw.index", ART_DIR/"faiss_pq.index", ART_DIR/"faiss_ivfpq.index"]:
        if not need.exists():
            missing.append(str(need))
    if missing:
        raise RuntimeError(f"Assets still missing after extraction: {missing}")

    print("✅ Assets ready.")
