import os
import tarfile
from pathlib import Path
import urllib.request

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ART_DIR = ROOT_DIR / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

ASSET_FILE = ROOT_DIR / "musicmatch_assets.tar.gz"

def ensure_assets():
    asset_url = os.environ.get("ASSET_URL")

    # If assets already exist, do nothing
    if (DATA_DIR / "X.npy").exists() and (ART_DIR / "hnsw.index").exists():
        print("Assets already present. Skipping download.")
        return

    if not asset_url:
        raise RuntimeError("ASSET_URL environment variable not set")

    print(f"Downloading assets from {asset_url} ...")
    urllib.request.urlretrieve(asset_url, ASSET_FILE)

    print("Extracting assets...")
    with tarfile.open(ASSET_FILE, "r:gz") as tar:
        tar.extractall(ROOT_DIR)

    print("Assets ready.")
