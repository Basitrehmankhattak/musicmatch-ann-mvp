import os
import tarfile
import urllib.request
from pathlib import Path

from app.paths import (
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

REQUIRED_FILES = [
    X_PATH,
    META_PATH,
    FEATURES_PATH,
    SCALER_PATH,
    HNSW_PATH,
    PQ_PATH,
    IVFPQ_PATH,
]


def _assets_present() -> bool:
    return all(p.exists() for p in REQUIRED_FILES)


def _pick_extracted_base(extract_dir: Path) -> Path:
    """
    Support multiple tar structures:

    Case A (flat):
      extract_dir/X.npy

    Case B (nested):
      extract_dir/<one-folder>/X.npy
      extract_dir/deploy_assets/X.npy

    We choose:
    - extract_dir if X.npy exists there
    - else, the first subdir that contains X.npy
    """
    if (extract_dir / "X.npy").exists():
        return extract_dir

    # Look for a subfolder that contains X.npy
    for child in extract_dir.iterdir():
        if child.is_dir() and (child / "X.npy").exists():
            return child

    # If not found, we will fail later with a clear error
    return extract_dir


def ensure_assets():
    print("🚀 Bootstrap: ensuring assets...")
    print("DATA_DIR:", DATA_DIR)
    print("ART_DIR:", ART_DIR)

    # Ensure folders exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    if _assets_present():
        print("✅ Assets already present. Skipping download.")
        return

    asset_url = os.environ.get("ASSET_URL")
    if not asset_url:
        missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
        raise RuntimeError(
            "ASSET_URL not set and assets missing:\n" + "\n".join(missing)
        )

    tmp_dir = Path("/tmp/musicmatch_assets")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tar_path = tmp_dir / "musicmatch_assets.tar.gz"
    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("⬇️ Downloading assets from:", asset_url)
    urllib.request.urlretrieve(asset_url, tar_path)

    print("📦 Extracting assets to:", extract_dir)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    base = _pick_extracted_base(extract_dir)
    print("📁 Detected extracted base:", base)

    # Move/copy files into their final locations
    def copy_file(src: Path, dst: Path):
        if not src.exists():
            raise RuntimeError(f"Missing expected file in tar: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        # copy2 preserves metadata; also works across filesystems
        import shutil
        shutil.copy2(src, dst)

    copy_file(base / "X.npy", X_PATH)
    copy_file(base / "meta.parquet", META_PATH)
    copy_file(base / "features.json", FEATURES_PATH)
    copy_file(base / "scaler.pkl", SCALER_PATH)
    copy_file(base / "hnsw.index", HNSW_PATH)
    copy_file(base / "faiss_pq.index", PQ_PATH)
    copy_file(base / "faiss_ivfpq.index", IVFPQ_PATH)

    if not _assets_present():
        missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
        raise RuntimeError("❌ Assets still missing after extraction:\n" + "\n".join(missing))

    print("✅ Assets ready.")
