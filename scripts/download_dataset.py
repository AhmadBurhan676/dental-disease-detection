"""
Download and prepare the DENTEX dataset from Hugging Face.
Extracts:
  1. train_quadrant_enumeration_disease.json (annotations)
  2. validation_triple.json (test ground truth annotations)
  3. 705 training images via direct range-streaming
  4. 50 validation/test images

Usage:
    python scripts/download_dataset.py
"""

import io
import json
import os
import struct
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/dentex")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_XRAY_DIR = DATA_DIR / "training" / "training_data" / "quadrant-enumeration-disease" / "xrays"
VAL_XRAY_DIR = DATA_DIR / "validation" / "validation_data" / "quadrant_enumeration_disease" / "xrays"
TRAIN_XRAY_DIR.mkdir(parents=True, exist_ok=True)
VAL_XRAY_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_ZIP_URL = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/training_data.zip"
VAL_ZIP_URL = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/validation_data.zip"
VAL_JSON_URL = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/raw/main/DENTEX/validation_triple.json"

def download_file(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[Skip] {dest} already exists.")
        return
    print(f"[Download] {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))

def extract_validation_data():
    val_images = list(VAL_XRAY_DIR.glob("*.png"))
    if len(val_images) >= 50:
        print(f"[Skip] Validation xrays already extracted ({len(val_images)} images).")
        return
    val_zip = DATA_DIR / "validation_data.zip"
    download_file(VAL_ZIP_URL, val_zip)
    print(f"[Extract] Extracting validation xrays from {val_zip}...")
    with zipfile.ZipFile(val_zip) as zf:
        for member in zf.namelist():
            if "quadrant_enumeration_disease/xrays" in member and member.endswith(".png"):
                bname = Path(member).name
                (VAL_XRAY_DIR / bname).write_bytes(zf.read(member))
    print(f"[Done] Validation xrays extracted.")

def main():
    # 1. Validation JSON
    val_json_path = DATA_DIR / "validation_triple.json"
    download_file(VAL_JSON_URL, val_json_path)

    # 2. Validation images
    extract_validation_data()

    # 3. Training images check
    train_images = list(TRAIN_XRAY_DIR.glob("*.png"))
    print(f"[Status] Training disease images present: {len(train_images)} / 705")
    if len(train_images) < 705:
        print("[Info] To stream-extract remaining training images, run:")
        print("curl -r 7063293442-9210418631 -L " + TRAIN_ZIP_URL + " | python3 scripts/extract_stream.py")

    # 4. Build splits
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.dataset_prep import setup_dataset
    try:
        splits = setup_dataset()
        print(f"\n[Splits Ready] Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    except Exception as e:
        print(f"[Notice] Splits build: {e}")

if __name__ == "__main__":
    main()
