"""
Dental OPG Dataset Preparation — DENTEX (MICCAI 2023)
Loads real COCO-style annotations and creates image-level multi-label targets.
No labels are fabricated; every record comes from ground-truth bounding-box annotations.
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH FOR DISEASE CLASSES
# Matches DENTEX categories_3 exactly. Periodontitis is NOT in DENTEX.
# ---------------------------------------------------------------------------
DISEASE_CLASSES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
# Map DENTEX category_id_3 → index in DISEASE_CLASSES
DENTEX_CAT_TO_IDX = {1: 0, 3: 1, 2: 2, 0: 3}

DATA_DIR = Path("data/dentex")
TRAIN_ANNOTATION_FILE = DATA_DIR / "train_quadrant_enumeration_disease.json"
VALIDATION_IMAGE_DIR = DATA_DIR / "validation" / "validation_data" / "quadrant_enumeration_disease" / "xrays"
VALIDATION_ANNOTATION_FILE = DATA_DIR / "validation_triple.json"
TRAINING_IMAGE_DIR = DATA_DIR / "training" / "training_data" / "quadrant-enumeration-disease" / "xrays"
SPLITS_FILE = Path("data/splits.json")


def load_coco_annotations(annotation_path: Path) -> dict:
    """Load a COCO-format JSON file. Raises FileNotFoundError if missing."""
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}\n"
            "Run scripts/download_dataset.py first."
        )
    with open(annotation_path) as f:
        return json.load(f)


def build_image_level_labels(coco_data: dict) -> list:
    """
    Convert COCO bounding-box annotations to per-image multi-label targets.
    Returns a list of records with keys: filename, image_id, labels (list of 0/1).
    No labels are invented; only real annotations from DENTEX are used.
    """
    image_diseases = defaultdict(set)
    for ann in coco_data["annotations"]:
        cat_id = ann["category_id_3"]
        idx = DENTEX_CAT_TO_IDX.get(cat_id)
        if idx is not None:
            image_diseases[ann["image_id"]].add(idx)

    records = []
    for img_info in coco_data["images"]:
        img_id = img_info["id"]
        fname = img_info["file_name"]
        disease_indices = image_diseases.get(img_id, set())
        labels = [0] * len(DISEASE_CLASSES)
        for idx in disease_indices:
            labels[idx] = 1
        records.append({
            "image_id": img_id,
            "filename": fname,
            "labels": labels,
            "is_healthy": (sum(labels) == 0),
        })
    return records


def make_splits(records: list, val_ratio: float = 0.20, seed: int = 42) -> dict:
    """
    Stratified random split of records into train / val subsets.
    Uses a fixed seed for reproducibility.
    Returns {"train": [...], "val": [...]}
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for rec in records:
        key = tuple(rec["labels"])
        strata[key].append(rec)

    train_recs, val_recs = [], []
    for group in strata.values():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio))
        val_recs.extend(group[:n_val])
        train_recs.extend(group[n_val:])

    rng.shuffle(train_recs)
    rng.shuffle(val_recs)
    return {"train": train_recs, "val": val_recs}


def setup_dataset() -> dict:
    """
    Load annotations, build image-level labels, create splits, persist to data/splits.json.
    Returns the splits dict {"train": [...], "val": [...], "test": [...]}.
    Raises FileNotFoundError if annotations are missing.
    """
    coco_train = load_coco_annotations(TRAIN_ANNOTATION_FILE)
    train_records = build_image_level_labels(coco_train)
    splits = make_splits(train_records, val_ratio=0.20, seed=42)

    # Official validation set (50 images with ground truth) used as test
    val_ann = DATA_DIR / "validation_triple.json"
    if val_ann.exists():
        with open(val_ann) as f:
            coco_val = json.load(f)
        splits["test"] = build_image_level_labels(coco_val)
    else:
        splits["test"] = []

    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    _print_split_summary(splits)
    return splits


def load_splits() -> dict:
    """Load persisted splits from data/splits.json."""
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"Splits file not found: {SPLITS_FILE}\n"
            "Run scripts/download_dataset.py first."
        )
    with open(SPLITS_FILE) as f:
        return json.load(f)


def _print_split_summary(splits: dict):
    n_classes = len(DISEASE_CLASSES)
    for split_name, recs in splits.items():
        if not recs:
            print(f"[Dataset] {split_name:5s}: 0 images (unavailable)")
            continue
        counts = [sum(r["labels"][i] for r in recs) for i in range(n_classes)]
        total = len(recs)
        print(f"[Dataset] {split_name:5s}: {total} images", end="")
        for cls, cnt in zip(DISEASE_CLASSES, counts):
            print(f" | {cls}={cnt}", end="")
        print()


def get_image_path(filename: str, split: str = "train") -> Path:
    """Return absolute path to a radiograph PNG for the given split."""
    if split == "test":
        return VALIDATION_IMAGE_DIR / filename
    return TRAINING_IMAGE_DIR / filename
