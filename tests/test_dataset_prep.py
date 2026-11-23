"""
Unit tests for backend.dataset_prep module.
Tests COCO label parsing, multi-label binarization, stratified split logic, and leakage prevention.
"""

import json
from pathlib import Path
import pytest
from backend.dataset_prep import (
    DISEASE_CLASSES,
    DENTEX_CAT_TO_IDX,
    build_image_level_labels,
    make_splits,
    load_splits,
    SPLITS_FILE
)

def test_disease_classes():
    """Verify single source of truth for disease classes matches DENTEX."""
    assert DISEASE_CLASSES == ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
    assert len(DISEASE_CLASSES) == 4
    # Ensure Periodontitis is NOT included (honest dataset reflection)
    assert "Periodontitis" not in DISEASE_CLASSES

def test_dentex_category_mapping():
    """Verify DENTEX category_id_3 IDs map to correct indexes."""
    # 0: Impacted -> idx 3
    # 1: Caries -> idx 0
    # 2: Periapical Lesion -> idx 2
    # 3: Deep Caries -> idx 1
    assert DENTEX_CAT_TO_IDX[1] == 0
    assert DENTEX_CAT_TO_IDX[3] == 1
    assert DENTEX_CAT_TO_IDX[2] == 2
    assert DENTEX_CAT_TO_IDX[0] == 3

def test_build_image_level_labels_synthetic():
    """Test COCO annotation to multi-label encoding without external files."""
    mock_coco = {
        "images": [
            {"id": 1, "file_name": "opg_1.png"},
            {"id": 2, "file_name": "opg_2.png"},
            {"id": 3, "file_name": "opg_3.png"},
        ],
        "annotations": [
            {"image_id": 1, "category_id_3": 1},  # Caries
            {"image_id": 1, "category_id_3": 3},  # Deep Caries
            {"image_id": 2, "category_id_3": 2},  # Periapical
            # Image 3 has no annotations (healthy)
        ]
    }
    records = build_image_level_labels(mock_coco)
    assert len(records) == 3

    # Image 1: Caries (idx 0) and Deep Caries (idx 1)
    rec1 = next(r for r in records if r["image_id"] == 1)
    assert rec1["labels"] == [1, 1, 0, 0]
    assert rec1["is_healthy"] is False

    # Image 2: Periapical Lesion (idx 2)
    rec2 = next(r for r in records if r["image_id"] == 2)
    assert rec2["labels"] == [0, 0, 1, 0]
    assert rec2["is_healthy"] is False

    # Image 3: Healthy (all 0)
    rec3 = next(r for r in records if r["image_id"] == 3)
    assert rec3["labels"] == [0, 0, 0, 0]
    assert rec3["is_healthy"] is True

def test_make_splits_no_leakage():
    """Test that make_splits creates mutually exclusive subsets."""
    dummy_records = [
        {"image_id": i, "filename": f"img_{i}.png", "labels": [i % 2, (i + 1) % 2, 0, 0], "is_healthy": False}
        for i in range(100)
    ]
    splits = make_splits(dummy_records, val_ratio=0.20, seed=42)
    train_ids = set(r["image_id"] for r in splits["train"])
    val_ids = set(r["image_id"] for r in splits["val"])

    # Disjoint check
    overlap = train_ids.intersection(val_ids)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping IDs: {overlap}"
    assert len(train_ids) + len(val_ids) == 100

def test_persisted_splits_integrity():
    """Verify existing data/splits.json if present has zero leakage between train, val, and test."""
    if not SPLITS_FILE.exists():
        pytest.skip("data/splits.json not yet created")

    splits = load_splits()
    assert "train" in splits and "val" in splits and "test" in splits

    train_files = set(r["filename"] for r in splits["train"])
    val_files = set(r["filename"] for r in splits["val"])
    test_files = set(r["filename"] for r in splits["test"])

    # Zero leakage check
    assert len(train_files.intersection(val_files)) == 0, "Leakage between train and val!"
    assert len(train_files.intersection(test_files)) == 0, "Leakage between train and test!"
    assert len(val_files.intersection(test_files)) == 0, "Leakage between val and test!"
