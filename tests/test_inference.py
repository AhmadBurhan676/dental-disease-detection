"""
Unit tests for backend.inference module.
Ensures preprocessing, Grad-CAM overlays, and honest failure modes work as expected.
"""

import pytest
import numpy as np
from PIL import Image
import torch
from backend.inference import (
    preprocess_image,
    run_inference,
    run_gradcam,
    composite_heatmap,
    overlay_heatmap,
    INPUT_H,
    INPUT_W
)
from backend.model import DentalOPGNet, NUM_CLASSES

def test_preprocess_image():
    """Test preprocessing converts PIL image to normalized tensor of shape (1, 3, 512, 1024)."""
    img = Image.new("RGB", (800, 400), color=(100, 150, 200))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, INPUT_H, INPUT_W)
    assert isinstance(tensor, torch.Tensor)

def test_run_inference_raises_without_model():
    """Inference must fail explicitly if model is None."""
    img = Image.new("RGB", (400, 200), color=(50, 50, 50))
    with pytest.raises(RuntimeError, match="Model weights not found"):
        run_inference(None, img)

def test_run_gradcam_raises_without_model():
    """Grad-CAM must fail explicitly if model is None."""
    img = Image.new("RGB", (400, 200), color=(50, 50, 50))
    with pytest.raises(RuntimeError, match="Model weights not found"):
        run_gradcam(None, img, class_idx=0)

def test_run_inference_with_model():
    """Test inference with unweighted model returns probabilities between 0 and 1."""
    model = DentalOPGNet(num_classes=NUM_CLASSES, pretrained=False)
    img = Image.new("RGB", (600, 300), color=(128, 128, 128))
    res = run_inference(model, img, device="cpu")
    assert "probabilities" in res
    assert len(res["probabilities"]) == NUM_CLASSES
    for p in res["probabilities"]:
        assert 0.0 <= p <= 1.0

def test_composite_heatmap():
    """Test composite weighted average of heatmaps."""
    cams = {
        "Caries": np.ones((512, 1024), dtype=np.float32) * 0.2,
        "Deep Caries": np.ones((512, 1024), dtype=np.float32) * 0.4,
        "Periapical Lesion": np.ones((512, 1024), dtype=np.float32) * 0.6,
        "Impacted": np.ones((512, 1024), dtype=np.float32) * 0.8,
    }
    probs = [0.1, 0.2, 0.3, 0.4]
    comp = composite_heatmap(cams, probs)
    assert comp.shape == (512, 1024)
    assert not np.isnan(comp).any()

def test_overlay_heatmap():
    """Test overlay blends with RGB image."""
    rgb = np.zeros((512, 1024, 3), dtype=np.uint8)
    cam = np.ones((512, 1024), dtype=np.float32) * 0.5
    overlay = overlay_heatmap(rgb, cam, alpha=0.4)
    assert overlay.shape == (512, 1024, 3)
    assert overlay.dtype == np.uint8
