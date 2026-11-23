"""
Unit tests for DentalOPGNet deep learning model and Grad-CAM hooks.
"""

import pytest
import numpy as np
import torch
from backend.model import DentalOPGNet, DISEASE_CLASSES, NUM_CLASSES, load_trained_model

def test_model_initialization():
    """Verify DentalOPGNet can be initialized with 4 classes without pretrained weights."""
    model = DentalOPGNet(num_classes=NUM_CLASSES, pretrained=False)
    assert model is not None
    assert len(DISEASE_CLASSES) == 4

def test_model_forward_pass():
    """Test forward pass with batch size 2 and standard panoramic dimensions."""
    model = DentalOPGNet(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 512, 1024)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (2, NUM_CLASSES)
    assert not torch.isnan(logits).any()

def test_gradcam_generation():
    """Test that real Grad-CAM generates valid heatmap without crashing."""
    model = DentalOPGNet(num_classes=NUM_CLASSES, pretrained=False)
    dummy_tensor = torch.randn(1, 3, 512, 1024, requires_grad=True)

    cam = model.generate_gradcam(dummy_tensor, target_class_idx=0)
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (512, 1024)
    assert not np.isnan(cam).any()
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-5

def test_load_trained_model_missing_file():
    """load_trained_model should return None if file does not exist (never silent fake)."""
    model = load_trained_model(weights_path="models/non_existent_weights.pth")
    assert model is None
