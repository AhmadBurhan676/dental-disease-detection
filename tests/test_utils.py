"""
Unit tests for backend.utils module.
"""

import numpy as np
import pytest
from backend.utils import enhance_radiograph_clahe, build_clinical_report
from backend.model import DISEASE_CLASSES

def test_clahe_enhancement():
    """Test CLAHE output preserves shape and dtype."""
    img = np.random.randint(50, 200, (512, 1024, 3), dtype=np.uint8)
    enhanced = enhance_radiograph_clahe(img)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8

def test_build_clinical_report():
    """Test clinical report builder with 4 class probabilities."""
    probs = [0.85, 0.70, 0.20, 0.10]
    report = build_clinical_report(probs, elapsed_ms=15.0, device_name="CPU Mode")
    assert "overall_health" in report
    assert "diseases" in report
    assert len(report["diseases"]) == len(DISEASE_CLASSES)
    assert "disclaimer" in report
    
    # Check first disease (Caries) is marked DETECTED
    assert report["diseases"][0]["badge"] == "DETECTED"
    # Check 4th disease (Impacted) is marked CLEAR
    assert report["diseases"][3]["badge"] == "CLEAR"
