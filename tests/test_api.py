"""
Unit tests for FastAPI endpoints in backend.app.
Ensures honest HTTP 503 response when model is not trained (no mock simulation).
"""

import pytest
from starlette.testclient import TestClient
from backend.app import app
from backend.model import DISEASE_CLASSES

client = TestClient(app)

def test_model_info_endpoint():
    """GET /api/model-info should return honest architecture and 4 DENTEX classes."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "disease_classes" in data
    returned_classes = [c["name"] for c in data["disease_classes"]]
    assert returned_classes == DISEASE_CLASSES
    assert "Periodontitis" not in returned_classes
    assert "disclaimer" in data

def test_predict_endpoint_unweighted_returns_503():
    """POST /api/predict must return HTTP 503 if weights are not trained."""
    import backend.app
    backend.app.model_instance = None  # Ensure no cached model
    
    # Upload dummy image bytes
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), color=(100, 100, 100)).save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/api/predict", files={"file": ("test.png", buf.getvalue(), "image/png")})
    # If dental_model.pth does not exist, status must be 503
    weights_path = backend.app.MODELS_DIR / "dental_model.pth"
    if not weights_path.exists():
        assert response.status_code == 503
        assert "not found" in response.json()["detail"].lower()
    else:
        # If weights exist, status should be 200
        assert response.status_code == 200

def test_training_status_endpoint():
    """GET /api/training-status returns genuine training status."""
    response = client.get("/api/training-status")
    assert response.status_code == 200
    data = response.json()
    assert "hardware" in data
