"""
FastAPI Backend Application for Dental OPG Disease Detection (DENTEX Benchmark)
Detects real DENTEX classes:
1. Caries
2. Deep Caries
3. Periapical Lesion
4. Impacted
Provides authentic model inference, Grad-CAM overlays, and evaluation metrics.
No fabricated labels, simulated probabilities, or mock GPU claims.
"""

import os
import sys
import io
import time
import json
import threading
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from backend.model import DentalOPGNet, DISEASE_CLASSES, DISEASE_METADATA, load_trained_model
from backend.utils import preprocess_opg, generate_overlay_base64, image_to_base64, build_clinical_report
from backend.inference import run_inference, run_all_gradcams, composite_heatmap, overlay_heatmap

app = FastAPI(
    title="DentalVision AI - Panoramic Radiography Analysis",
    description="Multi-label Dental Radiography Classifier trained on the DENTEX Benchmark (Hamamci et al., MICCAI 2023).",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "samples").mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
model_instance = None
model_lock = threading.Lock()

def get_or_load_model():
    """Lazily load model weights. Returns None if weights file is missing."""
    global model_instance
    with model_lock:
        if model_instance is None and TORCH_AVAILABLE:
            weights_file = MODELS_DIR / "dental_model.pth"
            if weights_file.exists():
                model_instance = load_trained_model(
                    weights_path=str(weights_file),
                    device=device
                )
        return model_instance

def get_system_info():
    """Return honest hardware and environment information."""
    cuda_avail = TORCH_AVAILABLE and torch.cuda.is_available()
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {
            "device": "cuda",
            "name": gpu_name,
            "vram_gb": round(total_vram, 2),
            "pytorch_version": torch.__version__
        }
    return {
        "device": "cpu",
        "name": "CPU Mode",
        "pytorch_version": torch.__version__ if TORCH_AVAILABLE else "Not installed"
    }

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>DentalVision AI Backend Ready</h1><p>API docs available at <a href='/docs'>/docs</a>.</p>")

@app.get("/api/model-info")
async def get_model_info():
    """Return model configuration and evaluation metrics."""
    metrics_file = MODELS_DIR / "metrics.json"
    metrics = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass

    weights_file = MODELS_DIR / "dental_model.pth"
    model_ready = weights_file.exists() and (get_or_load_model() is not None)

    return {
        "status": "ready" if model_ready else "not_trained",
        "model_architecture": "DentalOPGNet (ResNet50 Backbone + Multi-Label Classification Head)",
        "input_resolution": "1024 x 512 (Standard Panoramic 2:1 Aspect Ratio)",
        "dataset": "DENTEX (MICCAI 2023 Challenge)",
        "disease_classes": [
            {
                "name": meta["name"],
                "category": meta["category"],
                "hallmarks": meta["hallmarks"]
            }
            for meta in DISEASE_METADATA.values()
        ],
        "hardware": get_system_info(),
        "metrics": metrics,
        "disclaimer": "Research prototype. Not a diagnostic device. Findings must be reviewed by a qualified dentist."
    }

@app.get("/api/samples")
async def list_sample_opgs():
    """List sample panoramic radiographs from static/samples."""
    sample_files = sorted(list((STATIC_DIR / "samples").glob("*.png")) + list((STATIC_DIR / "samples").glob("*.jpg")))
    samples = []
    for idx, s_path in enumerate(sample_files):
        samples.append({
            "id": idx + 1,
            "filename": s_path.name,
            "url": f"/static/samples/{s_path.name}"
        })
    return samples

@app.post("/api/predict")
async def predict_dental_opg(file: UploadFile = File(...), enhance: bool = True):
    """
    Inference endpoint:
    Accepts panoramic radiograph, runs ResNet50 model, computes Grad-CAM.
    Returns HTTP 503 if model weights are not trained yet.
    """
    model = get_or_load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model weights not found. Run 'python -m backend.train' to train the model first."
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image uploaded.")

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    t0 = time.time()
    try:
        res = run_inference(model, pil_img, device=device)
        probs = res["probabilities"]
        cams = run_all_gradcams(model, pil_img, device=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    elapsed_ms = (time.time() - t0) * 1000.0

    # Build Grad-CAM overlays
    np_img = np.array(pil_img.resize((1024, 512), Image.Resampling.BILINEAR))
    if enhance:
        from backend.utils import enhance_radiograph_clahe
        np_img = enhance_radiograph_clahe(np_img)

    composite_cam = composite_heatmap(cams, probs)
    overlay_composite_b64 = generate_overlay_base64(np_img, composite_cam)

    per_class_cams = {}
    for cls_name, cam_arr in cams.items():
        per_class_cams[cls_name] = generate_overlay_base64(np_img, cam_arr)

    original_b64 = image_to_base64(np_img)
    sys_info = get_system_info()
    report = build_clinical_report(probs, elapsed_ms=elapsed_ms, device_name=sys_info["name"])

    return {
        "success": True,
        "filename": file.filename,
        "report": report,
        "visualizations": {
            "original_image": original_b64,
            "composite_heatmap": overlay_composite_b64,
            "per_class_heatmaps": per_class_cams
        }
    }

@app.get("/api/training-status")
async def get_training_status():
    """Return genuine training history from models/training_history.json."""
    history_file = MODELS_DIR / "training_history.json"
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
            return {
                "trained": True,
                "history": history,
                "hardware": get_system_info()
            }
        except Exception:
            pass

    return {
        "trained": False,
        "message": "Model has not been trained yet. Run 'python -m backend.train' to begin training.",
        "hardware": get_system_info()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
