"""
Shared inference module — single source of truth for preprocessing, prediction, and Grad-CAM.
Used by both streamlit_app.py and backend/app.py.
Never fabricates predictions. Returns None / raises if model is unavailable.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from backend.model import DISEASE_CLASSES, DentalOPGNet, load_trained_model

# Input resolution — panoramic aspect ratio preserved
INPUT_H = 512
INPUT_W = 1024

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_preprocess = None


def _get_transform():
    global _preprocess
    if _preprocess is None and TORCH_AVAILABLE:
        _preprocess = T.Compose([
            T.Resize((INPUT_H, INPUT_W)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return _preprocess


def preprocess_image(pil_image: Image.Image) -> "torch.Tensor":
    """Convert a PIL Image to a (1, 3, H, W) normalised tensor."""
    img_rgb = pil_image.convert("RGB")
    transform = _get_transform()
    return transform(img_rgb).unsqueeze(0)


def run_inference(model, pil_image: Image.Image, device: str = "cpu") -> dict:
    """
    Run the model on a PIL image.
    Returns:
        {
          "probabilities": [float, ...],      # sigmoid scores per class
          "classes": [str, ...],              # DISEASE_CLASSES
        }
    Raises RuntimeError if model is None (weights not found).
    """
    if model is None:
        raise RuntimeError(
            "Model weights not found. Run 'python -m backend.train' first "
            "to train the model and save models/dental_model.pth."
        )
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")

    tensor = preprocess_image(pil_image).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()

    return {
        "probabilities": probs,
        "classes": DISEASE_CLASSES,
    }


def run_gradcam(model, pil_image: Image.Image, class_idx: int, device: str = "cpu") -> np.ndarray:
    """
    Compute real Grad-CAM heatmap for the given class index.
    Returns a float32 array (H, W) in [0, 1].
    Raises RuntimeError if model is None.
    """
    if model is None:
        raise RuntimeError("Model weights not found — cannot compute Grad-CAM.")
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")

    tensor = preprocess_image(pil_image).to(device)
    tensor.requires_grad_(True)
    cam = model.generate_gradcam(tensor, class_idx)
    return cam


def run_all_gradcams(model, pil_image: Image.Image, device: str = "cpu") -> dict[str, np.ndarray]:
    """
    Compute Grad-CAM for every disease class.
    Returns {class_name: heatmap_array}.
    """
    result = {}
    for idx, cls_name in enumerate(DISEASE_CLASSES):
        result[cls_name] = run_gradcam(model, pil_image, idx, device)
    return result


def overlay_heatmap(np_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM heatmap over an RGB numpy image."""
    heatmap = np.uint8(255 * cam)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    # Resize to match image if needed
    if heatmap_rgb.shape[:2] != np_rgb.shape[:2]:
        H, W = np_rgb.shape[:2]
        heatmap_rgb = cv2.resize(heatmap_rgb, (W, H))
    return cv2.addWeighted(np_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)


def composite_heatmap(cams: dict[str, np.ndarray], probs: list[float]) -> np.ndarray:
    """Weighted average of per-class Grad-CAM maps using sigmoid probabilities."""
    arrays = list(cams.values())
    weights = np.array(probs, dtype=np.float32)
    total_weight = weights.sum()
    if total_weight < 1e-6:
        return arrays[0] if arrays else np.zeros((INPUT_H, INPUT_W), dtype=np.float32)
    stacked = np.stack(arrays, axis=0)  # (C, H, W)
    weighted = (stacked * weights[:, None, None]).sum(axis=0) / total_weight
    return weighted.astype(np.float32)
