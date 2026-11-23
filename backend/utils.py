"""
Dental Radiology Utility Functions:
- Image loading, CLAHE radiographic contrast enhancement
- Multi-channel preprocessing preserving 2:1 panoramic aspect ratio
- Grad-CAM heatmap visualization and base64 rendering
- Clinical research report generation
"""

import io
import base64
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    T = None
    TORCH_AVAILABLE = False

from backend.model import DISEASE_CLASSES, DISEASE_METADATA

RADIOLOGY_HEIGHT = 512
RADIOLOGY_WIDTH = 1024

def enhance_radiograph_clahe(img_np):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Standard radiograph enhancement for panoramic dental views.
    """
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
        
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb

def preprocess_opg(image_input, enhance=True):
    """
    Preprocess raw image (PIL Image, bytes, or file path) for model inference.
    Returns:
    - tensor: [1, 3, H, W] normalized PyTorch tensor
    - original_rgb: np.ndarray [H, W, 3]
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    pil_resized = pil_img.resize((RADIOLOGY_WIDTH, RADIOLOGY_HEIGHT), Image.Resampling.BILINEAR)
    np_img = np.array(pil_resized)

    if enhance:
        np_img = enhance_radiograph_clahe(np_img)

    if TORCH_AVAILABLE and T is not None:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(Image.fromarray(np_img)).unsqueeze(0)
    else:
        arr = np_img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(arr, axis=0)
    
    return tensor, np_img

def generate_overlay_base64(original_rgb, cam_map, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM activation heatmap onto the original radiograph.
    Returns base64 encoded JPEG data URI.
    """
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam_map, (w, h))
    
    heatmap = np.uint8(255 * np.clip(cam_resized, 0, 1))
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    
    success, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return ""
        
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

def image_to_base64(pil_or_np_img):
    """Convert image to base64 data URI."""
    if isinstance(pil_or_np_img, np.ndarray):
        pil_img = Image.fromarray(pil_or_np_img)
    else:
        pil_img = pil_or_np_img
        
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=90)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

def build_clinical_report(probabilities, elapsed_ms=0, device_name="CPU"):
    """
    Assemble clinical research findings from model probabilities.
    probabilities: list or array of floats corresponding to DISEASE_CLASSES
    """
    results = []
    high_risk_count = 0
    max_prob = 0.0

    for i, key in enumerate(DISEASE_CLASSES):
        prob = float(probabilities[i]) if i < len(probabilities) else 0.0
        max_prob = max(max_prob, prob)
        meta = DISEASE_METADATA.get(key, {})
        
        severity_levels = meta.get("severity_levels", {})
        if prob >= 0.65:
            status = "Possible Finding / Elevated Score"
            severity = severity_levels.get("severe", "Significant radiographic signal")
            high_risk_count += 1
            color = "#ef4444"
            badge = "DETECTED"
        elif prob >= 0.35:
            status = "Borderline / Moderate Score"
            severity = severity_levels.get("moderate", "Intermediate radiographic signal")
            color = "#f59e0b"
            badge = "SUSPICIOUS"
        elif prob >= 0.15:
            status = "Low / Incipient Signal"
            severity = severity_levels.get("mild", "Subtle radiographic irregularity")
            color = "#3b82f6"
            badge = "MONITOR"
        else:
            status = "Unlikely / Below Threshold"
            severity = "No definitive radiographic evidence above threshold"
            color = "#10b981"
            badge = "CLEAR"

        results.append({
            "index": i,
            "disease": key,
            "category": meta.get("category", "Dental Pathology"),
            "probability": round(prob * 100, 1),
            "raw_prob": prob,
            "status": status,
            "badge": badge,
            "color": color,
            "severity": severity,
            "description": meta.get("description", ""),
            "hallmarks": meta.get("hallmarks", ""),
            "clinical_guidance": meta.get("clinical_guidance", "")
        })

    if max_prob < 0.25:
        overall_health = "No High-Confidence Findings"
        health_color = "#10b981"
        summary = "Model did not detect findings above threshold. This research tool does not replace clinical dental examination."
    elif high_risk_count >= 2:
        overall_health = "Multiple Possible Findings"
        health_color = "#ef4444"
        summary = "Multiple findings detected above threshold. Comprehensive in-person examination by a licensed dentist is essential."
    elif max_prob >= 0.65:
        overall_health = "Possible Radiographic Finding"
        health_color = "#f97316"
        summary = "Radiographic pattern consistent with potential pathology identified. Professional dental review recommended."
    else:
        overall_health = "Borderline Observations"
        health_color = "#eab308"
        summary = "Low-to-moderate confidence signal. Clinical correlation and follow-up recommended."

    return {
        "overall_health": overall_health,
        "health_color": health_color,
        "summary": summary,
        "diseases": results,
        "inference_time_ms": round(elapsed_ms, 1),
        "hardware_accelerator": device_name,
        "disclaimer": "Research prototype. Not a diagnostic device. Findings must be reviewed by a qualified dentist."
    }
