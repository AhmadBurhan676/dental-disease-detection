"""
Dental Radiology Utility Functions:
- Image loading, CLAHE radiographic contrast enhancement
- Multi-channel preprocessing preserving 2:1 panoramic aspect ratio
- Grad-CAM heatmap visualization and base64 rendering
- Clinical diagnostic report generation
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
from backend.model import DISEASE_METADATA

# Radiology transform: 512x1024 preserves the authentic 2:1 panoramic ratio
RADIOLOGY_HEIGHT = 512
RADIOLOGY_WIDTH = 1024

def enhance_radiograph_clahe(img_np):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    This is the radiological standard for panoramic dental radiographs,
    revealing subtle caries shadows and periapical bone trabeculation.
    """
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
        
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Convert back to 3-channel RGB
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb

def preprocess_opg(image_input, enhance=True):
    """
    Preprocess raw image (PIL Image or bytes or file path) for model inference.
    Returns:
    - tensor: [1, 3, H, W] normalized PyTorch tensor
    - original_rgb: np.ndarray [H, W, 3] of the resized original image
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Resize to standardized panoramic dimension
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
        # Fallback normalized array [1, 3, H, W]
        arr = np_img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(arr, axis=0)
    
    return tensor, np_img

def generate_overlay_base64(original_rgb, cam_map, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM activation heatmap onto the original panoramic radiograph.
    Returns base64 encoded PNG data URI.
    """
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam_map, (w, h))
    
    # Convert CAM to 8-bit heatmap
    heatmap = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Alpha blend
    overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    
    # Encode as JPEG/PNG base64
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

def build_clinical_report(probabilities, elapsed_ms=0, device_name="NVIDIA GPU"):
    """
    Assemble clinical findings and risk stratifications from model probabilities.
    probabilities: list or array of 3 floats [p_caries, p_periapical, p_periodontitis]
    """
    disease_keys = ["Dental Caries", "Periapical Lesion", "Periodontitis"]
    results = []
    
    high_risk_count = 0
    max_prob = 0.0

    for i, key in enumerate(disease_keys):
        prob = float(probabilities[i])
        max_prob = max(max_prob, prob)
        meta = DISEASE_METADATA[key]
        
        # Clinical status
        if prob >= 0.70:
            status = "High Risk / Detected"
            severity = meta["severity_levels"]["severe"]
            high_risk_count += 1
            color = "#ef4444"  # Red
            badge = "DETECTED"
        elif prob >= 0.45:
            status = "Moderate / Suspicious"
            severity = meta["severity_levels"]["moderate"]
            color = "#f59e0b"  # Amber
            badge = "SUSPICIOUS"
        elif prob >= 0.20:
            status = "Low / Incipient"
            severity = meta["severity_levels"]["mild"]
            color = "#3b82f6"  # Blue
            badge = "MONITOR"
        else:
            status = "Negative / Intact"
            severity = "No radiological evidence of active disease"
            color = "#10b981"  # Green
            badge = "CLEAR"

        results.append({
            "index": i,
            "disease": key,
            "category": meta["category"],
            "probability": round(prob * 100, 1),
            "raw_prob": prob,
            "status": status,
            "badge": badge,
            "color": color,
            "severity": severity,
            "description": meta["description"],
            "hallmarks": meta["hallmarks"],
            "clinical_guidance": meta["clinical_guidance"]
        })

    # Overall Diagnosis Assessment
    if max_prob < 0.25:
        overall_health = "Intact / Healthy Dentition"
        health_color = "#10b981"
        summary = "No significant radiographic evidence of dental caries, periapical lesions, or advanced periodontal bone resorption detected across the panoramic view."
    elif high_risk_count >= 2:
        overall_health = "Multi-Disease Dental Pathologies"
        health_color = "#ef4444"
        summary = "Multiple concurrent pathologies identified. Comprehensive clinical examination and prompt multidisciplinary dental intervention recommended."
    elif max_prob >= 0.70:
        overall_health = "Targeted Pathology Detected"
        health_color = "#f97316"
        summary = "Definitive radiolucencies detected corresponding to primary dental disease. Clinical correlation and treatment plan initiation advised."
    else:
        overall_health = "Early / Incipient Observations"
        health_color = "#eab308"
        summary = "Mild/incipient radiographic irregularities observed. Clinical bitewing confirmation and follow-up monitoring suggested."

    return {
        "overall_health": overall_health,
        "health_color": health_color,
        "summary": summary,
        "diseases": results,
        "inference_time_ms": round(elapsed_ms, 1),
        "hardware_accelerator": device_name
    }
