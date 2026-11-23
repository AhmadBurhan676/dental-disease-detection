"""
DentalVision AI - Streamlit Web Application
Panoramic Dental Radiography Pathology Diagnostics for 3 Diseases:
1. Dental Caries
2. Periapical Lesion
3. Periodontitis
Using Panoramic Dental Radiography Benchmark Dataset by Román et al. / Orvile with an AI-accelerated backend.

Design: Soft Claymorphic / Neumorphic Clinical Interface
Features: Dual Theme Engine (Light & Dark), SVG Vector Clinical Icons, Fluid Micro-Interactions.
"""

import os
import sys
import time
import json
import io
import textwrap
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="DentalVision AI - Panoramic Radiography",
    page_icon=":material/dentistry:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"
sys.path.insert(0, str(BASE_DIR))

# Dynamic imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from backend.model import DISEASE_METADATA, DISEASE_CLASSES, load_trained_model
from backend.utils import enhance_radiograph_clahe
from backend.inference import run_inference, run_all_gradcams, overlay_heatmap, composite_heatmap

# ------------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH FOR DISEASE CLASSES (driven by backend/model.py)
# ------------------------------------------------------------------------------
DISEASE_KEYS = DISEASE_CLASSES  # ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
CLASS_ICONS = {
    "Caries": "caries",
    "Deep Caries": "caries",
    "Periapical Lesion": "periapical",
    "Impacted": "tooth",
}
CLASS_COLORS = {
    "Caries": "#F59E0B",
    "Deep Caries": "#fb923c",
    "Periapical Lesion": "#F43F5E",
    "Impacted": "var(--accent)",
}

# ------------------------------------------------------------------------------
# CACHED MODEL RESOURCE — returns None if weights not yet trained
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Load and cache the trained model. Returns None if weights are missing."""
    weights_path = MODELS_DIR / "dental_model.pth"
    if weights_path.exists():
        return load_trained_model(str(weights_path))
    return None  # No fake predictions when weights are missing

# ------------------------------------------------------------------------------
# CACHED METRICS DATA LOADER
# ------------------------------------------------------------------------------
@st.cache_data
def load_metrics_data():
    """
    Load evaluation metrics from models/metrics.json.
    Returns None if file does not exist (no hardcoded fallback numbers).
    """
    metrics_file = MODELS_DIR / "metrics.json"
    if not metrics_file.exists():
        return None
    try:
        raw = json.loads(metrics_file.read_text(encoding="utf-8"))
        # Normalize format: map new evaluate.py output to what the UI expects
        normalized = {
            "n_samples_eval": raw.get("n_samples_eval", 0),
            "macro_auc": raw.get("macro_auc"),
            "mean_auc_roc": raw.get("macro_auc"),
            "macro_f1": raw.get("macro_f1"),
            "macro_precision": raw.get("macro_precision"),
            "macro_recall": raw.get("macro_recall"),
            "per_class": raw.get("per_class", {}),
            "disease_classes": raw.get("disease_classes", DISEASE_KEYS),
            "is_validated": raw.get("is_validated", False),
            "is_demo": raw.get("is_fabricated", True),  # mark fabricated metrics as demo
            "generated_at": raw.get("generated_at", ""),
            "split": raw.get("split", ""),
        }
        # history is in training_history.json, not metrics.json
        history_file = MODELS_DIR / "training_history.json"
        if history_file.exists():
            try:
                hist_raw = json.loads(history_file.read_text(encoding="utf-8"))
                # Build history dict for chart
                per_class_f1 = {}
                for epoch_data in hist_raw.get("per_class_auc", []):
                    for cls, auc in (epoch_data or {}).items():
                        if cls not in per_class_f1:
                            per_class_f1[cls] = []
                        per_class_f1[cls].append(auc if auc is not None else 0.0)
                normalized["history"] = {
                    "epochs": hist_raw.get("epochs", []),
                    "val_accuracy": [v * 100 for v in hist_raw.get("val_macro_auc", [])],
                    "per_class_f1": per_class_f1,
                }
            except Exception:
                pass
        return normalized
    except Exception:
        return None

# ------------------------------------------------------------------------------
# SVG ICON HELPER (Unified Clinical Line Icons using currentColor)
# ------------------------------------------------------------------------------
def icon(name: str, size: int = 18, color: str = "currentColor", extra_class: str = "") -> str:
    """Returns a crisp inline SVG line icon with consistent 2px stroke, round caps/joins."""
    icons = {
        "tooth": """<path d="M7 2c-2.5 0-4 2-4 4.5 0 2.5 1.2 5 2 7.5.8 2.5 1.2 6 2 8 0 0 1-1 2-4 .5-1.5 1-3 1-3s.5 1.5 1 3c1 3 2 4 2 4 .8-2 1.2-5.5 2-8 .8-2.5 2-5 2-7.5C19 4 17.5 2 15 2c-1.5 0-2.5.8-3 1.5C11.5 2.8 10.5 2 7 2z"/>""",
        "caries": """<path d="M7 2c-2.5 0-4 2-4 4.5 0 2.5 1.2 5 2 7.5.8 2.5 1.2 6 2 8 0 0 1-1 2-4 .5-1.5 1-3 1-3s.5 1.5 1 3c1 3 2 4 2 4 .8-2 1.2-5.5 2-8 .8-2.5 2-5 2-7.5C19 4 17.5 2 15 2c-1.5 0-2.5.8-3 1.5C11.5 2.8 10.5 2 7 2z"/><circle cx="10" cy="8" r="2.2" fill="currentColor"/>""",
        "periapical": """<path d="M7 2c-2.5 0-4 2-4 4.5 0 2.5 1.2 5 2 7.5.8 2.5 1.2 6 2 8 0 0 1-1 2-4 .5-1.5 1-3 1-3s.5 1.5 1 3c1 3 2 4 2 4 .8-2 1.2-5.5 2-8 .8-2.5 2-5 2-7.5C19 4 17.5 2 15 2c-1.5 0-2.5.8-3 1.5C11.5 2.8 10.5 2 7 2z"/><circle cx="7" cy="19" r="3" fill="none" stroke-dasharray="2 2"/>""",
        "periodontitis": """<path d="M4 14c2-1 4-1 6 0 2 1 4 1 6 0 2-1 3-1 4 0"/><path d="M4 18c2-1 4-1 6 0 2 1 4 1 6 0 2-1 3-1 4 0"/><path d="M7 2c-1.5 0-2.5 1-3 2.5-.5 2 0 4.5.5 6.5"/><path d="M17 2c1.5 0 2.5 1 3 2.5.5 2 0 4.5-.5 6.5"/>""",
        "microscope": """<path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><circle cx="9" cy="9" r="2"/><path d="M9 7V2"/><path d="M7 4h4"/><path d="M11 12l2.5 2.5"/>""",
        "analytics": """<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>""",
        "database": """<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>""",
        "chip": """<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>""",
        "sliders": """<path d="M4 21v-7"/><path d="M4 10V3"/><path d="M12 21v-9"/><path d="M12 8V3"/><path d="M20 21v-5"/><path d="M20 12V3"/><path d="M1 14h6"/><path d="M9 8h6"/><path d="M17 16h6"/>""",
        "palette": """<circle cx="13.5" cy="6.5" r=".5"/><circle cx="17.5" cy="10.5" r=".5"/><circle cx="8.5" cy="7.5" r=".5"/><circle cx="6.5" cy="12.5" r=".5"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.9 0 1.6-.7 1.6-1.6 0-.4-.2-.8-.4-1.1-.3-.4-.4-.8-.4-1.3 0-.9.7-1.6 1.6-1.6h2c3.3 0 6-2.7 6-6 0-5.5-4.5-10-10-10z"/>""",
        "shield": """<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>""",
        "shield_check": """<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>""",
        "alert": """<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>""",
        "download": """<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>""",
        "crosshair": """<circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/>""",
        "sparkles": """<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>""",
        "contrast": """<circle cx="12" cy="12" r="10"/><path d="M12 18a6 6 0 0 0 0-12v12z"/>""",
        "sun": """<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>""",
        "moon": """<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>""",
        "check": """<polyline points="20 6 9 17 4 12"/>"""
    }
    path_data = icons.get(name, "")
    cls_attr = f' class="{extra_class}"' if extra_class else ""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;display:inline-block;"{cls_attr}>{path_data}</svg>"""

# ------------------------------------------------------------------------------
# CLEAN HTML RENDERING HELPER (PREVENTS MARKDOWN RAW CODE BLOCKS)
# ------------------------------------------------------------------------------
def clean_html(html_str: str) -> str:
    """Strips leading and trailing whitespace from each line, removes blank lines, and joins without indentation."""
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    return "".join(lines)

def html_block(html_str: str):
    """
    Renders HTML snippets safely via st.markdown(..., unsafe_allow_html=True).
    Eliminates Markdown 4-space code block triggers and blank line breaks.
    """
    st.markdown(clean_html(html_str), unsafe_allow_html=True)

def card_html(content: str, extra_class: str = "", style: str = "") -> str:
    style_attr = f' style="{style}"' if style else ""
    cls = f"clay-card {extra_class}".strip()
    return f'<div class="{cls}"{style_attr}>{clean_html(content)}</div>'

def badge_html(text: str, level: str = "clear", icon_name: str = "") -> str:
    icon_svg = f"{icon(icon_name, size=13)} " if icon_name else ""
    return f'<span class="clay-badge badge-{level}">{icon_svg}{text}</span>'

def chip_html(text: str, icon_name: str = "", extra_class: str = "") -> str:
    icon_svg = f"{icon(icon_name, size=13)} " if icon_name else ""
    cls = f"clay-chip {extra_class}".strip()
    return f'<span class="{cls}">{icon_svg}{text}</span>'

# ------------------------------------------------------------------------------
# RADIOGRAPH VALIDATION HELPER
# ------------------------------------------------------------------------------
def validate_radiograph(image: Image.Image, file_size_bytes: int = 0) -> tuple[bool, str]:
    """Validates that uploaded file is an authentic panoramic dental radiograph."""
    MAX_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
    if file_size_bytes > MAX_SIZE_BYTES:
        return False, f"File size exceeds 15 MB limit ({file_size_bytes / (1024*1024):.1f} MB uploaded). Please upload a standard radiograph file."

    w, h = image.size
    if w < 200 or h < 100:
        return False, f"Image resolution ({w}x{h} px) is too low for radiographic inspection. Minimum required dimensions are 200x100 px."

    aspect = w / float(h)
    if aspect < 0.5 or aspect > 4.5:
        return False, f"Image aspect ratio ({aspect:.2f}:1) deviates from panoramic dental radiographies (typically 1.5:1 to 3.0:1)."

    np_img = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    if mean_sat > 60.0:
        return False, f"Uploaded image exhibits high color saturation ({mean_sat:.1f}/255). Dental radiographs are grayscale X-rays. Please upload an authentic radiograph."

    return True, ""


# ------------------------------------------------------------------------------
# CLINICAL REPORT BUILDER (uses real model outputs, not hardcoded values)
# ------------------------------------------------------------------------------
def _build_report(probs: list) -> dict:
    """
    Build a structured clinical report dict from real sigmoid probabilities.
    probs: list of float, one per DISEASE_CLASSES entry.
    """
    diseases = []
    for idx, cls_name in enumerate(DISEASE_KEYS):
        prob = float(probs[idx])
        pct = round(prob * 100, 1)
        meta = DISEASE_METADATA.get(cls_name, {})
        if prob >= 0.65:
            badge = "DETECTED"
            severity_key = "severe"
        elif prob >= 0.35:
            badge = "SUSPICIOUS"
            severity_key = "moderate"
        else:
            badge = "UNLIKELY"
            severity_key = "mild"
        severity_levels = meta.get("severity_levels", {})
        severity_str = severity_levels.get(severity_key, severity_key.title())
        diseases.append({
            "disease": cls_name,
            "category": meta.get("category", "Dental Pathology"),
            "raw_prob": prob,
            "probability": pct,
            "badge": badge,
            "severity": severity_str,
            "hallmarks": meta.get("hallmarks", ""),
            "clinical_guidance": meta.get("clinical_guidance", ""),
        })

    detected = [d["disease"] for d in diseases if d["badge"] == "DETECTED"]
    suspicious = [d["disease"] for d in diseases if d["badge"] == "SUSPICIOUS"]

    if detected:
        overall = "Possible Findings Detected"
        summary = f"Model identified possible {', '.join(detected)}. Review required by qualified dentist."
        health_color = "#F43F5E"
    elif suspicious:
        overall = "Borderline Findings — Further Review Advised"
        summary = f"Low-confidence signals for {', '.join(suspicious)}. Clinical correlation recommended."
        health_color = "#F59E0B"
    else:
        overall = "No High-Confidence Findings"
        summary = "Model did not detect findings above threshold. This does not exclude pathology."
        health_color = "#10B981"

    return {
        "overall_health": overall,
        "summary": summary,
        "health_color": health_color,
        "diseases": diseases,
    }


# ------------------------------------------------------------------------------
# REPORT GENERATION (PNG EXPORT WITH EMBEDDED DISCLAIMER)
# ------------------------------------------------------------------------------
def generate_report_image(orig_img: np.ndarray, overlay_img: np.ndarray, report_data: dict, image_label: str) -> bytes:
    """Creates a high-contrast printable clinical analysis summary sheet (PNG)."""
    W_SHEET, H_SHEET = 950, 720
    canvas = Image.new("RGB", (W_SHEET, H_SHEET), color=(245, 246, 248))
    draw = ImageDraw.Draw(canvas)

    # Header Card
    draw.rectangle([(20, 20), (W_SHEET - 20, 95)], fill=(30, 32, 38))
    draw.text((40, 32), "DentalVision AI — Radiographic Analysis Report", fill=(255, 255, 255))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw.text((40, 62), f"Analysis Timestamp: {timestamp}  |  Source: {image_label}", fill=(180, 185, 195))

    # Thumbnail Previews (Original vs Overlay)
    TH_W, TH_H = 435, 215
    orig_pil = Image.fromarray(orig_img).resize((TH_W, TH_H))
    over_pil = Image.fromarray(overlay_img).resize((TH_W, TH_H))

    draw.rectangle([(20, 110), (20 + TH_W + 10, 110 + TH_H + 35)], fill=(255, 255, 255), outline=(220, 222, 228))
    draw.text((30, 118), "Original Radiograph (Raw)", fill=(40, 45, 55))
    canvas.paste(orig_pil, (25, 140))

    draw.rectangle([(485, 110), (485 + TH_W + 10, 110 + TH_H + 35)], fill=(255, 255, 255), outline=(220, 222, 228))
    draw.text((495, 118), "Grad-CAM Attention Overlay", fill=(40, 45, 55))
    canvas.paste(over_pil, (490, 140))

    # Findings Summary Block
    draw.rectangle([(20, 375), (W_SHEET - 20, 615)], fill=(255, 255, 255), outline=(220, 222, 228))
    draw.text((40, 390), "Evaluated Pathologies & Estimated Probabilities", fill=(25, 28, 35))
    draw.line([(40, 415), (W_SHEET - 40, 415)], fill=(230, 233, 240), width=1)

    y_pos = 430
    for idx, d in enumerate(report_data.get("diseases", [])):
        draw.text((40, y_pos), f"{idx+1}. {d['disease']} ({d['category']})", fill=(30, 35, 45))
        draw.text((380, y_pos), f"Probability: {d['probability']}%", fill=(20, 25, 35))
        draw.text((540, y_pos), f"Status: {d['badge']}", fill=(200, 50, 70) if d['badge'] == "DETECTED" else (30, 160, 100))
        draw.text((680, y_pos), f"Grade: {d['severity'][:24]}...", fill=(90, 95, 105))
        y_pos += 45

    draw.text((40, 575), f"Overall Assessment: {report_data.get('overall_health', 'Evaluated')} — {report_data.get('summary', '')[:85]}...", fill=(60, 65, 75))

    # Persistent Medical Disclaimer Banner
    draw.rectangle([(20, 630), (W_SHEET - 20, 695)], fill=(254, 242, 242), outline=(254, 202, 202))
    draw.text((35, 642), "MEDICAL DISCLAIMER", fill=(185, 28, 28))
    draw.text((35, 665), "Research prototype. Not a diagnostic device. Findings must be reviewed by a qualified dentist.", fill=(153, 27, 27))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()

# ------------------------------------------------------------------------------
# CACHED CHART GENERATOR (SITS FLAT ON CARD SURFACE, NO BEZEL)
# ------------------------------------------------------------------------------
@st.cache_data
def generate_training_chart_figure(theme: str, history_dict: dict):
    """Generates theme-aware matplotlib convergence chart figure cached by theme & data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_dark = (theme == "dark")
    bg = "#1E2025" if is_dark else "#E8E8EB"
    surface = "#24272E" if is_dark else "#DFDFE3"
    text_color = "#E6EDF3" if is_dark else "#1F2328"
    muted_color = "#9DA7B3" if is_dark else "#4B535D"
    grid_color = (1.0, 1.0, 1.0, 0.08) if is_dark else (0.0, 0.0, 0.0, 0.08)

    epochs = history_dict.get("epochs", list(range(1, 11)))
    val_acc = history_dict.get("val_accuracy", [])
    per_class_f1 = history_dict.get("per_class_f1", {})

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(surface)

    if val_acc:
        ax.plot(epochs, val_acc, label="Val Accuracy (%)", color="#F26B7A",
                linewidth=2.4, marker="o", markersize=4, markerfacecolor="#F26B7A", markeredgewidth=0)

    curve_colors = ["#F59E0B", "#10B981", "#0EA5E9", "#8B5CF6"]
    for idx, (cls_name, f1_scores) in enumerate(per_class_f1.items()):
        col = curve_colors[idx % len(curve_colors)]
        scaled_scores = [s * 100 if max(f1_scores) <= 1.0 else s for s in f1_scores]
        ax.plot(epochs, scaled_scores, label=f"{cls_name} F1 (%)", color=col,
                linewidth=2.2, marker="o", markersize=4, markerfacecolor=col, markeredgewidth=0)

    ax.set_xlabel("Epoch", color=text_color, fontsize=10)
    ax.set_ylabel("Score (%)", color=text_color, fontsize=10)
    ax.tick_params(colors=muted_color, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(muted_color)
        sp.set_alpha(0.3)
    ax.grid(True, color=grid_color, linewidth=0.8)
    ax.set_ylim(65, 95)
    ax.set_xlim(1, max(epochs) if epochs else 10)
    ax.legend(
        facecolor=surface, edgecolor=muted_color,
        labelcolor=text_color, fontsize=8.5,
        loc="lower right", framealpha=0.9
    )
    plt.tight_layout(pad=1.5)
    return fig

# ------------------------------------------------------------------------------
# THEME CONFIGURATION & CSS VARIABLES (WCAG AA COMPLIANT IN BOTH THEMES)
# ------------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

query_params = st.query_params
if "theme" in query_params:
    param_theme = query_params.get("theme", "").lower()
    if param_theme in ("light", "dark"):
        st.session_state["theme"] = param_theme

default_mode_idx = 0
if "tab" in query_params:
    tab_param = query_params.get("tab", "").lower()
    if "metric" in tab_param:
        default_mode_idx = 1
    elif "dataset" in tab_param:
        default_mode_idx = 2

def get_theme_css(theme: str) -> str:
    is_dark = (theme == "dark")
    if is_dark:
        bg = "#1E2025"
        surface = "#1E2025"
        surface_sec = "#17181C"
        shadow_dark = "#111215"
        shadow_light = "#2B2E35"
        accent = "#E06372"
        accent_hover = "#D45464"
        text = "#E6EDF3"
        text_muted = "#9DA7B3"  # WCAG AA contrast against #1E2025 (4.8:1)
        border_subtle = "rgba(255, 255, 255, 0.08)"
        viewer_bg = "#0B0C0E"
    else:
        bg = "#E8E8EB"
        surface = "#E8E8EB"
        surface_sec = "#DFDFE3"
        shadow_dark = "#C2C2C7"
        shadow_light = "#FFFFFF"
        accent = "#F26B7A"
        accent_hover = "#E85B6B"
        text = "#1F2328"
        text_muted = "#4B535D"  # WCAG AA contrast against #E8E8EB (4.6:1)
        border_subtle = "rgba(0, 0, 0, 0.06)"
        viewer_bg = "#0B0C0E"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {{
            --bg: {bg};
            --surface: {surface};
            --surface-sec: {surface_sec};
            --shadow-dark: {shadow_dark};
            --shadow-light: {shadow_light};
            --shadow-raised: 6px 6px 14px var(--shadow-dark), -6px -6px 14px var(--shadow-light);
            --shadow-raised-sm: 4px 4px 10px var(--shadow-dark), -4px -4px 10px var(--shadow-light);
            --shadow-raised-lg: 10px 14px 28px var(--shadow-dark), -8px -8px 20px var(--shadow-light);
            --shadow-inset: inset 4px 4px 8px var(--shadow-dark), inset -4px -4px 8px var(--shadow-light);
            --shadow-inset-sm: inset 2px 2px 5px var(--shadow-dark), inset -2px -2px 5px var(--shadow-light);
            --accent: {accent};
            --accent-hover: {accent_hover};
            --text: {text};
            --text-muted: {text_muted};
            --radius-card: 22px;
            --radius-control: 14px;
            --radius-pill: 9999px;
            --border-subtle: {border_subtle};
            --viewer-bg: {viewer_bg};
        }}

        /* ── Universal Typography Overrides for Dual Theme WCAG AA ── */
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
            background-color: var(--bg) !important;
            color: var(--text) !important;
        }}

        .stApp,
        .stApp * {{
            color: var(--text);
        }}

        .stMarkdown,
        .stMarkdown p,
        .stMarkdown span,
        .stMarkdown h1,
        .stMarkdown h2,
        .stMarkdown h3,
        .stMarkdown h4,
        .stMarkdown h5,
        .stMarkdown h6,
        p, span, label, h1, h2, h3, h4, h5, h6,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] span,
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3,
        div[data-testid="stMarkdownContainer"] h4,
        div[data-testid="stMarkdownContainer"] h5,
        div[data-testid="stMarkdownContainer"] h6 {{
            color: var(--text) !important;
        }}

        #MainMenu, footer {{
            display: none !important;
        }}

        header[data-testid="stHeader"] {{
            background: transparent !important;
            z-index: 10 !important;
        }}

        /* Top & Bottom clearance: toolbar never overlaps top card, fixed disclaimer never covers content */
        .main .block-container {{
            padding-top: 4.5rem !important;
            padding-bottom: 9rem !important;
            max-width: 1400px !important;
        }}

        *, *::before, *::after {{
            transition: background-color 0.18s cubic-bezier(0.4, 0, 0.2, 1),
                        box-shadow 0.18s cubic-bezier(0.4, 0, 0.2, 1),
                        transform 0.18s cubic-bezier(0.4, 0, 0.2, 1),
                        color 0.18s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        *:focus-visible {{
            outline: 2px solid var(--accent) !important;
            outline-offset: 2px !important;
        }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                transition: none !important;
                animation: none !important;
            }}
        }}

        .clay-card {{
            background: var(--surface) !important;
            border-radius: var(--radius-card) !important;
            box-shadow: var(--shadow-raised) !important;
            padding: 1.4rem 1.6rem !important;
            margin-bottom: 1.2rem !important;
            border: 1px solid var(--border-subtle) !important;
        }}

        .clay-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-raised-lg) !important;
        }}

        .clay-card-hero {{
            background: var(--surface) !important;
            border-radius: var(--radius-card) !important;
            box-shadow: var(--shadow-raised) !important;
            padding: 1.6rem 2rem !important;
            margin-bottom: 1.5rem !important;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.2rem;
            border: 1px solid var(--border-subtle) !important;
        }}

        .clay-badge {{
            border-radius: var(--radius-pill);
            padding: 5px 14px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            box-shadow: var(--shadow-raised-sm);
        }}

        .badge-high {{
            background: #F43F5E !important;
            color: #FFFFFF !important;
            box-shadow: 3px 3px 8px rgba(244, 63, 94, 0.45), -2px -2px 6px rgba(255, 255, 255, 0.4) !important;
        }}

        .badge-mod {{
            background: #F59E0B !important;
            color: #FFFFFF !important;
            box-shadow: 3px 3px 8px rgba(245, 158, 11, 0.4), -2px -2px 6px rgba(255, 255, 255, 0.4) !important;
        }}

        .badge-clear {{
            background: #10B981 !important;
            color: #FFFFFF !important;
            box-shadow: 3px 3px 8px rgba(16, 185, 129, 0.35), -2px -2px 6px rgba(255, 255, 255, 0.4) !important;
        }}

        .clay-chip {{
            background: var(--surface) !important;
            color: var(--text-muted) !important;
            border-radius: var(--radius-pill) !important;
            padding: 5px 13px !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            box-shadow: var(--shadow-inset-sm) !important;
        }}

        .clay-chip-accent {{
            background: var(--surface) !important;
            color: var(--accent) !important;
            box-shadow: var(--shadow-raised-sm) !important;
            font-weight: 700 !important;
        }}

        /* ── Sidebar Styling & Contrast Overrides ── */
        section[data-testid="stSidebar"] {{
            background-color: var(--surface) !important;
            border-right: none !important;
            box-shadow: 6px 0 16px var(--shadow-dark) !important;
            color: var(--text) !important;
        }}

        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: var(--text) !important;
        }}

        /* Checkbox labels in sidebar */
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label span,
        div[data-testid="stCheckbox"] label p {{
            color: var(--text) !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
        }}

        /* Slider labels and values */
        div[data-testid="stSlider"] label,
        div[data-testid="stSlider"] label p,
        div[data-testid="stSlider"] label span,
        div[data-testid="stSlider"] div[data-testid="stTickBar"] *,
        div[data-testid="stSlider"] [data-testid="stWidgetLabel"] p {{
            color: var(--text) !important;
        }}

        /* Radio section headers ("Navigation") - subtle plain text label */
        div[data-testid="stRadio"] > label,
        div[data-testid="stRadio"] > label p,
        div[data-testid="stRadio"] label[data-testid="stWidgetLabel"],
        div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] p,
        div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] span {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 0 0.5rem 0 !important;
            margin: 0 !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            color: var(--text-muted) !important;
            cursor: default !important;
            width: auto !important;
            display: block !important;
        }}

        div[data-testid="stRadio"] > label:hover,
        div[data-testid="stRadio"] label[data-testid="stWidgetLabel"]:hover {{
            box-shadow: none !important;
            transform: none !important;
        }}

        /* Radio group container and options - full & equal width */
        div[data-testid="stRadio"] div[role="radiogroup"] {{
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 8px !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] label {{
            width: 100% !important;
            box-sizing: border-box !important;
            display: flex !important;
            align-items: center !important;
            background: var(--surface) !important;
            border-radius: var(--radius-control) !important;
            padding: 0.75rem 1.1rem !important;
            box-shadow: var(--shadow-raised-sm) !important;
            border: 1px solid var(--border-subtle) !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            margin: 0 !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] label,
        div[data-testid="stRadio"] div[role="radiogroup"] label p,
        div[data-testid="stRadio"] div[role="radiogroup"] label span {{
            color: var(--text) !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
            box-shadow: var(--shadow-raised) !important;
            transform: translateY(-1px) !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {{
            box-shadow: var(--shadow-inset) !important;
            color: var(--accent) !important;
            font-weight: 700 !important;
            transform: none !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p,
        div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] span {{
            color: var(--accent) !important;
            font-weight: 700 !important;
        }}

        /* Horizontal radiogroup (e.g. Theme toggle) */
        div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] {{
            flex-direction: row !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] label {{
            flex: 1 1 0 !important;
            justify-content: center !important;
        }}

        div[data-testid="stButton"] button {{
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-control) !important;
            padding: 0.65rem 1.4rem !important;
            font-weight: 700 !important;
            box-shadow: var(--shadow-raised-sm) !important;
        }}

        div[data-testid="stButton"] button:hover {{
            box-shadow: var(--shadow-raised) !important;
            color: var(--accent) !important;
            transform: translateY(-1px);
        }}

        div[data-testid="stButton"] button:active {{
            box-shadow: var(--shadow-inset-sm) !important;
            transform: translateY(1px);
        }}

        div[data-testid="stFileUploader"] {{
            background: var(--surface) !important;
            border-radius: var(--radius-card) !important;
            padding: 1.4rem !important;
            box-shadow: var(--shadow-inset) !important;
            border: 1px dashed var(--border-subtle) !important;
        }}

        div[data-baseweb="select"] > div {{
            background: var(--surface) !important;
            color: var(--text) !important;
            border: none !important;
            border-radius: var(--radius-control) !important;
            box-shadow: var(--shadow-raised-sm) !important;
            padding: 0.2rem 0.4rem !important;
        }}

        div[data-baseweb="slider"] div[role="slider"] {{
            background-color: var(--accent) !important;
            box-shadow: var(--shadow-raised-sm) !important;
            border: 2px solid var(--surface) !important;
            width: 19px !important;
            height: 19px !important;
        }}

        /* ── Stat Card (st.metric) Overrides ── */
        div[data-testid="stMetric"] {{
            background: var(--surface) !important;
            border-radius: var(--radius-card) !important;
            padding: 1.2rem 1.4rem !important;
            box-shadow: var(--shadow-raised) !important;
            border: 1px solid var(--border-subtle) !important;
        }}

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] span {{
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            color: var(--text-muted) !important;
        }}

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {{
            font-size: clamp(1.25rem, 2.2vw, 1.85rem) !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em !important;
            color: var(--text) !important;
            white-space: normal !important;
            word-break: break-word !important;
            overflow-wrap: break-word !important;
            line-height: 1.25 !important;
        }}

        div[data-testid="stImage"] {{
            background: var(--viewer-bg) !important;
            border-radius: var(--radius-card) !important;
            padding: 10px !important;
            box-shadow: var(--shadow-inset) !important;
            border: 1px solid var(--border-subtle) !important;
            overflow: hidden !important;
        }}

        div[data-testid="stImage"] img {{
            border-radius: var(--radius-control) !important;
        }}

        /* ── Training Convergence Chart: sits flat on card surface without black bezel ── */
        div[data-testid="stPyplot"],
        div[data-testid="stPyplot"] div[data-testid="stImage"],
        div[data-testid="stPyplot"] [data-testid="stImage"] {{
            background: transparent !important;
            padding: 0 !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: var(--radius-card) !important;
            overflow: hidden !important;
        }}

        div[data-testid="stPyplot"] img {{
            background: var(--surface) !important;
            border-radius: var(--radius-card) !important;
            box-shadow: var(--shadow-raised) !important;
            border: 1px solid var(--border-subtle) !important;
            display: block !important;
        }}

        [data-testid="StyledFullScreenButton"] {{
            background: var(--surface) !important;
            border-radius: var(--radius-pill) !important;
            box-shadow: var(--shadow-raised-sm) !important;
            border: 1px solid var(--border-subtle) !important;
            padding: 4px 7px !important;
            opacity: 1 !important;
        }}

        [data-testid="StyledFullScreenButton"] svg {{
            color: var(--text) !important;
            stroke: var(--text) !important;
            fill: none !important;
        }}

        [data-testid="StyledFullScreenButton"]:hover {{
            background: var(--accent) !important;
        }}

        [data-testid="StyledFullScreenButton"]:hover svg {{
            color: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }}

        [data-testid="stElementToolbar"] {{
            background: var(--surface) !important;
            border-radius: var(--radius-control) !important;
            box-shadow: var(--shadow-raised-sm) !important;
            border: 1px solid var(--border-subtle) !important;
            padding: 2px 4px !important;
        }}

        /* Persistent Medical Disclaimer Footer */
        .clinical-disclaimer-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: var(--surface);
            border-top: 1px solid var(--border-subtle);
            box-shadow: 0 -4px 12px var(--shadow-dark);
            padding: 0.65rem 1.5rem;
            z-index: 999;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .clinical-disclaimer-pill {{
            background: rgba(242, 107, 122, 0.12);
            color: var(--accent);
            padding: 2px 8px;
            border-radius: 9999px;
            font-weight: 700;
            margin-right: 8px;
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.05em;
        }}

        @media (max-width: 768px) {{
            .main .block-container {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}
            .clay-card-hero {{
                flex-direction: column;
                align-items: flex-start;
            }}
        }}
    </style>
    """

# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION & THEME CONTROLS
# ------------------------------------------------------------------------------
with st.sidebar:
    html_block(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:0.4rem;">
        <div style="background:var(--accent); color:#FFFFFF; border-radius:14px; width:44px; height:44px; display:flex; align-items:center; justify-content:center; box-shadow:var(--shadow-raised-sm);">
            {icon("tooth", size=24, color="#FFFFFF")}
        </div>
        <div>
            <h3 style="margin:0; font-weight:800; letter-spacing:-0.02em; font-size:1.15rem; color:var(--text);">DentalVision AI</h3>
            <span style="font-size:0.75rem; color:var(--accent); font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">Panoramic Analysis</span>
        </div>
    </div>
    """)

    mode = st.radio(
        "Navigation",
        [
            ":material/medical_services: Analysis Studio",
            ":material/analytics: Model Metrics & Evaluation",
            ":material/dataset: Dataset Info"
        ],
        index=default_mode_idx
    )

    html_block("<hr style='border-color:var(--border-subtle); margin:1.2rem 0;'>")
    html_block(f"""
    <div style="font-size:0.85rem; font-weight:700; margin-bottom:0.6rem; display:flex; align-items:center; gap:6px; color:var(--text);">
        {icon("sliders", size=16, color="var(--accent)")} Radiograph Tools
    </div>
    """)

    enhance_clahe = st.checkbox("CLAHE Contrast Enhancement", value=True, help="Applies Contrast Limited Adaptive Histogram Equalization to maximize radiodensity differentiation.")
    invert_colors = st.checkbox("Invert Grayscale (Negative View)", value=False, help="Inverts pixel brightness for enhanced root canal and osseous margin inspection.")
    heatmap_opacity = st.slider(
        "Grad-CAM Overlay Opacity",
        min_value=0.1, max_value=1.0, value=0.65, step=0.05,
        help="Adjusts the blend transparency of the Grad-CAM deep learning heatmaps on top of the radiograph."
    )

    html_block("<hr style='border-color:var(--border-subtle); margin:1.2rem 0;'>")
    html_block(f"""
    <div style="font-size:0.85rem; font-weight:700; margin-bottom:0.6rem; display:flex; align-items:center; gap:6px; color:var(--text);">
        {icon("palette", size=16, color="var(--accent)")} Appearance
    </div>
    """)

    theme_choice = st.radio(
        "Theme Toggle",
        [":material/light_mode: Light", ":material/dark_mode: Dark"],
        index=0 if st.session_state["theme"] == "light" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )

    selected_theme = "light" if "Light" in theme_choice else "dark"
    if selected_theme != st.session_state["theme"]:
        st.session_state["theme"] = selected_theme
        st.rerun()

    html_block("<hr style='border-color:var(--border-subtle); margin:1.2rem 0;'>")
    st.caption("DentalVision AI v2.2 &bull; Benchmark Dataset: Román et al. Sensors 2021")

st.markdown(get_theme_css(st.session_state["theme"]), unsafe_allow_html=True)

# ==============================================================================
# TAB 1: ANALYSIS STUDIO
# ==============================================================================
if "Analysis Studio" in mode:
    html_block(f"""
    <div class="clay-card-hero">
        <div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; flex-wrap:wrap;">
                <span class="clay-chip clay-chip-accent">
                    {icon("sparkles", size=14, color="var(--accent)")} AI ANALYSIS ENGINE
                </span>
                <span class="clay-chip">
                    {icon("database", size=14, color="var(--text-muted)")} BENCHMARK OPG
                </span>
                <span class="clay-chip">
                    {icon("crosshair", size=14, color="var(--text-muted)")} GRAD-CAM XAI
                </span>
            </div>
            <h1 style="margin:0; font-size:1.85rem; font-weight:800; letter-spacing:-0.02em; color:var(--text);">Panoramic Dental Radiography Analysis Studio</h1>
            <p style="margin:4px 0 0 0; font-size:0.88rem; color:var(--text-muted);">Multi-Disease Radiographic Inspection with Explainable Visual Attention Heatmaps</p>
        </div>
        <div>
            <span class="clay-chip" style="color:#10B981; font-weight:700;">
                {icon("shield_check", size=14, color="#10B981")} Real-time Analysis
            </span>
        </div>
    </div>
    """)

    col_view, col_diag = st.columns([1.3, 1.0], gap="large")

    with col_view:
        html_block(f"<h3 style='font-size:1.2rem; font-weight:700; margin-bottom:0.8rem; display:flex; align-items:center; gap:8px; color:var(--text);'>{icon('microscope', size=20, color='var(--accent)')} 1. Radiograph Input</h3>")

        input_method = st.radio("Choose Input:", ["Benchmark Sample Radiographs", "Upload Custom OPG"], horizontal=True)

        selected_image = None
        image_name = "sample"

        if input_method == "Benchmark Sample Radiographs":
            st.caption("Quick-select a reference benchmark radiograph to inspect:")
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                if st.button("Sample 1: Caries", use_container_width=True):
                    st.session_state["chosen_sample"] = "sample_1.png"
            with s_col2:
                if st.button("Sample 2: Periapical", use_container_width=True):
                    st.session_state["chosen_sample"] = "sample_2.png"
            with s_col3:
                if st.button("Sample 3: Multi-finding", use_container_width=True):
                    st.session_state["chosen_sample"] = "sample_3.png"

            sample_options = {
                "Sample 1: Caries & Deep Caries (DENTEX test #33)": "sample_1.png",
                "Sample 2: Periapical Lesion & Impacted (DENTEX test #38)": "sample_2.png",
                "Sample 3: Multi-morbidity — Caries, Deep Caries, Periapical (DENTEX test #30)": "sample_3.png"
            }
            default_choice_idx = 0
            if "chosen_sample" in st.session_state:
                for idx, fname in enumerate(sample_options.values()):
                    if fname == st.session_state["chosen_sample"]:
                        default_choice_idx = idx

            sample_choice = st.selectbox("Or choose from dropdown:", list(sample_options.keys()), index=default_choice_idx)
            sample_file_to_load = sample_options[sample_choice]
            sample_path = SAMPLES_DIR / sample_file_to_load
            if sample_path.exists():
                selected_image = Image.open(sample_path)
                image_name = sample_file_to_load
        else:
            uploaded_file = st.file_uploader(
                "Upload OPG Radiograph (PNG, JPG, JPEG — max 15MB)",
                type=["jpg", "jpeg", "png"],
                help="Accepts panoramic dental radiographs up to 15MB in size."
            )
            if uploaded_file is not None:
                try:
                    img_candidate = Image.open(uploaded_file)
                    is_valid, err_msg = validate_radiograph(img_candidate, uploaded_file.size)
                    if not is_valid:
                        st.error(f"Validation Notice: {err_msg}")
                    else:
                        selected_image = img_candidate
                        image_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Error opening image file: {e}")

        if selected_image is not None:
            model = get_cached_model()
            model_ready = model is not None

            with st.spinner("Analyzing radiograph..."):
                np_img_raw = np.array(selected_image.convert("RGB"))
                np_img = np_img_raw.copy()
                if enhance_clahe:
                    np_img = enhance_radiograph_clahe(np_img)
                if invert_colors:
                    np_img = 255 - np_img

                H, W = np_img.shape[:2]
                probs = None
                cams = {}
                inference_error = None

                if model_ready:
                    try:
                        import time as _time
                        t0 = _time.time()
                        result = run_inference(model, selected_image)
                        probs = result["probabilities"]
                        cams = run_all_gradcams(model, selected_image)
                        elapsed_ms = (_time.time() - t0) * 1000
                    except Exception as e:
                        inference_error = str(e)
                else:
                    inference_error = (
                        "Model weights not found. Run `python -m backend.train` to train the "
                        "model first, then restart this app."
                    )

                if inference_error:
                    st.error(f"**Analysis unavailable:** {inference_error}")
                    report = None
                    overlay_display = np_img.copy()
                    composite_cam = np.zeros((H, W), dtype=np.float32)
                else:
                    # Build report from real model outputs
                    report = _build_report(probs)
                    # Real composite Grad-CAM
                    if cams:
                        composite_cam = composite_heatmap(cams, probs)
                    else:
                        composite_cam = np.zeros((H, W), dtype=np.float32)
                    overlay_display = overlay_heatmap(np_img, composite_cam, alpha=heatmap_opacity)

            # View modes
            html_block("<div style='margin-top:0.8rem;'></div>")
            view_mode = st.selectbox(
                "Radiograph Viewing Mode:",
                ["Composite Grad-CAM Overlay", "Side-by-Side (Raw vs CLAHE vs Overlay)", "Original Radiograph (Raw)", "CLAHE Enhanced View"]
            )

            if view_mode == "Side-by-Side (Raw vs CLAHE vs Overlay)":
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.image(np_img_raw, caption="Original Raw OPG", use_container_width=True)
                with col_s2:
                    st.image(enhance_radiograph_clahe(np_img_raw), caption="CLAHE Enhanced", use_container_width=True)
                with col_s3:
                    st.image(overlay_display, caption="Grad-CAM Overlay", use_container_width=True)
            elif view_mode == "Original Radiograph (Raw)":
                st.image(np_img_raw, caption="Original Panoramic Radiograph", use_container_width=True)
            elif view_mode == "CLAHE Enhanced View":
                st.image(np_img, caption="CLAHE Contrast Enhanced Radiograph", use_container_width=True)
            else:
                st.image(overlay_display, caption="Panoramic Radiograph with Grad-CAM Attention Map", use_container_width=True)

            clahe_label = "CLAHE Active" if enhance_clahe else "Raw Density"
            invert_label = "Negative View" if invert_colors else "Positive View"
            html_block(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; flex-wrap:wrap; gap:8px;">
                <span class="clay-chip">{icon("crosshair", size=14, color="var(--accent)")} {W} &times; {H} px</span>
                <span class="clay-chip">{icon("contrast", size=14, color="var(--accent)")} {clahe_label}</span>
                <span class="clay-chip">{icon("sparkles", size=14, color="var(--accent)")} {invert_label}</span>
                <span class="clay-chip clay-chip-accent">{view_mode}</span>
            </div>
            """)

    with col_diag:
        html_block(f"<h3 style='font-size:1.2rem; font-weight:700; margin-bottom:0.8rem; display:flex; align-items:center; gap:8px; color:var(--text);'>{icon('shield_check', size=20, color='var(--accent)')} 2. Possible Findings</h3>")
        if selected_image is not None:
            if report is None:
                # Model not available — already showed error above
                html_block("""
                <div class="clay-card" style="text-align:center; padding:2rem 1.5rem;">
                <h4 style="margin:0 0 0.5rem 0; font-weight:700; color:var(--text);">Model Not Ready</h4>
                <p style="color:var(--text-muted); font-size:0.88rem; max-width:320px; margin:0 auto;">
                Train the model first with <code>python -m backend.train</code>,
                then restart the app to enable real AI analysis.
                </p>
                </div>""")
            else:
                health_color = report.get('health_color', '#10B981')
                html_block(f"""
                <div class="clay-card" style="border-left: 6px solid {health_color}; margin-bottom: 1.25rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                <span style="font-size:0.75rem; text-transform:uppercase; color:var(--text-muted); font-weight:800; letter-spacing:0.06em;">Primary Radiographic Observation</span>
                <span class="clay-chip" style="color:var(--accent); font-size:0.72rem;">
                {icon("analytics", size=12, color="var(--accent)")} AI Analysis
                </span>
                </div>
                <h3 style="color:{health_color}; margin:0.2rem 0 0.4rem 0; font-size:1.35rem; font-weight:800;">{report['overall_health']}</h3>
                <p style="font-size:0.88rem; margin:0; line-height:1.45; color:var(--text);">{report['summary']}</p>
                <div style="margin-top:0.75rem; font-size:0.76rem; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                {icon("shield_check", size=13, color="#10B981")} {len(DISEASE_KEYS)} pathologies evaluated &bull; DentalVision AI
                </div>
                </div>
                """)

                # Confidence Threshold Slider
                conf_threshold = st.slider(
                    "Filter Findings by Confidence Threshold:",
                    min_value=0.10, max_value=0.90, value=0.25, step=0.05,
                    help="Only show pathology findings whose model confidence meets or exceeds this threshold."
                )

                # Findings Driven by DISEASE_METADATA
                displayed_count = 0
                for idx, d in enumerate(report["diseases"]):
                    if d["raw_prob"] < conf_threshold:
                        continue
                    displayed_count += 1
                    cls_name = d["disease"]
                    color_code = CLASS_COLORS.get(cls_name, "var(--accent)")
                    icon_key = CLASS_ICONS.get(cls_name, "tooth")
                    badge_level = "high" if d["badge"] == "DETECTED" else ("mod" if d["badge"] == "SUSPICIOUS" else "clear")
                    icon_markup = icon(icon_key, size=22, color=color_code)
                    badge_markup = badge_html(d["badge"], level=badge_level)

                    html_block(f"""
                    <div class="clay-card" style="border-left: 5px solid {color_code}; margin-bottom: 0.85rem; padding: 1.15rem 1.3rem;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="display:flex; align-items:center; gap:10px;">
                    <div style="background:var(--surface); border-radius:12px; width:40px; height:40px; display:flex; align-items:center; justify-content:center; box-shadow:var(--shadow-inset-sm);">
                    {icon_markup}
                    </div>
                    <div>
                    <span class="disease-title" style="display:block; color:var(--text); font-weight:700;">{idx+1}. {d['disease']}</span>
                    <span class="disease-sub" style="color:var(--text-muted);">{d['category']}</span>
                    </div>
                    </div>
                    {badge_markup}
                    </div>
                    <div style="margin: 0.75rem 0 0.35rem 0; display:flex; justify-content:space-between; font-size:0.82rem;">
                    <span style="color:var(--text-muted); font-weight:600;">Estimated Probability</span>
                    <strong style="font-size:0.9rem; color:var(--text);">{d['probability']}%</strong>
                    </div>
                    </div>
                    """)
                    st.progress(float(d["raw_prob"]))

                    with st.expander(f"Clinical Guidance & Hallmarks: {d['disease']}"):
                        st.markdown(f"**Severity Grade:** {d['severity']}")
                        st.markdown(f"**Radiographic Hallmarks:** {d['hallmarks']}")
                        st.markdown(f"**Recommended Follow-up:** {d['clinical_guidance']}")

                if displayed_count == 0:
                    st.info(f"No findings meet the selected {int(conf_threshold*100)}% confidence threshold.")

                # Export Buttons (JSON + PNG printable report)
                html_block("<div style='margin-top:1.2rem;'></div>")
                report_png_bytes = generate_report_image(np_img_raw, overlay_display, report, image_name)

                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    st.download_button(
                        "Download Report (PNG)",
                        data=report_png_bytes,
                        file_name=f"dental_report_{image_name}.png",
                        mime="image/png",
                        icon=":material/download:",
                        use_container_width=True
                    )
                with exp_col2:
                    st.download_button(
                        "Export Data (JSON)",
                        data=json.dumps(report, indent=2),
                        file_name=f"dental_report_{image_name}.json",
                        mime="application/json",
                        icon=":material/download:",
                        use_container_width=True
                    )
        else:
            html_block(f"""
            <div class="clay-card" style="text-align:center; padding:3rem 1.8rem;">
                <div style="margin-bottom:1rem;">
                    {icon("tooth", size=48, color="var(--accent)")}
                </div>
                <h4 style="margin:0 0 0.5rem 0; font-weight:700; color:var(--text);">No Radiograph Selected</h4>
                <p style="color:var(--text-muted); font-size:0.88rem; max-width:320px; margin:0 auto;">
                    Select a benchmark sample or upload a panoramic radiograph (PNG/JPG) to initiate AI-assisted pathology inspection.
                </p>
            </div>
            """)

# ==============================================================================
# TAB 2: MODEL METRICS & EVALUATION
# ==============================================================================
elif "Model Metrics" in mode:
    metrics_data = load_metrics_data()

    if metrics_data is None:
        html_block(f"""
        <div class="clay-card" style="text-align:center; padding:3.5rem 2rem;">
            <div style="margin-bottom:1rem;">
                {icon("alert", size=48, color="var(--accent)")}
            </div>
            <h3 style="margin:0 0 0.5rem 0; font-weight:700; color:var(--text);">No evaluation results found</h3>
            <p style="color:var(--text-muted); font-size:0.9rem; max-width:440px; margin:0 auto 1.5rem auto;">
                No evaluation results found. Run <code>python scripts/evaluate.py</code> to generate them.
            </p>
        </div>
        """)
    else:
        is_validated = metrics_data.get("validated", False)
        is_demo = metrics_data.get("is_demo", False)
        badge_text = "VALIDATED BENCHMARK" if is_validated else "EVALUATION RESULTS"
        badge_level = "clear" if is_validated else "mod"

        demo_badge_markup = f'<span class="clay-chip" style="color:#F59E0B; font-weight:700;">{icon("alert", size=13, color="#F59E0B")} DEMO VALUES</span>' if is_demo else ""

        overall_acc = metrics_data.get("overall_accuracy")
        mean_auc = metrics_data.get("mean_auc_roc")
        baseline_label = metrics_data.get("baseline_label")
        baseline_acc = metrics_data.get("baseline_overall_accuracy")
        baseline_auc = metrics_data.get("baseline_mean_auc_roc")

        acc_delta = f"{overall_acc - baseline_acc:+.1f}% vs baseline" if (baseline_acc is not None and overall_acc is not None) else None
        auc_delta = f"{mean_auc - baseline_auc:+.3f} vs baseline" if (baseline_auc is not None and mean_auc is not None) else None

        html_block(f"""
        <div class="clay-card-hero">
            <div>
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; flex-wrap:wrap;">
                    {badge_html(badge_text, level=badge_level, icon_name="shield_check")}
                    {demo_badge_markup}
                    <span class="clay-chip">
                        {icon("database", size=14, color="var(--text-muted)")} {metrics_data.get("n_samples_eval", 598)} OPG SAMPLES
                    </span>
                    <span class="clay-chip">
                        DENTEX (MICCAI 2023)
                    </span>
                </div>
                <h1 style="margin:0; font-size:1.85rem; font-weight:800; letter-spacing:-0.02em; color:var(--text);">Model Performance & Clinical Evaluation</h1>
                <p style="margin:4px 0 0 0; font-size:0.88rem; color:var(--text-muted);">Evaluated on the DENTEX Dataset (Hamamci et al., MICCAI 2023) · ibrahimhamamci/DENTEX on Hugging Face</p>
            </div>
            <div>
                <span class="clay-chip clay-chip-accent">
                    {icon("analytics", size=14, color="var(--accent)")} Mean AUC: {f"{mean_auc:.3f}" if mean_auc is not None else 'N/A'}
                </span>
            </div>
        </div>
        """)

        # 5 Metric Cards Driven by Single Source of Truth
        per_class_metrics = metrics_data.get("per_class", {})
        metric_cols = st.columns(2 + len(DISEASE_KEYS))

        with metric_cols[0]:
            st.metric(
                "Overall Accuracy",
                f"{overall_acc:.1f}%" if overall_acc is not None else "N/A",
                delta=acc_delta,
                help=f"Percentage of correctly classified radiographs across all pathologies. Baseline: {baseline_label or 'None'}"
            )

        for col_idx, cls_name in enumerate(DISEASE_KEYS, start=1):
            cls_data = per_class_metrics.get(cls_name, {})
            f1_val = cls_data.get("f1")
            base_f1 = cls_data.get("baseline_f1")
            f1_str = f"{f1_val * 100:.1f}%" if (f1_val is not None and f1_val <= 1.0) else (f"{f1_val:.1f}%" if f1_val is not None else "N/A")
            f1_delta = None
            if base_f1 is not None and f1_val is not None:
                diff = (f1_val - base_f1) * 100 if f1_val <= 1.0 else (f1_val - base_f1)
                f1_delta = f"{diff:+.1f}% vs baseline"

            with metric_cols[col_idx]:
                st.metric(
                    f"{cls_name} F1",
                    f1_str,
                    delta=f1_delta,
                    help=f"F1-Score (harmonic mean of precision and recall) for {cls_name}. Balances missed detections against false alarms."
                )

        with metric_cols[-1]:
            st.metric(
                "Mean ROC-AUC",
                f"{mean_auc:.3f}" if mean_auc is not None else "N/A",
                delta=auc_delta,
                help="Area Under the Receiver Operating Characteristic curve measures disease discrimination across all decision thresholds."
            )

        html_block("<hr style='border-color:var(--border-subtle); margin:1.6rem 0;'>")

        col_chart, col_matrix = st.columns([1.2, 1.0], gap="large")

        with col_chart:
            html_block(f"<h3 style='font-size:1.2rem; font-weight:700; margin-bottom:0.8rem; display:flex; align-items:center; gap:8px; color:var(--text);'>{icon('analytics', size=20, color='var(--accent)')} Training Convergence & Disease F1 Progression</h3>")
            history = metrics_data.get("history")
            if history:
                chart_fig = generate_training_chart_figure(st.session_state["theme"], history)
                st.pyplot(chart_fig, use_container_width=True)
            else:
                st.info("No training history data recorded in metrics file.")

        with col_matrix:
            html_block(f"<h3 style='font-size:1.2rem; font-weight:700; margin-bottom:0.8rem; display:flex; align-items:center; gap:8px; color:var(--text);'>{icon('microscope', size=20, color='var(--accent)')} Clinical Confusion Matrix</h3>")
            table_rows = []
            for cls_name in DISEASE_KEYS:
                c_data = per_class_metrics.get(cls_name, {})
                tp = c_data.get("tp", "-")
                fp = c_data.get("fp", "-")
                fn = c_data.get("fn", "-")
                rec = c_data.get("recall")
                rec_str = f"{rec*100:.1f}%" if (rec is not None and rec <= 1.0) else (f"{rec:.1f}%" if rec is not None else "-")
                spec = c_data.get("specificity")
                spec_str = f"{spec*100:.1f}%" if (spec is not None and spec <= 1.0) else (f"{spec:.1f}%" if spec is not None else "-")

                table_rows.append(
                    f'<tr style="border-bottom:1px solid var(--border-subtle);">'
                    f'<td style="text-align:left; padding:8px; font-weight:600; color:var(--text);">{cls_name}</td>'
                    f'<td style="color:var(--text);">{tp}</td><td style="color:var(--text);">{fp}</td><td style="color:var(--text);">{fn}</td>'
                    f'<td style="color:#10B981; font-weight:700;">{rec_str}</td>'
                    f'<td style="color:var(--text);">{spec_str}</td>'
                    f'</tr>'
                )

            rows_markup = "".join(table_rows)

            html_block(f"""
            <div class="clay-card" style="padding:1.2rem; margin-bottom:1rem;">
                <table style="width:100%; border-collapse:collapse; font-size:0.86rem; text-align:center;">
                    <thead>
                        <tr style="border-bottom:2px solid var(--border-subtle); color:var(--accent);">
                            <th style="text-align:left; padding:8px; color:var(--accent);">Target Pathology</th>
                            <th style="padding:8px; color:var(--accent);" title="True Positives">TP</th>
                            <th style="padding:8px; color:var(--accent);" title="False Positives">FP</th>
                            <th style="padding:8px; color:var(--accent);" title="False Negatives">FN</th>
                            <th style="padding:8px; color:var(--accent);" title="Sensitivity / Recall">Recall</th>
                            <th style="padding:8px; color:var(--accent);" title="Specificity">Spec.</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_markup}
                    </tbody>
                </table>
            </div>
            """)

            html_block(f"""
            <div class="clay-card" style="border-left: 4px solid var(--accent);">
                <h4 style="margin:0 0 0.4rem 0; font-size:0.95rem; color:var(--accent); display:flex; align-items:center; gap:6px;">
                    {icon("shield_check", size=18, color="var(--accent)")} Radiographic Reliability Note
                </h4>
                <p style="margin:0; font-size:0.84rem; line-height:1.45; color:var(--text);">
                    The model achieves <strong>{f"{mean_auc:.3f} Mean AUC" if mean_auc is not None else "evaluated discrimination"}</strong> on the benchmark dataset. Performance figures reflect validation on the Owandy I-max Touch clinical dataset.
                </p>
            </div>
            """)

        html_block("<hr style='border-color:var(--border-subtle); margin:1.6rem 0;'>")
        html_block("<h3 style='font-size:1.2rem; font-weight:700; margin-bottom:0.8rem; color:var(--text);'>Pathology Classification Summary</h3>")
        import pandas as pd
        summary_rows = []
        for cls_name in DISEASE_KEYS:
            c_data = per_class_metrics.get(cls_name, {})
            p_val = c_data.get("precision")
            r_val = c_data.get("recall")
            f_val = c_data.get("f1")
            meta = DISEASE_METADATA.get(cls_name, {})
            summary_rows.append({
                "Disease Target": cls_name,
                "Precision": f"{p_val*100:.1f}%" if (p_val is not None and p_val <= 1.0) else (f"{p_val:.1f}%" if p_val is not None else "N/A"),
                "Recall (Sensitivity)": f"{r_val*100:.1f}%" if (r_val is not None and r_val <= 1.0) else (f"{r_val:.1f}%" if r_val is not None else "N/A"),
                "F1-Score": f"{f_val*100:.1f}%" if (f_val is not None and f_val <= 1.0) else (f"{f_val:.1f}%" if f_val is not None else "N/A"),
                "Clinical Hallmark": meta.get("hallmarks", "Characteristic radiolucency")
            })
        st.table(pd.DataFrame(summary_rows))

elif "Dataset Info" in mode:
    html_block(f"""
    <div class="clay-card-hero">
        <div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; flex-wrap:wrap;">
                <span class="clay-chip" style="color:var(--accent); font-weight:700;">
                    {icon("database", size=14, color="var(--accent)")} DENTEX DATASET
                </span>
                <span class="clay-chip">MICCAI 2023 CHALLENGE</span>
                <span class="clay-chip">CC BY-NC-SA 4.0</span>
            </div>
            <h1 style="margin:0; font-size:1.85rem; font-weight:800; letter-spacing:-0.02em; color:var(--text);">DENTEX — Dental Enumeration &amp; Diagnosis</h1>
            <p style="margin:4px 0 0 0; font-size:0.88rem; color:var(--text-muted);">Hamamci et al. (2023) · Hugging Face: ibrahimhamamci/DENTEX · Panoramic OPG Multi-label Dataset</p>
        </div>
        <div>
            <span class="clay-chip" style="color:#10B981; font-weight:700;">
                {icon("sparkles", size=14, color="#10B981")} 755 Annotated OPGs
            </span>
        </div>
    </div>
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        html_block("""
        <div class="clay-card" style="padding:1.2rem 1.4rem; height:100%; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-muted); margin-bottom:4px;">Annotated OPGs</div>
            <div style="font-size:1.85rem; font-weight:800; color:var(--text); line-height:1.2;">755</div>
            <div style="font-size:0.82rem; color:var(--text-muted); margin-top:6px;">705 train + 50 test</div>
        </div>
        """)
    with c2:
        html_block("""
        <div class="clay-card" style="padding:1.2rem 1.4rem; height:100%; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-muted); margin-bottom:4px;">Resolution</div>
            <div style="font-size:1.5rem; font-weight:800; color:var(--text); line-height:1.2;">~2900 &times; 1316</div>
            <div style="font-size:0.82rem; color:var(--text-muted); margin-top:6px;">PNG panoramic format</div>
        </div>
        """)
    with c3:
        html_block("""
        <div class="clay-card" style="padding:1.2rem 1.4rem; height:100%; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-muted); margin-bottom:4px;">Institutions</div>
            <div style="font-size:1.85rem; font-weight:800; color:var(--text); line-height:1.2;">3 Clinics</div>
            <div style="font-size:0.82rem; color:var(--text-muted); margin-top:6px;">Diverse equipment</div>
        </div>
        """)
    with c4:
        disease_names = " &bull; ".join([k.split()[0] for k in DISEASE_KEYS])
        html_block(f"""
        <div class="clay-card" style="padding:1.2rem 1.4rem; height:100%; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-muted); margin-bottom:4px;">Annotated Classes</div>
            <div style="font-size:1.85rem; font-weight:800; color:var(--text); line-height:1.2;">{len(DISEASE_KEYS)} Classes</div>
            <div style="font-size:0.82rem; color:var(--text-muted); margin-top:6px;">{disease_names}</div>
        </div>
        """)

    html_block("<hr style='border-color:var(--border-subtle); margin:1.6rem 0;'>")

    col_meta, col_hallmarks = st.columns([1.1, 1.0], gap="large")

    with col_meta:
        html_block(f"""
        <div class="clay-card">
            <h3 style="margin:0 0 0.8rem 0; font-size:1.2rem; font-weight:700; display:flex; align-items:center; gap:8px; color:var(--text);">
                {icon("database", size=20, color="var(--accent)")} Scientific Reference &amp; Origin
            </h3>
            <ul style="margin:0; padding-left:1.2rem; font-size:0.88rem; line-height:1.6; color:var(--text);">
                <li><strong>Dataset:</strong> DENTEX — <a href="https://huggingface.co/datasets/ibrahimhamamci/DENTEX" target="_blank" style="color:var(--accent); font-weight:600;">Hugging Face: ibrahimhamamci/DENTEX</a></li>
                <li><strong>Challenge:</strong> MICCAI 2023 Dental Enumeration and Diagnosis Challenge</li>
                <li><strong>License:</strong> CC BY-NC-SA 4.0 (non-commercial research use)</li>
                <li><strong>Split used:</strong> 705 training images + 50 official validation (as test)</li>
                <li><strong>Note:</strong> Periodontitis is NOT present in DENTEX annotations and is excluded from this model.</li>
                <li><strong>Citation:</strong>
                    <blockquote style="margin:0.5rem 0 0 0; border-left:3px solid var(--accent); padding-left:0.8rem; color:var(--text-muted); font-size:0.82rem;">
                        Hamamci, I.E. et al. DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays. <em>arXiv:2305.19112</em>, 2023.
                    </blockquote>
                </li>
            </ul>
        </div>
        """)

    with col_hallmarks:
        hallmarks_items = []
        for idx, (cls_name, meta) in enumerate(DISEASE_METADATA.items(), start=1):
            col_code = CLASS_COLORS.get(cls_name, "var(--accent)")
            icon_code = CLASS_ICONS.get(cls_name, "tooth")
            hallmark_text = meta.get('hallmarks', '')
            hallmarks_items.append(
                f'<div style="margin-bottom:0.75rem; display:flex; gap:10px; align-items:flex-start;">'
                f'<div style="margin-top:2px;">{icon(icon_code, size=20, color=col_code)}</div>'
                f'<div><strong style="color:{col_code};">{idx}. {cls_name}:</strong><br>'
                f'<span style="color:var(--text);">{hallmark_text}</span></div>'
                f'</div>'
            )

        hallmarks_markup = "".join(hallmarks_items)

        html_block(
            f'<div class="clay-card">'
            f'<h3 style="margin:0 0 0.8rem 0; font-size:1.2rem; font-weight:700; display:flex; align-items:center; gap:8px; color:var(--text);">'
            f'{icon("microscope", size=20, color="var(--accent)")} Pathological Hallmarks in Panoramic X-rays'
            f'</h3>'
            f'<div style="font-size:0.85rem; line-height:1.5; color:var(--text);">'
            f'{hallmarks_markup}'
            f'</div>'
            f'</div>'
        )

# ------------------------------------------------------------------------------
# PERSISTENT TASTEFUL MEDICAL DISCLAIMER FOOTER (ON ALL PAGES)
# ------------------------------------------------------------------------------
html_block(f"""
<div class="clinical-disclaimer-footer">
    <div>
        <span class="clinical-disclaimer-pill">Medical Disclaimer</span>
        Research prototype. Not a diagnostic device. Findings must be reviewed by a qualified dentist.
    </div>
</div>
""")
