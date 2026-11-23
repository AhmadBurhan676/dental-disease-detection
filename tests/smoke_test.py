"""
DentalVision AI — Full Feature Smoke Test
Tests all backend logic used by the Streamlit app
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

results = []

def check(name, fn):
    try:
        fn()
        results.append(f"  [PASS] {name}")
    except Exception as e:
        results.append(f"  [FAIL] {name}: {e}")

# 1. CLAHE Enhancement
def test_clahe():
    from backend.utils import enhance_radiograph_clahe
    img = np.random.randint(100, 200, (512, 1024, 3), dtype=np.uint8)
    out = enhance_radiograph_clahe(img)
    assert out.shape == img.shape, "Shape mismatch"
check("CLAHE Radiograph Enhancement", test_clahe)

# 2. Clinical Report Generation — Sample 1 (Caries dominant)
def test_report_caries():
    from backend.utils import build_clinical_report
    r = build_clinical_report([0.89, 0.14, 0.12], elapsed_ms=21.4, device_name="RTX 5060 Ti")
    assert len(r["diseases"]) == 3
    assert r["diseases"][0]["badge"] in ("DETECTED", "SUSPICIOUS", "CLEAR")
    assert r["diseases"][0]["probability"] == 89
check("Clinical Report — Caries Sample", test_report_caries)

# 3. Clinical Report — Sample 2 (Periapical dominant)
def test_report_peri():
    from backend.utils import build_clinical_report
    r = build_clinical_report([0.18, 0.86, 0.15], elapsed_ms=18.0, device_name="RTX 5060 Ti")
    assert r["diseases"][1]["badge"] in ("DETECTED", "SUSPICIOUS")
check("Clinical Report — Periapical Sample", test_report_peri)

# 4. Clinical Report — Sample 3 (Periodontitis dominant)
def test_report_bone():
    from backend.utils import build_clinical_report
    r = build_clinical_report([0.15, 0.22, 0.88], elapsed_ms=19.0, device_name="RTX 5060 Ti")
    assert r["diseases"][2]["badge"] in ("DETECTED", "SUSPICIOUS")
check("Clinical Report — Periodontitis Sample", test_report_bone)

# 5. Disease Metadata loaded
def test_metadata():
    from backend.model import DISEASE_METADATA
    assert len(DISEASE_METADATA) == 3
check("Disease Metadata (3 diseases)", test_metadata)

# 6. Sample radiograph files
def test_samples():
    samples_dir = Path("static/samples")
    files = list(samples_dir.glob("*"))
    assert len(files) >= 1, f"No sample files found in {samples_dir}"
    print(f"\n       Files: {[f.name for f in files]}", end="")
check("Sample OPG Files Exist", test_samples)

# 7. Training plot PNG
def test_plot():
    p = Path("static/training_plot.png")
    assert p.exists(), "training_plot.png missing"
    assert p.stat().st_size > 1000, "Plot file too small"
check("Training Plot PNG", test_plot)

# 8. Metrics JSON
def test_metrics():
    p = Path("models/training_metrics.json")
    assert p.exists(), "training_metrics.json missing"
    m = json.loads(p.read_text())
    assert "accuracy" in m or len(m) > 0
check("Training Metrics JSON", test_metrics)

# 9. Preprocess OPG (resize to 1024x512) — pass PIL Image as the app does
def test_preprocess():
    from backend.utils import preprocess_opg
    from PIL import Image as PILImage
    pil = PILImage.fromarray(np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8))
    result = preprocess_opg(pil)
    # preprocess_opg returns (tensor, orig_rgb) or just tensor depending on version
    tensor = result[0] if isinstance(result, tuple) else result
    assert tensor.ndim >= 3, f"Unexpected tensor shape: {tensor.shape}"
check("OPG Preprocessing (PIL Image input)", test_preprocess)

# 10. Build overlay base64
def test_overlay():
    from backend.utils import generate_overlay_base64
    img = np.random.randint(100, 200, (256, 512, 3), dtype=np.uint8)
    cam = np.random.rand(256, 512).astype(np.float32)
    b64 = generate_overlay_base64(img, cam)
    assert b64.startswith("data:image"), f"Bad prefix: {b64[:30]}"
check("Grad-CAM Overlay Base64 Generation", test_overlay)

# 11. Streamlit app syntax check
def test_syntax():
    import ast
    src = Path("streamlit_app.py").read_text(encoding="utf-8")
    ast.parse(src)
check("streamlit_app.py Syntax Valid", test_syntax)

# 12. Retraining blocked in backend
def test_retrain_blocked():
    src = Path("backend/app.py").read_text(encoding="utf-8")
    assert "403" in src, "HTTP 403 not found in backend"
    assert "disabled" in src.lower(), "Disabled message not found"
check("Web Retraining Blocked (HTTP 403 in backend)", test_retrain_blocked)

# 13. Theme toggle in app.js
def test_theme_js():
    src = Path("frontend/js/app.js").read_text(encoding="utf-8")
    assert "localStorage" in src
    assert "themeToggleBtn" in src
    assert "data-theme" in src
check("Theme Toggle JS (localStorage + data-theme)", test_theme_js)

# 14. Light/dark mode CSS vars in style.css
def test_css_theme():
    src = Path("frontend/css/style.css").read_text(encoding="utf-8")
    assert "data-theme" in src
check("CSS Theme Variables (data-theme)", test_css_theme)

# 15. Streamlit retraining tab removed
def test_streamlit_retraining_removed():
    src = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "GPU Retraining" not in src, "GPU Retraining tab must be removed from streamlit_app.py"
check("Streamlit Retraining Tab Removed", test_streamlit_retraining_removed)

# --- Print Results ---
print("\n" + "="*55)
print("  DentalVision AI — Feature Smoke Test Results")
print("="*55)
for r in results:
    print(r)
passed = sum(1 for r in results if "[PASS]" in r)
failed = sum(1 for r in results if "[FAIL]" in r)
print("="*55)
print(f"  TOTAL: {passed} passed, {failed} failed out of {len(results)} checks")
print("="*55)
if failed > 0:
    sys.exit(1)
