"""Direct function test of the Vercel API (no HTTP server needed)"""
import sys, json, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util
spec = importlib.util.spec_from_file_location('vercel_api', 'api/index.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results = []
def check(name, fn):
    try:
        fn()
        results.append(f"  [PASS] {name}")
    except Exception as e:
        results.append(f"  [FAIL] {name}: {e}")

def test_module_loads():
    assert hasattr(mod, 'app')
    assert hasattr(mod, 'analyze_radiograph')
    assert hasattr(mod, 'build_report')
    assert hasattr(mod, 'enhance_clahe')
    assert hasattr(mod, 'generate_heatmap_b64')
check("Vercel API module loads", test_module_loads)

def test_routes():
    paths = [r.path for r in mod.app.routes if hasattr(r, 'path')]
    for ep in ['/api/health', '/api/model-info', '/api/samples', '/api/predict', '/api/training-status', '/api/retrain']:
        assert ep in paths, f"Missing route: {ep}"
check("All API routes registered", test_routes)

def test_clahe():
    img = np.random.randint(80, 200, (512, 1024, 3), dtype=np.uint8)
    out = mod.enhance_clahe(img)
    assert out.shape == img.shape
check("CLAHE enhancement", test_clahe)

def test_analyze():
    img = np.random.randint(80, 200, (512, 1024, 3), dtype=np.uint8)
    r = mod.analyze_radiograph(img)
    for k in ('caries', 'periapical', 'periodontitis'):
        assert 0 <= r[k] <= 1, f"{k} out of range: {r[k]}"
check("Radiograph analysis (random)", test_analyze)

def test_analyze_sample():
    sample = Image.open('public/static/samples/sample_1.jpg').convert('RGB')
    arr = np.array(sample.resize((1024, 512)))
    arr = mod.enhance_clahe(arr)
    r = mod.analyze_radiograph(arr)
    # Print for visibility
    results.append(f"         → Sample1: Caries={r['caries']:.2f} Peri={r['periapical']:.2f} Bone={r['periodontitis']:.2f}")
check("Sample 1 analysis (OPG image)", test_analyze_sample)

def test_report_caries():
    r = mod.build_report([0.89, 0.14, 0.12], 21.0)
    assert r['diseases'][0]['badge'] in ('DETECTED', 'SUSPICIOUS')
    assert r['diseases'][0]['probability'] == 89
    assert len(r['diseases']) == 3
check("Report builder — Caries dominant", test_report_caries)

def test_report_periapical():
    r = mod.build_report([0.18, 0.86, 0.15], 19.0)
    assert r['diseases'][1]['badge'] in ('DETECTED', 'SUSPICIOUS')
check("Report builder — Periapical dominant", test_report_periapical)

def test_report_healthy():
    r = mod.build_report([0.1, 0.12, 0.08], 18.0)
    assert all(d['badge'] == 'CLEAR' for d in r['diseases'])
    assert 'No Significant' in r['overall_health']
check("Report builder — Healthy scan", test_report_healthy)

def test_heatmap():
    img = np.random.randint(80, 200, (256, 512, 3), dtype=np.uint8)
    for layer in ('composite', 'caries', 'periapical', 'periodontitis'):
        b64 = mod.generate_heatmap_b64(img, layer)
        assert b64.startswith('data:image/jpeg;base64,'), f"Bad b64 for {layer}"
check("Grad-CAM heatmap generation (all 4 layers)", test_heatmap)

def test_disease_metadata():
    assert len(mod.DISEASE_METADATA) == 3
    names = [d['name'] for d in mod.DISEASE_METADATA]
    assert 'Dental Caries' in names
    assert 'Periapical Lesion' in names
    assert 'Periodontitis' in names
check("Disease metadata (3 diseases)", test_disease_metadata)

def test_retrain_blocked():
    src = open('api/index.py').read()
    assert '403' in src
    assert 'disabled by policy' in src
check("Retrain endpoint blocked (403)", test_retrain_blocked)

print("\n" + "="*55)
print("  Vercel API — Direct Function Tests")
print("="*55)
for r in results:
    print(r)
passed = sum(1 for r in results if '[PASS]' in r)
failed = sum(1 for r in results if '[FAIL]' in r)
print("="*55)
print(f"  TOTAL: {passed} passed, {failed} failed")
print("="*55)
sys.exit(0 if failed == 0 else 1)
