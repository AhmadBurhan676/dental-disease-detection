"""Quick test of the Vercel API endpoints"""
import urllib.request, json, io, sys
from pathlib import Path

BASE = "http://127.0.0.1:8001"
PASS, FAIL = 0, 0

def test(name, url, method="GET", data=None, headers=None, expected_key=None, expected_status=200):
    global PASS, FAIL
    try:
        req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        resp = urllib.request.urlopen(req, timeout=10)
        body = json.loads(resp.read())
        status = resp.status
        if expected_key and expected_key not in body:
            print(f"  [FAIL] {name}: missing key '{expected_key}' in response")
            FAIL += 1
        else:
            print(f"  [PASS] {name}: HTTP {status}")
            PASS += 1
    except urllib.error.HTTPError as e:
        if e.code == expected_status:
            print(f"  [PASS] {name}: HTTP {e.code} (expected)")
            PASS += 1
        else:
            print(f"  [FAIL] {name}: HTTP {e.code} - {e.read()[:100]}")
            FAIL += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        FAIL += 1

print("\n" + "="*50)
print("  Vercel API Endpoint Tests")
print("="*50)

test("GET /api/health", f"{BASE}/api/health", expected_key="status")
test("GET /api/model-info", f"{BASE}/api/model-info", expected_key="model_name")
test("GET /api/samples", f"{BASE}/api/samples")
test("GET /api/training-status", f"{BASE}/api/training-status", expected_key="metrics")
test("POST /api/retrain (must be 403)", f"{BASE}/api/retrain",
     method="POST", data=b'{}', headers={"Content-Type":"application/json"},
     expected_status=403)

# Test predict with a real image
sample = Path("public/static/samples/sample_1.jpg")
if sample.exists():
    import urllib.request as ur
    boundary = b"----VercelTestBoundary"
    img_bytes = sample.read_bytes()
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="sample_1.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n" + img_bytes + b"\r\n"
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="enhance"\r\n\r\ntrue\r\n'
        b"--" + boundary + b"--\r\n"
    )
    test("POST /api/predict (sample image)",
         f"{BASE}/api/predict", method="POST", data=body,
         headers={"Content-Type": f"multipart/form-data; boundary=----VercelTestBoundary"},
         expected_key="report")

print("="*50)
print(f"  TOTAL: {PASS} passed, {FAIL} failed")
print("="*50)
sys.exit(0 if FAIL == 0 else 1)
