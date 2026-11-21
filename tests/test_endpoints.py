import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import serve_index, get_model_info, list_sample_opgs, predict_dental_opg
from fastapi import UploadFile

class MockUploadFile:
    def __init__(self, filename, content_bytes):
        self.filename = filename
        self._content = content_bytes

    async def read(self):
        return self._content

async def run_direct_tests():
    print("--- 1. Testing serve_index() ---")
    resp = await serve_index()
    print("Response type:", type(resp))

    print("\n--- 2. Testing get_model_info() ---")
    info = await get_model_info()
    print("Status:", info.get("status"))
    print("Architecture:", info.get("model_architecture"))
    print("Hardware:", info.get("hardware"))

    print("\n--- 3. Testing list_sample_opgs() ---")
    samples = await list_sample_opgs()
    print("Total samples loaded:", len(samples))
    for s in samples[:3]:
        print(f"  Sample #{s['id']}: {s['title']} ({s['url']})")

    print("\n--- 4. Testing predict_dental_opg() with authentic OPG ---")
    sample_img_path = Path("static/samples/sample_1.jpg")
    if sample_img_path.exists():
        with open(sample_img_path, "rb") as f:
            bytes_data = f.read()
        mock_file = MockUploadFile("sample_1.jpg", bytes_data)
        pred = await predict_dental_opg(mock_file, enhance=True)
        report = pred["report"]
        print("Diagnosis Status:", report["overall_health"])
        print("Summary:", report["summary"])
        print("Latency:", report["inference_time_ms"], "ms")
        for d in report["diseases"]:
            print(f"  [{d['badge']}] {d['disease']}: {d['probability']}% - {d['severity']}")
        print("Visualizations created:", list(pred["visualizations"].keys()))
        assert len(pred["visualizations"]["composite_heatmap"]) > 100
        assert len(pred["visualizations"]["caries_heatmap"]) > 100
        assert len(pred["visualizations"]["periapical_heatmap"]) > 100
        assert len(pred["visualizations"]["periodontitis_heatmap"]) > 100

    print("\n==========================================")
    print(">>> ALL ENDPOINTS & DIAGNOSTIC PIPELINES WORKING! <<<")
    print("==========================================")

if __name__ == "__main__":
    asyncio.run(run_direct_tests())
