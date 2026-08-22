"""
Verification test for Manga Color Studio LoRA & Dataset endpoints.
"""
import sys
from pathlib import Path

# Add root directory to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.server import app, load_loras_config

client = TestClient(app)

def test_api_loras_get():
    resp = client.get("/api/loras")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "loras" in data, "No 'loras' field in response"
    print(f"[PASS] GET /api/loras returned {len(data['loras'])} LoRAs")
    for lora in data['loras']:
        print(f"  - {lora['name']} ({lora.get('filename')}) scale={lora['scale']} enabled={lora['enabled']}")

def test_api_datasets_status():
    resp = client.get("/api/datasets/status")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "loras" in data and "models" in data, "Dataset status structure mismatch"
    print(f"[PASS] GET /api/datasets/status:")
    print(f"  - Loras Dataset Exists: {data['loras']['exists']} ({data['loras']['dataset_id']})")
    print(f"  - Models Dataset Exists: {data['models']['exists']} ({data['models']['dataset_id']})")

def test_api_lora_update():
    resp = client.post("/api/loras/update", json={"id": "color_manga", "scale": 1.25, "enabled": True})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    updated = resp.json()["lora"]
    assert updated["scale"] == 1.25, f"Expected scale 1.25, got {updated['scale']}"
    print(f"[PASS] POST /api/loras/update updated scale to 1.25")

    # Reset back to 1.0
    client.post("/api/loras/update", json={"id": "color_manga", "scale": 1.00, "enabled": True})

def test_notebook_build():
    from scripts.make_notebook import build_notebook
    loras = load_loras_config()
    out = build_notebook(
        loras=loras,
        max_dim=1440,
        num_steps=8,
        guidance_scale=1.5,
        prompt="masterpiece, vibrant shonen anime style, dynamic lighting",
        negative_prompt="blurry, bad anatomy, greyscale"
    )
    assert Path(out).exists(), f"Notebook file {out} was not generated"
    with open(out, "r", encoding="utf-8") as f:
        content = f.read()
        assert "vibrant shonen anime style" in content, "Custom prompt not found in generated notebook"
        assert "blurry, bad anatomy" in content, "Custom negative prompt not found in generated notebook"
    print(f"[PASS] build_notebook verified with custom prompt and negative prompt: {out}")

def test_api_settings():
    resp = client.get("/api/settings")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "kaggle_authenticated" in data, "No 'kaggle_authenticated' field in settings response"
    print(f"[PASS] GET /api/settings: authenticated={data['kaggle_authenticated']} user=@{data['kaggle_username']}")

if __name__ == "__main__":
    print("Running verification suite...")
    test_api_settings()
    test_api_loras_get()
    test_api_datasets_status()
    test_api_lora_update()
    test_notebook_build()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
