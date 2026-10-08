"""
Comprehensive Test Script for Vehicle Damage Detection Flask Server.
Tests:
1. GET / (Homepage rendering)
2. POST /predict with damaged car JPG
3. POST /predict with undamaged car JPG
4. POST /predict with PNG (alpha channel)
5. POST /predict with invalid file type (TXT)
6. POST /predict with missing file payload
"""

import io
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app, predictor

def run_tests():
    print("=" * 60)
    print("RUNNING FLASK API & PIPELINE VALIDATION TESTS")
    print("=" * 60)

    assert predictor.is_ready(), "Model is not loaded!"
    client = app.test_client()

    # 1. Test GET /
    print("\n[TEST 1] Testing GET / (Homepage HTML)")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    html = res.get_data(as_text=True)
    assert "Vehicle Damage Detection" in html, "Page title missing"
    assert "style.css" in html, "CSS link missing"
    assert "script.js" in html, "JS link missing"
    print("  -> PASS: Homepage rendered successfully with static assets linked.")

    # 2. Test Damaged Car JPG
    print("\n[TEST 2] Testing POST /predict with damaged_car.jpg")
    img_path = PROJECT_ROOT / "samples" / "damaged_car.jpg"
    with open(img_path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), "damaged_car.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.data}"
    json_data = res.get_json()
    assert json_data["success"] is True
    assert "predicted_class" in json_data
    assert "confidence_percentage" in json_data
    assert "probabilities" in json_data
    assert len(json_data["probabilities"]) == 3
    assert json_data["gradcam_image"].startswith("data:image/jpeg;base64,")
    assert json_data["original_image"].startswith("data:image/jpeg;base64,")
    print(f"  -> PASS: Class='{json_data['predicted_class']}', Conf={json_data['confidence_percentage']}, LowConf={json_data['is_low_confidence']}")

    # 3. Test Undamaged Car JPG
    print("\n[TEST 3] Testing POST /predict with undamaged_car.jpg")
    img_path = PROJECT_ROOT / "samples" / "undamaged_car.jpg"
    with open(img_path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), "undamaged_car.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.data}"
    json_data = res.get_json()
    assert json_data["success"] is True
    assert "predicted_class" in json_data
    print(f"  -> PASS: Class='{json_data['predicted_class']}', Conf={json_data['confidence_percentage']}, LowConf={json_data['is_low_confidence']}")

    # 4. Test PNG with Alpha
    print("\n[TEST 4] Testing POST /predict with car_alpha.png")
    img_path = PROJECT_ROOT / "samples" / "car_alpha.png"
    with open(img_path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), "car_alpha.png")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.data}"
    json_data = res.get_json()
    assert json_data["success"] is True
    print(f"  -> PASS: Class='{json_data['predicted_class']}', Conf={json_data['confidence_percentage']}")

    # 5. Test Invalid File Format (.txt)
    print("\n[TEST 5] Testing POST /predict with invalid file type (test.txt)")
    data = {"image": (io.BytesIO(b"This is not a vehicle image."), "test.txt")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 400, f"Expected 400 for invalid format, got {res.status_code}"
    json_data = res.get_json()
    assert json_data["success"] is False
    assert "error" in json_data
    print(f"  -> PASS: Handled gracefully with message: '{json_data['error']}'")

    # 6. Test Corrupted Image Content with .jpg extension
    print("\n[TEST 6] Testing POST /predict with corrupted JPEG image content")
    data = {"image": (io.BytesIO(b"CORRUPTED_NON_IMAGE_DATA_BYTES_12345"), "broken.jpg")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 400, f"Expected 400 for corrupted image, got {res.status_code}"
    json_data = res.get_json()
    assert json_data["success"] is False
    assert "error" in json_data
    print(f"  -> PASS: Handled gracefully with message: '{json_data['error']}'")

    # 7. Test Missing File Field
    print("\n[TEST 7] Testing POST /predict with empty request body")
    res = client.post("/predict", data={}, content_type="multipart/form-data")
    assert res.status_code == 400, f"Expected 400 for missing file, got {res.status_code}"
    json_data = res.get_json()
    assert json_data["success"] is False
    print(f"  -> PASS: Handled gracefully with message: '{json_data['error']}'")

    print("\n" + "=" * 60)
    print("ALL 7 FLASK API TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_tests()
