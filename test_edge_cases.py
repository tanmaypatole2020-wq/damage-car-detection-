"""
QA Test runner: Verifies inference and edge cases on 5 distinct test images:
1. Damaged car image
2. Undamaged car image
3. PNG with Alpha Transparency
4. Tiny image (32x32)
5. Non-car image
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import DamagePredictor

def test_samples():
    predictor = DamagePredictor()
    assert predictor.is_ready(), "Model failed to load!"

    samples = [
        "samples/damaged_car.jpg",
        "samples/undamaged_car.jpg",
        "samples/car_alpha.png",
        "samples/tiny_car.jpg",
        "samples/non_car.jpg",
    ]

    print("\n" + "=" * 60)
    print("RUNNING EDGE CASE INFERENCE TESTS ON 5 SAMPLES")
    print("=" * 60)

    for sample_rel in samples:
        sample_path = PROJECT_ROOT / sample_rel
        assert sample_path.exists(), f"Sample image {sample_path} does not exist!"

        res = predictor.predict(sample_path)
        print(f"\n[PASS] Testing: {sample_rel}")
        print(f"  - Predicted Class : {res['display_name']} ({res['class_name']})")
        print(f"  - Confidence      : {res['confidence_percentage']}")
        print(f"  - Low Confidence? : {res['is_low_confidence']}")
        print(f"  - Probabilities   : {res['probabilities']}")
        print(f"  - Summary         : {res['summary']}")
        print(f"  - Grad-CAM Shape  : {res['gradcam_overlay_rgb'].shape}")

        # Sanity assertions
        assert len(res["probabilities"]) == 3, "Must output probabilities for all 3 classes!"
        assert res["gradcam_overlay_rgb"] is not None, "Grad-CAM overlay cannot be None!"
        assert res["original_rgb"] is not None, "Original RGB array cannot be None!"

    print("\n" + "=" * 60)
    print("ALL 5 SAMPLE IMAGE TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    test_samples()
