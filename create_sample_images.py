"""
Generates 5 distinct sample images in samples/ directory for testing all edge cases:
1. damaged_car.jpg (Vehicle with visible collision damage)
2. undamaged_car.jpg (Clean, intact vehicle)
3. car_alpha.png (PNG image with transparent RGBA alpha channel)
4. tiny_car.jpg (Very small 32x32 pixel image)
5. non_car.jpg (Non-vehicle natural landscape/abstract object)
"""

from pathlib import Path
import numpy as np
import cv2
from PIL import Image

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def draw_car_base(width: int = 224, height: int = 224) -> np.ndarray:
    """Draws a base car silhouette."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 210
    # Road
    cv2.rectangle(img, (0, int(height * 0.8)), (width, height), (70, 70, 70), -1)
    # Car Body
    cv2.rectangle(img, (int(width * 0.15), int(height * 0.45)), (int(width * 0.85), int(height * 0.78)), (160, 160, 170), -1)
    # Cabin / Roof
    cv2.rectangle(img, (int(width * 0.3), int(height * 0.25)), (int(width * 0.7), int(height * 0.45)), (110, 130, 150), -1)
    # Windows
    cv2.rectangle(img, (int(width * 0.34), int(height * 0.28)), (int(width * 0.48), int(height * 0.42)), (200, 230, 255), -1)
    cv2.rectangle(img, (int(width * 0.52), int(height * 0.28)), (int(width * 0.66), int(height * 0.42)), (200, 230, 255), -1)
    # Wheels
    r = int(height * 0.1)
    cv2.circle(img, (int(width * 0.3), int(height * 0.78)), r, (30, 30, 30), -1)
    cv2.circle(img, (int(width * 0.7), int(height * 0.78)), r, (30, 30, 30), -1)
    return img


# 1. Damaged Car Image
def make_damaged_car():
    img = draw_car_base(224, 224)
    # Add dent and collision lines
    pts = np.array([[40, 100], [90, 160], [130, 110]], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (40, 40, 40))
    cv2.polylines(img, [pts], True, (0, 0, 220), 3)
    cv2.line(img, (60, 120), (140, 140), (20, 20, 20), 4)
    cv2.imwrite(str(SAMPLES_DIR / "damaged_car.jpg"), img)


# 2. Undamaged Car Image
def make_undamaged_car():
    img = draw_car_base(224, 224)
    cv2.imwrite(str(SAMPLES_DIR / "undamaged_car.jpg"), img)


# 3. PNG with Alpha Channel
def make_car_alpha():
    img = draw_car_base(224, 224)
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # Make top-left and top-right transparent
    rgba[:50, :50, 3] = 0
    rgba[:50, -50:, 3] = 0
    cv2.imwrite(str(SAMPLES_DIR / "car_alpha.png"), rgba)


# 4. Tiny Car Image (32x32)
def make_tiny_car():
    img = draw_car_base(32, 32)
    cv2.imwrite(str(SAMPLES_DIR / "tiny_car.jpg"), img)


# 5. Non-car image (Landscape sunset / abstract gradient)
def make_non_car():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        img[y, :, 0] = int(255 * (y / 224))  # Blue gradient
        img[y, :, 1] = int(180 * (1 - y / 224))
        img[y, :, 2] = int(240 * (1 - y / 224))  # Sunset orange/red
    cv2.circle(img, (112, 112), 40, (0, 240, 255), -1)  # Sun
    cv2.imwrite(str(SAMPLES_DIR / "non_car.jpg"), img)


if __name__ == "__main__":
    make_damaged_car()
    make_undamaged_car()
    make_car_alpha()
    make_tiny_car()
    make_non_car()
    print("Generated 5 sample images in samples/ directory.")
