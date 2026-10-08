"""
Utility script to generate synthetic dummy images for testing and pipeline verification.
Creates sample training and validation images across all 3 target classes:
- no_damage
- minor_damage
- severe_damage
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import cv2

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_synthetic_image(class_name: str, seed: int = 42) -> np.ndarray:
    """
    Draws a synthetic vehicle-like image with class-specific characteristics.
    """
    np.random.seed(seed)

    # Base car background
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    img[:, :, 0] = np.clip(img[:, :, 0] + np.random.randint(-20, 20, (224, 224)), 0, 255)

    # Basic vehicle body outline
    cv2.rectangle(img, (30, 80), (194, 170), (160, 160, 160), -1)  # Main body
    cv2.rectangle(img, (60, 40), (164, 80), (120, 140, 160), -1)   # Roof/Windows
    cv2.circle(img, (65, 170), 22, (30, 30, 30), -1)               # Left Wheel
    cv2.circle(img, (159, 170), 22, (30, 30, 30), -1)              # Right Wheel

    if class_name == "minor_damage":
        # Draw minor scratch line & small paint scuff
        cv2.line(img, (70, 110), (140, 130), (50, 50, 50), 3)
        cv2.line(img, (72, 112), (138, 128), (220, 220, 220), 1)
        cv2.ellipse(img, (100, 120), (15, 8), 15, 0, 360, (80, 80, 80), -1)

    elif class_name == "severe_damage":
        # Draw jagged collision dent & crushed impact zone
        pts = np.array([[40, 70], [90, 140], [130, 90], [170, 160], [110, 165], [30, 120]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (40, 40, 40))
        cv2.polylines(img, [pts], True, (255, 50, 50), 3)
        cv2.circle(img, (100, 130), 25, (0, 0, 180), -1)

    # Return RGB image
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def generate_dummy_dataset(train_per_class: int = 6, val_per_class: int = 3) -> None:
    """
    Generates dummy dataset images in data/train and data/val folders for 3 classes.
    """
    logging.info("Generating synthetic dummy dataset for 3 classes...")

    idx = 0
    for split, count in [("train", train_per_class), ("val", val_per_class)]:
        for class_name in config.CLASS_NAMES:
            target_dir = config.DATA_DIR / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                idx += 1
                img_array = create_synthetic_image(class_name, seed=idx)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                file_path = target_dir / f"dummy_{class_name}_{i+1:02d}.jpg"
                cv2.imwrite(str(file_path), img_bgr)

    logging.info(f"Successfully generated dummy dataset at {config.DATA_DIR}")


def cleanup_dummy_dataset() -> None:
    """
    Removes generated dummy image files from data directory.
    """
    if config.DATA_DIR.exists():
        import shutil
        shutil.rmtree(config.DATA_DIR)
        logging.info("Cleaned up dataset directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for vehicle damage detection")
    parser.add_argument("--cleanup", action="store_true", help="Clean up generated dataset")
    args = parser.parse_args()

    if args.cleanup:
        cleanup_dummy_dataset()
    else:
        generate_dummy_dataset()
