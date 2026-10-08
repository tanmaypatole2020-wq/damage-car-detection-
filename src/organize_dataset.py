"""
Dataset organization utility script for Binary Vehicle Damage Detection (Damaged vs Non-Damaged).
Organizes raw downloaded image directories into data/train and data/val folder structure.
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LABEL_KEYWORDS = {
    "no_damage": ["no_damage", "nodamage", "clean", "intact", "normal", "undamaged", "00-damage"],
    "damaged": ["damaged", "damage", "minor", "severe", "scratch", "dent", "scuff", "crash", "wreck", "broken"],
}


def categorize_filename_or_folder(name: str) -> str:
    """Infers target binary class label ('no_damage' or 'damaged')."""
    name_lower = name.lower()
    if any(kw in name_lower for kw in LABEL_KEYWORDS["no_damage"]):
        return "no_damage"
    return "damaged"


def organize_dataset(source_dir: Path, val_split: float = 0.2, seed: int = config.RANDOM_SEED) -> None:
    """Organizes images from source directory into data/train and data/val binary folders."""
    if not source_dir.exists():
        logging.error(f"Source directory '{source_dir}' does not exist.")
        return

    random.seed(seed)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_files: List[Path] = [p for p in source_dir.rglob("*") if p.suffix.lower() in valid_exts]

    if not image_files:
        logging.warning(f"No image files found in {source_dir}")
        return

    logging.info(f"Found {len(image_files)} image files in source directory.")

    class_groups: dict = {c: [] for c in config.CLASS_NAMES}
    for img_path in image_files:
        text_to_check = f"{img_path.parent.name}_{img_path.name}"
        label = categorize_filename_or_folder(text_to_check)
        class_groups[label].append(img_path)

    for split in ["train", "val"]:
        for cls_name in config.CLASS_NAMES:
            (config.DATA_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for cls_name, files in class_groups.items():
        random.shuffle(files)
        val_count = int(len(files) * val_split)
        val_files = files[:val_count]
        train_files = files[val_count:]

        logging.info(f"Class '{cls_name}': {len(train_files)} Train images, {len(val_files)} Val images.")

        for p in train_files:
            shutil.copy2(p, config.TRAIN_DIR / cls_name / p.name)
            total_copied += 1

        for p in val_files:
            shutil.copy2(p, config.VAL_DIR / cls_name / p.name)
            total_copied += 1

    logging.info(f"Dataset organization complete! Copied {total_copied} images into {config.DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize vehicle damage dataset into binary train/val structure")
    parser.add_argument("--source", type=str, required=True, help="Path to raw extracted dataset directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation ratio")
    args = parser.parse_args()

    organize_dataset(source_dir=Path(args.source), val_split=args.val_split)
