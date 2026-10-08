"""
Data loader module for Vehicle Damage Detection.
Loads dataset directories, applies data augmentation, and creates optimized tf.data.Dataset pipelines.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras import layers, Sequential

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def verify_dataset_structure(train_dir: Path = config.TRAIN_DIR, val_dir: Path = config.VAL_DIR) -> Tuple[bool, str]:
    """
    Checks if dataset directories exist and contain images for all required classes.
    """
    missing_info = []

    if not train_dir.exists():
        missing_info.append(f"Training directory missing: {train_dir}")
    if not val_dir.exists():
        missing_info.append(f"Validation directory missing: {val_dir}")

    if missing_info:
        msg = (
            "\n" + "=" * 70 + "\n"
            "DATASET DIRECTORY MISSING!\n"
            "Expected Folder Layout:\n"
            "  data/\n"
            "    ├── train/\n"
            "    │   ├── minor_damage/\n"
            "    │   ├── no_damage/\n"
            "    │   └── severe_damage/\n"
            "    └── val/\n"
            "        ├── minor_damage/\n"
            "        ├── no_damage/\n"
            "        └── severe_damage/\n\n"
            "Tip: You can download the Kaggle Car Damage dataset or run generate_dummy_data.py.\n"
            "=" * 70
        )
        return False, msg

    # Check if all class subdirectories exist and contain images
    for class_name in config.CLASS_NAMES:
        t_sub = train_dir / class_name
        v_sub = val_dir / class_name
        t_count = len(list(t_sub.glob("*.*"))) if t_sub.exists() else 0
        v_count = len(list(v_sub.glob("*.*"))) if v_sub.exists() else 0

        if t_count == 0 or v_count == 0:
            msg = f"Class folder '{class_name}' lacks images (Train: {t_count}, Val: {v_count})."
            return False, msg

    return True, "Dataset structure verified successfully."


def get_data_augmentation() -> tf.keras.Sequential:
    """
    Returns data augmentation pipeline using Keras Preprocessing Layers.
    """
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    return Sequential(
        [
            layers.RandomFlip("horizontal", seed=config.RANDOM_SEED),
            layers.RandomRotation(0.15, seed=config.RANDOM_SEED),
            layers.RandomZoom(0.1, seed=config.RANDOM_SEED),
            layers.RandomContrast(0.1, seed=config.RANDOM_SEED),
        ],
        name="data_augmentation",
    )


def load_datasets(
    train_dir: Path = config.TRAIN_DIR,
    val_dir: Path = config.VAL_DIR,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
    batch_size: int = config.BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Loads train and validation tf.data.Dataset instances from disk.
    """
    valid, msg = verify_dataset_structure(train_dir, val_dir)
    if not valid:
        logging.error(msg)
        raise FileNotFoundError(msg)

    logging.info(f"Loading training dataset from {train_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=config.RANDOM_SEED,
    )

    logging.info(f"Loading validation dataset from {val_dir}...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=config.RANDOM_SEED,
    )

    class_names = train_ds.class_names
    logging.info(f"Inferred class names: {class_names}")

    # Ensure output models directory exists and save class names mapping
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=4)

    # Optimize datasets performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(buffer_size=1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names
