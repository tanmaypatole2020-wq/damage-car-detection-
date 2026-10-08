"""
Configuration parameters for Vehicle Damage Detection.
Centralizes paths, hyperparameters, class labels, and UI display configurations.
Supports cross-platform path handling for Windows, macOS, and Linux.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Project Directories (Cross-platform using pathlib.Path)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR: Path = PROJECT_ROOT / "data"
TRAIN_DIR: Path = DATA_DIR / "train"
VAL_DIR: Path = DATA_DIR / "val"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "vehicle_damage_model.keras"
CLASS_NAMES_PATH: Path = MODELS_DIR / "class_names.json"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CONFUSION_MATRIX_PATH: Path = REPORTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH: Path = REPORTS_DIR / "training_history.png"
REPORT_TEXT_PATH: Path = REPORTS_DIR / "classification_report.txt"

SAMPLES_DIR: Path = PROJECT_ROOT / "samples"

# Model & Image Configurations
IMAGE_SIZE: Tuple[int, int] = (224, 224)
INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)
BATCH_SIZE: int = 16
RANDOM_SEED: int = 42

# Training Hyperparameters
EPOCHS_INITIAL: int = 10
EPOCHS_FINE_TUNE: int = 10
INITIAL_LEARNING_RATE: float = 1e-3
FINE_TUNE_LEARNING_RATE: float = 1e-5

# 3 Target Classes (Alphabetical order as inferred by image_dataset_from_directory)
CLASS_NAMES: List[str] = ["minor_damage", "no_damage", "severe_damage"]

# Visual & Business Information Mapping
CLASS_DETAILS: Dict[str, Dict[str, str]] = {
    "no_damage": {
        "display_name": "No Damage",
        "badge_color": "#22c55e",  # Green
        "bg_color": "#f0fdf4",
        "border_color": "#86efac",
        "severity": "No Damage",
        "summary": "No visible exterior damage detected on vehicle panels or bodywork.",
        "action": "Vehicle appearance and body condition are optimal. No repair required."
    },
    "minor_damage": {
        "display_name": "Minor Damage",
        "badge_color": "#f59e0b",  # Amber/Yellow
        "bg_color": "#fffbeb",
        "border_color": "#fde68a",
        "severity": "Minor Damage",
        "summary": "Minor surface scratches, small paint scuffs, or shallow dents detected.",
        "action": "Cosmetic detailing, paintless dent repair, or touch-up recommended."
    },
    "severe_damage": {
        "display_name": "Severe Damage",
        "badge_color": "#ef4444",  # Red
        "bg_color": "#fef2f2",
        "border_color": "#fca5a5",
        "severity": "Severe Damage",
        "summary": "Significant structural deformation, crushed body panels, or collision impact detected.",
        "action": "Professional collision repair shop inspection and insurance appraisal required."
    }
}

# Image upload validation settings
MAX_IMAGE_SIZE_MB: int = 10
ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
MIN_IMAGE_DIMENSION: int = 32
CONFIDENCE_THRESHOLD: float = 0.50

# Live Demo & Tunnel Settings
DEFAULT_DEMO_SUBDOMAIN: str = "vehicle-damage-ai"
TUNNEL_INFO_PATH: Path = PROJECT_ROOT / "static" / "tunnel_info.json"

