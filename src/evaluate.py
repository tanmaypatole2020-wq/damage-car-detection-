"""
Evaluation script for Vehicle Damage Detection model.
Loads trained model, computes accuracy, classification metrics report, and confusion matrix visual.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.data_loader import load_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path = config.CONFUSION_MATRIX_PATH,
) -> None:
    """
    Renders and saves a polished Confusion Matrix heatmap plot.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    display_names = [config.CLASS_DETAILS.get(c, {}).get("display_name", c) for c in class_names]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=display_names,
        yticklabels=display_names,
        title="Confusion Matrix - Vehicle Damage Detection",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate matrix values
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
                fontsize=12,
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved confusion matrix plot to {save_path}")


def evaluate_model() -> None:
    """
    Loads saved model, evaluates performance metrics on validation set, and generates reports.
    """
    if not config.MODEL_PATH.exists():
        logging.error(f"Model file not found at {config.MODEL_PATH}. Train the model first.")
        print(f"Model file not found at {config.MODEL_PATH}. Please run python src/train.py first.")
        return

    logging.info(f"Loading trained model from {config.MODEL_PATH}...")
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        custom_objs = {"preprocess_input": preprocess_input}
        model = tf.keras.models.load_model(config.MODEL_PATH, custom_objects=custom_objs)
    except Exception:
        model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)

    _, val_ds, class_names = load_datasets()

    logging.info("Evaluating model on validation dataset...")
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(pred_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    display_names = [config.CLASS_DETAILS.get(c, {}).get("display_name", c) for c in class_names]

    # Classification Report
    report_str = classification_report(y_true, y_pred, target_names=display_names, digits=4)
    print("\n" + "=" * 60)
    print("CLASSIFICATION EVALUATION REPORT")
    print("=" * 60)
    print(report_str)
    print("=" * 60)

    # Save report to text file
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORT_TEXT_PATH, "w") as f:
        f.write("VEHICLE DAMAGE DETECTION - MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report_str + "\n")
    logging.info(f"Saved classification report text to {config.REPORT_TEXT_PATH}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    evaluate_model()
