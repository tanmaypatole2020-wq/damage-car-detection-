"""
Training script for Vehicle Damage Detection.
Performs 2-stage transfer learning (initial feature extraction head training followed by fine-tuning).
Auto-generates dummy dataset if data directory is empty.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import tensorflow as tf

from src import config
from src.data_loader import load_datasets, verify_dataset_structure
from src.model import build_model, compile_model, unfreeze_top_layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def ensure_data_available() -> None:
    """Checks if dataset exists; if empty, automatically generates synthetic dummy dataset."""
    valid, msg = verify_dataset_structure(config.TRAIN_DIR, config.VAL_DIR)
    if not valid:
        logging.warning("Data directory is empty or incomplete. Auto-generating dummy images for 3 classes to test the pipeline...")
        try:
            from generate_dummy_data import generate_dummy_dataset
            generate_dummy_dataset(train_per_class=6, val_per_class=3)
            logging.info("Auto-generated dummy dataset successfully.")
        except Exception as e:
            logging.error(f"Failed to auto-generate dummy dataset: {e}")


def plot_training_history(history_initial: tf.keras.callbacks.History, history_fine: Any = None) -> None:
    """
    Plots and saves training & validation accuracy/loss curves.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    acc = history_initial.history.get("accuracy", [])
    val_acc = history_initial.history.get("val_accuracy", [])
    loss = history_initial.history.get("loss", [])
    val_loss = history_initial.history.get("val_loss", [])

    if history_fine and hasattr(history_fine, "history"):
        acc += history_fine.history.get("accuracy", [])
        val_acc += history_fine.history.get("val_accuracy", [])
        loss += history_fine.history.get("loss", [])
        val_loss += history_fine.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", linewidth=2)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#16a34a", linewidth=2, linestyle="--")
    if history_fine:
        ax1.axvline(x=len(history_initial.history["accuracy"]), color="gray", linestyle=":", label="Fine-Tuning Start")
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Loss Plot
    ax2.plot(epochs_range, loss, label="Training Loss", color="#dc2626", linewidth=2)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#ea580c", linewidth=2, linestyle="--")
    if history_fine:
        ax2.axvline(x=len(history_initial.history["loss"]), color="gray", linestyle=":", label="Fine-Tuning Start")
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.TRAINING_HISTORY_PATH, dpi=300)
    plt.close()
    logging.info(f"Saved training curves plot to {config.TRAINING_HISTORY_PATH}")


def train_model(epochs_initial: int = config.EPOCHS_INITIAL, epochs_fine: int = config.EPOCHS_FINE_TUNE) -> None:
    """
    Main training execution pipeline.
    """
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    # Step 1: Ensure dataset is available (auto-generate dummy dataset if empty)
    ensure_data_available()

    # Step 2: Load Data
    train_ds, val_ds, class_names = load_datasets()

    # Step 3: Build & Compile Model
    logging.info("Building MobileNetV2 Transfer Learning Model...")
    model = build_model(num_classes=len(class_names))
    model = compile_model(model, learning_rate=config.INITIAL_LEARNING_RATE)

    # Callbacks setup
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    # Stage 1: Feature Extraction (Frozen Base)
    logging.info(f"\n--- STAGE 1: Training Classification Head for {epochs_initial} Epochs ---")
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_initial,
        callbacks=callbacks,
    )

    # Stage 2: Fine-Tuning Top Layers (if requested)
    history_fine = None
    if epochs_fine > 0:
        logging.info(f"\n--- STAGE 2: Fine-Tuning Top MobileNetV2 Layers for {epochs_fine} Epochs ---")
        model = unfreeze_top_layers(model, num_layers_to_unfreeze=30)
        model = compile_model(model, learning_rate=config.FINE_TUNE_LEARNING_RATE)

        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_initial + epochs_fine,
            initial_epoch=history_initial.epoch[-1] + 1,
            callbacks=callbacks,
        )

    # Save final model state
    model.save(config.MODEL_PATH)
    logging.info(f"Model successfully saved to {config.MODEL_PATH}")

    # Plot metrics
    plot_training_history(history_initial, history_fine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vehicle Damage Detection Model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_INITIAL, help="Initial training epochs")
    parser.add_argument("--fine-epochs", type=int, default=config.EPOCHS_FINE_TUNE, help="Fine-tuning epochs")
    args = parser.parse_args()

    train_model(epochs_initial=args.epochs, epochs_fine=args.fine_epochs)
