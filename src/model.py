"""
Model architecture definition for Vehicle Damage Detection.
Uses MobileNetV2 pre-trained on ImageNet with Transfer Learning and Fine-Tuning options.
"""

import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src import config
from src.data_loader import get_data_augmentation


def build_model(
    num_classes: int = len(config.CLASS_NAMES),
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    use_augmentation: bool = True,
) -> Model:
    """
    Builds a MobileNetV2 transfer learning classification model.

    Args:
        num_classes: Number of output target classes.
        input_shape: Target input image dimensions (height, width, channels).
        use_augmentation: Whether to include data augmentation layers at input.

    Returns:
        Compiled Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Optional Augmentation layer
    if use_augmentation:
        augmentation = get_data_augmentation()
        x = augmentation(inputs)
    else:
        x = inputs

    # Preprocessing layer expected by MobileNetV2 (scales pixels from [0, 255] to [-1, 1])
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(x)

    # Base MobileNetV2 feature extractor
    base_model = MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze base feature extractor initially

    # Classification Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="head_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.2, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="VehicleDamageDetector")

    return model


def unfreeze_top_layers(model: Model, num_layers_to_unfreeze: int = 30) -> Model:
    """
    Unfreezes the top N layers of the MobileNetV2 base model for fine-tuning.

    Args:
        model: Trained Keras model containing MobileNetV2 base.
        num_layers_to_unfreeze: Number of top layers to unfreeze.

    Returns:
        Model with fine-tuning layer permissions updated.
    """
    # Locate the MobileNetV2 base model or search inner layers
    mobilenet_layer = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            mobilenet_layer = layer
            break

    if mobilenet_layer and hasattr(mobilenet_layer, "layers"):
        mobilenet_layer.trainable = True
        # Freeze initial layers and unfreeze only top N layers
        for layer in mobilenet_layer.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        for layer in mobilenet_layer.layers[-num_layers_to_unfreeze:]:
            layer.trainable = False if isinstance(layer, layers.BatchNormalization) else True
    else:
        # Fallback if layers flat
        model.trainable = True
        for layer in model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

    return model


def compile_model(model: Model, learning_rate: float = config.INITIAL_LEARNING_RATE) -> Model:
    """
    Compiles the model with Adam optimizer and Sparse Categorical Crossentropy.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
