"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for Vehicle Damage Detection.
Generates heatmap visualizations explaining model predictions by highlighting key damaged regions.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from src import config


def find_target_conv_layer(model: Model) -> str:
    """
    Dynamically finds the last 4D convolutional layer in the model architecture.
    """
    # Check known MobileNetV2 last conv layer names first
    target_names = ["Conv_1", "out_relu", "top_conv"]

    # Search through sub-models or flat layer list
    for layer in reversed(model.layers):
        if layer.name in target_names:
            return layer.name
        # If layer is a submodel (like base MobileNetV2)
        if isinstance(layer, Model):
            for sub_layer in reversed(layer.layers):
                if sub_layer.name in target_names or len(getattr(sub_layer, "output_shape", ())) == 4:
                    return sub_layer.name

    # Fallback search for any 4D output layer
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            if len(output_shape) == 4:
                return layer.name
        except Exception:
            continue

    # Default MobileNetV2 layer name
    return "Conv_1"


def compute_gradcam_heatmap(
    model: Model,
    img_array: np.ndarray,
    pred_index: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """
    Computes Grad-CAM heatmap array for a given input image and model.

    Args:
        model: Trained Keras classification model.
        img_array: Preprocessed image batch of shape (1, H, W, C).
        pred_index: Index of target class score. If None, uses top predicted class.
        layer_name: Name of target convolutional layer. If None, auto-detected.

    Returns:
        2D heatmap array normalized between 0.0 and 1.0.
    """
    if layer_name is None:
        layer_name = find_target_conv_layer(model)

    # Build a functional sub-model mapping input -> (target_conv_output, model_predictions)
    grad_model = None

    # Try direct extraction if layer is at top level
    try:
        target_layer = model.get_layer(layer_name)
        grad_model = Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
    except Exception:
        # If target layer resides inside a nested base model (e.g. MobileNetV2)
        for layer in model.layers:
            if isinstance(layer, Model):
                try:
                    target_layer = layer.get_layer(layer_name)
                    # Reconstruct forward graph
                    sub_grad_model = Model(inputs=layer.inputs, outputs=target_layer.output)
                    # Connect top head
                    x = sub_grad_model(model.inputs)
                    # Find downstream layers after base model
                    head_layers = model.layers[model.layers.index(layer) + 1 :]
                    y = x
                    for h_layer in head_layers:
                        y = h_layer(y)
                    grad_model = Model(inputs=model.inputs, outputs=[x, y])
                    break
                except Exception:
                    continue

    if grad_model is None:
        # Final resilient fallback: build feature extraction sub-model
        target_layer = model.get_layer(index=-4)
        grad_model = Model(inputs=model.inputs, outputs=[target_layer.output, model.output])

    # Record gradients using tf.GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of target class score w.r.t target conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Channel-wise mean intensity of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply feature map channels by gradient importance weights
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain positive activation regions and normalize
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_gradcam(
    original_img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays Grad-CAM heatmap onto the original RGB image.

    Args:
        original_img_rgb: Original image as uint8 RGB numpy array (H, W, 3).
        heatmap: 2D float heatmap normalized [0, 1].
        alpha: Heatmap blend ratio (0.0 = only image, 1.0 = only heatmap).
        colormap: OpenCV colormap enum.

    Returns:
        Tuple of (blended_rgb_image, colored_heatmap_rgb)
    """
    h, w = original_img_rgb.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to 0-255 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply OpenCV color map (returns BGR)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Convert original RGB image to float/uint8 for blending
    original_uint8 = np.uint8(original_img_rgb)

    # Blend images
    blended = cv2.addWeighted(original_uint8, 1.0 - alpha, heatmap_rgb, alpha, 0)

    return blended, heatmap_rgb
