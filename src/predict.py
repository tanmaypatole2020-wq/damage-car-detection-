"""
Inference & Prediction module for Vehicle Damage Detection.
Loads model, validates input, performs classification, generates Grad-CAM heatmaps, and formats results.
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from src import config
from src.gradcam import compute_gradcam_heatmap, overlay_gradcam

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DamagePredictor:
    """
    Predictor managing model loading, image preprocessing, classification, and Grad-CAM generation.
    """

    def __init__(self, model_path: Path = config.MODEL_PATH, class_names_path: Path = config.CLASS_NAMES_PATH):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model: Optional[tf.keras.Model] = None
        self.class_names: list = config.CLASS_NAMES
        self.load_model_and_classes()

    def load_model_and_classes(self) -> None:
        """
        Loads trained Keras model and target class names from disk.
        """
        if not self.model_path.exists():
            logging.warning(f"Model file not found at {self.model_path}.")
            return

        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            custom_objs = {"preprocess_input": preprocess_input}
            self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objs)
            logging.info(f"Loaded model successfully from {self.model_path}")
        except Exception:
            try:
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                logging.info(f"Loaded model (compile=False) from {self.model_path}")
            except Exception as e:
                logging.error(f"Failed to load model from {self.model_path}: {e}")
                self.model = None

        if self.class_names_path.exists():
            try:
                with open(self.class_names_path, "r") as f:
                    self.class_names = json.load(f)
                logging.info(f"Loaded class names: {self.class_names}")
            except Exception as e:
                logging.warning(f"Could not parse class names file: {e}")

    def is_ready(self) -> bool:
        """Checks if model is loaded and ready for predictions."""
        return self.model is not None

    def validate_and_load_image(
        self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray]
    ) -> Image.Image:
        """
        Validates input format, size, and corruption, returning a clean PIL RGB image.
        """
        pil_img = None

        # 1. Handle file paths
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            suffix = path.suffix.lower()
            if suffix not in config.ALLOWED_IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Unsupported image format '{suffix}'. Allowed: {', '.join(config.ALLOWED_IMAGE_EXTENSIONS)}"
                )
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > config.MAX_IMAGE_SIZE_MB:
                raise ValueError(f"Image file exceeds maximum limit of {config.MAX_IMAGE_SIZE_MB}MB.")
            try:
                pil_img = Image.open(path)
                pil_img.load()  # Verify integrity
            except Exception as e:
                raise ValueError(f"Corrupted or unreadable image file: {e}")

        # 2. Handle raw bytes / Byte streams
        elif isinstance(image_input, (bytes, io.BytesIO)):
            stream = io.BytesIO(image_input) if isinstance(image_input, bytes) else image_input
            stream.seek(0, io.SEEK_END)
            size_mb = stream.tell() / (1024 * 1024)
            stream.seek(0)
            if size_mb > config.MAX_IMAGE_SIZE_MB:
                raise ValueError(f"Uploaded image exceeds maximum limit of {config.MAX_IMAGE_SIZE_MB}MB.")
            try:
                pil_img = Image.open(stream)
                pil_img.load()
            except Exception as e:
                raise ValueError(f"Corrupted or invalid image data: {e}")

        # 3. Handle PIL Image
        elif isinstance(image_input, Image.Image):
            pil_img = image_input

        # 4. Handle Numpy Array
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(np.uint8(image_input))

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if pil_img is None:
            raise ValueError("Failed to load image.")

        # Handle EXIF rotation and convert transparency (RGBA / Palette) to RGB with white background
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            # Alpha composite onto white background so transparent areas do not become black
            rgba = pil_img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            pil_img = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            pil_img = pil_img.convert("RGB")

        return pil_img

    def preprocess_image(
        self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses image into RGB uint8 array and batch tensor of shape (1, 224, 224, 3).
        """
        pil_img = self.validate_and_load_image(image_input)

        # Original RGB image
        original_rgb = np.array(pil_img, dtype=np.uint8)

        # Resize to model input shape (224, 224)
        resized_pil = pil_img.resize(config.IMAGE_SIZE, Image.Resampling.BILINEAR)
        resized_array = np.array(resized_pil, dtype=np.float32)

        # Batch tensor (1, 224, 224, 3)
        batch_tensor = np.expand_dims(resized_array, axis=0)

        return original_rgb, batch_tensor

    def predict(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
        gradcam_alpha: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Performs damage classification and returns complete prediction metadata.
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded. Please train the model first by running `python src/train.py`.")

        original_rgb, batch_tensor = self.preprocess_image(image_input)

        # Forward prediction
        preds = self.model.predict(batch_tensor, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_class = self.class_names[top_idx]
        confidence = float(preds[top_idx])

        # Probabilities dictionary across all 3 classes
        probabilities = {
            self.class_names[i]: float(preds[i]) for i in range(len(self.class_names))
        }

        # Check for low confidence (< 50%)
        is_low_confidence = confidence < config.CONFIDENCE_THRESHOLD

        # Class details
        details = config.CLASS_DETAILS.get(
            top_class,
            {
                "display_name": top_class.replace("_", " ").title(),
                "badge_color": "#64748b",
                "bg_color": "#f8fafc",
                "border_color": "#cbd5e1",
                "severity": "Unknown",
                "summary": "Vehicle damage analysis complete.",
                "action": "Consult automotive repair specialist.",
            },
        )

        if is_low_confidence:
            summary = "Low confidence - try a clearer, well-lit photo of the car."
            action = "Please upload an unobstructed exterior photo taken in adequate daylight."
        else:
            summary = details.get("summary", "")
            action = details.get("action", "")

        # Compute Grad-CAM Heatmap
        try:
            heatmap = compute_gradcam_heatmap(self.model, batch_tensor, pred_index=top_idx)
            overlay_rgb, heatmap_colored_rgb = overlay_gradcam(original_rgb, heatmap, alpha=gradcam_alpha)
        except Exception as e:
            logging.warning(f"Grad-CAM generation failed: {e}. Falling back to clean overlay.")
            overlay_rgb = original_rgb
            heatmap_colored_rgb = original_rgb

        return {
            "class_name": top_class,
            "display_name": details.get("display_name", top_class),
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "is_low_confidence": is_low_confidence,
            "probabilities": probabilities,
            "badge_color": details.get("badge_color", "#3b82f6"),
            "bg_color": details.get("bg_color", "#f8fafc"),
            "border_color": details.get("border_color", "#cbd5e1"),
            "severity": details.get("severity", "N/A"),
            "summary": summary,
            "action": action,
            "original_rgb": original_rgb,
            "gradcam_overlay_rgb": overlay_rgb,
            "gradcam_heatmap_rgb": heatmap_colored_rgb,
        }


def main():
    parser = argparse.ArgumentParser(description="Vehicle Damage Detection Inference")
    parser.add_argument("image", nargs="?", default=None, help="Path to input car image")
    parser.add_argument("--image", dest="opt_image", type=str, default=None, help="Path to input car image")
    parser.add_argument("--alpha", type=float, default=0.4, help="Grad-CAM blend factor")
    args = parser.parse_args()

    image_path = args.image or args.opt_image
    if not image_path:
        print("Usage: python src/predict.py <image_path>")
        return

    predictor = DamagePredictor()
    if not predictor.is_ready():
        print("Error: Model is not loaded. Train the model first using: python src/train.py")
        return

    try:
        result = predictor.predict(image_path, gradcam_alpha=args.alpha)
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"Predicted Class : {result['display_name']} ({result['class_name']})")
        print(f"Confidence Score: {result['confidence_percentage']}")
        if result["is_low_confidence"]:
            print("Notice          : Low confidence - try a clearer, well-lit photo of the car.")
        print(f"Damage Severity : {result['severity']}")
        print(f"Summary         : {result['summary']}")
        print(f"Suggested Action: {result['action']}")
        print("\nProbabilities Distribution (3 Classes):")
        for k, v in result["probabilities"].items():
            disp = config.CLASS_DETAILS.get(k, {}).get("display_name", k)
            print(f"  - {disp} ({k}): {v:.2%}")
        print("=" * 50)
    except Exception as e:
        print(f"Prediction Error: {e}")


if __name__ == "__main__":
    main()
