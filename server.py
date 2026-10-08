"""
Flask Web Server for Vehicle Damage Detection.
Serves a modern HTML/CSS/JS web page and REST API for damage prediction and Grad-CAM visualizations.
"""

import base64
from io import BytesIO
import logging
from pathlib import Path
import sys

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.predict import DamagePredictor

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global predictor instance
predictor = DamagePredictor()


def numpy_to_base64(img_array: np.ndarray, format: str = "JPEG") -> str:
    """Converts a numpy RGB image array into a base64 encoded data URI string."""
    pil_img = Image.fromarray(np.uint8(img_array))
    buffered = BytesIO()
    pil_img.save(buffered, format=format, quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


@app.route("/")
def index():
    """Renders the HTML web page."""
    model_ready = predictor.is_ready()
    return render_template("index.html", model_ready=model_ready)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    """API endpoint for vehicle damage analysis."""
    if not predictor.is_ready():
        return jsonify({"error": "Model is not loaded. Please train the model first."}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Selected image is empty."}), 400

    try:
        alpha = float(request.form.get("alpha", 0.4))
        pil_img = Image.open(file.stream)

        # Run prediction pipeline
        result = predictor.predict(pil_img, gradcam_alpha=alpha)

        # Convert images to base64 Data URIs for browser display
        original_b64 = numpy_to_base64(result["original_rgb"])
        overlay_b64 = numpy_to_base64(result["gradcam_overlay_rgb"])
        heatmap_b64 = numpy_to_base64(result["gradcam_heatmap_rgb"])

        return jsonify(
            {
                "status": "success",
                "class_name": result["class_name"],
                "display_name": result["display_name"],
                "confidence": result["confidence"],
                "confidence_percentage": f"{result['confidence'] * 100:.1f}%",
                "probabilities": result["probabilities"],
                "badge_color": result["badge_color"],
                "bg_color": result["bg_color"],
                "severity": result["severity"],
                "summary": result["summary"],
                "action": result["action"],
                "original_image": original_b64,
                "gradcam_overlay": overlay_b64,
                "gradcam_heatmap": heatmap_b64,
            }
        )

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" VEHICLE DAMAGE DETECTION WEB SERVER RUNNING")
    print(" Open Web Page in Browser: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
