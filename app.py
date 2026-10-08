"""
Vehicle Damage Detection - Flask Application
Serves a lightweight local web dashboard for vehicle damage analysis and Grad-CAM interpretability.
Accessible locally and over Wi-Fi on Mobile Phones (iOS Safari & Android Chrome).
Runs entirely offline with zero external cloud dependencies.
"""

import base64
import io
import logging
from pathlib import Path
import socket
import sys
import threading
import webbrowser

from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image, ImageOps

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.predict import DamagePredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("vehicle_damage_app")

# Initialize Flask app
app = Flask(__name__)

# Max upload payload configuration (10 MB)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_IMAGE_SIZE_MB * 1024 * 1024

# Instantiate DamagePredictor once at startup
predictor = DamagePredictor()


def get_local_ip() -> str:
    """Finds the local network IPv4 address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_qr_code_base64(url: str) -> str:
    """Generates an in-memory QR code PNG encoded as a base64 Data URI."""
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            box_size=6,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#0a0f1d", back_color="#ffffff")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        logger.warning(f"Could not generate QR code: {e}")
        return ""


def numpy_to_base64_data_uri(img_array: np.ndarray, img_format: str = "JPEG") -> str:
    """Converts a numpy RGB array into a base64 Data URI."""
    pil_img = Image.fromarray(np.uint8(img_array))
    buffer = io.BytesIO()
    pil_img.save(buffer, format=img_format, quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{img_format.lower()};base64,{encoded}"


@app.route("/", methods=["GET"])
def index():
    """Renders the main dashboard page with mobile access details."""
    model_ready = predictor.is_ready()
    local_ip = get_local_ip()
    mobile_url = f"http://{local_ip}:5000"
    mobile_qr = generate_qr_code_base64(mobile_url)

    return render_template(
        "index.html",
        model_ready=model_ready,
        mobile_url=mobile_url,
        mobile_qr=mobile_qr,
        local_ip=local_ip,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an uploaded car image, performs 3-class damage classification,
    computes Grad-CAM heatmap overlay, and returns JSON results.
    """
    # 1. Verify model is loaded
    if not predictor.is_ready():
        return jsonify({
            "success": False,
            "error": "Model file not found. Please ensure 'models/vehicle_damage_model.keras' exists."
        }), 503

    # 2. Verify file presence
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided. Please choose a car image to analyze."
        }), 400

    file = request.files["image"]
    if not file or not file.filename or file.filename.strip() == "":
        return jsonify({
            "success": False,
            "error": "No file selected. Please choose a car photo."
        }), 400

    # 3. Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.ALLOWED_IMAGE_EXTENSIONS:
        allowed_list = ", ".join(ext.upper().replace(".", "") for ext in config.ALLOWED_IMAGE_EXTENSIONS)
        return jsonify({
            "success": False,
            "error": f"Invalid file format '{file_ext}'. Allowed formats: {allowed_list}."
        }), 400

    try:
        # 4. Read bytes and validate size
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({
                "success": False,
                "error": "The uploaded file is empty. Please select a valid photo."
            }), 400

        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > config.MAX_IMAGE_SIZE_MB:
            return jsonify({
                "success": False,
                "error": f"File size ({size_mb:.1f} MB) exceeds maximum limit of {config.MAX_IMAGE_SIZE_MB} MB."
            }), 400

        # 5. Open image with PIL and verify integrity
        try:
            raw_img = Image.open(io.BytesIO(image_bytes))
            raw_img.verify()
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            return jsonify({
                "success": False,
                "error": "Uploaded file is corrupted or not a readable image."
            }), 400

        # 6. Run prediction pipeline through DamagePredictor
        result = predictor.predict(pil_img, gradcam_alpha=0.45)

        # 7. Convert output images to base64 data URIs
        original_b64 = numpy_to_base64_data_uri(result["original_rgb"])
        gradcam_b64 = numpy_to_base64_data_uri(result["gradcam_overlay_rgb"])

        # Determine badge type & styling
        class_key = result["class_name"]
        confidence = float(result["confidence"])
        is_low_conf = confidence < config.CONFIDENCE_THRESHOLD

        # Tailored one-line suggestions
        if is_low_conf:
            suggestion = "Low confidence - try a clearer photo."
        elif class_key == "no_damage":
            suggestion = "Vehicle exterior is intact — no body panel repairs required."
        elif class_key == "minor_damage":
            suggestion = "Paintless dent repair or cosmetic touch-up recommended."
        elif class_key == "severe_damage":
            suggestion = "Structural collision repair inspection and insurance appraisal recommended."
        else:
            suggestion = result.get("action", "Further inspection recommended.")

        # Class display order for probability distribution
        ordered_classes = [
            ("no_damage", "No Damage", "#22c55e"),
            ("minor_damage", "Minor Damage", "#f59e0b"),
            ("severe_damage", "Severe Damage", "#ef4444"),
        ]

        prob_list = []
        raw_probs = result.get("probabilities", {})
        for key, display_name, color in ordered_classes:
            val = float(raw_probs.get(key, 0.0))
            prob_list.append({
                "key": key,
                "label": display_name,
                "value": round(val, 4),
                "percentage": f"{val * 100:.1f}%",
                "color": color,
            })

        response_data = {
            "success": True,
            "predicted_class": result["display_name"],
            "class_key": class_key,
            "badge_color": result["badge_color"],
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "is_low_confidence": is_low_conf,
            "low_confidence_notice": "Low confidence - try a clearer photo" if is_low_conf else None,
            "suggestion": suggestion,
            "probabilities": prob_list,
            "original_image": original_b64,
            "gradcam_image": gradcam_b64,
        }

        return jsonify(response_data)

    except Exception as e:
        logger.exception("Error processing prediction")
        return jsonify({
            "success": False,
            "error": f"Failed to analyze image: {str(e)}"
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handles requests exceeding the Flask content length limit."""
    return jsonify({
        "success": False,
        "error": f"File is too large. Maximum allowed size is {config.MAX_IMAGE_SIZE_MB} MB."
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


def open_browser():
    """Opens localhost in user's default browser after a brief delay."""
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception as e:
        logger.warning(f"Could not open browser automatically: {e}")


if __name__ == "__main__":
    local_ip = get_local_ip()
    print("\n" + "=" * 65)
    print("  VEHICLE DAMAGE DETECTION AI - FLASK MULTI-DEVICE SERVER")
    print("=" * 65)
    print(f"  Laptop Browser  : http://127.0.0.1:5000  (or localhost:5000)")
    print(f"  Mobile Phone    : http://{local_ip}:5000  (iOS Safari & Android)")
    print("  Model Status    :", "Ready (MobileNetV2)" if predictor.is_ready() else "Not Found")
    print("  Note: Connect phone to the same Wi-Fi network as this laptop.")
    print("=" * 65 + "\n")

    # Launch browser only once in a timer thread
    threading.Timer(1.25, open_browser).start()

    # Bind to 0.0.0.0 to accept connections from iPhones and Androids on the local network
    app.run(host="0.0.0.0", port=5000, debug=False)
