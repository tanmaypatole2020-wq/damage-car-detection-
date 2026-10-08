"""
Streamlit Web Application for AI Vehicle Damage Detection.
Provides image upload, 3-class damage classification, confidence breakdown, and Grad-CAM explainability.
Deployable on Streamlit Community Cloud and local environments.
"""

from pathlib import Path
import sys
from io import BytesIO

from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.predict import DamagePredictor

# Page Setup
st.set_page_config(
    page_title="Vehicle Damage Detection AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        max-width: 1250px;
        margin: 0 auto;
    }
    .header-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        max-width: 800px;
        line-height: 1.5;
    }
    .result-card {
        background-color: white;
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .action-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
    }
    .low-confidence-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        color: #92400e;
    }
    .instruction-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        padding: 1.75rem;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_predictor() -> DamagePredictor:
    """Instantiates and caches the predictor model to load only once."""
    return DamagePredictor()


def render_sidebar(predictor: DamagePredictor) -> float:
    """Renders the Streamlit sidebar with system status and controls."""
    with st.sidebar:
        st.header("⚙️ System Status")

        # Model Status Indicator
        if predictor.is_ready():
            st.success("🟢 Model Ready: MobileNetV2 Active")
        else:
            st.warning("🟡 Model Missing: Please Train Model")

        st.divider()

        # How it works section
        st.subheader("💡 How It Works")
        st.markdown(
            """
            1. **Upload Photo**: Upload a car photo (JPG, PNG, WEBP).
            2. **Neural Network**: Pre-trained **MobileNetV2** extracts deep visual features.
            3. **Classification**: Evaluates probability across 3 classes:
               - 🟢 **No Damage**
               - 🟡 **Minor Damage**
               - 🔴 **Severe Damage**
            4. **Grad-CAM**: Computes activation heatmap overlay highlighting damage location.
            """
        )

        st.divider()

        # Heatmap Transparency Slider
        st.subheader("🔥 Visual Overlay")
        gradcam_alpha = st.slider(
            "Grad-CAM Heatmap Blend (Alpha)",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
            help="Adjust the transparency of the Grad-CAM heatmap layer",
        )

        st.divider()
        st.caption("Vehicle Damage Detection AI | MobileNetV2 + Grad-CAM")

    return gradcam_alpha


def main():
    # Header Banner
    st.markdown(
        """
        <div class="header-card">
            <div class="header-title">🚗 Vehicle Damage Detection AI</div>
            <div class="header-subtitle">
                Automated vehicle exterior damage classification and explainable AI powered by MobileNetV2 Transfer Learning and Grad-CAM Heatmap overlays.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        predictor = get_cached_predictor()
    except Exception as e:
        st.error(f"Error initializing predictor: {e}")
        return

    gradcam_alpha = render_sidebar(predictor)

    # Missing model edge case handling
    if not predictor.is_ready():
        st.warning("⚠️ No Trained Model Found in `models/`")
        st.markdown(
            """
            <div class="instruction-card">
                <h3>🚀 Quick Training Instructions</h3>
                <p>A trained model is required to perform live inferences. You can generate sample data and train the model in 1 minute:</p>
                <ol>
                    <li><b>Generate demo dataset:</b><br><code>python generate_dummy_data.py</code></li>
                    <li><b>Train the model (1-2 epochs for testing):</b><br><code>python src/train.py --epochs 2 --fine-epochs 1</code></li>
                    <li><b>Refresh this page</b> to start detecting vehicle damage!</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Main Grid Layout
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📷 Upload Vehicle Image")
        uploaded_file = st.file_uploader(
            "Upload car photo (JPG, JPEG, PNG, WEBP, max 10MB)...",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear photo of the car exterior or damaged region",
        )

        pil_image = None
        if uploaded_file is not None:
            # Check file size limit (10MB)
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > config.MAX_IMAGE_SIZE_MB:
                st.error(f"File size ({file_size_mb:.1f}MB) exceeds the maximum limit of {config.MAX_IMAGE_SIZE_MB}MB.")
                return

            try:
                # Open image using BytesIO stream
                file_bytes = uploaded_file.read()
                pil_image = Image.open(BytesIO(file_bytes))
                pil_image.load()

                # Handle tiny images (< 32x32)
                if pil_image.width < config.MIN_IMAGE_DIMENSION or pil_image.height < config.MIN_IMAGE_DIMENSION:
                    st.warning(f"Image resolution ({pil_image.width}x{pil_image.height}) is very small. Classification accuracy may be reduced.")

                st.image(pil_image, caption="Uploaded Image Preview", use_container_width=True)

            except Exception as e:
                st.error(f"Invalid or corrupted image file: {e}")
                return
        else:
            st.info("👆 Upload a car photo above to begin automated damage analysis.")
            st.markdown(
                """
                **Evaluation Supported:**
                - Scratches, scuffs, and small body dents
                - Bumper cracks and side panel impacts
                - Clear/undamaged vehicle verification
                """
            )
            return

    with col_right:
        st.subheader("🔍 Damage Analysis & Diagnosis")

        with st.spinner("Analyzing damage features and computing Grad-CAM heatmap..."):
            try:
                result = predictor.predict(pil_image, gradcam_alpha=gradcam_alpha)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return

        badge_color = result["badge_color"]
        bg_color = result["bg_color"]
        border_color = result["border_color"]
        display_name = result["display_name"]
        confidence_pct = result["confidence"] * 100
        is_low_confidence = result["is_low_confidence"]

        # Classification Badge Card
        st.markdown(
            f"""
            <div style="background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">AI Assessment</div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
                    <span style="background-color: {badge_color}; color: white; padding: 0.4rem 1.1rem; border-radius: 9999px; font-weight: 800; font-size: 1.2rem;">
                        {display_name}
                    </span>
                    <span style="font-size: 1.5rem; font-weight: 800; color: #1e293b;">
                        {confidence_pct:.1f}% Confidence
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Plain-English Diagnostic Summary
        if is_low_confidence:
            st.markdown(
                """
                <div class="result-card">
                    <h4 style="margin-top:0; color: #b45309;">⚠️ Low Confidence Verdict</h4>
                    <p style="color: #78350f;"><strong>Low confidence - try a clearer, well-lit photo of the car.</strong></p>
                    <div class="low-confidence-box">
                        <strong>💡 Suggestion:</strong> Ensure the car exterior is well lit, unobstructed, and centered in frame.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4 style="margin-top:0; color: #1e293b;">📋 Diagnostic Assessment</h4>
                    <p style="color: #334155; font-size: 0.95rem; line-height: 1.5;">{result['summary']}</p>
                    <div class="action-box">
                        <strong>💡 Recommended Next Step:</strong><br>
                        <span style="color: #1e293b; font-size: 0.95rem;">{result['action']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Probability Bar Distribution for All 3 Classes
        st.subheader("📊 Probability Breakdown (All 3 Classes)")
        probs = result["probabilities"]
        formatted_probs = {
            config.CLASS_DETAILS.get(k, {}).get("display_name", k): round(v * 100, 1)
            for k, v in probs.items()
        }
        st.bar_chart(formatted_probs, color="#3b82f6")

    # Grad-CAM Heatmap Visual Explainability Section
    st.divider()
    st.subheader("🔥 Grad-CAM Visual Heatmap (Explainable AI)")
    st.caption(
        "Grad-CAM computes gradients at the final convolutional layer to visualize the spatial regions that triggered the model's damage verdict."
    )

    g_col1, g_col2 = st.columns([1, 1], gap="medium")

    with g_col1:
        st.image(result["original_rgb"], caption="Original Vehicle Image", use_container_width=True)

    with g_col2:
        st.image(
            result["gradcam_overlay_rgb"],
            caption=f"Grad-CAM Heatmap Overlay (Blend Alpha = {gradcam_alpha})",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
