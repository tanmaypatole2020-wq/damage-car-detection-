# 🚗 Vehicle Damage Detection AI

An offline, local Deep Learning application that inspects car photos and classifies vehicle exterior damage severity into **No Damage**, **Minor Damage**, and **Severe Damage** with **Grad-CAM** visual explainability heatmaps.

Built with **Flask**, **TensorFlow/Keras (MobileNetV2)**, and clean **HTML5/CSS3/Vanilla JS** with zero external CDN dependencies.

> 🚀 **Live Local Demo Link**:  
> 👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**  
> *(Ensure `python app.py` is running, or double-click `Launch_App.url` / `open_app.bat`)*

---

## 1. Setup

### Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **Windows, macOS, or Linux**

### Automated Setup
- **Windows**: Double-click `run.bat` or run:
  ```cmd
  run.bat
  ```
- **macOS / Linux**: Make executable and run:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```
The script will automatically create a virtual environment (`.venv`), install all dependencies from `requirements.txt`, and launch the web app.

### Manual Setup (Optional)
```bash
# 1. Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 2. Run

Start the local Flask application:

```bash
python app.py
```

- Server starts at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
- Your default web browser opens automatically.
- Or double-click **`Launch_App.url`** or **`open_app.bat`** anytime to jump straight into the running app.
- Completely offline: no Node, no build step, no CDN libraries, and no cloud deployment required.

---

## 3. How to Demo It

1. **Open the App**:
   - Navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.
2. **Upload an Image**:
   - Drag and drop a vehicle exterior photo or click the upload box to browse.
   - You can test with ready-made photos in the `samples/` directory:
     - `samples/damaged_car.jpg` (Damaged vehicle)
     - `samples/undamaged_car.jpg` (Clean vehicle)
     - `samples/car_alpha.png` (PNG with transparency)
3. **Review Preview & Analyze**:
   - Check the instant image preview and click **"Analyze Damage"**.
4. **Inspect the AI Results**:
   - **Severity Badge**: Color-coded diagnosis (🟢 No Damage, 🟡 Minor Damage, 🔴 Severe Damage).
   - **Confidence Score**: Model confidence percentage (warns if under 50%).
   - **Grad-CAM Heatmap**: Side-by-side view comparing original photo with AI activation regions.
   - **Probability Breakdown**: Real-time distribution across all 3 damage classes.
   - **Suggested Action**: Practical recommendation for vehicle appraisal or repair.
5. **Analyze Another Image**:
   - Click **"Analyze another image"** to reset and test a new photo.
