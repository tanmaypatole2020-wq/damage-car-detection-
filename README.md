# 🚗 Vehicle Damage Detection AI

An offline Deep Learning application that inspects car photos and classifies vehicle exterior damage severity into **No Damage**, **Minor Damage**, and **Severe Damage** with **Grad-CAM** visual explainability heatmaps.

Built with **Flask**, **TensorFlow/Keras (MobileNetV2)**, and clean **HTML5/CSS3/Vanilla JS** with zero external CDN dependencies. Supports desktop browsers, **tablets, and mobile phones (iOS Safari & Android Chrome)**.

---

## 🌐 Live Demo Links (Works on Any Device)

| Demo Method | Target Device | Access URL / Instructions |
| :--- | :--- | :--- |
| **Custom Named Public Demo** | **ANY Device** (iPhone, Android, iPad, PC, 4G/5G, Worldwide) | **`https://vehicle-damage-ai.loca.lt`** *(or any custom name)*<br>Run `run_live_tunnel.bat` to choose your own name! |
| **Local Laptop Browser** | Host Computer | **[http://127.0.0.1:5000](http://127.0.0.1:5000)** *(or [http://localhost:5000](http://localhost:5000))* |
| **Direct Local Wi-Fi** | Smartphones & Tablets on same Wi-Fi | **`http://<laptop-ip>:5000`** *(Scan instant QR Code in app)* |

---

## 🎯 Name Your Demo Link Any Name

You can customize the public demo link to **any name you want** (e.g. `car-damage-ai`, `tanmay-vehicle-detector`, `vehicle-damage-demo`, etc.):

1. **Option A: Interactive Prompt**  
   Double-click `run_live_tunnel.bat` on Windows.  
   When prompted:
   ```text
   Enter any custom name for your demo link (default: vehicle-damage-ai): <TYPE_ANY_NAME>
   ```
2. **Option B: One-Line Command**  
   Run from PowerShell or Command Prompt:
   ```cmd
   run_live_tunnel.bat your-custom-name
   ```
3. **Instant Access**:
   - The script automatically displays your link:
     ```text
     https://your-custom-name.loca.lt
     ```
   - Fetches your **Tunnel IP Password** and automatically copies it to your clipboard (`clip.exe`)!
   - Saves `static/tunnel_info.json` and creates a desktop shortcut `Live_Public_Demo.url`.

---

## 📱 How to Open on Any Device (Phones, Tablets, Remote PCs)

### Method 1: Worldwide Public Link (Cellular 4G/5G, iPhone & Android)
1. Ensure `run_web.bat` (or `python app.py`) and `run_live_tunnel.bat` are running.
2. Send or open your named link (e.g. `https://vehicle-damage-ai.loca.lt`) on **any device anywhere in the world**.
3. When prompted on the first visit for **"Tunnel Password"**, paste your IP password (copied to clipboard by `run_live_tunnel.bat`).
4. Click **"Submit"** — the dashboard opens instantly!

### Method 2: Instant QR Code Scan
1. Open the dashboard on your laptop: [http://127.0.0.1:5000](http://127.0.0.1:5000).
2. Click **"📱 Open on Phone / Any Device"** in the top header.
3. Switch between **🌐 Public Demo Link** (for mobile data / remote users) and **📶 Local Wi-Fi** (for same-room LAN testing).
4. Point your iPhone Camera or Android Google Lens at the on-screen QR Code to open it immediately.
5. Tap **"📋 Copy Password"** if prompted for tunnel verification.

---

## ⚙️ Quick Setup & Run

### 1. Automated Setup (Recommended)
- **Windows**: Double-click `run.bat` or run:
  ```cmd
  run.bat
  ```
- **macOS / Linux**:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```
The script automatically provisions a Python virtual environment (`.venv`), installs dependencies from `requirements.txt`, and launches the multi-device server.

### 2. Manual Setup
```bash
# 1. Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
python app.py
```

---

## 🔬 How to Demo & Test

1. **Upload Vehicle Photo**:
   - **On Phone**: Tap the drop zone to snap a live photo with your camera or select from photo library.
   - **On Desktop**: Drag-and-drop any image, or use ready samples:
     - `samples/damaged_car.jpg` (Damaged vehicle)
     - `samples/undamaged_car.jpg` (Clean vehicle)
     - `samples/car_alpha.png` (PNG with transparency)
2. **Click "Analyze Damage"**:
   - MobileNetV2 evaluates the photo and extracts Grad-CAM heatmaps.
3. **Inspect Output**:
   - **Diagnosis Badge**: 🟢 No Damage, 🟡 Minor Damage, 🔴 Severe Damage.
   - **Confidence Score**: Prediction confidence percentage.
   - **Grad-CAM Heatmap**: Side-by-side / stacked overlay showing AI focus zones.
   - **Class Probabilities**: Visual bar charts for all 3 categories.
   - **Action Recommendation**: Tailored repair and appraisal advice.

---

## 🧪 Verification & Testing Suite

Run all validation tests:

```cmd
# 1. Flask API test suite (7 tests)
python test_server_api.py

# 2. Edge case inference tests (5 samples)
python test_edge_cases.py

# 3. Model validation & metrics evaluation
python src/evaluate.py
```
All tests run with **zero errors**.
