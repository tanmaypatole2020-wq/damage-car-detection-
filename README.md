# 🚗 Vehicle Damage Detection AI

An offline Deep Learning application that inspects car photos and classifies vehicle exterior damage severity into **No Damage**, **Minor Damage**, and **Severe Damage** with **Grad-CAM** visual explainability heatmaps.

Built with **Flask**, **TensorFlow/Keras (MobileNetV2)**, and clean **HTML5/CSS3/Vanilla JS** with zero external CDN dependencies. Supports desktop browsers and **mobile phones (iOS Safari & Android Chrome)**.

> 🚀 **Live Demo Links**:  
> - **Desktop Laptop**: **[http://127.0.0.1:5000](http://127.0.0.1:5000)** *(or [http://localhost:5000](http://localhost:5000))*  
> - **Mobile Phones (iOS & Android)**: **`http://<laptop-ip>:5000`** *(Click "📱 Open on Phone" on the web page to scan the instant QR code)*  
> - **Internet / Mobile Data (4G/5G)**: Run `run_live_tunnel.bat` for an instant public HTTPS link.

---

## 📱 Mobile Phone Access (iOS & Android)

You can use this app directly from your smartphone to snap car photos with your camera and inspect damage:

1. **Same Wi-Fi Network**:
   - Connect your iPhone or Android phone to the same Wi-Fi as your laptop.
   - On your laptop, click the **"📱 Open on Phone"** button in the top header.
   - Scan the displayed **QR Code** using your phone camera (iOS Camera or Android Google Lens / Chrome), or navigate to:
     ```text
     http://192.168.10.12:5000
     ```
2. **Camera Upload**:
   - Tap the upload area on your phone to either select a picture from your gallery or take a live photo of a car using your phone's camera.
3. **Over Cellular / Remote Internet (4G/5G)**:
   - Double-click `run_live_tunnel.bat` to generate a public HTTPS tunnel link (e.g. `https://xxxx.loca.lt`) accessible from anywhere in the world on any device.

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

Start the local multi-device Flask application:

```bash
python app.py
```

- Server starts on `0.0.0.0:5000`
- Opens automatically in your laptop's default browser at **http://127.0.0.1:5000**
- Accessible across your local Wi-Fi from iPhones, Androids, and tablets.

---

## 3. How to Demo It

1. **Open the App**:
   - Laptop: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.
   - Phone: **`http://<laptop-ip>:5000`** or scan the QR code.
2. **Upload an Image**:
   - Drag and drop or take a photo with your mobile camera.
   - You can also test with samples in the `samples/` directory:
     - `samples/damaged_car.jpg` (Damaged vehicle)
     - `samples/undamaged_car.jpg` (Clean vehicle)
     - `samples/car_alpha.png` (PNG with transparency)
3. **Analyze Damage**:
   - Click/tap **"Analyze Damage"**.
4. **Inspect the AI Results**:
   - **Severity Badge**: Color-coded diagnosis (🟢 No Damage, 🟡 Minor Damage, 🔴 Severe Damage).
   - **Confidence Score**: Model confidence percentage.
   - **Grad-CAM Heatmap**: Side-by-side (or stacked on mobile) view comparing photo with AI focus areas.
   - **Probability Breakdown**: Distribution across all 3 damage classes with animated bars.
   - **Suggested Action**: Practical recommendation for vehicle appraisal or repair.
5. **Reset**:
   - Tap **"Analyze another image"** to test a new vehicle.
