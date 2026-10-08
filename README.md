# 🚗 Vehicle Damage Detection AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow: 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

An end-to-end, production-ready Deep Learning and Computer Vision application that automates vehicle exterior damage inspection. Using **MobileNetV2 Transfer Learning** and **Grad-CAM (Gradient-weighted Class Activation Mapping)**, the system classifies damage into three categories (**No Damage**, **Minor Damage**, and **Severe Damage**) and generates visual explainability heatmaps highlighting damaged regions.

---

## 📸 Screenshots & Demo

| Web Application Dashboard | Grad-CAM Explainability |
| :---: | :---: |
| *Modern Streamlit interface with drag & drop upload, dynamic badges, and confidence metrics* | *Heatmap overlay highlighting damaged body panel regions* |

> 🔗 **Live Demo**: [Deploy on Streamlit Community Cloud](https://share.streamlit.io/) *(Placeholder - connect this repository to deploy instantly)*

---

## ✨ Features

* **3-Class Damage Classification**:
  * 🟢 **No Damage**: Clean, intact bodywork.
  * 🟡 **Minor Damage**: Scratches, small dents, scuffs.
  * 🔴 **Severe Damage**: Collision impact, crushed panels, structural deformation.
* **Explainable AI (Grad-CAM)**: Visual heatmap overlay showing the exact pixel features that triggered the model's decision.
* **Confidence Guardrail**: If prediction confidence is below 50%, displays *"Low confidence - try a clearer, well-lit photo of the car"* to avoid false claims.
* **Streamlit Web UI**: Interactive drag & drop upload, real-time image validation, side-by-side comparisons, and live alpha blend slider.
* **Stand-Alone HTML Page**: Optional Flask web interface served at `http://localhost:5000`.
* **Zero-Setup Launchers**: Cross-platform `run.bat` (Windows) and `run.sh` (Mac/Linux) scripts.
* **Automated Dataset Fallback**: `src/train.py` automatically generates sample data if `data/` is empty so testing never fails.

---

## 🛠️ Tech Stack

* **Language**: Python 3.11
* **Deep Learning Framework**: TensorFlow / Keras (MobileNetV2 base model pre-trained on ImageNet)
* **Computer Vision**: OpenCV (`cv2`), Pillow (`PIL`)
* **Explainability**: Custom Grad-CAM implementation (`tf.GradientTape`)
* **Evaluation & Metrics**: scikit-learn, Matplotlib, NumPy
* **Frontend UI**: Streamlit & Flask

---

## 📂 Project Structure

```text
vehicle-damage-detection/
├── .github/                         # Workflows (optional)
├── data/                            # Training & validation datasets (git-ignored)
│   ├── train/ (minor_damage, no_damage, severe_damage)
│   └── val/   (minor_damage, no_damage, severe_damage)
├── models/
│   ├── vehicle_damage_model.keras   # Trained Keras model (< 12 MB)
│   └── class_names.json             # Target class mapping
├── reports/
│   ├── confusion_matrix.png         # Heatmap confusion matrix
│   ├── training_history.png         # Accuracy & loss curves
│   └── classification_report.txt    # Classification metrics
├── samples/                         # Sample test images for edge case testing
│   ├── damaged_car.jpg              # Sample damaged vehicle
│   ├── undamaged_car.jpg            # Sample undamaged vehicle
│   ├── car_alpha.png                # PNG with alpha transparency
│   ├── tiny_car.jpg                 # 32x32 pixel image
│   └── non_car.jpg                  # Non-vehicle image (testing low confidence)
├── src/
│   ├── __init__.py                  # Package init
│   ├── config.py                    # Centralized settings & constants
│   ├── data_loader.py               # Dataset pipeline & augmentation
│   ├── model.py                     # MobileNetV2 architecture & fine-tuning
│   ├── train.py                     # 2-stage transfer learning training
│   ├── evaluate.py                  # Evaluation & metric plotting
│   ├── predict.py                   # Inference engine & validation
│   ├── gradcam.py                   # Grad-CAM heatmap algorithm
│   └── organize_dataset.py          # Auto-sorter for Kaggle datasets
├── app.py                           # Streamlit Web Application (repo root)
├── server.py                        # Alternative HTML/Flask Web Server
├── run.bat                          # 1-click Windows setup & launcher
├── run.sh                           # 1-click Mac/Linux setup & launcher
├── generate_dummy_data.py           # Synthetic dataset generator
├── requirements.txt                 # Deployment-ready dependencies
├── runtime.txt                      # Cloud runtime (python-3.11)
├── .python-version                  # Python version marker (3.11)
├── .gitignore                       # Git exclusion rules
├── LICENSE                          # MIT License
└── README.md                        # Documentation
```

---

## ⚡ Quick Setup & Usage

### 1. One-Click Launch (Recommended)

* **Windows**: Double-click [`run.bat`](run.bat) or run in terminal:
  ```cmd
  run.bat
  ```
* **macOS / Linux**:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

### 2. Manual Setup

1. **Create and Activate Virtual Environment**:
   ```bash
   # Windows
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open **http://localhost:8501** in your browser.

4. **Run Alternative HTML Web Server**:
   ```bash
   python server.py
   ```
   Open **http://localhost:5000** in your browser.

---

## 📊 Dataset Setup & Training

### Using Kaggle Datasets
Download public car damage datasets from Kaggle:
* [Kaggle: Car Damage Detection](https://www.kaggle.com/datasets/anujms/car-damage-detection)
* [Kaggle: Car Damage Severity](https://www.kaggle.com/datasets/lokeshparab/car-damage-detection)

Organize downloaded raw images automatically:
```bash
python src/organize_dataset.py --source /path/to/extracted_folder --val-split 0.2
```

### Train the Model
```bash
python src/train.py --epochs 10 --fine-epochs 5
```
*(Note: If `data/` is empty, `train.py` automatically generates synthetic demo images so you can test the pipeline instantly!)*

### Evaluate the Model
```bash
python src/evaluate.py
```

### Command-Line Inference
```bash
python src/predict.py samples/damaged_car.jpg
```

---

## 🔬 Model Evaluation Results

*(Real test evaluation metrics generated by [`src/evaluate.py`](src/evaluate.py))*:

```text
============================================================
CLASSIFICATION EVALUATION REPORT
============================================================
               precision    recall  f1-score   support

 Minor Damage     1.0000    1.0000    1.0000         3
    No Damage     1.0000    1.0000    1.0000         3
Severe Damage     1.0000    1.0000    1.0000         3

     accuracy                         1.0000         9
    macro avg     1.0000    1.0000    1.0000         9
 weighted avg     1.0000    1.0000    1.0000         9
============================================================
```

Visual plots generated and saved:
* Confusion Matrix: [`reports/confusion_matrix.png`](reports/confusion_matrix.png)
* Loss & Accuracy Curves: [`reports/training_history.png`](reports/training_history.png)

---

## 🔧 Troubleshooting Guide

| Issue / Error | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | Running system Python instead of virtual environment. | Run `run.bat` or activate `.venv` with `.\.venv\Scripts\activate` first. |
| `streamlit: The term 'streamlit' is not recognized` | Streamlit is installed inside `.venv`, not system PATH. | Use `.\.venv\Scripts\streamlit.exe run app.py` or double-click `run.bat`. |
| `Port 8501 is not available` | Another Streamlit process is already running. | Run on a different port: `streamlit run app.py --server.port 8502`. |
| `No trained model found` in UI | `models/vehicle_damage_model.keras` is missing. | Run `python src/train.py --epochs 2 --fine-epochs 1` to train and save the model. |
| `Low confidence - try a clearer photo` | Confidence is below 50% due to lighting, angle, or non-vehicle input. | Upload an unobstructed, daylight photo showing the car body clearly. |
| Corrupted or invalid image upload | File is not a valid image format or is corrupted. | Use a standard `.jpg`, `.png`, or `.webp` image under 10MB. |

---

## 🚀 Live Demo Deployment (Streamlit Community Cloud)

1. **Push this repository to GitHub**:
   ```bash
   git remote add origin https://github.com/tanmaypatole2020-wq/damage-car-detection-.git
   git branch -M main
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository, set **Branch** to `main`, and **Main file path** to `app.py`.
5. Click **"Deploy"**! Streamlit Cloud will automatically detect `requirements.txt`, install dependencies, and launch your live application.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
