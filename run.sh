#!/usr/bin/env bash
# Vehicle Damage Detection - Setup and Launch Script for macOS and Linux

set -e

echo "============================================================"
echo "  Vehicle Damage Detection - Auto Setup & Launch (POSIX)"
echo "============================================================"

# 1. Determine Python 3 binary
if command -v python3.11 &>/dev/null; then
    PY_BIN="python3.11"
elif command -v python3 &>/dev/null; then
    PY_BIN="python3"
elif command -v python &>/dev/null; then
    PY_BIN="python"
else
    echo "[ERROR] Python 3 not found. Please install Python 3.10 or 3.11."
    exit 1
fi

# 2. Create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment with $PY_BIN..."
    $PY_BIN -m venv .venv
    echo "[SUCCESS] Virtual environment created in .venv."
fi

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Install dependencies
echo "[INFO] Installing requirements..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# 5. Launch Streamlit Application
echo "============================================================"
echo "[SUCCESS] Launching Streamlit Web App..."
echo "Open your browser at http://localhost:8501"
echo "============================================================"
streamlit run app.py
