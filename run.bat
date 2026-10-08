@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Vehicle Damage Detection - Auto Setup & Launch (Windows)
echo ============================================================

:: 1. Check or create virtual environment
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment in .venv...
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        py -3.11 -m venv .venv 2>nul || py -m venv .venv
    ) else (
        python -m venv .venv
    )
    if not exist ".venv" (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.10+ is installed.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
)

:: 2. Upgrade pip and install dependencies
echo [INFO] Checking and installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\pip.exe" install -r requirements.txt --quiet
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Retrying dependency installation...
    ".venv\Scripts\pip.exe" install -r requirements.txt
)

:: 3. Launch Streamlit Application
echo ============================================================
echo [SUCCESS] Setup complete! Launching Streamlit Web App...
echo Open your browser at http://localhost:8501
echo ============================================================
".venv\Scripts\streamlit.exe" run app.py
pause
