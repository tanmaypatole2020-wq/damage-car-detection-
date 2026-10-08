@echo off
echo ============================================================
echo Training Vehicle Damage Detection Model...
echo ============================================================
".venv\Scripts\python.exe" src/train.py --epochs 10 --fine-epochs 5
pause
