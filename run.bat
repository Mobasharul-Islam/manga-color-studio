@echo off
cd /d "%~dp0"
title Manga Color Studio

echo ========================================================
echo   Launching Manga Color Studio (TPU v5-8 Accelerator)
echo ========================================================

python -c "import fastapi, uvicorn, kaggle, PIL" >nul 2>&1
if %errorlevel% neq 0 (
    echo [SETUP] Installing required Python packages...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies. Please ensure Python is installed and added to PATH.
        pause
        exit /b 1
    )
)

python run_app.py
pause