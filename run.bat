@echo off
cd /d "%~dp0"
title Manga Color Studio

echo ========================================================
echo   Launching Manga Color Studio (TPU v5-8 Accelerator)
echo ========================================================

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.9+ from https://www.python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

:: 2. Auto-create and configure .venv on first launch
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] First-time launch detected! Creating isolated virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SETUP] Installing required dependencies into .venv...
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

:: 3. Launch Studio
python run_app.py
pause