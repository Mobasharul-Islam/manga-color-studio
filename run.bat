@echo off
setlocal
cd /d "%~dp0"
title Manga Color Studio

echo ========================================================
echo   Launching Manga Color Studio - TPU v5-8 Accelerator
echo ========================================================

:: 1. Check Python installation
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.9+ from https://www.python.org and check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

:: 2. Auto-create and configure .venv on first launch
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] First-time launch detected!
    echo [SETUP] Creating isolated virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create virtual environment.
        echo.
        pause
        exit /b 1
    )
    echo [SETUP] Installing required dependencies into .venv ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies.
        echo.
        pause
        exit /b 1
    )
)

:: 3. Check if core dependencies are present in .venv
".venv\Scripts\python.exe" -c "import uvicorn, fastapi, kaggle" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Installing required dependencies into .venv ...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

:: 4. Run application using .venv python directly
".venv\Scripts\python.exe" run_app.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    echo.
    pause
)