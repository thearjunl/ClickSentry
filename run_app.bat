@echo off
echo Starting ClickSentry - Phishing URL Detection
echo =============================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Check if model files exist
if not exist "phishing_model.pkl" (
    echo Model files not found. Training model...
    python train_model.py
    if %errorlevel% neq 0 (
        echo Error: Failed to train model
        pause
        exit /b 1
    )
)

:: Start the Flask application
echo Starting Flask application...
python app.py

pause