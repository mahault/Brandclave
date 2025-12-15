@echo off
title BrandClave Demo
color 0A

echo.
echo  ============================================
echo       BrandClave Aggregator - Demo Mode
echo  ============================================
echo.

:: Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Conda not found!
    echo.
    echo Please install Miniconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

echo [1/4] Activating environment...
call conda activate brandclave
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Environment 'brandclave' not found!
    echo.
    echo Run this command first to create it:
    echo    conda env create -f environment.yml
    echo.
    pause
    exit /b 1
)

echo [2/4] Environment activated!
echo.

:: Check if .env exists
if not exist ".env" (
    echo [WARNING] No .env file found!
    echo.
    echo Please create a .env file with your MISTRAL_API_KEY
    echo You can copy .env.example and add your key.
    echo.
    pause
    exit /b 1
)

echo [3/4] Starting API server...
echo.
echo  ============================================
echo.
echo   Demo is starting! Opening browser...
echo.
echo   API Docs:      http://localhost:8000/docs
echo   Social Pulse:  http://localhost:8000/api/social-pulse
echo   Hotelier Bets: http://localhost:8000/api/hotelier-bets
echo.
echo  ============================================
echo.
echo   Press Ctrl+C to stop the server
echo.

:: Wait a moment then open browser
start "" timeout /t 3 /nobreak >nul & start http://localhost:8000/docs

:: Start the server (this will block until Ctrl+C)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

echo.
echo Demo server stopped.
pause
