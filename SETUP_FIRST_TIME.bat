@echo off
title BrandClave - First Time Setup
color 0E

echo.
echo  ============================================
echo    BrandClave - First Time Setup
echo  ============================================
echo.
echo  This will set up everything you need.
echo  It may take 5-10 minutes.
echo.
pause

:: Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Conda not found!
    echo.
    echo Please install Miniconda first:
    echo.
    echo 1. Go to: https://docs.conda.io/en/latest/miniconda.html
    echo 2. Download "Miniconda3 Windows 64-bit"
    echo 3. Run the installer (use default settings)
    echo 4. Restart your computer
    echo 5. Run this script again
    echo.
    pause
    exit /b 1
)

echo.
echo [1/5] Creating conda environment...
echo       (This may take a few minutes)
echo.
call conda env create -f environment.yml
if %errorlevel% neq 0 (
    echo.
    echo Environment may already exist, trying to update...
    call conda env update -f environment.yml
)

echo.
echo [2/5] Activating environment...
call conda activate brandclave

echo.
echo [3/5] Installing dependencies...
pip install -r requirements.txt

echo.
echo [4/5] Setting up configuration...
if not exist ".env" (
    copy .env.example .env
    echo.
    echo [IMPORTANT] Created .env file from template.
    echo.
    echo You need to add your MISTRAL_API_KEY to the .env file!
    echo.
    echo 1. Open the .env file in Notepad
    echo 2. Replace "your_mistral_api_key_here" with your actual key
    echo 3. Save and close
    echo.
    notepad .env
    pause
)

echo.
echo [5/5] Initializing database...
python scripts/init_db.py

echo.
echo  ============================================
echo.
echo   Setup complete!
echo.
echo   Next steps:
echo   1. Make sure your MISTRAL_API_KEY is in the .env file
echo   2. Double-click START_DEMO.bat to run the demo
echo.
echo  ============================================
echo.
pause
