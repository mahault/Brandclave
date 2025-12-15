@echo off
title BrandClave - First Time Setup
color 0E

echo.
echo  ============================================
echo    BrandClave - First Time Setup
echo  ============================================
echo.
echo  This will set up everything you need.
echo.
pause

:: Set local conda install location
set "LOCAL_CONDA=%~dp0conda"
set "CONDA_BASE="

echo.
echo [1/6] Checking for conda...

:: First check if conda is already in PATH
where conda >nul 2>nul
if %errorlevel% equ 0 (
    echo Found conda in PATH
    goto :conda_ready
)

:: Check if we already installed it locally
if exist "%LOCAL_CONDA%\Scripts\conda.exe" (
    echo Found local conda installation
    set "CONDA_BASE=%LOCAL_CONDA%"
    goto :found_conda
)

:: Check common locations
for %%L in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\Miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\Anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "C:\miniconda3"
    "C:\anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
) do (
    if exist "%%~L\Scripts\conda.exe" (
        echo Found conda at: %%~L
        set "CONDA_BASE=%%~L"
        goto :found_conda
    )
)

:: Not found - install it locally
echo.
echo Conda not found. Installing Miniconda locally...
echo (This is a one-time download of ~80MB)
echo.

:: Create temp directory for download
if not exist "%~dp0temp" mkdir "%~dp0temp"

:: Download Miniconda using PowerShell
echo Downloading Miniconda...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%~dp0temp\miniconda_installer.exe'}"

if not exist "%~dp0temp\miniconda_installer.exe" (
    echo.
    echo [ERROR] Failed to download Miniconda!
    echo.
    echo Please check your internet connection and try again.
    echo Or download manually from: https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

:: Install Miniconda silently to local folder
echo.
echo Installing Miniconda (this may take a few minutes)...
"%~dp0temp\miniconda_installer.exe" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%LOCAL_CONDA%

if not exist "%LOCAL_CONDA%\Scripts\conda.exe" (
    echo.
    echo [ERROR] Miniconda installation failed!
    echo.
    pause
    exit /b 1
)

:: Cleanup installer
del "%~dp0temp\miniconda_installer.exe" >nul 2>nul
rmdir "%~dp0temp" >nul 2>nul

echo Miniconda installed successfully!
set "CONDA_BASE=%LOCAL_CONDA%"

:found_conda
echo.
echo [2/6] Initializing conda...
if exist "%CONDA_BASE%\Scripts\activate.bat" (
    call "%CONDA_BASE%\Scripts\activate.bat" "%CONDA_BASE%"
) else (
    call "%CONDA_BASE%\condabin\activate.bat" "%CONDA_BASE%"
)

:conda_ready
echo.
echo [3/6] Creating environment...
echo       (This may take a few minutes)
echo.
call conda env create -f environment.yml
if %errorlevel% neq 0 (
    echo.
    echo Environment may already exist, trying to update...
    call conda env update -f environment.yml
)

echo.
echo [4/6] Activating environment...
call conda activate brandclave
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Could not activate environment!
    pause
    exit /b 1
)

echo.
echo [5/6] Installing dependencies...
pip install -r requirements.txt

echo.
echo [6/6] Setting up...
if not exist "data" mkdir data

if not exist ".env" (
    copy .env.example .env
    echo.
    echo  ============================================
    echo.
    echo   IMPORTANT: Add your API key!
    echo.
    echo   Notepad will open. Replace the placeholder
    echo   with your MISTRAL_API_KEY, then save.
    echo.
    echo  ============================================
    echo.
    pause
    notepad .env
    echo.
    echo Press any key after saving your API key...
    pause >nul
)

:: Initialize database
python scripts/init_db.py

echo.
echo  ============================================
echo.
echo   Setup complete!
echo.
echo   To run the demo, double-click:
echo   START_DEMO.bat
echo.
echo  ============================================
echo.
pause
