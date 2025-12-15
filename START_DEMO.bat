@echo off
title BrandClave Demo
color 0A

echo.
echo  ============================================
echo       BrandClave Aggregator - Demo Mode
echo  ============================================
echo.

:: Set local conda location
set "LOCAL_CONDA=%~dp0conda"
set "CONDA_BASE="

:: First check if conda is already in PATH
where conda >nul 2>nul
if %errorlevel% equ 0 (
    goto :conda_ready
)

:: Check local installation first
if exist "%LOCAL_CONDA%\Scripts\conda.exe" (
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
        set "CONDA_BASE=%%~L"
        goto :found_conda
    )
)

:: Not found
echo.
echo [ERROR] Conda not found!
echo.
echo Run SETUP_FIRST_TIME.bat first.
echo.
pause
exit /b 1

:found_conda
echo Initializing...
if exist "%CONDA_BASE%\Scripts\activate.bat" (
    call "%CONDA_BASE%\Scripts\activate.bat" "%CONDA_BASE%"
) else (
    call "%CONDA_BASE%\condabin\activate.bat" "%CONDA_BASE%"
)

:conda_ready
call conda activate brandclave
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Environment not found!
    echo Run SETUP_FIRST_TIME.bat first.
    echo.
    pause
    exit /b 1
)

:: Check requirements
if not exist ".env" (
    echo.
    echo [ERROR] No .env file!
    echo Run SETUP_FIRST_TIME.bat first.
    echo.
    pause
    exit /b 1
)

if not exist "data" mkdir data

echo Starting server...
echo.
echo  ============================================
echo.
echo   Dashboard:     http://localhost:8000/api/monitoring/dashboard
echo   API Docs:      http://localhost:8000/docs
echo.
echo   Social Pulse:  http://localhost:8000/api/social-pulse
echo   Hotelier Bets: http://localhost:8000/api/hotelier-bets
echo   Demand Scan:   http://localhost:8000/api/demand-scan
echo.
echo  ============================================
echo.
echo   Press Ctrl+C to stop
echo.

:: Open browser after delay
start "" timeout /t 3 /nobreak >nul & start http://localhost:8000/api/monitoring/dashboard

:: Start server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
