@echo off
title BrandClave Dashboard
color 0A

echo.
echo  ============================================
echo       BrandClave Intelligence Dashboard
echo  ============================================
echo.

:: Find conda
set "LOCAL_CONDA=%~dp0conda"

where conda >nul 2>nul
if %errorlevel% equ 0 goto :conda_ready

if exist "%LOCAL_CONDA%\Scripts\conda.exe" (
    call "%LOCAL_CONDA%\Scripts\activate.bat" "%LOCAL_CONDA%"
    goto :conda_ready
)

for %%L in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "C:\miniconda3"
) do (
    if exist "%%~L\Scripts\conda.exe" (
        call "%%~L\Scripts\activate.bat" "%%~L"
        goto :conda_ready
    )
)

echo [ERROR] Conda not found! Run SETUP_FIRST_TIME.bat first.
pause
exit /b 1

:conda_ready
call conda activate brandclave
if %errorlevel% neq 0 (
    echo [ERROR] Environment not found! Run SETUP_FIRST_TIME.bat first.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [ERROR] No .env file! Run SETUP_FIRST_TIME.bat first.
    pause
    exit /b 1
)

if not exist "data" mkdir data

:: Check if we have any data
echo Checking database...
for /f %%i in ('python -c "from db.database import SessionLocal; from db.models import RawContentModel; db=SessionLocal(); count=db.query(RawContentModel).count(); db.close(); print(count)"') do set CONTENT_COUNT=%%i

echo Found %CONTENT_COUNT% content items in database.

if %CONTENT_COUNT% LSS 10 (
    echo.
    echo  ============================================
    echo   WARNING: Database has very little data!
    echo  ============================================
    echo.
    echo   The dashboard will be mostly empty.
    echo   For the best experience, run POPULATE_DATA.bat first.
    echo.
    echo   Press any key to continue anyway, or close this window
    echo   and run POPULATE_DATA.bat first.
    echo.
    pause
)

echo.
echo  ============================================
echo   Starting Dashboard Server
echo  ============================================
echo.
echo   Dashboard:     http://localhost:8000/api/monitoring/dashboard-v2
echo   API Docs:      http://localhost:8000/docs
echo.
echo   The scheduler will run background updates automatically.
echo.
echo   If you need fresh data, close this and run:
echo   POPULATE_DATA.bat
echo.
echo  ============================================
echo.
echo   Press Ctrl+C to stop the server
echo.

:: Open browser after delay
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000/api/monitoring/dashboard-v2"

:: Start server (scheduler runs automatically)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
