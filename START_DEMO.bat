@echo off
title BrandClave Demo
color 0A

echo.
echo  ============================================
echo       BrandClave Aggregator - Demo Mode
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

echo.
echo  ============================================
echo   Collecting Fresh Data (15-20 minutes)
echo  ============================================
echo.
echo   Scraping 24 hospitality sources...
echo.

:: News sources - Primary (13)
echo   [1/25] Skift...
python scripts/run_crawlers.py --source skift >nul 2>&1
echo   [2/25] HospitalityNet...
python scripts/run_crawlers.py --source hospitalitynet >nul 2>&1
echo   [3/25] HotelDive...
python scripts/run_crawlers.py --source hoteldive >nul 2>&1
echo   [4/25] PhocusWire...
python scripts/run_crawlers.py --source phocuswire >nul 2>&1
echo   [5/25] HotelManagement...
python scripts/run_crawlers.py --source hotelmanagement >nul 2>&1
echo   [6/25] TravelWeekly...
python scripts/run_crawlers.py --source travelweekly >nul 2>&1
echo   [7/25] HotelNewsResource...
python scripts/run_crawlers.py --source hotelnewsresource >nul 2>&1
echo   [8/25] TravelDailyNews...
python scripts/run_crawlers.py --source traveldailynews >nul 2>&1
echo   [9/25] BusinessTravelNews...
python scripts/run_crawlers.py --source businesstravelnews >nul 2>&1
echo   [10/25] BoutiqueHotelier...
python scripts/run_crawlers.py --source boutiquehotelier >nul 2>&1
echo   [11/25] HotelOnline...
python scripts/run_crawlers.py --source hotelonline >nul 2>&1
echo   [12/25] HotelTechReport...
python scripts/run_crawlers.py --source hoteltechreport >nul 2>&1
echo   [13/25] TopHotelNews...
python scripts/run_crawlers.py --source tophotelnews >nul 2>&1

:: News sources - Research & Insights (6)
echo   [14/25] SiteMinder...
python scripts/run_crawlers.py --source siteminder >nul 2>&1
echo   [15/25] EHL Insights...
python scripts/run_crawlers.py --source ehlinsights >nul 2>&1
echo   [16/25] CBRE Hotels...
python scripts/run_crawlers.py --source cbrehotels >nul 2>&1
echo   [17/25] Cushman Wakefield...
python scripts/run_crawlers.py --source cushmanwakefield >nul 2>&1
echo   [18/25] CoStar...
python scripts/run_crawlers.py --source costar >nul 2>&1
echo   [19/25] TravelDaily...
python scripts/run_crawlers.py --source traveldaily >nul 2>&1

:: Social sources (3)
echo   [20/25] Reddit...
python scripts/run_crawlers.py --source reddit >nul 2>&1
echo   [21/25] YouTube...
python scripts/run_crawlers.py --source youtube >nul 2>&1
echo   [22/25] Quora...
python scripts/run_crawlers.py --source quora >nul 2>&1

:: Review sources (2)
echo   [23/25] TripAdvisor...
python scripts/run_crawlers.py --source tripadvisor >nul 2>&1
echo   [24/25] Booking.com...
python scripts/run_crawlers.py --source booking >nul 2>&1

:: Process and generate insights
echo   [25/25] Analyzing with AI...
python scripts/run_crawlers.py --process --limit 400 >nul 2>&1
python scripts/run_crawlers.py --trends --days 30 >nul 2>&1
python scripts/run_crawlers.py --moves --days 30 --limit 100 >nul 2>&1

echo.
echo   Data collection complete!
echo.
echo  ============================================
echo   Starting Dashboard
echo  ============================================
echo.
echo   Opening browser...
echo.
echo   Dashboard:     http://localhost:8000/api/monitoring/dashboard
echo   API Docs:      http://localhost:8000/docs
echo.
echo  ============================================
echo.
echo   Press Ctrl+C to stop the server
echo.

:: Open browser after delay
start "" timeout /t 2 /nobreak >nul & start http://localhost:8000/api/monitoring/dashboard

:: Start server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
