@echo off
title BrandClave - Populate Data
color 0B

echo.
echo  ============================================
echo    BrandClave - Populate Demo Data
echo  ============================================
echo.
echo  This will scrape content from 12 reliable sources
echo  and generate AI insights.
echo.
echo  Takes about 10-15 minutes.
echo.
pause

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

echo.
echo  ============================================
echo   STEP 1: Scraping News Sources (Reliable)
echo  ============================================
echo.

echo [1/14] Skift...
python scripts/run_crawlers.py --source skift
echo [2/14] Hotel Dive...
python scripts/run_crawlers.py --source hoteldive
echo [3/14] Hotel Management...
python scripts/run_crawlers.py --source hotelmanagement
echo [4/14] Top Hotel News...
python scripts/run_crawlers.py --source tophotelnews
echo [5/14] SiteMinder...
python scripts/run_crawlers.py --source siteminder
echo [6/14] EHL Insights...
python scripts/run_crawlers.py --source ehlinsights
echo [7/14] eHotelier...
python scripts/run_crawlers.py --source ehotelier
echo [8/14] Lodging Magazine...
python scripts/run_crawlers.py --source lodgingmagazine
echo [9/14] Luxury Hospitality...
python scripts/run_crawlers.py --source luxuryhospitality
echo [10/14] Hotel Business...
python scripts/run_crawlers.py --source hotelbusiness

echo.
echo  ============================================
echo   STEP 2: Scraping Social Sources
echo  ============================================
echo.

echo [11/14] Reddit...
python scripts/run_crawlers.py --source reddit
echo [12/14] YouTube...
python scripts/run_crawlers.py --source youtube

echo.
echo  ============================================
echo   STEP 3: Processing Content
echo  ============================================
echo.

echo [13/16] Running NLP Pipeline...
python scripts/run_crawlers.py --process --limit 300

echo.
echo  ============================================
echo   STEP 4: Generating Intelligence
echo  ============================================
echo.

echo [14/16] Generating Social Pulse trends (with quality filtering)...
python scripts/regenerate_trends.py

echo [15/16] Extracting Hotelier Bets moves...
python scripts/run_crawlers.py --moves --days 30 --limit 100

echo [16/16] Scanning sample property...
python scripts/run_crawlers.py --scan "https://acehotel.com/new-york/"

echo.
echo  ============================================
echo.
echo   Data population complete!
echo.
echo   Run START_DEMO.bat to see your dashboard.
echo.
echo  ============================================
echo.
pause
