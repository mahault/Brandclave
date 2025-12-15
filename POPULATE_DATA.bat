@echo off
title BrandClave - Populate Data
color 0B

echo.
echo  ============================================
echo    BrandClave - Populate Demo Data
echo  ============================================
echo.
echo  This will scrape content from 15+ sources
echo  and generate AI insights.
echo.
echo  Takes about 15-20 minutes.
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
echo   STEP 1: Scraping News Sources
echo  ============================================
echo.

echo [1/12] Skift...
python scripts/run_crawlers.py --source skift
echo [2/12] Hotel Dive...
python scripts/run_crawlers.py --source hoteldive
echo [3/12] Hotel Management...
python scripts/run_crawlers.py --source hotelmanagement
echo [4/12] PhocusWire...
python scripts/run_crawlers.py --source phocuswire
echo [5/12] Travel Weekly...
python scripts/run_crawlers.py --source travelweekly
echo [6/12] Hospitality Net...
python scripts/run_crawlers.py --source hospitalitynet
echo [7/12] Hotel News Resource...
python scripts/run_crawlers.py --source hotelnewsresource
echo [8/12] Boutique Hotelier...
python scripts/run_crawlers.py --source boutiquehotelier
echo [9/12] Hotel Tech Report...
python scripts/run_crawlers.py --source hoteltechreport

echo.
echo  ============================================
echo   STEP 2: Scraping Social Sources
echo  ============================================
echo.

echo [10/12] Reddit...
python scripts/run_crawlers.py --source reddit
echo [11/12] YouTube...
python scripts/run_crawlers.py --source youtube

echo.
echo  ============================================
echo   STEP 3: Processing Content
echo  ============================================
echo.

echo [12/12] Running NLP Pipeline...
python scripts/run_crawlers.py --process --limit 300

echo.
echo  ============================================
echo   STEP 4: Generating Intelligence
echo  ============================================
echo.

echo Generating Social Pulse trends...
python scripts/run_crawlers.py --trends --days 30

echo Extracting Hotelier Bets moves...
python scripts/run_crawlers.py --moves --days 30 --limit 100

echo Scanning sample property...
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
