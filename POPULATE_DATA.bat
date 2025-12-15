@echo off
title BrandClave - Populate Data
color 0B

echo.
echo  ============================================
echo    BrandClave - Populate Demo Data
echo  ============================================
echo.
echo  This will scrape content and generate insights.
echo  It may take 10-15 minutes.
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
echo   Step 1/5: Scraping News (Skift)
echo  ============================================
echo.
python scripts/run_crawlers.py --source skift -v
if %errorlevel% neq 0 echo [WARNING] Skift scraper had issues

echo.
echo  ============================================
echo   Step 2/5: Scraping Social (Reddit)
echo  ============================================
echo.
python scripts/run_crawlers.py --source reddit -v
if %errorlevel% neq 0 echo [WARNING] Reddit scraper had issues

echo.
echo  ============================================
echo   Step 3/5: Scraping Social (YouTube)
echo  ============================================
echo.
python scripts/run_crawlers.py --source youtube -v
if %errorlevel% neq 0 echo [WARNING] YouTube scraper had issues

echo.
echo  ============================================
echo   Step 4/5: Processing with NLP Pipeline
echo  ============================================
echo.
python scripts/run_crawlers.py --process --limit 200 -v
if %errorlevel% neq 0 echo [WARNING] NLP pipeline had issues

echo.
echo  ============================================
echo   Step 5/5: Generating Insights
echo  ============================================
echo.
echo Generating Social Pulse trends...
python scripts/run_crawlers.py --trends --days 30 -v
if %errorlevel% neq 0 echo [WARNING] Trend generation had issues

echo.
echo Extracting Hotelier Bets moves...
python scripts/run_crawlers.py --moves --days 30 --limit 50 -v
if %errorlevel% neq 0 echo [WARNING] Move extraction had issues

echo.
echo Scanning sample property...
python scripts/run_crawlers.py --scan "https://acehotel.com/new-york/" -v
if %errorlevel% neq 0 echo [WARNING] Property scan had issues

echo.
echo  ============================================
echo.
echo   Data population complete!
echo.
echo   Now run START_DEMO.bat to see your data
echo   in the dashboard.
echo.
echo  ============================================
echo.
pause
