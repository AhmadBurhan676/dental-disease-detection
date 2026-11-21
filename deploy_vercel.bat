@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo   DentalVision AI ^| Vercel Deployment
echo ============================================================
echo.

cd /d "c:\Users\pc\Downloads\dental detection1"

:: Refresh PATH to include newly installed Node/Git
for /f "tokens=*" %%i in ('powershell -Command "[System.Environment]::GetEnvironmentVariable(\"Path\",\"Machine\")"') do set MACHINEPATH=%%i
for /f "tokens=*" %%i in ('powershell -Command "[System.Environment]::GetEnvironmentVariable(\"Path\",\"User\")"') do set USERPATH=%%i
set PATH=%MACHINEPATH%;%USERPATH%;%PATH%

:: Enable npm scripts
powershell -Command "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force" >nul 2>&1

:: Verify Node
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org then re-run.
    pause & exit /b 1
)
for /f %%v in ('node --version') do echo [OK] Node.js %%v

:: Install Vercel CLI if needed
vercel --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/4] Installing Vercel CLI...
    call npm install -g vercel
) else (
    for /f %%v in ('vercel --version') do echo [OK] Vercel CLI %%v
)

:: Initialize git if available, else use Vercel drag-and-drop
git --version >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo [2/4] Initializing Git repository...
    git init
    git add vercel.json requirements.txt .vercelignore api/ public/
    git config --local user.email "dental@vision.ai"
    git config --local user.name "DentalVision AI"
    git commit -m "DentalVision AI v2.0 - Vercel deployment" --allow-empty
) else (
    echo [2/4] Git not found — Vercel will deploy directly without git.
)

:: Run Vercel deploy
echo.
echo [3/4] Deploying to Vercel (browser login may open)...
echo.
vercel --prod --yes

echo.
echo ============================================================
echo   Done! Check the URL printed above.
echo ============================================================
pause
