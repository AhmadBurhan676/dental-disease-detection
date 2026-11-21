@echo off
title DentalVision AI - Dual Web Suite (Streamlit + FastAPI)
color 0b

echo =====================================================================
echo           DentalVision AI - Web Application & GPU Engine
echo       3 Dental Diseases Detection: Caries, Periapical, Periodontitis
echo                 Optimized for NVIDIA GeForce RTX 5060 Ti
echo =====================================================================
echo.

set PYTHON_CMD=C:\Users\pc\AppData\Local\Programs\Python\Python311\python.exe
if not exist "%PYTHON_CMD%" (
    set PYTHON_CMD=python
)

echo [1/3] Python Environment:
"%PYTHON_CMD%" --version
echo.

echo [2/3] Choose which application to launch:
echo.
echo   1. Launch Streamlit Web App (Recommended - http://localhost:8501)
echo   2. Launch FastAPI Full Clinical Web App (http://localhost:8000)
echo   3. Launch Both Simultaneously
echo.
set /p APP_CHOICE="Enter your choice (1, 2, or 3) [Default: 1]: "

if "%APP_CHOICE%"=="2" goto launch_fastapi
if "%APP_CHOICE%"=="3" goto launch_both

:launch_streamlit
echo.
echo Launching Streamlit Web App on http://localhost:8501 ...
"%PYTHON_CMD%" -m streamlit run streamlit_app.py --server.port 8501
goto end

:launch_fastapi
echo.
echo Launching FastAPI Clinical Web App on http://localhost:8000 ...
"%PYTHON_CMD%" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
goto end

:launch_both
echo.
echo Starting FastAPI Backend on http://localhost:8000 in background...
start /b "" "%PYTHON_CMD%" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
echo Starting Streamlit Frontend on http://localhost:8501 ...
"%PYTHON_CMD%" -m streamlit run streamlit_app.py --server.port 8501
goto end

:end
pause
