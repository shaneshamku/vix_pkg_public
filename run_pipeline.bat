@echo off
REM ===== CONFIG =====
set ROOT=C:\Users\shane\OneDrive\Desktop\Shane Shamku\Projects\New VIX\vix_pkg
set PY=%ROOT%\venv\Scripts\python.exe
set SCRIPT=%ROOT%\vix_pkg\finalorders\run_daily_pipeline.py
set LOGDIR=%ROOT%\logs
set LOGFILE=%LOGDIR%\pipeline.log

REM ===== PREP =====
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
cd /d "%ROOT%"

REM ===== RUN =====
echo ========================= >> "%LOGFILE%"
echo START %DATE% %TIME% >> "%LOGFILE%"
"%PY%" "%SCRIPT%" >> "%LOGFILE%" 2>&1
echo END   %DATE% %TIME% >> "%LOGFILE%"
