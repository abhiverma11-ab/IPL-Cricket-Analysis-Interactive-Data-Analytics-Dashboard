@echo off
echo ============================================
echo   IPL Cricket Analysis - Starting up...
echo ============================================
cd /d "%~dp0"

echo.
echo Installing required packages (this may take a minute the first time)...
pip install -r requirements.txt

echo.
echo Launching the app - your browser will open automatically...
streamlit run app.py

pause
