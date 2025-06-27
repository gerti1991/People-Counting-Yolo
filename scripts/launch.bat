@echo off
REM People Counting System - Windows Launcher
echo 🚀 People Counting System - Starting...

REM Set environment variable
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

REM Check if main.py exists
if not exist "main.py" (
    echo ❌ main.py not found. Please run from project root directory.
    pause
    exit /b 1
)

echo 🌐 Starting web application...
echo 📍 URL: http://localhost:8501
echo ⏹️ Press Ctrl+C to stop

REM Launch application
python -m streamlit run main.py --server.port 8501

echo 👋 Application stopped
pause
