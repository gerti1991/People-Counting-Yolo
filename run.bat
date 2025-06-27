@echo off
echo 🚀 Starting People Counting System...

REM Set environment variables to avoid PyTorch-Streamlit conflicts
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

echo 🌐 Launching web application...
echo 📍 URL: http://localhost:8501
echo ⏹️ Press Ctrl+C to stop

streamlit run app.py --server.port 8501

echo 👋 Application stopped
pause
