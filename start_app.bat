@echo off
echo ===============================================
echo  🚀 PEOPLE COUNTING SYSTEM LAUNCHER
echo ===============================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in PATH
    echo 💡 Please install Python or add it to your PATH
    pause
    exit /b 1
)

echo ✅ Python found
echo.

:: Check if Streamlit is available
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit not installed
    echo 💡 Installing Streamlit...
    pip install streamlit
    if errorlevel 1 (
        echo ❌ Failed to install Streamlit
        pause
        exit /b 1
    )
)

echo ✅ Streamlit ready
echo.

:: Quick system check
echo 🔍 Running quick system check...
python quick_test.py

echo.
echo ===============================================
echo  🚀 STARTING PEOPLE COUNTING SYSTEM...
echo ===============================================
echo.
echo 💡 The app will open in your web browser
echo 💡 Press Ctrl+C to stop the app
echo.

:: Start the application
streamlit run app.py

echo.
echo ===============================================
echo  👋 PEOPLE COUNTING SYSTEM STOPPED
echo ===============================================
pause
