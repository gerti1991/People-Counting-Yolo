# People Counting System - PowerShell Launcher
Write-Host "🚀 People Counting System - Starting..." -ForegroundColor Green

# Set environment variable
$env:STREAMLIT_SERVER_FILE_WATCHER_TYPE = "none"

# Check if main.py exists
if (-not (Test-Path "main.py")) {
    Write-Host "❌ main.py not found. Please run from project root directory." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🌐 Starting web application..." -ForegroundColor Cyan
Write-Host "📍 URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host "⏹️ Press Ctrl+C to stop" -ForegroundColor Magenta
Write-Host ""

try {
    # Launch application
    python -m streamlit run main.py --server.port 8501
}
catch {
    Write-Host "❌ Error launching application: $_" -ForegroundColor Red
    Write-Host "💡 Try running manually: streamlit run main.py" -ForegroundColor Yellow
}

Write-Host "👋 Application stopped" -ForegroundColor Green
Read-Host "Press Enter to exit"
