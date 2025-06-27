# People Counter Launcher Script
Write-Host "🚀 Starting People Counting System..." -ForegroundColor Green

# Set environment variables to avoid PyTorch-Streamlit conflicts
$env:STREAMLIT_SERVER_FILE_WATCHER_TYPE = "none"

Write-Host "🌐 Launching web application..." -ForegroundColor Cyan
Write-Host "📍 URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host "⏹️ Press Ctrl+C to stop" -ForegroundColor Magenta

streamlit run app.py --server.port 8501
