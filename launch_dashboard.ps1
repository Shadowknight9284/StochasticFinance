# Interactive Dashboard Launcher
# Launch the web-based strategy visualization dashboard

Write-Host "🚀 Launching Interactive Strategy Dashboard..." -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Check if Python environment is configured
$pythonPath = "C:/Users/prana/OneDrive/Desktop/ALGO/StochasticFinance/.venv/Scripts/python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Host "❌ Python environment not found!" -ForegroundColor Red
    Write-Host "💡 Please run: configure_python_environment first" -ForegroundColor Yellow
    exit 1
}

Write-Host "🐍 Python environment: OK" -ForegroundColor Green
Write-Host "📊 Starting dashboard server..." -ForegroundColor Cyan

try {
    # Launch the dashboard
    & $pythonPath interactive_dashboard.py
}
catch {
    Write-Host "❌ Error launching dashboard: $_" -ForegroundColor Red
    Write-Host "💡 Make sure all packages are installed" -ForegroundColor Yellow
}

Write-Host "`n🎉 Dashboard session ended." -ForegroundColor Green
