# Build script for HestonVolSurface strategy
# This script configures and builds the C++ strategy using CMake and GCC

Write-Host "Building HestonVolSurface Strategy..." -ForegroundColor Green

# Create build directory if it doesn't exist
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Name "build"
}

# Change to build directory
Set-Location "build"

# Clean previous build
if (Test-Path "CMakeCache.txt") {
    Remove-Item -Recurse -Force CMakeCache.txt, CMakeFiles 2>$null
    Write-Host "Cleaned previous build files" -ForegroundColor Yellow
}

# Configure with CMake
Write-Host "Configuring with CMake..." -ForegroundColor Cyan
cmake .. -G "Unix Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

if ($LASTEXITCODE -eq 0) {
    Write-Host "Configuration successful!" -ForegroundColor Green
    
    # Build
    Write-Host "Building..." -ForegroundColor Cyan
    make
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build successful!" -ForegroundColor Green
        Write-Host "Executables created:" -ForegroundColor White
        Write-Host "  - hestonvolsurface_strategy.exe" -ForegroundColor White
        Write-Host "  - hestonvolsurface_backtest.exe" -ForegroundColor White
        
        # Test the main strategy
        Write-Host "`nTesting strategy..." -ForegroundColor Cyan
        .\hestonvolsurface_strategy.exe
    } else {
        Write-Host "Build failed!" -ForegroundColor Red
    }
} else {
    Write-Host "Configuration failed!" -ForegroundColor Red
}

# Return to parent directory
Set-Location ..
