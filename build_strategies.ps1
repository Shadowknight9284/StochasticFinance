# Master build script for all C++ strategies
# This script builds all strategy implementations in the strategies directory

param(
    [string]$Strategy = "all",
    [switch]$Clean = $false
)

$strategies = @("HestonVolSurface", "JumpDiffusion", "LogNormalJumpMeanReversion", "OrnsteinUhlenbeck")

function Build-Strategy {
    param([string]$StrategyName)
    
    $strategyPath = "strategies\$StrategyName"
    
    if (-not (Test-Path $strategyPath)) {
        Write-Host "Strategy '$StrategyName' not found!" -ForegroundColor Red
        return $false
    }
    
    Write-Host "`n=== Building $StrategyName ===" -ForegroundColor Green
    
    Push-Location $strategyPath
    
    try {
        # Create build directory if it doesn't exist
        if (-not (Test-Path "build")) {
            New-Item -ItemType Directory -Name "build" | Out-Null
        }
        
        Set-Location "build"
        
        # Clean if requested
        if ($Clean -and (Test-Path "CMakeCache.txt")) {
            Remove-Item -Recurse -Force CMakeCache.txt, CMakeFiles 2>$null
            Write-Host "Cleaned previous build files" -ForegroundColor Yellow
        }
        
        # Configure with CMake
        Write-Host "Configuring with CMake..." -ForegroundColor Cyan
        cmake .. -G "Unix Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ 2>$null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Configuration successful!" -ForegroundColor Green
            
            # Build
            Write-Host "Building..." -ForegroundColor Cyan
            make 2>$null
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Build successful for $StrategyName!" -ForegroundColor Green
                return $true
            } else {
                Write-Host "Build failed for $StrategyName!" -ForegroundColor Red
                return $false
            }
        } else {
            Write-Host "Configuration failed for $StrategyName!" -ForegroundColor Red
            return $false
        }
    }
    finally {
        Pop-Location
    }
}

# Main execution
Write-Host "C++ Strategy Builder" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan

$builtCount = 0
$totalCount = 0

if ($Strategy -eq "all") {
    foreach ($strat in $strategies) {
        $totalCount++
        if (Build-Strategy $strat) {
            $builtCount++
        }
    }
} else {
    $totalCount = 1
    if (Build-Strategy $Strategy) {
        $builtCount = 1
    }
}

Write-Host "`n=== Build Summary ===" -ForegroundColor Cyan
Write-Host "Successfully built: $builtCount/$totalCount strategies" -ForegroundColor $(if ($builtCount -eq $totalCount) { "Green" } else { "Yellow" })

if ($builtCount -gt 0) {
    Write-Host "`nTo run a strategy, navigate to its build directory:" -ForegroundColor White
    Write-Host "  cd strategies\[StrategyName]\build" -ForegroundColor Gray
    Write-Host "  .\[strategyname]_strategy.exe" -ForegroundColor Gray
}
