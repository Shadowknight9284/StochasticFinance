# Simple Strategy Demo Script
# Demonstrates all C++ trading strategies

Write-Host "STOCHASTIC FINANCE STRATEGY DEMO" -ForegroundColor Cyan
Write-Host "High-Performance C++ Trading Strategies" -ForegroundColor White
Write-Host "=======================================" -ForegroundColor Cyan

$strategies = @(
    @{Name="HestonVolSurface"; Path="strategies\HestonVolSurface\build\hestonvolsurface_strategy.exe"},
    @{Name="JumpDiffusion"; Path="strategies\JumpDiffusion\build\jumpdiffusion_strategy.exe"},
    @{Name="LogNormalJumpMeanReversion"; Path="strategies\LogNormalJumpMeanReversion\build\lognormaljumpmeanreversion_strategy.exe"},
    @{Name="OrnsteinUhlenbeck"; Path="strategies\OrnsteinUhlenbeck\build\ornsteinuhlenbeck_strategy.exe"}
)

$successful = 0
$total = $strategies.Count

foreach ($strategy in $strategies) {
    Write-Host "`n=== TESTING STRATEGY: $($strategy.Name) ===" -ForegroundColor Yellow
    
    if (Test-Path $strategy.Path) {
        try {
            $output = & $strategy.Path
            if ($LASTEXITCODE -eq 0) {
                Write-Host $output -ForegroundColor White
                Write-Host "STATUS: SUCCESS" -ForegroundColor Green
                $successful++
            } else {
                Write-Host "STATUS: FAILED" -ForegroundColor Red
            }
        }
        catch {
            Write-Host "ERROR: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "ERROR: Executable not found at $($strategy.Path)" -ForegroundColor Red
        Write-Host "Run: .\build_strategies.ps1 -Strategy $($strategy.Name)" -ForegroundColor Yellow
    }
}

Write-Host "`n=======================================" -ForegroundColor Cyan
Write-Host "SUMMARY: $successful/$total strategies executed successfully" -ForegroundColor $(if ($successful -eq $total) { "Green" } else { "Yellow" })
Write-Host "=======================================" -ForegroundColor Cyan
