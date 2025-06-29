# Stochastic Finance Strategy Demo
# This script demonstrates all C++ trading strategies in action

param(
    [int]$Runs = 1,
    [switch]$Detailed = $false
)

$strategies = @(
    @{Name="HestonVolSurface"; Path="strategies\HestonVolSurface\build\hestonvolsurface_strategy.exe"; Description="Heston Volatility Surface Model"},
    @{Name="JumpDiffusion"; Path="strategies\JumpDiffusion\build\jumpdiffusion_strategy.exe"; Description="Jump Diffusion Process Model"},
    @{Name="LogNormalJumpMeanReversion"; Path="strategies\LogNormalJumpMeanReversion\build\lognormaljumpmeanreversion_strategy.exe"; Description="Log-Normal Jump Mean Reversion Model"},
    @{Name="OrnsteinUhlenbeck"; Path="strategies\OrnsteinUhlenbeck\build\ornsteinuhlenbeck_strategy.exe"; Description="Ornstein-Uhlenbeck Process Model"}
)

function Show-Banner {
    Write-Host @"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        STOCHASTIC FINANCE STRATEGY DEMO                      ║
║                     High-Performance C++ Trading Strategies                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan
}

function Show-StrategyHeader {
    param([string]$Name, [string]$Description)
    
    Write-Host "`n" + "="*80 -ForegroundColor Yellow
    Write-Host "🚀 STRATEGY: $Name" -ForegroundColor Green
    Write-Host "📈 MODEL: $Description" -ForegroundColor White
    Write-Host "="*80 -ForegroundColor Yellow
}

function Test-Strategy {
    param($Strategy, [int]$RunNumber)
    
    Show-StrategyHeader -Name $Strategy.Name -Description $Strategy.Description
    
    if (-not (Test-Path $Strategy.Path)) {
        Write-Host "❌ Strategy executable not found: $($Strategy.Path)" -ForegroundColor Red
        Write-Host "💡 Run: .\build_strategies.ps1 -Strategy $($Strategy.Name)" -ForegroundColor Yellow
        return $false
    }
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    try {
        if ($Detailed) {
            Write-Host "⚡ Executing strategy (Run $RunNumber)..." -ForegroundColor Magenta
        }
        
        $output = & $Strategy.Path 2>&1
        $stopwatch.Stop()
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host $output -ForegroundColor White
            
            if ($Detailed) {
                Write-Host "✅ Execution completed successfully in $($stopwatch.ElapsedMilliseconds)ms" -ForegroundColor Green
                
                # Extract key metrics from output
                $lines = $output -split "`n"
                $buySignals = ($lines | Where-Object { $_ -match "Buy signals: (\d+)" } | ForEach-Object { $matches[1] })
                $sellSignals = ($lines | Where-Object { $_ -match "Sell signals: (\d+)" } | ForEach-Object { $matches[1] })
                $latency = ($lines | Where-Object { $_ -match "Average latency: ([\d.]+)" } | ForEach-Object { $matches[1] })
                
                if ($buySignals -and $sellSignals -and $latency) {
                    $totalSignals = [int]$buySignals + [int]$sellSignals
                    $signalRatio = if ($totalSignals -gt 0) { [math]::Round([double]$buySignals / $totalSignals * 100, 1) } else { 0 }
                    
                    Write-Host "📊 Key Metrics:" -ForegroundColor Cyan
                    Write-Host "   • Total Active Signals: $totalSignals" -ForegroundColor White
                    Write-Host "   • Buy/Total Ratio: $signalRatio%" -ForegroundColor White
                    Write-Host "   • Ultra-Low Latency: $latency μs" -ForegroundColor White
                }
            }
            return $true
        } else {
            Write-Host "❌ Strategy execution failed!" -ForegroundColor Red
            Write-Host $output -ForegroundColor Red
            return $false
        }
    }
    catch {
        $stopwatch.Stop()
        Write-Host "❌ Error executing strategy: $_" -ForegroundColor Red
        return $false
    }
}

# Main execution
Show-Banner

Write-Host "🔧 Configuration:" -ForegroundColor Cyan
Write-Host "   • Number of runs per strategy: $Runs" -ForegroundColor White
Write-Host "   • Detailed output: $($Detailed.ToString())" -ForegroundColor White
Write-Host "   • Total strategies: $($strategies.Count)" -ForegroundColor White

$totalTests = $strategies.Count * $Runs
$passedTests = 0
$startTime = Get-Date

for ($run = 1; $run -le $Runs; $run++) {
    if ($Runs -gt 1) {
        Write-Host "`n" + "🔄 EXECUTION ROUND $run of $Runs" -ForegroundColor Magenta
        Write-Host "="*50 -ForegroundColor Magenta
    }
    
    foreach ($strategy in $strategies) {
        if (Test-Strategy -Strategy $strategy -RunNumber $run) {
            $passedTests++
        }
        
        if ($strategy -ne $strategies[-1] -or $run -lt $Runs) {
            Start-Sleep -Milliseconds 500  # Brief pause between strategies
        }
    }
}

$endTime = Get-Date
$totalTime = ($endTime - $startTime).TotalSeconds

# Summary
Write-Host "`n" + "="*80 -ForegroundColor Cyan
Write-Host "📈 EXECUTION SUMMARY" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

Write-Host "✅ Successful executions: $passedTests/$totalTests" -ForegroundColor $(if ($passedTests -eq $totalTests) { "Green" } else { "Yellow" })
Write-Host "⏱️  Total execution time: $([math]::Round($totalTime, 2)) seconds" -ForegroundColor White
Write-Host "🚀 Average time per strategy: $([math]::Round($totalTime / $strategies.Count, 2)) seconds" -ForegroundColor White

if ($passedTests -eq $totalTests) {
    Write-Host "`n🎉 ALL STRATEGIES EXECUTED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "💡 Your C++ quantitative trading system is fully operational!" -ForegroundColor White
} else {
    Write-Host "`n⚠️  Some strategies failed to execute properly." -ForegroundColor Yellow
    Write-Host "💡 Check build configuration or run: .\build_strategies.ps1" -ForegroundColor White
}

Write-Host "`n🔧 Environment Details:" -ForegroundColor Cyan
Write-Host "   • GCC Version: $(gcc --version | Select-Object -First 1)" -ForegroundColor Gray
Write-Host "   • CMake Version: $(cmake --version | Select-Object -First 1)" -ForegroundColor Gray
Write-Host "   • Eigen Library: v3.4.0 (Header-only)" -ForegroundColor Gray
