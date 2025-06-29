# C++ Build Environment Setup

This document describes the C++ build environment for the Stochastic Finance strategies.

## Prerequisites

✅ **GCC 13.2.0** - C++ compiler (already installed)
✅ **CMake 4.0.3** - Build system generator (already installed)  
✅ **Eigen 3.4.0** - Linear algebra library (already installed in `external/eigen-3.4.0`)

## Project Structure

```
StochasticFinance/
├── external/
│   └── eigen-3.4.0/           # Eigen header-only library
├── strategies/
│   ├── HestonVolSurface/       # Heston volatility surface strategy
│   │   ├── model.cpp           # Main strategy implementation
│   │   ├── hestonvolsurface.hpp # Header file
│   │   ├── CMakeLists.txt      # Build configuration
│   │   ├── build/              # Build directory
│   │   └── backtest/           # Backtesting code
│   ├── JumpDiffusion/          # Jump diffusion strategy
│   ├── LogNormalJumpMeanReversion/ # Log-normal jump mean reversion
│   └── OrnsteinUhlenbeck/      # Ornstein-Uhlenbeck process
├── build_strategies.ps1       # Master build script
└── README_CPP_BUILD.md        # This file
```

## Building Strategies

### Option 1: Build All Strategies
```powershell
.\build_strategies.ps1
```

### Option 2: Build Specific Strategy
```powershell
.\build_strategies.ps1 -Strategy HestonVolSurface
```

### Option 3: Clean Build
```powershell
.\build_strategies.ps1 -Strategy HestonVolSurface -Clean
```

### Option 4: Manual Build
```powershell
cd strategies\HestonVolSurface\build
cmake .. -G "Unix Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
make
```

## Running Strategies

After building, navigate to the strategy's build directory and run:

```powershell
cd strategies\HestonVolSurface\build
.\hestonvolsurface_strategy.exe    # Main strategy
.\hestonvolsurface_backtest.exe    # Backtesting engine
```

## Build Configuration

The CMakeLists.txt files are configured to:
- Use C++17 standard
- Include Eigen from `external/eigen-3.4.0`
- Apply GCC-specific optimizations
- Enable debugging symbols in debug mode
- Use aggressive optimizations in release mode

## Performance Features

- **Low Latency**: Sub-microsecond execution times
- **Vectorized Operations**: Using Eigen for matrix computations
- **Compile-time Optimizations**: Template-based implementations
- **Memory Efficient**: Header-only Eigen library

## Troubleshooting

### Missing Includes
If you see compilation errors about missing std::cout, std::accumulate, etc., add these includes:
```cpp
#include <iostream>
#include <numeric>
#include <random>
```

### CMake Configuration Issues
If CMake can't find Eigen, verify the path in CMakeLists.txt:
```cmake
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/eigen-3.4.0")
```

### Build Errors
1. Clean the build directory: `Remove-Item -Recurse -Force build\*`
2. Reconfigure: `cmake .. -G "Unix Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++`
3. Rebuild: `make`

## Example Output

```
Strategy calibrated with parameters:
Mean reversion speed: -198.728
Long-term mean: 99.9392
Variance: 3.97062
Signal distribution:
Buy signals: 581
Sell signals: 729
Hold signals: 8690
Performance metrics:
Average latency: 0.089 μs
Meets latency constraint: YES
```

# Visual Backtesting System

The project now includes a comprehensive visual backtesting system to understand how the stochastic models work in practice.

## 🎨 Visualization Tools

#### 1. Enhanced Visual Backtester (`enhanced_visual_backtester.py`)
Static comprehensive analysis with matplotlib:
```powershell
C:/Users/prana/OneDrive/Desktop/ALGO/StochasticFinance/.venv/Scripts/python.exe enhanced_visual_backtester.py
```

**Features:**
- 📈 Price charts with trading signals
- 💰 P&L evolution comparison
- ⚖️ Risk vs Return analysis
- 🎯 Trading statistics
- 📊 Performance metrics dashboard
- 💾 Saves results as PNG file

#### 2. Interactive Web Dashboard (`interactive_dashboard.py`)
Real-time web-based visualization:
```powershell
.\launch_dashboard.ps1
```

**Features:**
- 🔄 Real-time updates every 5 seconds
- 📊 Interactive charts with Plotly
- ⚙️ Configurable time periods
- 🎯 Hover data and zooming
- 📱 Responsive web interface
- 🌐 Accessible at `http://localhost:8050`

## 📊 Strategy Visualization Components

#### Market Price & Trading Signals
- Shows market price evolution
- Overlays buy/sell signals for each strategy
- Color-coded by strategy type
- Triangular markers for signal points

#### P&L Evolution
- Compares portfolio performance over time
- Shows cumulative returns
- Identifies best/worst performing strategies
- Risk-adjusted performance metrics

#### Performance Metrics Dashboard
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Calmar Ratio**: Return vs maximum drawdown
- **Number of Trades**: Trading frequency

#### Risk Analysis
- Risk vs Return scatter plot
- Volatility measurements
- Drawdown analysis
- Strategy comparison matrix

## 🧮 Model Implementations

Each strategy is visually simulated with realistic trading logic:

#### 1. **Heston Volatility Surface**
- **Logic**: Volatility-sensitive mean reversion
- **Signals**: Buy on low vol + oversold, Sell on high vol + overbought
- **Characteristics**: Moderate trading frequency, volatility-aware

#### 2. **Jump Diffusion**
- **Logic**: Jump detection with momentum
- **Signals**: Counter-trend on jumps, trend-following otherwise
- **Characteristics**: High trading frequency, jump-reactive

#### 3. **Log-Normal Jump Mean Reversion**
- **Logic**: Log-normal mean reversion
- **Signals**: Based on log price deviations
- **Characteristics**: Mathematical precision, longer holding periods

#### 4. **Ornstein-Uhlenbeck**
- **Logic**: Strong mean reversion
- **Signals**: Buy extreme oversold, Sell extreme overbought
- **Characteristics**: Patient strategy, high conviction trades

## 📈 Understanding the Results

#### Performance Interpretation
- **Positive Returns**: Strategy generated profit
- **Sharpe Ratio > 1**: Good risk-adjusted performance
- **Low Max Drawdown**: Conservative risk management
- **High Win Rate**: Consistent profitable trades

#### Visual Patterns
- **Clustered Signals**: High volatility periods
- **Smooth P&L**: Stable performance
- **Signal Divergence**: Strategies disagree on market direction
- **Performance Spread**: Different risk/return profiles

## 🚀 Quick Start

1. **Build all strategies**:
   ```powershell
   .\build_strategies.ps1
   ```

2. **Run static analysis**:
   ```powershell
   C:/Users/prana/OneDrive/Desktop/ALGO/StochasticFinance/.venv/Scripts/python.exe enhanced_visual_backtester.py
   ```

3. **Launch interactive dashboard**:
   ```powershell
   .\launch_dashboard.ps1
   ```

4. **View results**:
   - Static: `strategy_analysis_dashboard.png`
   - Interactive: `http://localhost:8050`
