# Pairs Trading Algorithm with Ornstein-Uhlenbeck Model

A com## Development Timeline

- ✅ **Day 1**: Project setup and configuration
- ✅ **Day 2**: Enhanced data infrastructure  
- 🔄 **Day 3**: Cointegration and pair selection + **Streamlit Dashboard Foundation**
- 📅 **Day 4-5**: Ornstein-Uhlenbeck model + **Interactive OU Model Validation**
- 📅 **Day 6**: Signal generation + **Real-time Signal Dashboard**
- 📅 **Day 7**: Trading strategy + **Strategy Performance Dashboard**
- 📅 **Day 8**: Risk management + **Risk Management Dashboard**
- 📅 **Day 9-11**: Backtesting engine + **Interactive Backtest Dashboard**
- 📅 **Day 12**: Optimization + **Parameter Optimization Interface**
- 📅 **Day 13-14**: Integration and testing + **Complete Dashboard Integration**
- 📅 **Day 15**: Production readiness + **Live Trading Dashboard**gorithmic trading system for pairs trading using the Ornstein-Uhlenbeck mean-reverting process model with **interactive Streamlit dashboard** for real-time model visualization and validation.

## Project Structure

```
AlgorithmicTradingScanner/
├── dashboard/                     # Streamlit Web Interface
│   ├── __init__.py
│   ├── app.py                    # Main Streamlit app
│   ├── pages/                    # Dashboard pages
│   │   ├── pair_selection.py     # Pair selection interface
│   │   ├── model_validation.py   # OU model validation
│   │   ├── live_trading.py       # Live trading monitor
│   │   └── backtesting.py        # Backtest results
│   └── components/               # Reusable components
│       ├── charts.py             # Interactive charts
│       ├── metrics.py            # Performance metrics
│       └── controls.py           # User controls
├── src/                          # Source code
│   ├── data/                      # Data management
│   │   ├── __init__.py
│   │   └── stock_data.py          # Enhanced Stock class (Day 2)
│   ├── models/                    # Mathematical models
│   │   ├── __init__.py
│   │   ├── ornstein_uhlenbeck.py  # OU model (Day 4-5)
│   │   └── cointegration.py       # Cointegration tests (Day 3)
│   ├── strategy/                  # Trading strategy
│   │   ├── __init__.py
│   │   ├── pairs_finder.py        # Pair identification (Day 3)
│   │   ├── pairs_trader.py        # Trading logic (Day 7)
│   │   └── signals.py             # Signal generation (Day 6)
│   ├── risk/                      # Risk management
│   │   ├── __init__.py
│   │   └── portfolio_manager.py   # Risk controls (Day 8)
│   ├── backtesting/              # Backtesting engine
│   │   ├── __init__.py
│   │   ├── backtester.py         # Main engine (Day 9-11)
│   │   ├── performance.py        # Metrics (Day 9-11)
│   │   └── visualization.py      # Charts (Day 9-11)
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── config.py             # Configuration ✓
│       └── helpers.py            # Helper functions ✓
├── tests/                        # Unit tests
│   └── test_setup.py            # Setup tests ✓
├── data/                         # Raw data storage
├── results/                      # Backtest results
├── notebooks/                    # Jupyter notebooks
├── Stock.py                      # Original Stock class ✓
├── requirements.txt              # Dependencies ✓
├── main.py                       # Main execution ✓
└── README.md                     # This file ✓
```

## Dependencies

- **Core**: pandas, numpy, yfinance
- **Statistics**: scipy, statsmodels
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Time Series**: arch

## Development Timeline

- ✅ **Day 1**: Project setup and configuration
- ✅ **Day 2**: Enhanced data infrastructure  
- � **Day 3**: Cointegration and pair selection
- 📅 **Day 4-5**: Ornstein-Uhlenbeck model
- 📅 **Day 6**: Signal generation
- 📅 **Day 7**: Trading strategy
- 📅 **Day 8**: Risk management
- 📅 **Day 9-11**: Backtesting engine
- 📅 **Day 12**: Optimization
- 📅 **Day 13-14**: Integration and testing
- 📅 **Day 15**: Production readiness

## Usage

```bash
# Run the Streamlit dashboard
streamlit run dashboard/app.py

# Run the main algorithm
python main.py

# With custom parameters
python main.py --start_date 2020-01-01 --end_date 2024-12-31 --verbose

# Run tests
python -m pytest tests/
```

## Streamlit Dashboard Features

### **🎯 Interactive Web Interface:**
- **Real-time pair selection** with filtering and ranking
- **Live model validation** with parameter adjustment
- **Interactive charts** with zoom, pan, and data export
- **Dynamic backtesting** with parameter optimization
- **Risk monitoring** with real-time alerts
- **Performance analytics** with drill-down capabilities

### **📊 Dashboard Pages:**
1. **Pair Selection** - Interactive correlation analysis and pair filtering
2. **Model Validation** - OU model diagnostics and goodness-of-fit tests  
3. **Live Trading** - Real-time signals and position monitoring
4. **Backtesting** - Historical performance analysis and optimization
5. **Risk Dashboard** - Portfolio risk metrics and alerts

## Visual Model Validation Features

### **Core Visualization Components:**
1. **Pair Selection Visualization** (Day 3)
   - Correlation heatmaps and scatter plots
   - Cointegration test results dashboard
   - Price spread evolution charts
   - Statistical significance validation plots

2. **Ornstein-Uhlenbeck Model Visualization** (Day 4-5)
   - OU parameter estimation convergence plots
   - Mean reversion visualization with confidence bands
   - Half-life analysis and validation
   - Model goodness-of-fit diagnostics

3. **Signal Generation Visualization** (Day 6)
   - Real-time spread z-score plots
   - Entry/exit signal markers
   - Threshold sensitivity analysis
   - Signal quality metrics dashboard

4. **Strategy Performance Visualization** (Day 7)
   - Live trading signals overlay
   - Position sizing visualization
   - Strategy performance attribution
   - Risk-adjusted returns analysis

5. **Interactive Performance Dashboard** (Day 9-11)
   - Real-time equity curve with drawdowns
   - Rolling performance metrics
   - Trade-level analysis with drill-down
   - Market regime detection plots
   - Parameter sensitivity heatmaps

### **Model Validation Suite:**
- **Statistical Tests Visualization**: ADF, Johansen, Granger causality
- **Residual Analysis**: QQ plots, autocorrelation, heteroscedasticity
- **Out-of-Sample Performance**: Walk-forward analysis charts
- **Risk Metrics Dashboard**: VaR, Expected Shortfall, Maximum Drawdown
- **Model Stability Analysis**: Parameter drift detection over time

## Configuration

The algorithm is highly configurable through `src/utils/config.py`. Key parameters:

- **Data**: Market filters, date ranges
- **OU Model**: Estimation methods, thresholds
- **Trading**: Entry/exit signals, position sizing
- **Risk**: Position limits, drawdown controls
- **Backtesting**: Transaction costs, benchmark

## Phase 2 Status: ✅ COMPLETE

### Completed:
- Enhanced Stock class with pairs trading capabilities
- Multi-stock data manager for batch operations
- Database schema with indexes and optimization
- Data validation and quality checks
- Correlation matrix calculations
- Aligned data retrieval for pairs
- Configuration updates for data module
- Unit tests for all components

### Ready for Day 3:
- Cointegration testing implementation
- Pair selection algorithms
- Statistical significance validation
