# StochasticFinance

Statistical-arbitrage research code in Python: a cached daily-price data layer and cointegration tests
for finding mean-reverting stock pairs, the groundwork for an Ornstein–Uhlenbeck pairs-trading strategy.

**Status:** the data layer and the statistics work and have tests. The pair-search pipeline, the OU model,
signals and backtesting are still in progress (see [Roadmap](#roadmap)).

## What works today
- **Price data with a local cache.** Downloads daily OHLCV and adjusted close from Yahoo Finance
  (`yfinance`) into SQLite, so repeat runs read from disk. It can align two tickers by date and compute
  returns, log returns and rolling volatility, and it checks data quality (by default at least 252
  observations). The code is in `src/data/`.
- **Cointegration tests.** `src/models/cointegration.py` has three parts:
  - The **Engle–Granger** two-step test: an OLS hedge ratio, then an ADF test on the spread.
  - The **Johansen** test for multivariate cointegration.
  - The **half-life of mean reversion**, estimated from the spread's AR(1) coefficient.
  - `PairValidator` also screens pairs by correlation band (0.5–0.95) before testing.
- **Configuration.** Typed dataclasses in `src/utils/config.py` cover data, cointegration, pair selection,
  the OU model, trading thresholds (z-score entry 2.0, exit 0.5, stop 3.5), risk limits and backtests.
- **Dashboard.** A Streamlit app (`dashboard/app.py`) with an overview page and a pair-selection page. The
  model-validation, live-trading, backtesting and risk pages are placeholders.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests           # 7 tests
streamlit run dashboard/app.py   # dashboard at http://localhost:8501
```

Test one pair for cointegration:
```python
from src.data.stock_data import MultiStockManager
from src.models.cointegration import CointegrationTester

m = MultiStockManager("stocks.db")
m.download_data_batch(["KO", "PEP"], "2021-01-01", "2024-12-31")
a, b = m.get_aligned_data("KO", "PEP")            # date-aligned price tables
r = CointegrationTester().engle_granger_test(a["adj_close"], b["adj_close"])
print(r.p_value, r.hedge_ratio, r.half_life, r.is_cointegrated)
```
A run on 2026-10-30 gave p = 0.647, hedge ratio 0.365, half-life ≈ 150 days and not cointegrated: KO and
PEP move together, but their spread did not mean-revert over 2021–2024. That kind of result is exactly what
the test is there to catch.

`python main.py --universe AAPL MSFT` is a setup check. It initializes the database and registers the tickers.

## Project layout
```
Stock.py                     base Stock class: yfinance download + SQLite storage
src/data/                    MultiStockManager (batch download, alignment, stats), DatabaseManager
src/models/cointegration.py  Engle–Granger, Johansen, half-life, PairValidator
src/strategy/pairs_finder.py universe-wide pair search (correlation filter, then parallel cointegration tests)
src/utils/                   config dataclasses, logging and helpers
dashboard/                   Streamlit app and chart components
tests/                       unit tests
```

## Roadmap
- [ ] Fix `PairsFinder`, which reads correlation settings from the wrong config section and fails on startup.
- [ ] Ornstein–Uhlenbeck fit of the spread (θ, μ, σ) and z-score signals.
- [ ] Backtester with transaction costs and the risk limits already in the config.
- [ ] Fill in the dashboard's validation, backtest and risk pages.

Research code: nothing here places orders or is investment advice.
