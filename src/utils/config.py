"""
Configuration settings for the Pairs Trading Algorithm.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataConfig:
    """Data-related configuration."""
    database_path: str = "stocks.db"
    data_directory: str = "data"
    default_start_date: str = "2020-01-01"
    default_end_date: str = "2024-12-31"
    min_price: float = 5.0  # Minimum stock price
    min_volume: int = 100000  # Minimum daily volume
    min_market_cap: float = 1e9  # Minimum market cap (1B)
    batch_size: int = 10  # Batch size for data downloads
    max_missing_data_pct: float = 0.05  # Maximum missing data percentage
    min_observations: int = 252  # Minimum observations for analysis
    data_validation: bool = True  # Enable data validation
    auto_cleanup_days: int = 1825  # Auto-cleanup data older than 5 years

@dataclass
class OUModelConfig:
    """Ornstein-Uhlenbeck model configuration."""
    lookback_window: int = 252  # 1 year of trading days
    min_half_life: int = 5  # Minimum half-life in days
    max_half_life: int = 120  # Maximum half-life in days
    confidence_level: float = 0.95
    estimation_method: str = "mle"  # "mle", "ols", or "kalman"
    
@dataclass
class CointegrationConfig:
    """Cointegration testing configuration."""
    min_correlation: float = 0.5
    max_correlation: float = 0.95
    pvalue_threshold: float = 0.05
    lookback_window: int = 252
    test_method: str = "engle_granger"  # "engle_granger" or "johansen"

@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    entry_threshold: float = 2.0  # Z-score threshold for entry
    exit_threshold: float = 0.5   # Z-score threshold for exit
    stop_loss_threshold: float = 3.5  # Stop-loss Z-score
    max_holding_period: int = 60  # Maximum holding period in days
    min_holding_period: int = 1   # Minimum holding period in days
    position_sizing_method: str = "equal_weight"  # "equal_weight", "volatility_adjusted", "ou_optimal"

@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_positions: int = 10
    max_sector_exposure: float = 0.3  # Maximum exposure to any sector
    max_single_position: float = 0.1  # Maximum single position size
    max_drawdown: float = 0.15  # Maximum portfolio drawdown
    var_confidence: float = 0.05  # VaR confidence level
    
@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% transaction cost
    slippage: float = 0.0005  # 0.05% slippage
    commission_per_share: float = 0.01
    benchmark: str = "SPY"
    rebalance_frequency: str = "daily"  # "daily", "weekly", "monthly"
    
@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = None
    ou_model: OUModelConfig = None
    cointegration: CointegrationConfig = None
    trading: TradingConfig = None
    risk: RiskConfig = None
    backtest: BacktestConfig = None
    
    def __post_init__(self):
        """Initialize default values after instantiation."""
        if self.data is None:
            self.data = DataConfig()
        if self.ou_model is None:
            self.ou_model = OUModelConfig()
        if self.cointegration is None:
            self.cointegration = CointegrationConfig()
        if self.trading is None:
            self.trading = TradingConfig()
        if self.risk is None:
            self.risk = RiskConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()
    
    # Paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_directory: str = "results"
    log_directory: str = "logs"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'ou_model': self.ou_model.__dict__,
            'cointegration': self.cointegration.__dict__,
            'trading': self.trading.__dict__,
            'risk': self.risk.__dict__,
            'backtest': self.backtest.__dict__,
            'project_root': self.project_root,
            'results_directory': self.results_directory,
            'log_directory': self.log_directory,
            'log_level': self.log_level,
            'log_to_file': self.log_to_file
        }

# Default configuration instance
default_config = Config()
