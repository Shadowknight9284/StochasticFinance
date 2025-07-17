"""
Enhanced stock data management for pairs trading.
Extends the original Stock class with multi-stock capabilities and advanced data handling.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import logging
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import sys
import os

# Import original Stock class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Stock import Stock

class EnhancedStockData(Stock):
    """
    Enhanced stock data management with multi-stock capabilities.
    Inherits from original Stock class and adds pairs trading functionality.
    """
    
    def __init__(self, ticker: str, db_path: str = "stocks.db"):
        """Initialize enhanced stock data manager."""
        self.db_path = db_path
        super().__init__(ticker)
        self._setup_enhanced_tables()
        
    def _setup_enhanced_tables(self):
        """Create additional tables for pairs trading."""
        # Pairs metadata table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pairs_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker1 TEXT NOT NULL,
                ticker2 TEXT NOT NULL,
                correlation REAL,
                cointegration_pvalue REAL,
                half_life REAL,
                last_updated TEXT,
                is_active INTEGER DEFAULT 1,
                UNIQUE(ticker1, ticker2)
            )
        """)
        
        # Spreads table for storing pair spreads
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pair_spreads (
                date TEXT,
                ticker1 TEXT,
                ticker2 TEXT,
                spread REAL,
                zscore REAL,
                PRIMARY KEY (date, ticker1, ticker2)
            )
        """)
        
        self.conn.commit()
    
    def get_returns(self, period: int = None) -> pd.Series:
        """Calculate returns for the stock."""
        self.get_data_from_db()
        prices = self.data['close']
        if period:
            prices = prices.tail(period)
        return prices.pct_change().dropna()
    
    def get_log_returns(self, period: int = None) -> pd.Series:
        """Calculate log returns for the stock."""
        self.get_data_from_db()
        prices = self.data['close']
        if period:
            prices = prices.tail(period)
        return np.log(prices / prices.shift(1)).dropna()
    
    def get_volatility(self, window: int = 252) -> float:
        """Calculate annualized volatility."""
        returns = self.get_returns()
        return returns.std() * np.sqrt(window)
    
    def validate_data(self, min_observations: int = 252) -> bool:
        """Validate data quality for pairs trading."""
        self.get_data_from_db()
        
        if len(self.data) < min_observations:
            logging.warning(f"{self.ticker}: Insufficient data ({len(self.data)} < {min_observations})")
            return False
            
        # Check for excessive missing values
        missing_pct = self.data['close'].isna().sum() / len(self.data)
        if missing_pct > 0.05:  # More than 5% missing
            logging.warning(f"{self.ticker}: Too many missing values ({missing_pct:.2%})")
            return False
            
        # Check for zero prices
        if (self.data['close'] <= 0).any():
            logging.warning(f"{self.ticker}: Contains zero or negative prices")
            return False
            
        return True


class MultiStockManager:
    """
    Manages multiple stocks for pairs trading analysis.
    Handles batch operations, data synchronization, and pair management.
    """
    
    def __init__(self, db_path: str = "stocks.db"):
        self.db_path = db_path
        self.stocks: Dict[str, EnhancedStockData] = {}
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
    def add_stock(self, ticker: str) -> EnhancedStockData:
        """Add a stock to the manager."""
        if ticker not in self.stocks:
            self.stocks[ticker] = EnhancedStockData(ticker, self.db_path)
        return self.stocks[ticker]
    
    def add_stocks(self, tickers: List[str]) -> Dict[str, EnhancedStockData]:
        """Add multiple stocks to the manager."""
        for ticker in tickers:
            self.add_stock(ticker)
        return self.stocks
    
    def download_data_batch(self, tickers: List[str], start_date: str, end_date: str, 
                           chunk_size: int = 10) -> Dict[str, bool]:
        """Download data for multiple stocks in batches."""
        results = {}
        
        # Process in chunks to avoid API limits
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            logging.info(f"Downloading data for batch {i//chunk_size + 1}: {chunk}")
            
            try:
                # Download batch using yfinance
                data = yf.download(chunk, start=start_date, end=end_date, 
                                 progress=False, group_by='ticker')
                
                for ticker in chunk:
                    try:
                        stock = self.add_stock(ticker)
                        
                        if len(chunk) == 1:
                            ticker_data = data
                        else:
                            ticker_data = data[ticker] if ticker in data.columns.levels[0] else pd.DataFrame()
                        
                        if not ticker_data.empty:
                            # Process and store data
                            ticker_data = ticker_data.dropna()
                            ticker_data.reset_index(inplace=True)
                            ticker_data['date'] = ticker_data['Date'].dt.strftime('%Y-%m-%d')
                            
                            # Rename columns to match our schema
                            column_map = {
                                'Open': 'open', 'High': 'high', 'Low': 'low', 
                                'Close': 'close', 'Volume': 'volume'
                            }
                            ticker_data = ticker_data.rename(columns=column_map)
                            ticker_data['adj_close'] = ticker_data['close']
                            
                            # Select required columns
                            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
                            ticker_data = ticker_data[required_cols]
                            
                            # Store to database
                            ticker_data.to_sql(stock.table_name, stock.conn, 
                                             if_exists='replace', index=False)
                            
                            results[ticker] = True
                            logging.info(f"Successfully downloaded {len(ticker_data)} records for {ticker}")
                        else:
                            results[ticker] = False
                            logging.warning(f"No data available for {ticker}")
                            
                    except Exception as e:
                        results[ticker] = False
                        logging.error(f"Error processing {ticker}: {str(e)}")
                        
            except Exception as e:
                logging.error(f"Error downloading batch {chunk}: {str(e)}")
                for ticker in chunk:
                    results[ticker] = False
        
        return results
    
    def get_aligned_data(self, ticker1: str, ticker2: str, 
                        start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get time-aligned data for two stocks."""
        stock1 = self.stocks.get(ticker1)
        stock2 = self.stocks.get(ticker2)
        
        if not stock1 or not stock2:
            raise ValueError(f"Stocks not found: {ticker1}, {ticker2}")
        
        # Get data from database
        stock1.get_data_from_db()
        stock2.get_data_from_db()
        
        data1 = stock1.data.copy()
        data2 = stock2.data.copy()
        
        # Filter by date range if specified
        if start_date:
            data1 = data1[data1['date'] >= start_date]
            data2 = data2[data2['date'] >= start_date]
        if end_date:
            data1 = data1[data1['date'] <= end_date]
            data2 = data2[data2['date'] <= end_date]
        
        # Align data by date
        data1.set_index('date', inplace=True)
        data2.set_index('date', inplace=True)
        
        # Inner join to get common dates
        aligned_data = data1.join(data2, how='inner', lsuffix='_1', rsuffix='_2')
        
        # Split back into separate DataFrames
        cols1 = [col for col in aligned_data.columns if col.endswith('_1')]
        cols2 = [col for col in aligned_data.columns if col.endswith('_2')]
        
        data1_aligned = aligned_data[cols1].copy()
        data2_aligned = aligned_data[cols2].copy()
        
        # Remove suffixes
        data1_aligned.columns = [col.replace('_1', '') for col in data1_aligned.columns]
        data2_aligned.columns = [col.replace('_2', '') for col in data2_aligned.columns]
        
        return data1_aligned, data2_aligned
    
    def calculate_correlation_matrix(self, tickers: List[str] = None, 
                                   method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for stocks."""
        if tickers is None:
            tickers = list(self.stocks.keys())
        
        # Collect closing prices
        price_data = {}
        for ticker in tickers:
            if ticker in self.stocks:
                self.stocks[ticker].get_data_from_db()
                data = self.stocks[ticker].data.set_index('date')['close']
                price_data[ticker] = data
        
        # Create aligned DataFrame
        prices_df = pd.DataFrame(price_data)
        prices_df = prices_df.dropna()
        
        # Calculate correlation
        correlation_matrix = prices_df.corr(method=method)
        return correlation_matrix
    
    def get_stock_universe_stats(self) -> pd.DataFrame:
        """Get statistics for all stocks in the universe."""
        stats = []
        
        for ticker, stock in self.stocks.items():
            try:
                stock.get_data_from_db()
                if len(stock.data) > 0:
                    last_price = stock.data['close'].iloc[-1]
                    avg_volume = stock.data['volume'].mean()
                    volatility = stock.get_volatility()
                    data_points = len(stock.data)
                    start_date = stock.data['date'].iloc[0]
                    end_date = stock.data['date'].iloc[-1]
                    
                    stats.append({
                        'ticker': ticker,
                        'last_price': last_price,
                        'avg_volume': avg_volume,
                        'volatility': volatility,
                        'data_points': data_points,
                        'start_date': start_date,
                        'end_date': end_date,
                        'valid_data': stock.validate_data()
                    })
            except Exception as e:
                logging.error(f"Error getting stats for {ticker}: {str(e)}")
        
        return pd.DataFrame(stats)
    
    def clean_data(self, remove_invalid: bool = True) -> Dict[str, bool]:
        """Clean and validate data for all stocks."""
        results = {}
        
        for ticker, stock in self.stocks.items():
            try:
                is_valid = stock.validate_data()
                
                if not is_valid and remove_invalid:
                    # Remove invalid stock
                    del self.stocks[ticker]
                    logging.info(f"Removed invalid stock: {ticker}")
                
                results[ticker] = is_valid
                
            except Exception as e:
                logging.error(f"Error cleaning data for {ticker}: {str(e)}")
                results[ticker] = False
        
        return results
    
    def close(self):
        """Close database connections."""
        self.conn.close()
        for stock in self.stocks.values():
            stock.conn.close()


class DataValidator:
    """Utility class for data validation and quality checks."""
    
    @staticmethod
    def check_price_consistency(data: pd.DataFrame) -> Dict[str, bool]:
        """Check price data consistency."""
        checks = {}
        
        # High >= Low
        checks['high_low_consistency'] = (data['high'] >= data['low']).all()
        
        # Close within High-Low range
        checks['close_in_range'] = ((data['close'] >= data['low']) & 
                                   (data['close'] <= data['high'])).all()
        
        # Open within High-Low range
        checks['open_in_range'] = ((data['open'] >= data['low']) & 
                                  (data['open'] <= data['high'])).all()
        
        # No zero volumes (for liquid stocks)
        checks['non_zero_volume'] = (data['volume'] > 0).all()
        
        # No zero prices
        checks['non_zero_prices'] = (data['close'] > 0).all()
        
        return checks
    
    @staticmethod
    def detect_outliers(prices: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
        """Detect price outliers."""
        if method == 'iqr':
            Q1 = prices.quantile(0.25)
            Q3 = prices.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (prices < lower_bound) | (prices > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((prices - prices.mean()) / prices.std())
            outliers = z_scores > threshold
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return outliers
