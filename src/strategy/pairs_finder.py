"""
Pairs finder module for identifying and ranking trading pairs.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.models.cointegration import PairValidator
from src.data.stock_data import MultiStockManager  
from src.utils.config import TradingConfig


class PairsFinder:
    """Find and rank potential trading pairs."""
    
    def __init__(self, config: TradingConfig):
        """
        Initialize pairs finder.
        
        Args:
            config: Trading configuration
        """
        self.config = config
        self.validator = PairValidator(
            min_correlation=config.data.min_correlation,
            max_correlation=config.data.max_correlation,
            min_data_points=100
        )
        self.stock_manager = MultiStockManager()
        self.logger = logging.getLogger(__name__)
        
    def find_pairs_from_universe(self, tickers: List[str], 
                                start_date: str, end_date: str,
                                max_pairs: int = 50) -> List[Dict]:
        """
        Find trading pairs from a universe of stocks.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date for data
            end_date: End date for data
            max_pairs: Maximum number of pairs to return
            
        Returns:
            List of validated pairs ranked by trading score
        """
        self.logger.info(f"Searching for pairs from {len(tickers)} stocks")
        
        # Step 1: Download data for all tickers
        self.logger.info("Downloading stock data...")
        stock_data = {}
        
        for ticker in tickers:
            try:
                data = self.stock_manager.get_stock_data(
                    ticker, start_date, end_date
                )
                if data is not None and len(data) >= 100:
                    stock_data[ticker] = data['close']
                else:
                    self.logger.warning(f"Insufficient data for {ticker}")
            except Exception as e:
                self.logger.warning(f"Failed to get data for {ticker}: {e}")
        
        self.logger.info(f"Successfully loaded data for {len(stock_data)} stocks")
        
        # Step 2: Pre-filter pairs by correlation
        valid_tickers = list(stock_data.keys())
        potential_pairs = list(combinations(valid_tickers, 2))
        
        self.logger.info(f"Testing {len(potential_pairs)} potential pairs")
        
        # Step 3: Filter by correlation first (fast pre-screening)
        correlation_filtered_pairs = self._filter_by_correlation(
            stock_data, potential_pairs
        )
        
        self.logger.info(f"Correlation filter passed: {len(correlation_filtered_pairs)} pairs")
        
        # Step 4: Full validation with cointegration testing
        validated_pairs = self._validate_pairs_parallel(
            stock_data, correlation_filtered_pairs
        )
        
        # Step 5: Rank and return top pairs
        ranked_pairs = sorted(
            validated_pairs, 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        top_pairs = ranked_pairs[:max_pairs]
        
        self.logger.info(f"Found {len(top_pairs)} valid trading pairs")
        
        return top_pairs
    
    def _filter_by_correlation(self, stock_data: Dict[str, pd.Series], 
                              pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Pre-filter pairs by correlation threshold."""
        filtered_pairs = []
        
        for ticker1, ticker2 in pairs:
            try:
                # Align data
                aligned = pd.concat([
                    stock_data[ticker1], 
                    stock_data[ticker2]
                ], axis=1).dropna()
                
                if len(aligned) < 50:
                    continue
                
                correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                
                # Check correlation threshold
                if (self.validator.min_correlation <= abs(correlation) <= 
                    self.validator.max_correlation):
                    filtered_pairs.append((ticker1, ticker2))
                    
            except Exception as e:
                self.logger.debug(f"Correlation filter failed for {ticker1}-{ticker2}: {e}")
                continue
        
        return filtered_pairs
    
    def _validate_pairs_parallel(self, stock_data: Dict[str, pd.Series],
                                pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Validate pairs using parallel processing."""
        validated_pairs = []
        
        # Use thread pool for parallel processing
        max_workers = min(4, len(pairs))  # Limit workers to avoid overwhelming
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit validation tasks
            future_to_pair = {
                executor.submit(
                    self._validate_single_pair, 
                    stock_data[ticker1], 
                    stock_data[ticker2], 
                    ticker1, 
                    ticker2
                ): (ticker1, ticker2)
                for ticker1, ticker2 in pairs
            }
            
            # Collect results
            for future in as_completed(future_to_pair):
                ticker1, ticker2 = future_to_pair[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    if result and result['is_valid']:
                        validated_pairs.append(result)
                        self.logger.debug(f"Validated pair: {ticker1}-{ticker2} (score: {result['score']:.1f})")
                except Exception as e:
                    self.logger.warning(f"Validation failed for {ticker1}-{ticker2}: {e}")
        
        return validated_pairs
    
    def _validate_single_pair(self, price1: pd.Series, price2: pd.Series,
                             ticker1: str, ticker2: str) -> Optional[Dict]:
        """Validate a single pair."""
        try:
            return self.validator.validate_pair(price1, price2, ticker1, ticker2)
        except Exception as e:
            self.logger.debug(f"Single pair validation failed for {ticker1}-{ticker2}: {e}")
            return None
    
    def find_sector_pairs(self, sector_tickers: Dict[str, List[str]],
                         start_date: str, end_date: str) -> Dict[str, List[Dict]]:
        """
        Find pairs within specific sectors.
        
        Args:
            sector_tickers: Dictionary mapping sector names to ticker lists
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping sector names to lists of valid pairs
        """
        sector_pairs = {}
        
        for sector, tickers in sector_tickers.items():
            self.logger.info(f"Finding pairs in {sector} sector ({len(tickers)} stocks)")
            
            pairs = self.find_pairs_from_universe(
                tickers, start_date, end_date, max_pairs=20
            )
            
            sector_pairs[sector] = pairs
            self.logger.info(f"Found {len(pairs)} pairs in {sector}")
        
        return sector_pairs
    
    def update_pair_rankings(self, existing_pairs: List[Dict],
                           start_date: str, end_date: str) -> List[Dict]:
        """
        Update rankings for existing pairs with fresh data.
        
        Args:
            existing_pairs: List of existing pair dictionaries
            start_date: Start date for fresh data
            end_date: End date for fresh data
            
        Returns:
            Updated list of pairs with new rankings
        """
        self.logger.info(f"Updating rankings for {len(existing_pairs)} pairs")
        
        updated_pairs = []
        
        for pair_info in existing_pairs:
            pair_name = pair_info['pair']
            ticker1, ticker2 = pair_name.split('-')
            
            try:
                # Get fresh data
                data1 = self.stock_manager.get_stock_data(ticker1, start_date, end_date)
                data2 = self.stock_manager.get_stock_data(ticker2, start_date, end_date)
                
                if data1 is not None and data2 is not None:
                    # Re-validate with fresh data
                    updated_result = self.validator.validate_pair(
                        data1['close'], data2['close'], ticker1, ticker2
                    )
                    updated_pairs.append(updated_result)
                else:
                    self.logger.warning(f"Could not update data for {pair_name}")
                    # Keep old result
                    updated_pairs.append(pair_info)
                    
            except Exception as e:
                self.logger.warning(f"Failed to update {pair_name}: {e}")
                updated_pairs.append(pair_info)
        
        # Re-rank
        updated_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        self.logger.info("Pair rankings updated")
        return updated_pairs


class PairScreener:
    """Screen and filter pairs based on various criteria."""
    
    def __init__(self):
        """Initialize pair screener."""
        self.logger = logging.getLogger(__name__)
    
    def screen_by_fundamentals(self, pairs: List[Dict], 
                              fundamental_data: Dict[str, Dict]) -> List[Dict]:
        """
        Screen pairs by fundamental criteria.
        
        Args:
            pairs: List of pair dictionaries
            fundamental_data: Dictionary with fundamental data for each ticker
            
        Returns:
            Filtered list of pairs
        """
        screened_pairs = []
        
        for pair in pairs:
            ticker1, ticker2 = pair['pair'].split('-')
            
            # Check if fundamental data available
            if ticker1 not in fundamental_data or ticker2 not in fundamental_data:
                continue
            
            fund1 = fundamental_data[ticker1]
            fund2 = fundamental_data[ticker2]
            
            # Screening criteria
            checks = {
                'market_cap_similar': self._check_market_cap_similarity(fund1, fund2),
                'same_sector': fund1.get('sector') == fund2.get('sector'),
                'liquidity_ok': self._check_liquidity(fund1, fund2),
                'financial_health': self._check_financial_health(fund1, fund2)
            }
            
            # Add fundamental score
            fundamental_score = sum(checks.values()) / len(checks) * 20  # Max 20 points
            pair['fundamental_score'] = fundamental_score
            pair['fundamental_checks'] = checks
            
            # Keep pairs that pass minimum criteria
            if checks['liquidity_ok'] and checks['financial_health']:
                screened_pairs.append(pair)
        
        return screened_pairs
    
    def _check_market_cap_similarity(self, fund1: Dict, fund2: Dict) -> bool:
        """Check if market caps are similar (within 5x difference)."""
        try:
            mcap1 = fund1.get('market_cap', 0)
            mcap2 = fund2.get('market_cap', 0)
            
            if mcap1 <= 0 or mcap2 <= 0:
                return False
            
            ratio = max(mcap1, mcap2) / min(mcap1, mcap2)
            return ratio <= 5.0
            
        except Exception:
            return False
    
    def _check_liquidity(self, fund1: Dict, fund2: Dict) -> bool:
        """Check if both stocks have sufficient liquidity."""
        try:
            vol1 = fund1.get('avg_volume', 0)
            vol2 = fund2.get('avg_volume', 0)
            
            # Minimum 1M average volume
            return vol1 >= 1_000_000 and vol2 >= 1_000_000
            
        except Exception:
            return False
    
    def _check_financial_health(self, fund1: Dict, fund2: Dict) -> bool:
        """Basic financial health check."""
        try:
            # Check debt-to-equity ratio
            de1 = fund1.get('debt_to_equity', float('inf'))
            de2 = fund2.get('debt_to_equity', float('inf'))
            
            # Check current ratio
            cr1 = fund1.get('current_ratio', 0)
            cr2 = fund2.get('current_ratio', 0)
            
            # Basic health criteria
            debt_ok = de1 < 2.0 and de2 < 2.0
            liquidity_ok = cr1 > 1.0 and cr2 > 1.0
            
            return debt_ok and liquidity_ok
            
        except Exception:
            return True  # If no data, don't filter out


def get_default_stock_universe() -> List[str]:
    """Get default universe of stocks for pair finding."""
    return [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
        'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO',
        
        # Finance
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'USB', 'PNC',
        
        # Healthcare
        'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN',
        
        # Consumer
        'WMT', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'DIS', 'HD', 'LOW',
        
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'PXD', 'KMI',
        
        # Industrials
        'GE', 'CAT', 'BA', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD'
    ]


def get_sector_universe() -> Dict[str, List[str]]:
    """Get sector-based stock universe."""
    return {
        'Technology': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO'
        ],
        'Finance': [
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'USB', 'PNC'
        ],
        'Healthcare': [
            'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN'
        ],
        'Consumer': [
            'WMT', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'DIS', 'HD', 'LOW'
        ],
        'Energy': [
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'PXD', 'KMI'
        ],
        'Industrials': [
            'GE', 'CAT', 'BA', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD'
        ]
    }
