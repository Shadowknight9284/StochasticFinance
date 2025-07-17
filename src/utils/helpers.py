"""
Helper utility functions for the pairs trading algorithm.
"""

import logging
import os
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def setup_logging(log_level: str = "INFO", log_to_file: bool = True, log_dir: str = "logs") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_to_file: Whether to log to file
        log_dir: Directory for log files
    """
    # Create logs directory if it doesn't exist
    if log_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up handlers
    handlers = [logging.StreamHandler()]  # Console handler
    
    if log_to_file:
        log_file = os.path.join(log_dir, f"pairs_trading_{datetime.now().strftime('%Y%m%d')}.log")
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

def save_results(results: Dict[str, Any], filename: str, results_dir: str = "results") -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary to save
        filename: Output filename
        results_dir: Results directory
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    filepath = os.path.join(results_dir, filename)
    
    if filename.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif filename.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    logging.info(f"Results saved to {filepath}")

def load_results(filename: str, results_dir: str = "results") -> Dict[str, Any]:
    """
    Load results from file.
    
    Args:
        filename: Input filename
        results_dir: Results directory
        
    Returns:
        Loaded results dictionary
    """
    filepath = os.path.join(results_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    if filename.endswith('.json'):
        with open(filepath, 'r') as f:
            results = json.load(f)
    elif filename.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    return results

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Returns series
    """
    return prices.pct_change().dropna()

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Log returns series
    """
    return np.log(prices / prices.shift(1)).dropna()

def calculate_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Calculate rolling z-score of a series.
    
    Args:
        series: Input series
        window: Rolling window size
        
    Returns:
        Z-score series
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def validate_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns and has valid data.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    # Check if all required columns exist
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for empty DataFrame
    if data.empty:
        logging.error("DataFrame is empty")
        return False
    
    # Check for all NaN columns
    for col in required_columns:
        if data[col].isna().all():
            logging.error(f"Column '{col}' contains only NaN values")
            return False
    
    return True

def format_performance_metrics(metrics: Dict[str, float]) -> str:
    """
    Format performance metrics for display.
    
    Args:
        metrics: Dictionary of performance metrics
        
    Returns:
        Formatted string
    """
    formatted_lines = []
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            if 'ratio' in metric_name.lower() or 'sharpe' in metric_name.lower():
                formatted_value = f"{value:.3f}"
            elif 'return' in metric_name.lower() or 'volatility' in metric_name.lower():
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        
        formatted_lines.append(f"{metric_name}: {formatted_value}")
    
    return "\n".join(formatted_lines)

def get_trading_days_between(start_date: str, end_date: str) -> int:
    """
    Calculate number of trading days between two dates.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        Number of trading days
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create business day range (excludes weekends)
    business_days = pd.bdate_range(start=start, end=end)
    return len(business_days)

def timestamp_to_string(timestamp: Any) -> str:
    """
    Convert various timestamp formats to string.
    
    Args:
        timestamp: Timestamp in various formats
        
    Returns:
        String representation
    """
    if pd.isna(timestamp):
        return "N/A"
    
    if isinstance(timestamp, str):
        return timestamp
    
    if hasattr(timestamp, 'strftime'):
        return timestamp.strftime('%Y-%m-%d')
    
    return str(timestamp)
