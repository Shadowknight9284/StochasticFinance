"""
Tests for enhanced data infrastructure.
"""

import unittest
import tempfile
import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import EnhancedStockData, MultiStockManager, DatabaseManager, DataValidator

class TestEnhancedData(unittest.TestCase):
    """Test enhanced data functionality."""
    
    def setUp(self):
        """Set up test environment with temporary database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_enhanced_stock_data_creation(self):
        """Test enhanced stock data class creation."""
        stock = EnhancedStockData("AAPL", self.db_path)
        self.assertEqual(stock.ticker, "AAPL")
        self.assertTrue(hasattr(stock, 'get_returns'))
        self.assertTrue(hasattr(stock, 'get_volatility'))
    
    def test_multi_stock_manager(self):
        """Test multi-stock manager functionality."""
        manager = MultiStockManager(self.db_path)
        
        # Add stocks
        manager.add_stock("AAPL")
        manager.add_stock("MSFT")
        
        self.assertEqual(len(manager.stocks), 2)
        self.assertIn("AAPL", manager.stocks)
        self.assertIn("MSFT", manager.stocks)
        
        manager.close()
    
    def test_database_manager(self):
        """Test database manager functionality."""
        db_manager = DatabaseManager(self.db_path)
        stats = db_manager.get_database_stats()
        
        self.assertIn('total_tables', stats)
        self.assertIsInstance(stats['database_size_mb'], float)
        
        db_manager.close()
    
    def test_data_validator(self):
        """Test data validation utilities."""
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        
        checks = DataValidator.check_price_consistency(sample_data)
        
        self.assertIn('high_low_consistency', checks)
        self.assertIn('close_in_range', checks)
        self.assertTrue(checks['high_low_consistency'])
        self.assertTrue(checks['close_in_range'])

if __name__ == '__main__':
    unittest.main()
