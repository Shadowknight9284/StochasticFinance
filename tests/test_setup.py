"""
Test suite for the Pairs Trading Algorithm.
"""

import sys
import os
import unittest

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestProjectSetup(unittest.TestCase):
    """Test project setup and configuration."""
    
    def test_config_import(self):
        """Test that configuration can be imported."""
        from src.utils.config import default_config
        self.assertIsNotNone(default_config)
        
    def test_helpers_import(self):
        """Test that helpers can be imported."""
        from src.utils.helpers import setup_logging
        self.assertTrue(callable(setup_logging))
        
    def test_directory_structure(self):
        """Test that required directories exist."""
        required_dirs = [
            'src',
            'src/data',
            'src/models',
            'src/strategy',
            'src/risk',
            'src/backtesting',
            'src/utils',
            'tests',
            'data',
            'results',
            'notebooks'
        ]
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        
        for dir_name in required_dirs:
            dir_path = os.path.join(project_root, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_name} does not exist")

if __name__ == '__main__':
    unittest.main()
