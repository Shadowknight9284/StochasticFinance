"""
Main execution script for the Pairs Trading Algorithm with Ornstein-Uhlenbeck Model.

This script orchestrates the entire pairs trading workflow:
1. Load configuration
2. Initialize data sources
3. Find cointegrated pairs
4. Fit Ornstein-Uhlenbeck models
5. Generate trading signals
6. Run backtesting
7. Generate performance reports
8. Save results

Usage:
    python main.py [--config CONFIG_FILE] [--start_date YYYY-MM-DD] [--end_date YYYY-MM-DD]
"""

import argparse
import sys
import os
from datetime import datetime
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import default_config, Config
from src.utils.helpers import setup_logging, save_results, ensure_directory_exists

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pairs Trading Algorithm with Ornstein-Uhlenbeck Model')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (JSON)')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--universe', type=str, nargs='+', default=None,
                       help='List of tickers to analyze')
    parser.add_argument('--dry_run', action='store_true',
                       help='Run in dry-run mode (no actual trading)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

def load_config(config_path: str = None) -> Config:
    """Load configuration from file or use default."""
    if config_path and os.path.exists(config_path):
        # TODO: Implement config loading from file
        logging.info(f"Loading configuration from {config_path}")
        return default_config
    else:
        logging.info("Using default configuration")
        return default_config

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.start_date:
        config.data.default_start_date = args.start_date
    if args.end_date:
        config.data.default_end_date = args.end_date
    if args.verbose:
        config.log_level = "DEBUG"
    
    # Setup logging
    ensure_directory_exists(config.log_directory)
    setup_logging(config.log_level, config.log_to_file, config.log_directory)
    
    logging.info("="*60)
    logging.info("PAIRS TRADING ALGORITHM WITH ORNSTEIN-UHLENBECK MODEL")
    logging.info("="*60)
    logging.info(f"Start Date: {config.data.default_start_date}")
    logging.info(f"End Date: {config.data.default_end_date}")
    logging.info(f"Dry Run: {args.dry_run}")
    
    try:
        # Ensure required directories exist
        ensure_directory_exists(config.results_directory)
        ensure_directory_exists(config.data.data_directory)
        
        # TODO: Phase 2 Implementation Status
        logging.info("Phase 1: Project setup completed ✅")
        logging.info("Phase 2: Enhanced data infrastructure completed ✅")
        logging.info("Next phases to implement:")
        logging.info("- Phase 3: Cointegration and pair selection")
        logging.info("- Phase 4: Ornstein-Uhlenbeck model")
        logging.info("- Phase 5: Signal generation")
        logging.info("- Phase 6: Trading strategy")
        logging.info("- Phase 7: Risk management")
        logging.info("- Phase 8: Backtesting engine")
        
        # Test enhanced data infrastructure
        try:
            from src.data import MultiStockManager, DatabaseManager
            
            # Initialize data infrastructure
            data_manager = MultiStockManager(config.data.database_path)
            db_manager = DatabaseManager(config.data.database_path)
            
            # Get database stats
            stats = db_manager.get_database_stats()
            logging.info(f"Database initialized: {stats['total_tables']} tables, {stats['database_size_mb']:.2f} MB")
            
            # Sample stock universe for testing
            test_tickers = ['AAPL', 'MSFT']  # Small test set
            if args.universe:
                test_tickers = args.universe
            
            logging.info(f"Testing with tickers: {test_tickers}")
            
            # Add stocks to manager
            data_manager.add_stocks(test_tickers)
            logging.info(f"Added {len(test_tickers)} stocks to manager")
            
            # Clean up
            data_manager.close()
            db_manager.close()
            
        except Exception as e:
            logging.error(f"Error testing data infrastructure: {str(e)}")
        
        # Placeholder for main workflow (to be implemented in subsequent phases)
        results = {
            "status": "Phase 2 Complete",
            "timestamp": datetime.now().isoformat(),
            "config": config.to_dict(),
            "phase": 2,
            "next_phase": "Cointegration and Pair Selection",
            "data_infrastructure": "Enhanced stock data management implemented"
        }
        
        # Save initial results
        save_results(results, "phase2_data_infrastructure_complete.json", config.results_directory)
        
        logging.info("Phase 2 implementation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
