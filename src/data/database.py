"""
Database schema management for pairs trading system.
"""

import sqlite3
import logging
from typing import Dict, List
import pandas as pd

class DatabaseManager:
    """Manages database schema and operations for pairs trading."""
    
    def __init__(self, db_path: str = "stocks.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for better performance."""
        try:
            # First ensure required tables exist
            self._ensure_core_tables()
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_pairs_tickers ON pairs_metadata (ticker1, ticker2)",
                "CREATE INDEX IF NOT EXISTS idx_spreads_date ON pair_spreads (date)",
                "CREATE INDEX IF NOT EXISTS idx_spreads_tickers ON pair_spreads (ticker1, ticker2)"
            ]
            
            # Create core table indexes
            for idx_sql in indexes:
                self.cursor.execute(idx_sql)
            
            # Get all stock tables and create date indexes
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT IN ('pairs_metadata', 'pair_spreads')")
            tables = self.cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                if not table_name.startswith('sqlite_'):
                    self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name} (date)")
            
            self.conn.commit()
            logging.info("Database indexes created successfully")
            
        except Exception as e:
            logging.error(f"Error creating indexes: {str(e)}")
    
    def _ensure_core_tables(self):
        """Ensure core tables exist."""
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
        
        # Spreads table
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
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        stats = {}
        
        # Get table information
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in self.cursor.fetchall()]
        
        stats['total_tables'] = len(tables)
        stats['stock_tables'] = len([t for t in tables if t not in ['pairs_metadata', 'pair_spreads']])
        
        # Get record counts
        table_stats = {}
        for table in tables:
            try:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cursor.fetchone()[0]
                table_stats[table] = count
            except Exception as e:
                logging.error(f"Error getting count for {table}: {str(e)}")
                table_stats[table] = 0
        
        stats['table_records'] = table_stats
        
        # Database size
        self.cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        size_bytes = self.cursor.fetchone()[0]
        stats['database_size_mb'] = size_bytes / (1024 * 1024)
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 365*5) -> int:
        """Remove old data beyond specified days."""
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_to_keep)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        
        # Get all stock tables
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT IN ('pairs_metadata', 'pair_spreads')")
        tables = self.cursor.fetchall()
        
        total_deleted = 0
        for table in tables:
            table_name = table[0]
            try:
                self.cursor.execute(f"DELETE FROM {table_name} WHERE date < ?", (cutoff_str,))
                deleted = self.cursor.rowcount
                total_deleted += deleted
                logging.info(f"Deleted {deleted} old records from {table_name}")
            except Exception as e:
                logging.error(f"Error cleaning {table_name}: {str(e)}")
        
        # Clean pair spreads
        try:
            self.cursor.execute("DELETE FROM pair_spreads WHERE date < ?", (cutoff_str,))
            deleted = self.cursor.rowcount
            total_deleted += deleted
            logging.info(f"Deleted {deleted} old spread records")
        except Exception as e:
            logging.error(f"Error cleaning pair spreads: {str(e)}")
        
        self.conn.commit()
        return total_deleted
    
    def vacuum_database(self):
        """Optimize database by reclaiming space."""
        try:
            self.cursor.execute("VACUUM")
            logging.info("Database vacuum completed")
        except Exception as e:
            logging.error(f"Error during vacuum: {str(e)}")
    
    def close(self):
        """Close database connection."""
        self.conn.close()
