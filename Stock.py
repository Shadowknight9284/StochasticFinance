import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import sqlite3
import os

class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = pd.DataFrame()
        self.conn = sqlite3.connect('stocks.db')
        self.cursor = self.conn.cursor()
        self.table_name = self.ticker.replace('.', '_')
        self.create_table()
        
    def create_table(self):
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} (date TEXT, open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume INTEGER)")
        self.conn.commit()
        
    def get_data(self, start_date, end_date):
        self.data = yf.download(self.ticker, start=start_date, end=end_date)
        self.data.reset_index(inplace=True)
        self.data['date'] = self.data['Date'].dt.strftime('%Y-%m-%d')
        self.data.drop(columns=['Date', 'Adj Close'], inplace=True)
        self.data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        self.data = self.data[['date', 'open', 'high', 'low', 'close', 'volume']]
        self.data['adj_close'] = self.data['close']
        self.data.to_sql(self.table_name, self.conn, if_exists='replace', index=False)
        
    def get_data_from_db(self):
        self.data = pd.read_sql(f"SELECT * FROM {self.table_name}", self.conn)
        
    def get_last_date(self):
        self.get_data_from_db()
        return self.data['date'].iloc[-1]
    
    def get_last_close(self):
        self.get_data_from_db()
        return self.data['close'].iloc[-1]
    
    def get_last_adj_close(self):
        self.get_data_from_db()
        return self.data['adj_close'].iloc[-1]
    
    def get_last_volume(self):
        self.get_data_from_db()
        return self.data['volume'].iloc[-1]
    
    def get_last_n_days(self, n):
        self.get_data_from_db()
        return self.data.iloc[-n:]
    
    def get_last_n_days_close(self, n):
        self.get_data_from_db()
        return self.data['close'].iloc[-n:]
    
    def get_last_n_days_adj_close(self, n):
        self.get_data_from_db()
        return self.data['adj_close'].iloc[-n:]