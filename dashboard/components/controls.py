"""
User interface controls and widgets for dashboard.
"""

import streamlit as st
import pandas as pd
from typing import List, Tuple, Optional
import datetime as dt

class DashboardControls:
    """Interactive controls and widgets for the dashboard."""
    
    @staticmethod
    def stock_selector(available_stocks: List[str]) -> Tuple[str, str]:
        """Stock pair selection widget."""
        st.subheader("📊 Stock Pair Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            stock1 = st.selectbox(
                "Select First Stock",
                available_stocks,
                index=0 if available_stocks else None,
                key="stock1_select"
            )
        
        with col2:
            stock2 = st.selectbox(
                "Select Second Stock", 
                [s for s in available_stocks if s != stock1],
                index=0 if len(available_stocks) > 1 else None,
                key="stock2_select"
            )
        
        return stock1, stock2
    
    @staticmethod
    def date_range_selector() -> Tuple[dt.date, dt.date]:
        """Date range selection widget."""
        st.subheader("📅 Date Range")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=dt.date.today() - dt.timedelta(days=365),
                max_value=dt.date.today(),
                key="start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=dt.date.today(),
                min_value=start_date,
                max_value=dt.date.today(),
                key="end_date"
            )
        
        return start_date, end_date
    
    @staticmethod
    def trading_parameters():
        """Trading strategy parameter controls."""
        st.subheader("⚙️ Trading Parameters")
        
        with st.expander("Strategy Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                entry_threshold = st.slider(
                    "Entry Z-Score Threshold",
                    min_value=1.0,
                    max_value=3.0,
                    value=2.0,
                    step=0.1,
                    help="Z-score threshold for entering trades"
                )
                
                exit_threshold = st.slider(
                    "Exit Z-Score Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Z-score threshold for exiting trades"
                )
            
            with col2:
                lookback_period = st.slider(
                    "Lookback Period (days)",
                    min_value=20,
                    max_value=250,
                    value=60,
                    step=10,
                    help="Period for calculating rolling statistics"
                )
                
                position_size = st.slider(
                    "Position Size (%)",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    help="Percentage of portfolio per trade"
                )
        
        return {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'lookback_period': lookback_period,
            'position_size': position_size / 100
        }
    
    @staticmethod
    def model_parameters():
        """Ornstein-Uhlenbeck model parameter controls."""
        st.subheader("🧮 Model Parameters")
        
        with st.expander("OU Model Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                theta = st.number_input(
                    "Mean Reversion Speed (θ)",
                    min_value=0.01,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    help="Speed of mean reversion"
                )
                
                mu = st.number_input(
                    "Long-term Mean (μ)",
                    min_value=-1.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    help="Long-term equilibrium level"
                )
            
            with col2:
                sigma = st.number_input(
                    "Volatility (σ)",
                    min_value=0.01,
                    max_value=2.0,
                    value=0.3,
                    step=0.01,
                    help="Volatility parameter"
                )
                
                confidence = st.slider(
                    "Confidence Level (%)",
                    min_value=90,
                    max_value=99,
                    value=95,
                    step=1,
                    help="Confidence level for bands"
                )
        
        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'confidence': confidence / 100
        }
    
    @staticmethod
    def data_controls():
        """Data refresh and update controls."""
        st.subheader("🔄 Data Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Data", help="Fetch latest market data"):
                st.success("Data refresh initiated!")
                return 'refresh'
        
        with col2:
            if st.button("💾 Save Config", help="Save current configuration"):
                st.success("Configuration saved!")
                return 'save'
        
        with col3:
            if st.button("📥 Load Config", help="Load saved configuration"):
                st.success("Configuration loaded!")
                return 'load'
        
        return None
    
    @staticmethod
    def analysis_options():
        """Analysis and visualization options."""
        st.subheader("📈 Analysis Options")
        
        options = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            options['show_confidence_bands'] = st.checkbox(
                "Show Confidence Bands",
                value=True,
                help="Display confidence bands around mean"
            )
            
            options['show_trade_signals'] = st.checkbox(
                "Show Trade Signals",
                value=True,
                help="Highlight entry/exit points"
            )
            
            options['show_rolling_stats'] = st.checkbox(
                "Show Rolling Statistics",
                value=False,
                help="Display rolling mean and std"
            )
        
        with col2:
            options['chart_type'] = st.selectbox(
                "Chart Type",
                ['Line', 'Candlestick', 'OHLC'],
                index=0,
                help="Type of price chart"
            )
            
            options['time_aggregation'] = st.selectbox(
                "Time Aggregation",
                ['Daily', 'Weekly', 'Monthly'],
                index=0,
                help="Data aggregation period"
            )
            
            options['normalize_prices'] = st.checkbox(
                "Normalize Prices",
                value=True,
                help="Normalize prices to same scale"
            )
        
        return options
