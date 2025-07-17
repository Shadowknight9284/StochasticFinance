"""
Performance metrics components for dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np

class PerformanceMetrics:
    """Performance metrics and KPI components."""
    
    @staticmethod
    def display_key_metrics(portfolio_value: float, total_return: float, 
                           sharpe_ratio: float, max_drawdown: float):
        """Display key performance metrics in cards."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💼 Portfolio Value",
                value=f"${portfolio_value:,.2f}",
                delta=f"{total_return:.2%}"
            )
        
        with col2:
            st.metric(
                label="📈 Sharpe Ratio",
                value=f"{sharpe_ratio:.3f}",
                delta="Risk-adjusted return"
            )
        
        with col3:
            st.metric(
                label="📉 Max Drawdown", 
                value=f"{max_drawdown:.2%}",
                delta="Maximum loss"
            )
        
        with col4:
            active_trades = np.random.randint(0, 5)  # Mock data
            st.metric(
                label="🎯 Active Trades",
                value=active_trades,
                delta=f"+{active_trades} positions"
            )
    
    @staticmethod
    def risk_metrics_table(metrics: dict):
        """Display risk metrics in a formatted table."""
        df = pd.DataFrame([
            {"Metric": "Value at Risk (95%)", "Value": f"{metrics.get('var_95', 0):.2%}"},
            {"Metric": "Expected Shortfall", "Value": f"{metrics.get('cvar', 0):.2%}"},
            {"Metric": "Volatility (Annualized)", "Value": f"{metrics.get('volatility', 0):.2%}"},
            {"Metric": "Correlation to Market", "Value": f"{metrics.get('beta', 0):.3f}"},
            {"Metric": "Maximum Leverage", "Value": f"{metrics.get('max_leverage', 1):.2f}x"}
        ])
        
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    @staticmethod
    def trade_summary(trades_df: pd.DataFrame):
        """Display trade summary statistics."""
        if trades_df.empty:
            st.info("No trades to display")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trade Statistics")
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            stats = {
                "Total Trades": total_trades,
                "Winning Trades": winning_trades,
                "Win Rate": f"{win_rate:.1%}",
                "Avg P&L": f"${trades_df['pnl'].mean():.2f}",
                "Best Trade": f"${trades_df['pnl'].max():.2f}",
                "Worst Trade": f"${trades_df['pnl'].min():.2f}"
            }
            
            for key, value in stats.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.subheader("Recent Trades")
            recent_trades = trades_df.tail(5)[['pair', 'entry_date', 'pnl', 'status']]
            st.dataframe(recent_trades, hide_index=True)
