"""
Pairs analysis page for cointegration testing and relationship analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from components.charts import InteractiveCharts
from components.controls import DashboardControls
from components.metrics import PerformanceMetrics

def render_analysis_page():
    """Render the pairs analysis page."""
    st.title("📊 Pairs Analysis")
    st.markdown("Analyze stock pairs for cointegration and trading opportunities.")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Controls")
        
        # Mock data for demonstration
        available_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        stock1, stock2 = DashboardControls.stock_selector(available_stocks)
        start_date, end_date = DashboardControls.date_range_selector()
        analysis_options = DashboardControls.analysis_options()
    
    # Main content area
    if stock1 and stock2:
        st.subheader(f"Analysis: {stock1} vs {stock2}")
        
        # Generate mock data for demonstration
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)  # For consistent demo data
        
        # Mock price data
        price1 = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        price2 = 95 * np.exp(np.cumsum(np.random.normal(0.0008, 0.018, len(dates))))
        
        data = pd.DataFrame({
            'date': dates,
            f'{stock1}_price': price1,
            f'{stock2}_price': price2
        })
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Price Analysis", 
            "🔗 Cointegration", 
            "📊 Spread Analysis", 
            "📋 Statistics"
        ])
        
        with tab1:
            st.subheader("Price Comparison")
            
            # Price comparison chart
            charts = InteractiveCharts()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = charts.price_comparison_chart(
                    data, stock1, stock2, 
                    normalize=analysis_options.get('normalize_prices', True)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Correlation", f"{np.corrcoef(price1, price2)[0,1]:.3f}")
                st.metric("Price Ratio", f"{price1[-1]/price2[-1]:.3f}")
                st.metric("Volatility Ratio", f"{np.std(price1)/np.std(price2):.3f}")
        
        with tab2:
            st.subheader("Cointegration Analysis")
            
            # Mock cointegration results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Engle-Granger Test**")
                st.write("ADF Statistic: -3.42")
                st.write("P-value: 0.012")
                st.write("Critical Values:")
                st.write("- 1%: -3.96")
                st.write("- 5%: -3.41") 
                st.write("- 10%: -3.13")
                
                if -3.42 < -3.41:
                    st.success("✅ Pairs are cointegrated (5% level)")
                else:
                    st.warning("⚠️ No cointegration detected")
            
            with col2:
                st.write("**Johansen Test**")
                st.write("Trace Statistic: 23.45")
                st.write("Critical Value (5%): 20.26")
                st.write("Max Eigenvalue: 18.92")
                st.write("Critical Value (5%): 15.89")
                
                st.success("✅ Strong cointegration evidence")
        
        with tab3:
            st.subheader("Spread Analysis")
            
            # Calculate spread
            spread = price1 - price2
            z_score = (spread - spread.mean()) / spread.std()
            
            # Spread chart
            spread_data = pd.DataFrame({
                'date': dates,
                'spread': spread,
                'z_score': z_score
            })
            
            fig = charts.spread_analysis_chart(spread_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Spread statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Z-Score", f"{z_score.iloc[-1]:.2f}")
            with col2:
                st.metric("Mean Spread", f"{spread.mean():.2f}")
            with col3:
                st.metric("Spread Std", f"{spread.std():.2f}")
            with col4:
                st.metric("Half-Life", "12.3 days")
        
        with tab4:
            st.subheader("Statistical Summary")
            
            # Summary statistics table
            summary_stats = pd.DataFrame({
                'Metric': [
                    'Mean Return (Daily)',
                    'Volatility (Annualized)', 
                    'Sharpe Ratio',
                    'Maximum Drawdown',
                    'Skewness',
                    'Kurtosis'
                ],
                stock1: [
                    f"{np.mean(np.diff(price1)/price1[:-1]):.4f}",
                    f"{np.std(np.diff(price1)/price1[:-1]) * np.sqrt(252):.3f}",
                    f"{np.mean(np.diff(price1)/price1[:-1]) / np.std(np.diff(price1)/price1[:-1]) * np.sqrt(252):.3f}",
                    f"{(np.min(np.cumsum(np.diff(price1)/price1[:-1]))):.3f}",
                    f"{pd.Series(np.diff(price1)/price1[:-1]).skew():.3f}",
                    f"{pd.Series(np.diff(price1)/price1[:-1]).kurtosis():.3f}"
                ],
                stock2: [
                    f"{np.mean(np.diff(price2)/price2[:-1]):.4f}",
                    f"{np.std(np.diff(price2)/price2[:-1]) * np.sqrt(252):.3f}",
                    f"{np.mean(np.diff(price2)/price2[:-1]) / np.std(np.diff(price2)/price2[:-1]) * np.sqrt(252):.3f}",
                    f"{(np.min(np.cumsum(np.diff(price2)/price2[:-1]))):.3f}",
                    f"{pd.Series(np.diff(price2)/price2[:-1]).skew():.3f}",
                    f"{pd.Series(np.diff(price2)/price2[:-1]).kurtosis():.3f}"
                ]
            })
            
            st.dataframe(summary_stats, hide_index=True, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Rolling Correlation")
            rolling_corr = pd.Series(price1).rolling(30).corr(pd.Series(price2))
            
            corr_data = pd.DataFrame({
                'date': dates[29:],  # Skip first 29 due to rolling window
                'correlation': rolling_corr.dropna()
            })
            
            fig = charts.create_line_chart(
                corr_data, 'date', 'correlation',
                title="30-Day Rolling Correlation",
                y_title="Correlation"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please select two different stocks to begin analysis.")

if __name__ == "__main__":
    render_analysis_page()
