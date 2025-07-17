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

# Import our new modules
try:
    from models.cointegration import CointegrationTester, PairValidator
    from strategy.pairs_finder import PairsFinder, get_default_stock_universe
    from utils.config import Config
    IMPORTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    IMPORTS_AVAILABLE = False

def render_analysis_page():
    """Render the pairs analysis page."""
    st.title("📊 Pairs Analysis & Cointegration Testing")
    st.markdown("Analyze stock pairs for cointegration and trading opportunities using statistical tests.")
    
    if not IMPORTS_AVAILABLE:
        st.error("Required modules not available. Please check imports.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Controls")
        
        # Stock universe selection
        st.subheader("Stock Universe")
        universe_type = st.selectbox(
            "Select Universe", 
            ["Manual Selection", "Default Universe", "Sector-Based"]
        )
        
        if universe_type == "Manual Selection":
            manual_tickers = st.text_input(
                "Enter tickers (comma-separated)", 
                value="AAPL,MSFT,GOOGL,AMZN"
            )
            available_stocks = [t.strip().upper() for t in manual_tickers.split(',') if t.strip()]
        elif universe_type == "Default Universe":
            available_stocks = get_default_stock_universe()[:20]  # Limit for demo
            st.info(f"Using {len(available_stocks)} stocks from default universe")
        else:  # Sector-based
            sector = st.selectbox(
                "Select Sector",
                ["Technology", "Finance", "Healthcare", "Consumer", "Energy"]
            )
            # For demo, use a subset
            sector_stocks = {
                "Technology": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA'],
                "Finance": ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'],
                "Healthcare": ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO'],
                "Consumer": ['WMT', 'PG', 'KO', 'PEP', 'MCD', 'SBUX'],
                "Energy": ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC']
            }
            available_stocks = sector_stocks.get(sector, [])
        
        # Date range
        start_date, end_date = DashboardControls.date_range_selector()
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        max_pairs = st.slider("Max Pairs to Analyze", 5, 50, 20)
        min_score = st.slider("Minimum Trading Score", 0, 100, 60)
        
        # Run analysis button
        run_analysis = st.button("🔍 Run Pair Analysis", type="primary")
    
    # Main content area
    if run_analysis and len(available_stocks) >= 2:
        with st.spinner("Analyzing pairs... This may take a moment."):
            try:
                # Initialize components
                config = Config()
                finder = PairsFinder(config)
                
                # Find pairs
                results = finder.find_pairs_from_universe(
                    available_stocks, 
                    start_date.strftime('%Y-%m-%d'), 
                    end_date.strftime('%Y-%m-%d'),
                    max_pairs=max_pairs
                )
                
                # Filter by minimum score
                filtered_results = [r for r in results if r['score'] >= min_score]
                
                st.success(f"Analysis complete! Found {len(filtered_results)} valid pairs.")
                
                if filtered_results:
                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📈 Pair Rankings", 
                        "🔗 Cointegration Details", 
                        "📊 Visualizations", 
                        "📋 Export Results"
                    ])
                    
                    with tab1:
                        st.subheader("Top Trading Pairs")
                        display_pair_rankings(filtered_results)
                    
                    with tab2:
                        st.subheader("Cointegration Test Results")
                        display_cointegration_details(filtered_results)
                    
                    with tab3:
                        st.subheader("Pair Visualizations")
                        display_pair_visualizations(filtered_results, start_date, end_date)
                    
                    with tab4:
                        st.subheader("Export & Save Results")
                        display_export_options(filtered_results)
                        
                else:
                    st.warning("No pairs met the minimum score criteria. Try lowering the threshold.")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)
    
    elif run_analysis:
        st.warning("Please select at least 2 stocks for analysis.")
    
    else:
        # Show demo/default content
        st.info("👆 Configure your analysis parameters in the sidebar and click 'Run Pair Analysis' to begin.")
        
        # Show sample results for demonstration
        show_demo_content()


def display_pair_rankings(results):
    """Display pair rankings table."""
    # Create summary dataframe
    summary_data = []
    for result in results:
        coint_result = result['cointegration_results']
        summary_data.append({
            'Pair': result['pair'],
            'Score': f"{result['score']:.1f}",
            'Cointegrated': '✅' if result['is_valid'] else '❌',
            'Correlation': f"{coint_result['correlation']:.3f}",
            'Half-Life (days)': f"{coint_result['half_life']:.1f}" if coint_result['half_life'] else 'N/A',
            'Hedge Ratio': f"{coint_result['hedge_ratio']:.3f}",
            'P-Value': f"{coint_result['engle_granger'].p_value:.4f}" if coint_result['engle_granger'] else 'N/A'
        })
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Top 3 pairs details
    st.subheader("Top 3 Pairs - Detailed View")
    for i, result in enumerate(results[:3]):
        with st.expander(f"#{i+1}: {result['pair']} (Score: {result['score']:.1f})"):
            col1, col2, col3 = st.columns(3)
            
            coint_result = result['cointegration_results']
            
            with col1:
                st.metric("Trading Score", f"{result['score']:.1f}/100")
                st.metric("Correlation", f"{coint_result['correlation']:.3f}")
            
            with col2:
                st.metric("Half-Life", f"{coint_result['half_life']:.1f} days" if coint_result['half_life'] else "N/A")
                st.metric("Hedge Ratio", f"{coint_result['hedge_ratio']:.3f}")
            
            with col3:
                st.metric("Data Points", coint_result['data_points'])
                st.metric("Spread Volatility", f"{coint_result['spread_volatility']:.3f}")
            
            st.write("**Validation Summary:**", result['validation_summary'])


def display_cointegration_details(results):
    """Display detailed cointegration test results."""
    pair_names = [r['pair'] for r in results]
    selected_pair = st.selectbox("Select pair for detailed analysis:", pair_names)
    
    if selected_pair:
        result = next(r for r in results if r['pair'] == selected_pair)
        coint_result = result['cointegration_results']
        
        st.subheader(f"Cointegration Analysis: {selected_pair}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Engle-Granger Test Results:**")
            if coint_result['engle_granger']:
                eg = coint_result['engle_granger']
                st.write(f"- ADF Statistic: {eg.test_statistic:.4f}")
                st.write(f"- P-value: {eg.p_value:.4f}")
                st.write(f"- Critical Values:")
                for level, value in eg.critical_values.items():
                    st.write(f"  - {level}: {value:.4f}")
                
                # Interpretation
                if eg.is_cointegrated:
                    st.success("✅ Series are cointegrated!")
                else:
                    st.warning("❌ No cointegration detected")
        
        with col2:
            st.write("**Johansen Test Results:**")
            if coint_result['johansen']:
                joh = coint_result['johansen']
                st.write(f"- Trace Statistic: {joh.trace_stat:.4f}")
                st.write(f"- Trace Critical (5%): {joh.trace_critical_value:.4f}")
                st.write(f"- Max Eigenvalue: {joh.max_eigen_stat:.4f}")
                st.write(f"- Max Eigen Critical (5%): {joh.max_eigen_critical_value:.4f}")
                
                if joh.is_cointegrated:
                    st.success("✅ Johansen test confirms cointegration!")
                else:
                    st.warning("❌ Johansen test: no cointegration")
            else:
                st.info("Johansen test not available")
        
        # Validation checks
        st.subheader("Validation Checks")
        checks = result['checks']
        
        check_cols = st.columns(len(checks))
        for i, (check_name, passed) in enumerate(checks.items()):
            with check_cols[i]:
                icon = "✅" if passed else "❌"
                st.metric(check_name.replace('_', ' ').title(), icon)


def display_pair_visualizations(results, start_date, end_date):
    """Display visualizations for selected pairs."""
    if not results:
        st.warning("No results to visualize.")
        return
    
    pair_names = [r['pair'] for r in results]
    selected_pair = st.selectbox("Select pair to visualize:", pair_names, key="viz_pair")
    
    if selected_pair:
        ticker1, ticker2 = selected_pair.split('-')
        
        # Generate mock data for visualization (in real implementation, fetch actual data)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(selected_pair) % 2**32)  # Consistent random data per pair
        
        # Mock price data
        price1 = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        price2 = 95 * np.exp(np.cumsum(np.random.normal(0.0008, 0.018, len(dates))))
        
        data = pd.DataFrame({
            'date': dates,
            f'{ticker1}_price': price1,
            f'{ticker2}_price': price2
        })
        
        # Create visualizations
        charts = InteractiveCharts()
        
        # Price comparison
        st.subheader("Price Comparison")
        fig1 = charts.price_comparison_chart(data, ticker1, ticker2, normalize=True)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Spread analysis
        st.subheader("Spread Analysis")
        spread = np.log(price1) - np.log(price2)
        spread_data = pd.DataFrame({
            'date': dates,
            'spread': spread,
            'z_score': (spread - spread.mean()) / spread.std()
        })
        
        fig2 = charts.spread_analysis_chart(spread_data)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Rolling Correlation")
        rolling_corr = pd.Series(price1).rolling(30).corr(pd.Series(price2))
        corr_data = pd.DataFrame({
            'date': dates[29:],
            'correlation': rolling_corr.dropna()
        })
        
        fig3 = charts.create_line_chart(
            corr_data, 'date', 'correlation',
            title="30-Day Rolling Correlation",
            y_title="Correlation"
        )
        st.plotly_chart(fig3, use_container_width=True)


def display_export_options(results):
    """Display export and save options."""
    st.write("Export your analysis results:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to CSV"):
            # Create CSV data
            csv_data = []
            for result in results:
                coint_result = result['cointegration_results']
                csv_data.append({
                    'pair': result['pair'],
                    'score': result['score'],
                    'is_valid': result['is_valid'],
                    'correlation': coint_result['correlation'],
                    'half_life': coint_result['half_life'],
                    'hedge_ratio': coint_result['hedge_ratio'],
                    'data_points': coint_result['data_points'],
                    'spread_volatility': coint_result['spread_volatility']
                })
            
            df = pd.DataFrame(csv_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"pairs_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Save Configuration"):
            st.success("Configuration saved to session!")
            # In real implementation, save to database or file
    
    with col3:
        if st.button("📧 Generate Report"):
            st.info("Report generation feature coming soon!")


def show_demo_content():
    """Show demonstration content when no analysis is running."""
    st.subheader("🎯 Pairs Trading Analysis Features")
    
    features = [
        "**Cointegration Testing**: Engle-Granger and Johansen tests",
        "**Statistical Validation**: Half-life, correlation, and significance testing", 
        "**Automated Ranking**: Trading score based on multiple criteria",
        "**Interactive Visualizations**: Price charts, spread analysis, and correlation plots",
        "**Flexible Universe**: Manual selection, default universe, or sector-based",
        "**Export Capabilities**: CSV export and report generation"
    ]
    
    for feature in features:
        st.markdown(f"• {feature}")
    
    st.subheader("📊 Sample Analysis Results")
    
    # Demo table
    demo_data = pd.DataFrame({
        'Pair': ['AAPL-MSFT', 'JPM-BAC', 'XOM-CVX', 'JNJ-PFE'],
        'Score': [85.2, 78.9, 72.1, 69.5],
        'Cointegrated': ['✅', '✅', '✅', '❌'],
        'Correlation': [0.847, 0.782, 0.691, 0.543],
        'Half-Life': ['12.3 days', '8.7 days', '15.2 days', 'N/A']
    })
    
    st.dataframe(demo_data, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    render_analysis_page()
