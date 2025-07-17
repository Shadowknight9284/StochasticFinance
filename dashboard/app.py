"""
Pairs Trading Algorithm - Streamlit Dashboard
Interactive web interface for model visualization and validation.
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure Streamlit page
st.set_page_config(
    page_title="Pairs Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = []
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}

def load_configuration():
    """Load configuration from the main application."""
    try:
        from src.utils.config import default_config
        return default_config
    except ImportError:
        st.error("Failed to load configuration. Please check the installation.")
        return None

def main():
    """Main dashboard application."""
    initialize_session_state()
    
    # Load configuration
    config = load_configuration()
    if config is None:
        return
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 Pairs Trading Algorithm Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("## 🧭 Navigation")
    
    # Navigation menu
    pages = {
        "🏠 Overview": "overview",
        "📊 Pair Selection": "pair_selection", 
        "🔬 Model Validation": "model_validation",
        "📈 Live Trading": "live_trading",
        "⚡ Backtesting": "backtesting",
        "⚠️ Risk Dashboard": "risk_dashboard"
    }
    
    selected_page = st.sidebar.selectbox(
        "Select Page",
        list(pages.keys()),
        index=0
    )
    
    # Configuration sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.strptime(config.data.default_start_date, '%Y-%m-%d').date(),
            key="start_date"
        )
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime.strptime(config.data.default_end_date, '%Y-%m-%d').date(),
            key="end_date"
        )
    
    # Stock universe input
    st.sidebar.markdown("### 📈 Stock Universe")
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"]
    
    universe_input = st.sidebar.text_area(
        "Enter tickers (comma-separated)",
        value=", ".join(default_tickers),
        height=100,
        help="Enter stock tickers separated by commas"
    )
    
    # Parse tickers
    tickers = [ticker.strip().upper() for ticker in universe_input.split(",") if ticker.strip()]
    
    # Model parameters
    st.sidebar.markdown("### 🔧 Model Parameters")
    correlation_threshold = st.sidebar.slider(
        "Min Correlation", 
        min_value=0.0, 
        max_value=1.0, 
        value=config.cointegration.min_correlation,
        step=0.05
    )
    
    pvalue_threshold = st.sidebar.slider(
        "P-value Threshold",
        min_value=0.01,
        max_value=0.1, 
        value=config.cointegration.pvalue_threshold,
        step=0.01
    )
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 System Status")
    
    # Mock system status for now
    st.sidebar.markdown("**Data Connection:** ✅ Connected")
    st.sidebar.markdown("**Model Status:** ⏳ Ready")
    st.sidebar.markdown("**Last Update:** " + datetime.now().strftime("%H:%M:%S"))
    
    # Main content area
    if pages[selected_page] == "overview":
        show_overview_page(config, tickers)
    elif pages[selected_page] == "pair_selection":
        show_pair_selection_page(tickers, correlation_threshold, pvalue_threshold)
    elif pages[selected_page] == "model_validation":
        show_model_validation_page()
    elif pages[selected_page] == "live_trading":
        show_live_trading_page()
    elif pages[selected_page] == "backtesting":
        show_backtesting_page()
    elif pages[selected_page] == "risk_dashboard":
        show_risk_dashboard_page()

def show_overview_page(config, tickers):
    """Show overview page with system summary."""
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Universe Size",
            value=len(tickers),
            delta=f"+{len(tickers)-5} from default"
        )
    
    with col2:
        # Mock data for now
        st.metric(
            label="🎯 Active Pairs", 
            value="0",
            delta="Ready to analyze"
        )
    
    with col3:
        st.metric(
            label="💼 Portfolio Value",
            value="$100,000",
            delta="Initial capital"
        )
    
    with col4:
        st.metric(
            label="📊 Model Status",
            value="Ready",
            delta="Initialized"
        )
    
    # System information
    st.markdown("## 🔧 System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Configuration")
        config_data = {
            "Database Path": config.data.database_path,
            "Min Price": f"${config.data.min_price}",
            "Min Volume": f"{config.data.min_volume:,}",
            "Min Observations": config.data.min_observations
        }
        
        for key, value in config_data.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.markdown("### Trading Configuration")
        trading_data = {
            "Entry Threshold": f"{config.trading.entry_threshold}σ",
            "Exit Threshold": f"{config.trading.exit_threshold}σ", 
            "Stop Loss": f"{config.trading.stop_loss_threshold}σ",
            "Max Positions": config.risk.max_positions
        }
        
        for key, value in trading_data.items():
            st.write(f"**{key}:** {value}")
    
    # Quick start guide
    st.markdown("## 🚀 Quick Start Guide")
    
    with st.expander("How to use this dashboard", expanded=True):
        st.markdown("""
        1. **📊 Pair Selection**: Analyze correlations and select trading pairs
        2. **🔬 Model Validation**: Validate Ornstein-Uhlenbeck model parameters 
        3. **📈 Live Trading**: Monitor real-time signals and positions
        4. **⚡ Backtesting**: Run historical performance analysis
        5. **⚠️ Risk Dashboard**: Monitor portfolio risk metrics
        
        Use the sidebar to configure parameters and navigate between pages.
        """)
    
    # Sample data preview
    st.markdown("## 📈 Stock Universe Preview")
    
    if st.button("🔄 Load Sample Data", type="primary"):
        with st.spinner("Loading sample data..."):
            # Create sample data for demonstration
            sample_data = []
            for ticker in tickers[:5]:  # Show first 5 tickers
                sample_data.append({
                    "Ticker": ticker,
                    "Last Price": f"${np.random.uniform(50, 300):.2f}",
                    "Volume": f"{np.random.randint(1000000, 50000000):,}",
                    "Volatility": f"{np.random.uniform(15, 40):.1f}%",
                    "Status": "✅ Valid"
                })
            
            df = pd.DataFrame(sample_data)
            st.dataframe(df, use_container_width=True)

def show_pair_selection_page(tickers, correlation_threshold, pvalue_threshold):
    """Show pair selection page."""
    st.markdown("## 📊 Pair Selection & Analysis")
    
    st.info("🔧 **Coming Soon**: Advanced pair selection with real-time correlation analysis")
    
    # Mock correlation matrix for demonstration
    if st.button("Generate Sample Correlation Matrix"):
        n_tickers = min(len(tickers), 8)  # Limit for display
        sample_tickers = tickers[:n_tickers]
        
        # Generate random correlation matrix
        np.random.seed(42)
        corr_matrix = np.random.rand(n_tickers, n_tickers)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1)  # Diagonal = 1
        
        corr_df = pd.DataFrame(corr_matrix, 
                              index=sample_tickers, 
                              columns=sample_tickers)
        
        st.plotly_chart(
            create_correlation_heatmap(corr_df),
            use_container_width=True
        )

def show_model_validation_page():
    """Show model validation page."""
    st.markdown("## 🔬 Model Validation")
    st.info("🔧 **Coming Soon**: Ornstein-Uhlenbeck model validation and diagnostics")

def show_live_trading_page():
    """Show live trading page."""
    st.markdown("## 📈 Live Trading Monitor")
    st.info("🔧 **Coming Soon**: Real-time trading signals and position monitoring")

def show_backtesting_page():
    """Show backtesting page."""
    st.markdown("## ⚡ Backtesting Engine")
    st.info("🔧 **Coming Soon**: Historical performance analysis and optimization")

def show_risk_dashboard_page():
    """Show risk dashboard page."""
    st.markdown("## ⚠️ Risk Management Dashboard")
    st.info("🔧 **Coming Soon**: Portfolio risk metrics and monitoring")

def create_correlation_heatmap(corr_matrix):
    """Create interactive correlation heatmap."""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Stock Correlation Matrix",
        width=600,
        height=500
    )
    
    return fig

if __name__ == "__main__":
    main()
