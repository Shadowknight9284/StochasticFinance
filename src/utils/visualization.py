"""
Core visualization module for pairs trading algorithm.
Provides comprehensive visual validation and monitoring capabilities.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PairsVisualization:
    """
    Comprehensive visualization suite for pairs trading analysis.
    Provides static and interactive plots for model validation.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_price_series(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                         ticker1: str, ticker2: str, 
                         save_path: str = None, show: bool = True) -> plt.Figure:
        """Plot normalized price series for two stocks."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Normalize prices to start at 100
        price1_norm = 100 * data1['close'] / data1['close'].iloc[0]
        price2_norm = 100 * data2['close'] / data2['close'].iloc[0]
        
        dates = pd.to_datetime(data1.index)
        
        # Price plots
        ax1.plot(dates, price1_norm, label=f'{ticker1} (Normalized)', 
                color=self.colors[0], linewidth=1.5)
        ax1.plot(dates, price2_norm, label=f'{ticker2} (Normalized)', 
                color=self.colors[1], linewidth=1.5)
        ax1.set_title(f'Normalized Price Comparison: {ticker1} vs {ticker2}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Normalized Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Spread plot
        spread = np.log(data1['close']) - np.log(data2['close'])
        ax2.plot(dates, spread, label='Log Price Spread', color=self.colors[2], linewidth=1.5)
        ax2.axhline(y=spread.mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
        ax2.fill_between(dates, spread.mean() - 2*spread.std(), spread.mean() + 2*spread.std(), 
                        alpha=0.2, color='gray', label='±2σ')
        ax2.set_title('Log Price Spread', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Log Spread')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            
        return fig
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                save_path: str = None, show: bool = True) -> plt.Figure:
        """Plot correlation heatmap for stock universe."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, ax=ax, cbar_kws={"shrink": .8})
        
        ax.set_title('Stock Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            
        return fig
    
    def plot_spread_analysis(self, spread: pd.Series, ticker1: str, ticker2: str,
                           save_path: str = None, show: bool = True) -> plt.Figure:
        """Comprehensive spread analysis visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        dates = pd.to_datetime(spread.index)
        
        # 1. Spread time series with statistics
        ax1.plot(dates, spread, color=self.colors[0], alpha=0.8)
        ax1.axhline(y=spread.mean(), color='red', linestyle='-', linewidth=2, label='Mean')
        ax1.axhline(y=spread.mean() + spread.std(), color='orange', linestyle='--', label='+1σ')
        ax1.axhline(y=spread.mean() - spread.std(), color='orange', linestyle='--', label='-1σ')
        ax1.axhline(y=spread.mean() + 2*spread.std(), color='red', linestyle='--', alpha=0.7, label='±2σ')
        ax1.axhline(y=spread.mean() - 2*spread.std(), color='red', linestyle='--', alpha=0.7)
        ax1.set_title(f'Spread Time Series: {ticker1} - {ticker2}')
        ax1.set_ylabel('Spread')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Spread distribution
        ax2.hist(spread.dropna(), bins=50, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.axvline(spread.mean(), color='red', linestyle='-', linewidth=2, label='Mean')
        ax2.axvline(spread.median(), color='green', linestyle='-', linewidth=2, label='Median')
        ax2.set_title('Spread Distribution')
        ax2.set_xlabel('Spread Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Z-score
        zscore = (spread - spread.mean()) / spread.std()
        ax3.plot(dates, zscore, color=self.colors[2], alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Entry/Exit Levels')
        ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=3, color='darkred', linestyle='--', alpha=0.7, label='Stop Loss')
        ax3.axhline(y=-3, color='darkred', linestyle='--', alpha=0.7)
        ax3.set_title('Spread Z-Score')
        ax3.set_ylabel('Z-Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Rolling statistics
        window = min(30, len(spread) // 10)
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        ax4.plot(dates, rolling_mean, label=f'Rolling Mean ({window}d)', color=self.colors[3])
        ax4.plot(dates, rolling_std, label=f'Rolling Std ({window}d)', color=self.colors[4])
        ax4.set_title('Rolling Statistics')
        ax4.set_ylabel('Value')
        ax4.set_xlabel('Date')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Format dates
        for ax in [ax1, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            
        return fig
    
    def plot_interactive_dashboard(self, data1: pd.DataFrame, data2: pd.DataFrame,
                                 ticker1: str, ticker2: str,
                                 save_path: str = None) -> go.Figure:
        """Create interactive dashboard using Plotly."""
        
        # Calculate spread and z-score
        spread = np.log(data1['close']) - np.log(data2['close'])
        zscore = (spread - spread.mean()) / spread.std()
        
        # Create subplots
        fig = sp.make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                f'Price Comparison: {ticker1} vs {ticker2}',
                'Log Price Spread',
                'Z-Score with Trading Signals'
            ],
            vertical_spacing=0.1
        )
        
        # Normalize prices
        price1_norm = 100 * data1['close'] / data1['close'].iloc[0]
        price2_norm = 100 * data2['close'] / data2['close'].iloc[0]
        
        # Price traces
        fig.add_trace(
            go.Scatter(x=data1.index, y=price1_norm, name=f'{ticker1} (Normalized)',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data2.index, y=price2_norm, name=f'{ticker2} (Normalized)',
                      line=dict(color='orange', width=2)),
            row=1, col=1
        )
        
        # Spread trace
        fig.add_trace(
            go.Scatter(x=spread.index, y=spread, name='Log Spread',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        # Add spread mean line
        fig.add_hline(y=spread.mean(), line_dash="dash", line_color="red",
                     annotation_text="Mean", row=2, col=1)
        
        # Z-score trace
        fig.add_trace(
            go.Scatter(x=zscore.index, y=zscore, name='Z-Score',
                      line=dict(color='purple', width=2)),
            row=3, col=1
        )
        
        # Add trading signal lines
        for level, color, name in [(2, 'red', 'Entry/Exit'), (-2, 'red', ''), 
                                  (3, 'darkred', 'Stop Loss'), (-3, 'darkred', '')]:
            fig.add_hline(y=level, line_dash="dash", line_color=color,
                         annotation_text=name if name else "", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'Pairs Trading Dashboard: {ticker1} vs {ticker2}',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
        fig.update_yaxes(title_text="Log Spread", row=2, col=1)
        fig.update_yaxes(title_text="Z-Score", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        if save_path:
            plot(fig, filename=save_path, auto_open=False)
            
        return fig
    
    def plot_model_diagnostics(self, residuals: pd.Series, fitted_values: pd.Series,
                             save_path: str = None, show: bool = True) -> plt.Figure:
        """Plot model diagnostic plots for validation."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Residuals vs Fitted
        ax1.scatter(fitted_values, residuals, alpha=0.6, color=self.colors[0])
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q plot
        from scipy import stats
        stats.probplot(residuals.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residual histogram
        ax3.hist(residuals.dropna(), bins=30, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Autocorrelation of residuals
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(residuals.dropna(), ax=ax4, color=self.colors[2])
        ax4.set_title('Residual Autocorrelation')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            
        return fig


class ModelValidation:
    """Model validation and goodness-of-fit testing."""
    
    @staticmethod
    def jarque_bera_test(residuals: pd.Series) -> Dict[str, float]:
        """Jarque-Bera test for normality."""
        from scipy.stats import jarque_bera
        stat, pvalue = jarque_bera(residuals.dropna())
        return {'statistic': stat, 'pvalue': pvalue, 'is_normal': pvalue > 0.05}
    
    @staticmethod
    def ljung_box_test(residuals: pd.Series, lags: int = 10) -> Dict[str, float]:
        """Ljung-Box test for autocorrelation."""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        result = acorr_ljungbox(residuals.dropna(), lags=lags, return_df=True)
        return {
            'statistic': result['lb_stat'].iloc[-1],
            'pvalue': result['lb_pvalue'].iloc[-1],
            'no_autocorr': result['lb_pvalue'].iloc[-1] > 0.05
        }
    
    @staticmethod
    def adf_stationarity_test(series: pd.Series) -> Dict[str, float]:
        """Augmented Dickey-Fuller test for stationarity."""
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna())
        return {
            'statistic': result[0],
            'pvalue': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }


# Update requirements for visualization
VISUALIZATION_REQUIREMENTS = [
    "plotly>=5.0.0",
    "matplotlib>=3.7.0", 
    "seaborn>=0.12.0"
]
