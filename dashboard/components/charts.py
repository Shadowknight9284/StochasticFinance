"""
Interactive chart components for Streamlit dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np

class InteractiveCharts:
    """Interactive chart components for the dashboard."""
    
    @staticmethod
    def correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix"):
        """Create interactive correlation heatmap."""
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
            title=title,
            height=500
        )
        
        return fig
    
    @staticmethod
    def price_comparison(data1: pd.DataFrame, data2: pd.DataFrame, 
                        ticker1: str, ticker2: str):
        """Create price comparison chart."""
        fig = sp.make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=[f'Price Comparison: {ticker1} vs {ticker2}', 'Spread Analysis'],
            vertical_spacing=0.1
        )
        
        # Normalize prices
        price1_norm = 100 * data1['close'] / data1['close'].iloc[0]
        price2_norm = 100 * data2['close'] / data2['close'].iloc[0]
        
        # Price traces
        fig.add_trace(
            go.Scatter(x=data1.index, y=price1_norm, name=f'{ticker1}',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data2.index, y=price2_norm, name=f'{ticker2}',
                      line=dict(color='orange', width=2)),
            row=1, col=1
        )
        
        # Spread
        spread = np.log(data1['close']) - np.log(data2['close'])
        fig.add_trace(
            go.Scatter(x=spread.index, y=spread, name='Log Spread',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        # Add spread statistics
        fig.add_hline(y=spread.mean(), line_dash="dash", line_color="red",
                     annotation_text="Mean", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        return fig
    
    @staticmethod
    def spread_analysis(spread: pd.Series, ticker1: str, ticker2: str):
        """Create comprehensive spread analysis."""
        zscore = (spread - spread.mean()) / spread.std()
        
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=['Spread Time Series', 'Z-Score', 
                          'Spread Distribution', 'Rolling Statistics'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Spread time series
        fig.add_trace(
            go.Scatter(x=spread.index, y=spread, name='Spread', 
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Z-score
        fig.add_trace(
            go.Scatter(x=zscore.index, y=zscore, name='Z-Score',
                      line=dict(color='purple')),
            row=1, col=2
        )
        
        # Distribution
        fig.add_trace(
            go.Histogram(x=spread.dropna(), name='Distribution', 
                        marker_color='lightblue'),
            row=2, col=1
        )
        
        # Rolling statistics
        window = min(30, len(spread) // 10)
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        fig.add_trace(
            go.Scatter(x=rolling_mean.index, y=rolling_mean, 
                      name=f'Rolling Mean ({window}d)',
                      line=dict(color='red')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=rolling_std.index, y=rolling_std,
                      name=f'Rolling Std ({window}d)', 
                      line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True,
                         title_text=f"Spread Analysis: {ticker1} - {ticker2}")
        return fig
