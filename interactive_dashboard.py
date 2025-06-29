"""
Interactive Web Dashboard for Stochastic Finance Strategies
Real-time visualization with Plotly Dash
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import os
import subprocess
import json

class InteractiveDashboard:
    """Interactive web dashboard for strategy visualization"""
    
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.strategies_data = {}
        self.market_data = None
        
        # Strategy configurations
        self.strategy_configs = {
            'HestonVolSurface': {'color': '#FF6B6B', 'name': 'Heston Vol Surface'},
            'JumpDiffusion': {'color': '#4ECDC4', 'name': 'Jump Diffusion'},
            'LogNormalJumpMeanReversion': {'color': '#45B7D1', 'name': 'LNJMR'},
            'OrnsteinUhlenbeck': {'color': '#96CEB4', 'name': 'Ornstein-Uhlenbeck'}
        }
        
        self.setup_layout()
        self.setup_callbacks()
    
    def generate_sample_data(self):
        """Generate sample trading data for demonstration"""
        np.random.seed(42)
        
        # Generate 100 days of hourly data
        dates = pd.date_range(start='2024-01-01', periods=2400, freq='H')
        
        # Market price simulation
        price = 100
        prices = [price]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0, 0.001)  # Small random changes
            price *= (1 + change)
            prices.append(price)
        
        self.market_data = pd.DataFrame({
            'timestamp': dates,
            'price': prices
        })
        
        # Generate strategy performance data
        for strategy_name, config in self.strategy_configs.items():
            # Simulate different strategy performance
            returns = np.random.normal(0.0001, 0.01, len(dates))
            
            if strategy_name == 'HestonVolSurface':
                returns += np.where(np.random.random(len(dates)) < 0.1, 
                                  np.random.normal(0, 0.02), 0)  # Volatility spikes
            elif strategy_name == 'JumpDiffusion':
                returns += np.where(np.random.random(len(dates)) < 0.05,
                                  np.random.normal(0, 0.05), 0)  # Jump events
            
            cumulative_returns = np.cumprod(1 + returns) - 1
            portfolio_value = 100000 * (1 + cumulative_returns)
            
            # Generate trading signals
            signals = np.random.choice([0, 1, -1], len(dates), p=[0.85, 0.075, 0.075])
            buy_signals = np.where(signals == 1)[0]
            sell_signals = np.where(signals == -1)[0]
            
            self.strategies_data[strategy_name] = {
                'portfolio_value': portfolio_value,
                'returns': cumulative_returns,
                'signals': signals,
                'buy_points': [(i, prices[i]) for i in buy_signals],
                'sell_points': [(i, prices[i]) for i in sell_signals],
                'performance': {
                    'total_return': cumulative_returns[-1],
                    'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252*24),
                    'max_drawdown': self.calculate_max_drawdown(portfolio_value),
                    'volatility': np.std(returns) * np.sqrt(252*24)
                }
            }
    
    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🚀 Stochastic Finance Strategy Dashboard", 
                           className="text-center mb-4", 
                           style={'color': '#2E4057', 'fontWeight': 'bold'})
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊 Real-Time Performance", className="card-title"),
                            dcc.Graph(id='performance-chart', style={'height': '400px'})
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("⚡ Strategy Metrics", className="card-title"),
                            html.Div(id='metrics-table')
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📈 Price & Signals", className="card-title"),
                            dcc.Graph(id='price-signals-chart', style={'height': '350px'})
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🎯 Risk Analysis", className="card-title"),
                            dcc.Graph(id='risk-analysis-chart', style={'height': '350px'})
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("⚙️ Strategy Controls", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Time Period:"),
                                    dcc.Dropdown(
                                        id='time-period',
                                        options=[
                                            {'label': 'Last 24 Hours', 'value': 24},
                                            {'label': 'Last 7 Days', 'value': 168},
                                            {'label': 'Last 30 Days', 'value': 720},
                                            {'label': 'All Data', 'value': -1}
                                        ],
                                        value=168
                                    )
                                ], width=6),
                                
                                dbc.Col([
                                    html.Label("Update Frequency:"),
                                    dcc.Dropdown(
                                        id='update-frequency',
                                        options=[
                                            {'label': '1 Second', 'value': 1000},
                                            {'label': '5 Seconds', 'value': 5000},
                                            {'label': '10 Seconds', 'value': 10000}
                                        ],
                                        value=5000
                                    )
                                ], width=6)
                            ])
                        ])
                    ])
                ])
            ]),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=5000,  # Update every 5 seconds
                n_intervals=0
            )
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            [Output('performance-chart', 'figure'),
             Output('metrics-table', 'children'),
             Output('price-signals-chart', 'figure'),
             Output('risk-analysis-chart', 'figure')],
            [Input('interval-component', 'n_intervals'),
             Input('time-period', 'value')]
        )
        def update_dashboard(n, time_period):
            # Generate fresh data on each update
            self.generate_sample_data()
            
            # Filter data based on time period
            if time_period > 0:
                data_slice = slice(-time_period, None)
            else:
                data_slice = slice(None)
            
            timestamps = self.market_data['timestamp'].iloc[data_slice]
            
            # 1. Performance Chart
            performance_fig = go.Figure()
            
            for strategy_name, data in self.strategies_data.items():
                config = self.strategy_configs[strategy_name]
                performance_fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=data['portfolio_value'][data_slice],
                    mode='lines',
                    name=config['name'],
                    line=dict(color=config['color'], width=3),
                    hovertemplate=f'<b>{config["name"]}</b><br>' +
                                 'Value: $%{y:,.0f}<br>' +
                                 'Time: %{x}<extra></extra>'
                ))
            
            performance_fig.update_layout(
                title='Portfolio Value Over Time',
                xaxis_title='Time',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True
            )
            
            # 2. Metrics Table
            metrics_data = []
            for strategy_name, data in self.strategies_data.items():
                config = self.strategy_configs[strategy_name]
                perf = data['performance']
                metrics_data.append(html.Tr([
                    html.Td(config['name'], style={'fontWeight': 'bold', 'color': config['color']}),
                    html.Td(f"{perf['total_return']*100:.1f}%"),
                    html.Td(f"{perf['sharpe_ratio']:.2f}"),
                    html.Td(f"{perf['max_drawdown']*100:.1f}%")
                ]))
            
            metrics_table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Strategy"),
                        html.Th("Return"),
                        html.Th("Sharpe"),
                        html.Th("Max DD")
                    ])
                ]),
                html.Tbody(metrics_data)
            ], striped=True, bordered=True, hover=True, size='sm')
            
            # 3. Price & Signals Chart
            price_fig = go.Figure()
            
            # Market price
            price_fig.add_trace(go.Scatter(
                x=timestamps,
                y=self.market_data['price'].iloc[data_slice],
                mode='lines',
                name='Market Price',
                line=dict(color='black', width=2)
            ))
            
            # Trading signals for each strategy
            for strategy_name, data in self.strategies_data.items():
                config = self.strategy_configs[strategy_name]
                
                # Buy signals
                if data['buy_points']:
                    buy_indices, buy_prices = zip(*data['buy_points'])
                    buy_times = [timestamps.iloc[i] for i in buy_indices if i < len(timestamps)]
                    buy_prices_filtered = [buy_prices[j] for j, i in enumerate(buy_indices) if i < len(timestamps)]
                    
                    if buy_times:
                        price_fig.add_trace(go.Scatter(
                            x=buy_times,
                            y=buy_prices_filtered,
                            mode='markers',
                            name=f'{config["name"]} Buy',
                            marker=dict(symbol='triangle-up', size=10, color=config['color'])
                        ))
                
                # Sell signals
                if data['sell_points']:
                    sell_indices, sell_prices = zip(*data['sell_points'])
                    sell_times = [timestamps.iloc[i] for i in sell_indices if i < len(timestamps)]
                    sell_prices_filtered = [sell_prices[j] for j, i in enumerate(sell_indices) if i < len(timestamps)]
                    
                    if sell_times:
                        price_fig.add_trace(go.Scatter(
                            x=sell_times,
                            y=sell_prices_filtered,
                            mode='markers',
                            name=f'{config["name"]} Sell',
                            marker=dict(symbol='triangle-down', size=10, color=config['color'])
                        ))
            
            price_fig.update_layout(
                title='Market Price with Trading Signals',
                xaxis_title='Time',
                yaxis_title='Price ($)',
                template='plotly_white',
                showlegend=True
            )
            
            # 4. Risk Analysis (Return vs Volatility)
            risk_fig = go.Figure()
            
            for strategy_name, data in self.strategies_data.items():
                config = self.strategy_configs[strategy_name]
                perf = data['performance']
                
                risk_fig.add_trace(go.Scatter(
                    x=[perf['volatility']*100],
                    y=[perf['total_return']*100],
                    mode='markers+text',
                    name=config['name'],
                    text=[config['name']],
                    textposition='top center',
                    marker=dict(size=15, color=config['color'])
                ))
            
            risk_fig.update_layout(
                title='Risk vs Return Analysis',
                xaxis_title='Volatility (%)',
                yaxis_title='Return (%)',
                template='plotly_white',
                showlegend=False
            )
            
            return performance_fig, metrics_table, price_fig, risk_fig
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        print("🚀 Starting Interactive Strategy Dashboard...")
        print(f"📊 Dashboard will be available at: http://localhost:{port}")
        print("🔄 Auto-refreshing every 5 seconds with simulated real-time data")
        print("💡 Use Ctrl+C to stop the server")
        
        self.app.run(debug=debug, port=port, host='0.0.0.0')

if __name__ == "__main__":
    dashboard = InteractiveDashboard()
    dashboard.run_server()
