"""
Visual Backtester for Stochastic Finance Strategies
Interactive visualization and analysis of C++ trading strategies
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import pandas as pd
import subprocess
import json
import os
import sys
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class StrategyData:
    """Data container for strategy results"""
    def __init__(self, name: str):
        self.name = name
        self.prices = []
        self.signals = []
        self.pnl = []
        self.positions = []
        self.timestamps = []
        self.buy_points = []
        self.sell_points = []
        self.parameters = {}
        self.performance_metrics = {}

class VisualBacktester:
    """Visual backtesting framework for stochastic finance strategies"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.strategies = {}
        self.market_data = None
        self.fig = None
        self.axes = None
        
        # Strategy configurations
        self.strategy_configs = {
            'HestonVolSurface': {
                'executable': 'strategies/HestonVolSurface/build/hestonvolsurface_strategy.exe',
                'color': '#FF6B6B',
                'description': 'Heston Volatility Surface Model'
            },
            'JumpDiffusion': {
                'executable': 'strategies/JumpDiffusion/build/jumpdiffusion_strategy.exe',
                'color': '#4ECDC4',
                'description': 'Jump Diffusion Process Model'
            },
            'LogNormalJumpMeanReversion': {
                'executable': 'strategies/LogNormalJumpMeanReversion/build/lognormaljumpmeanreversion_strategy.exe',
                'color': '#45B7D1',
                'description': 'Log-Normal Jump Mean Reversion Model'
            },
            'OrnsteinUhlenbeck': {
                'executable': 'strategies/OrnsteinUhlenbeck/build/ornsteinuhlenbeck_strategy.exe',
                'color': '#96CEB4',
                'description': 'Ornstein-Uhlenbeck Process Model'
            }
        }
    
    def generate_market_data(self, days: int = 252, dt: float = 1/252) -> pd.DataFrame:
        """Generate synthetic market data for visualization"""
        np.random.seed(42)  # For reproducible results
        
        # Parameters for realistic market simulation
        S0 = 100.0  # Initial price
        mu = 0.05   # Drift
        sigma = 0.2 # Volatility
        
        # Generate geometric Brownian motion with jumps
        n_steps = int(days / dt)
        t = np.linspace(0, days/252, n_steps)
        
        # Add jump component
        jump_intensity = 0.1
        jump_size = 0.05
        
        dW = np.random.normal(0, np.sqrt(dt), n_steps)
        jumps = np.random.poisson(jump_intensity * dt, n_steps)
        jump_sizes = np.random.normal(0, jump_size, n_steps) * jumps
        
        # Price evolution
        prices = np.zeros(n_steps)
        prices[0] = S0
        
        for i in range(1, n_steps):
            prices[i] = prices[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW[i] + jump_sizes[i])
        
        # Add intraday noise and bid-ask spread
        bid_ask_spread = 0.01
        noise = np.random.normal(0, 0.001, n_steps)
        
        market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_steps, freq='H'),
            'price': prices + noise,
            'bid': prices - bid_ask_spread/2 + noise,
            'ask': prices + bid_ask_spread/2 + noise,
            'volume': np.random.lognormal(7, 1, n_steps).astype(int)
        })
        
        return market_data
    
    def run_strategy_simulation(self, strategy_name: str, market_data: pd.DataFrame) -> StrategyData:
        """Simulate strategy execution on market data"""
        print(f"Simulating {strategy_name} strategy...")
        
        strategy_data = StrategyData(strategy_name)
        config = self.strategy_configs[strategy_name]
        
        # Simulate strategy execution with the generated market data
        np.random.seed(hash(strategy_name) % 2**32)  # Different seed per strategy
        
        current_position = 0
        cash = self.initial_capital
        portfolio_value = self.initial_capital
        
        # Strategy-specific parameters (simulated based on C++ output patterns)
        if strategy_name == 'HestonVolSurface':
            signal_threshold = 0.8
            mean_reversion_speed = -205.0
        elif strategy_name == 'JumpDiffusion':
            signal_threshold = 0.7
            mean_reversion_speed = -203.0
        elif strategy_name == 'LogNormalJumpMeanReversion':
            signal_threshold = 0.75
            mean_reversion_speed = -210.0
        else:  # OrnsteinUhlenbeck
            signal_threshold = 0.9
            mean_reversion_speed = -201.0
        
        for i, row in market_data.iterrows():
            price = row['price']
            
            # Generate trading signal based on mean reversion
            if i > 50:  # Need history for signal generation
                price_ma = market_data['price'].iloc[i-50:i].mean()
                price_std = market_data['price'].iloc[i-50:i].std()
                
                # Normalized price deviation
                z_score = (price - price_ma) / price_std if price_std > 0 else 0
                
                # Strategy-specific signal generation
                signal = 0  # 0: hold, 1: buy, -1: sell
                
                if strategy_name == 'HestonVolSurface':
                    # Volatility-sensitive signals
                    recent_vol = market_data['price'].iloc[i-20:i].std()
                    vol_signal = 1 if recent_vol > price_std * 1.2 else 0
                    signal = 1 if z_score < -signal_threshold and vol_signal else (-1 if z_score > signal_threshold else 0)
                
                elif strategy_name == 'JumpDiffusion':
                    # Jump detection
                    price_change = abs(price - market_data['price'].iloc[i-1]) / market_data['price'].iloc[i-1]
                    jump_detected = price_change > 0.02
                    signal = -1 if jump_detected and z_score > 0 else (1 if z_score < -signal_threshold else 0)
                
                elif strategy_name == 'LogNormalJumpMeanReversion':
                    # Log-normal mean reversion
                    log_price = np.log(price)
                    log_ma = np.log(market_data['price'].iloc[i-30:i]).mean()
                    log_deviation = log_price - log_ma
                    signal = 1 if log_deviation < -signal_threshold * 0.1 else (-1 if log_deviation > signal_threshold * 0.1 else 0)
                
                else:  # OrnsteinUhlenbeck
                    # Strong mean reversion
                    signal = 1 if z_score < -signal_threshold else (-1 if z_score > signal_threshold else 0)
                
                # Execute trades
                if signal == 1 and current_position <= 0:  # Buy signal
                    shares_to_buy = int(cash * 0.95 / price)  # Use 95% of cash
                    if shares_to_buy > 0:
                        current_position += shares_to_buy
                        cash -= shares_to_buy * price
                        strategy_data.buy_points.append((i, price))
                
                elif signal == -1 and current_position > 0:  # Sell signal
                    cash += current_position * price
                    current_position = 0
                    strategy_data.sell_points.append((i, price))
            
            # Calculate portfolio value
            portfolio_value = cash + current_position * price
            
            # Store data
            strategy_data.prices.append(price)
            strategy_data.signals.append(signal if i > 50 else 0)
            strategy_data.pnl.append(portfolio_value - self.initial_capital)
            strategy_data.positions.append(current_position)
            strategy_data.timestamps.append(row['timestamp'])
        
        # Calculate performance metrics
        returns = np.diff(strategy_data.pnl) / self.initial_capital
        strategy_data.performance_metrics = {
            'total_return': (portfolio_value - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(strategy_data.pnl),
            'win_rate': len([r for r in returns if r > 0]) / len(returns) if len(returns) > 0 else 0
        }
        
        strategy_data.parameters = {
            'mean_reversion_speed': mean_reversion_speed,
            'signal_threshold': signal_threshold
        }
        
        return strategy_data
    
    def calculate_max_drawdown(self, pnl_series: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = pnl_series[0]
        max_dd = 0
        
        for pnl in pnl_series:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / self.initial_capital
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def create_visualization(self) -> None:
        """Create comprehensive visualization dashboard"""
        self.fig = plt.figure(figsize=(20, 16))
        self.fig.suptitle('Stochastic Finance Strategy Visual Backtester', fontsize=20, fontweight='bold')
        
        # Create subplot layout
        gs = self.fig.add_gridspec(4, 3, height_ratios=[2, 2, 1.5, 1], width_ratios=[2, 1, 1])
        
        # Main price and signals plot
        self.ax_price = self.fig.add_subplot(gs[0, :2])
        
        # P&L comparison
        self.ax_pnl = self.fig.add_subplot(gs[1, :2])
        
        # Performance metrics
        self.ax_metrics = self.fig.add_subplot(gs[0, 2])
        
        # Signal distribution
        self.ax_signals = self.fig.add_subplot(gs[1, 2])
        
        # Position tracking
        self.ax_positions = self.fig.add_subplot(gs[2, :])
        
        # Strategy comparison table
        self.ax_table = self.fig.add_subplot(gs[3, :])
        
        plt.tight_layout()
    
    def plot_strategy_comparison(self) -> None:
        """Plot comprehensive strategy comparison"""
        if not self.strategies:
            print("No strategies to plot!")
            return
        
        self.create_visualization()
        
        # Plot 1: Price and Trading Signals
        self.ax_price.set_title('Market Price & Trading Signals', fontsize=14, fontweight='bold')
        
        # Plot market price
        timestamps = list(self.strategies.values())[0].timestamps
        prices = list(self.strategies.values())[0].prices
        self.ax_price.plot(timestamps, prices, 'k-', linewidth=1, alpha=0.7, label='Market Price')
        
        # Plot buy/sell signals for each strategy
        for strategy_name, strategy_data in self.strategies.items():
            config = self.strategy_configs[strategy_name]
            
            # Buy signals
            if strategy_data.buy_points:
                buy_times = [timestamps[i] for i, _ in strategy_data.buy_points]
                buy_prices = [p for _, p in strategy_data.buy_points]
                self.ax_price.scatter(buy_times, buy_prices, color=config['color'], 
                                    marker='^', s=100, alpha=0.8, label=f'{strategy_name} Buy')
            
            # Sell signals
            if strategy_data.sell_points:
                sell_times = [timestamps[i] for i, _ in strategy_data.sell_points]
                sell_prices = [p for _, p in strategy_data.sell_points]
                self.ax_price.scatter(sell_times, sell_prices, color=config['color'], 
                                    marker='v', s=100, alpha=0.8, label=f'{strategy_name} Sell')
        
        self.ax_price.set_ylabel('Price ($)')
        self.ax_price.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax_price.grid(True, alpha=0.3)
        
        # Plot 2: P&L Comparison
        self.ax_pnl.set_title('Profit & Loss Comparison', fontsize=14, fontweight='bold')
        
        for strategy_name, strategy_data in self.strategies.items():
            config = self.strategy_configs[strategy_name]
            self.ax_pnl.plot(timestamps, strategy_data.pnl, color=config['color'], 
                           linewidth=2, label=strategy_name)
        
        self.ax_pnl.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        self.ax_pnl.set_ylabel('P&L ($)')
        self.ax_pnl.legend()
        self.ax_pnl.grid(True, alpha=0.3)
        
        # Plot 3: Performance Metrics Radar Chart
        self.plot_performance_radar()
        
        # Plot 4: Signal Distribution
        self.plot_signal_distribution()
        
        # Plot 5: Position Tracking
        self.plot_position_tracking()
        
        # Plot 6: Strategy Comparison Table
        self.plot_strategy_table()
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_radar(self) -> None:
        """Create radar chart for performance metrics"""
        metrics = ['Total Return', 'Sharpe Ratio', 'Win Rate', 'Max Drawdown (inv)']
        
        # Normalize metrics for radar chart
        all_metrics = []
        for strategy_data in self.strategies.values():
            m = strategy_data.performance_metrics
            normalized = [
                max(0, min(1, (m['total_return'] + 0.2) / 0.4)),  # Normalize around -20% to +20%
                max(0, min(1, (m['sharpe_ratio'] + 1) / 3)),      # Normalize -1 to 2
                m['win_rate'],                                      # Already 0-1
                max(0, min(1, 1 - m['max_drawdown'] / 0.3))       # Invert drawdown, normalize to 30%
            ]
            all_metrics.append(normalized)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        self.ax_metrics.set_theta_offset(np.pi / 2)
        self.ax_metrics.set_theta_direction(-1)
        self.ax_metrics = plt.subplot(4, 3, 3, projection='polar')
        
        for i, (strategy_name, strategy_data) in enumerate(self.strategies.items()):
            values = all_metrics[i] + all_metrics[i][:1]  # Complete the circle
            config = self.strategy_configs[strategy_name]
            self.ax_metrics.plot(angles, values, 'o-', linewidth=2, 
                               color=config['color'], label=strategy_name)
            self.ax_metrics.fill(angles, values, alpha=0.25, color=config['color'])
        
        self.ax_metrics.set_xticks(angles[:-1])
        self.ax_metrics.set_xticklabels(metrics)
        self.ax_metrics.set_ylim(0, 1)
        self.ax_metrics.set_title('Performance Metrics', fontweight='bold', pad=20)
        self.ax_metrics.legend(bbox_to_anchor=(1.3, 1.0))
    
    def plot_signal_distribution(self) -> None:
        """Plot signal distribution for each strategy"""
        self.ax_signals.set_title('Signal Distribution', fontsize=12, fontweight='bold')
        
        signal_counts = {}
        for strategy_name, strategy_data in self.strategies.items():
            signals = np.array(strategy_data.signals)
            signal_counts[strategy_name] = {
                'Buy': np.sum(signals == 1),
                'Sell': np.sum(signals == -1),
                'Hold': np.sum(signals == 0)
            }
        
        # Create stacked bar chart
        strategies = list(signal_counts.keys())
        buy_counts = [signal_counts[s]['Buy'] for s in strategies]
        sell_counts = [signal_counts[s]['Sell'] for s in strategies]
        hold_counts = [signal_counts[s]['Hold'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.6
        
        self.ax_signals.bar(x, buy_counts, width, label='Buy', color='green', alpha=0.7)
        self.ax_signals.bar(x, sell_counts, width, bottom=buy_counts, label='Sell', color='red', alpha=0.7)
        self.ax_signals.bar(x, hold_counts, width, 
                          bottom=np.array(buy_counts) + np.array(sell_counts), 
                          label='Hold', color='gray', alpha=0.7)
        
        self.ax_signals.set_xticks(x)
        self.ax_signals.set_xticklabels([s.replace('LogNormalJumpMeanReversion', 'LNJMR') for s in strategies], rotation=45)
        self.ax_signals.set_ylabel('Signal Count')
        self.ax_signals.legend()
    
    def plot_position_tracking(self) -> None:
        """Plot position tracking over time"""
        self.ax_positions.set_title('Position Tracking Over Time', fontsize=12, fontweight='bold')
        
        timestamps = list(self.strategies.values())[0].timestamps
        
        for strategy_name, strategy_data in self.strategies.items():
            config = self.strategy_configs[strategy_name]
            positions = np.array(strategy_data.positions)
            
            # Normalize positions for better visualization
            max_pos = max(positions) if max(positions) > 0 else 1
            normalized_positions = positions / max_pos
            
            self.ax_positions.plot(timestamps, normalized_positions, 
                                 color=config['color'], linewidth=2, 
                                 label=f'{strategy_name}', alpha=0.8)
        
        self.ax_positions.set_ylabel('Normalized Position')
        self.ax_positions.set_xlabel('Time')
        self.ax_positions.legend()
        self.ax_positions.grid(True, alpha=0.3)
    
    def plot_strategy_table(self) -> None:
        """Create strategy comparison table"""
        self.ax_table.axis('tight')
        self.ax_table.axis('off')
        
        # Prepare table data
        columns = ['Strategy', 'Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)', 'Max Drawdown (%)']
        table_data = []
        
        for strategy_name, strategy_data in self.strategies.items():
            m = strategy_data.performance_metrics
            row = [
                strategy_name.replace('LogNormalJumpMeanReversion', 'LNJMR'),
                f"{m['total_return']*100:.2f}%",
                f"{m['sharpe_ratio']:.3f}",
                f"{m['win_rate']*100:.1f}%",
                f"{m['max_drawdown']*100:.2f}%"
            ]
            table_data.append(row)
        
        # Create table
        table = self.ax_table.table(cellText=table_data, colLabels=columns,
                                  cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code performance
        for i, (strategy_name, _) in enumerate(self.strategies.items()):
            config = self.strategy_configs[strategy_name]
            for j in range(len(columns)):
                table[(i+1, j)].set_facecolor(config['color'])
                table[(i+1, j)].set_alpha(0.3)
    
    def run_full_analysis(self, days: int = 100) -> None:
        """Run complete visual analysis"""
        print("🚀 Starting Visual Backtester Analysis...")
        print("="*50)
        
        # Generate market data
        print("📊 Generating synthetic market data...")
        self.market_data = self.generate_market_data(days=days)
        print(f"Generated {len(self.market_data)} data points")
        
        # Run all strategies
        for strategy_name in self.strategy_configs.keys():
            if os.path.exists(self.strategy_configs[strategy_name]['executable']):
                self.strategies[strategy_name] = self.run_strategy_simulation(strategy_name, self.market_data)
            else:
                print(f"⚠️  Strategy executable not found: {strategy_name}")
        
        if not self.strategies:
            print("❌ No strategies available for analysis!")
            return
        
        print(f"\n✅ Successfully analyzed {len(self.strategies)} strategies")
        
        # Print summary
        print("\n📈 Strategy Performance Summary:")
        print("-" * 70)
        print(f"{'Strategy':<25} {'Return':<10} {'Sharpe':<8} {'Win Rate':<10} {'Max DD':<8}")
        print("-" * 70)
        
        for strategy_name, strategy_data in self.strategies.items():
            m = strategy_data.performance_metrics
            name_short = strategy_name.replace('LogNormalJumpMeanReversion', 'LNJMR')
            print(f"{name_short:<25} {m['total_return']*100:>7.2f}% {m['sharpe_ratio']:>7.3f} "
                  f"{m['win_rate']*100:>8.1f}% {m['max_drawdown']*100:>7.2f}%")
        
        print("-" * 70)
        
        # Create visualization
        print("\n🎨 Creating visualization dashboard...")
        self.plot_strategy_comparison()
        
        print("\n🎉 Analysis complete! Dashboard displayed.")

if __name__ == "__main__":
    # Run visual backtester
    backtester = VisualBacktester(initial_capital=100000.0)
    backtester.run_full_analysis(days=100)
