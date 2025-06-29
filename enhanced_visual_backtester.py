"""
Enhanced Visual Backtester for Stochastic Finance Strategies
Interactive visualization and analysis with improved plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
plt.style.use('default')
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

class EnhancedVisualBacktester:
    """Enhanced visual backtesting framework"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.strategies = {}
        self.market_data = None
        
        # Strategy configurations with enhanced styling
        self.strategy_configs = {
            'HestonVolSurface': {
                'executable': 'strategies/HestonVolSurface/build/hestonvolsurface_strategy.exe',
                'color': '#FF6B6B',
                'linestyle': '-',
                'description': 'Heston Volatility Surface Model'
            },
            'JumpDiffusion': {
                'executable': 'strategies/JumpDiffusion/build/jumpdiffusion_strategy.exe',
                'color': '#4ECDC4',
                'linestyle': '--',
                'description': 'Jump Diffusion Process Model'
            },
            'LogNormalJumpMeanReversion': {
                'executable': 'strategies/LogNormalJumpMeanReversion/build/lognormaljumpmeanreversion_strategy.exe',
                'color': '#45B7D1',
                'linestyle': '-.',
                'description': 'Log-Normal Jump Mean Reversion Model'
            },
            'OrnsteinUhlenbeck': {
                'executable': 'strategies/OrnsteinUhlenbeck/build/ornsteinuhlenbeck_strategy.exe',
                'color': '#96CEB4',
                'linestyle': ':',
                'description': 'Ornstein-Uhlenbeck Process Model'
            }
        }
    
    def generate_market_data(self, days: int = 252, dt: float = 1/252) -> pd.DataFrame:
        """Generate synthetic market data with realistic patterns"""
        np.random.seed(42)
        
        # Enhanced market simulation with regime changes
        S0 = 100.0
        mu = 0.08  # Annual return
        sigma = 0.25  # Annual volatility
        
        n_steps = int(days / dt)
        t = np.linspace(0, days/252, n_steps)
        
        # Add market regimes (bull/bear periods)
        regime_changes = np.random.poisson(0.5, n_steps)  # Rare regime changes
        current_regime = 1  # 1 = bull, -1 = bear
        
        # Price evolution with regime-dependent parameters
        prices = np.zeros(n_steps)
        prices[0] = S0
        
        for i in range(1, n_steps):
            # Check for regime change
            if regime_changes[i] > 0:
                current_regime *= -1
            
            # Regime-dependent parameters
            regime_mu = mu * current_regime * 0.5
            regime_sigma = sigma * (1.5 if current_regime == -1 else 1.0)
            
            # Add jumps during volatile periods
            jump_prob = 0.001 * (2 if current_regime == -1 else 1)
            jump = np.random.exponential(0.05) * np.random.choice([-1, 1]) if np.random.random() < jump_prob else 0
            
            # Price update
            dW = np.random.normal(0, np.sqrt(dt))
            prices[i] = prices[i-1] * np.exp((regime_mu - 0.5*regime_sigma**2)*dt + regime_sigma*dW + jump)
        
        # Create realistic bid-ask spreads and volume
        spread_bps = 5  # 5 basis points
        volume_base = 1000000
        
        market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_steps, freq='H'),
            'price': prices,
            'bid': prices * (1 - spread_bps/10000),
            'ask': prices * (1 + spread_bps/10000),
            'volume': np.random.lognormal(np.log(volume_base), 0.5, n_steps).astype(int)
        })
        
        return market_data
    
    def run_strategy_simulation(self, strategy_name: str, market_data: pd.DataFrame) -> StrategyData:
        """Enhanced strategy simulation with realistic trading logic"""
        print(f"📊 Simulating {strategy_name} strategy...")
        
        strategy_data = StrategyData(strategy_name)
        config = self.strategy_configs[strategy_name]
        
        # Initialize portfolio
        position = 0  # Number of shares
        cash = self.initial_capital
        transaction_cost = 0.001  # 0.1% per trade
        
        # Strategy-specific parameters
        lookback_periods = {'HestonVolSurface': 20, 'JumpDiffusion': 15, 
                           'LogNormalJumpMeanReversion': 30, 'OrnsteinUhlenbeck': 25}
        lookback = lookback_periods.get(strategy_name, 20)
        
        # Seed for reproducible but different results per strategy
        np.random.seed(hash(strategy_name) % 2**32)
        
        for i, row in market_data.iterrows():
            price = row['price']
            
            if i >= lookback:
                # Calculate technical indicators
                recent_prices = market_data['price'].iloc[i-lookback:i].values
                sma = np.mean(recent_prices)
                std_dev = np.std(recent_prices)
                rsi = self.calculate_rsi(recent_prices)
                
                # Strategy-specific signal generation
                signal = self.generate_signal(strategy_name, price, sma, std_dev, rsi, recent_prices)
                
                # Portfolio management
                target_position = self.calculate_position_size(signal, cash, price, position)
                
                # Execute trades
                if target_position != position:
                    trade_size = target_position - position
                    trade_value = abs(trade_size * price)
                    cost = trade_value * transaction_cost
                    
                    if trade_size > 0 and cash >= trade_value + cost:  # Buy
                        position = target_position
                        cash -= trade_value + cost
                        strategy_data.buy_points.append((i, price))
                    elif trade_size < 0:  # Sell
                        position = target_position
                        cash += trade_value - cost
                        strategy_data.sell_points.append((i, price))
            else:
                signal = 0
            
            # Calculate portfolio value
            portfolio_value = cash + position * price
            
            # Store data
            strategy_data.prices.append(price)
            strategy_data.signals.append(signal)
            strategy_data.pnl.append(portfolio_value - self.initial_capital)
            strategy_data.positions.append(position)
            strategy_data.timestamps.append(row['timestamp'])
        
        # Calculate comprehensive performance metrics
        strategy_data.performance_metrics = self.calculate_performance_metrics(strategy_data)
        
        return strategy_data
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, strategy_name: str, price: float, sma: float, 
                       std_dev: float, rsi: float, recent_prices: np.ndarray) -> int:
        """Generate trading signals based on strategy type"""
        
        if strategy_name == 'HestonVolSurface':
            # Volatility-based mean reversion
            vol_threshold = std_dev * 1.5
            price_deviation = abs(price - sma)
            if price < sma - vol_threshold and rsi < 30:
                return 1  # Strong buy
            elif price > sma + vol_threshold and rsi > 70:
                return -1  # Strong sell
            
        elif strategy_name == 'JumpDiffusion':
            # Jump detection and momentum
            if len(recent_prices) >= 2:
                jump = abs(price - recent_prices[-1]) / recent_prices[-1]
                if jump > 0.02:  # Significant price jump
                    return -1 if price > sma else 1
                elif price < sma * 0.98 and rsi < 40:
                    return 1
                elif price > sma * 1.02 and rsi > 60:
                    return -1
        
        elif strategy_name == 'LogNormalJumpMeanReversion':
            # Log-normal mean reversion
            log_price = np.log(price)
            log_sma = np.log(sma) if sma > 0 else 0
            log_deviation = log_price - log_sma
            
            if log_deviation < -0.05 and rsi < 35:
                return 1
            elif log_deviation > 0.05 and rsi > 65:
                return -1
        
        else:  # OrnsteinUhlenbeck
            # Strong mean reversion with oversold/overbought
            z_score = (price - sma) / std_dev if std_dev > 0 else 0
            if z_score < -1.5 and rsi < 25:
                return 1
            elif z_score > 1.5 and rsi > 75:
                return -1
        
        return 0  # Hold
    
    def calculate_position_size(self, signal: int, cash: float, price: float, current_position: int) -> int:
        """Calculate optimal position size"""
        max_position_value = self.initial_capital * 0.95  # Max 95% invested
        max_shares = int(max_position_value / price)
        
        if signal == 1:  # Buy signal
            return min(max_shares, int(cash * 0.3 / price))  # Use 30% of available cash
        elif signal == -1:  # Sell signal
            return 0  # Close position
        else:
            return current_position  # Hold current position
    
    def calculate_performance_metrics(self, strategy_data: StrategyData) -> Dict:
        """Calculate comprehensive performance metrics"""
        pnl_series = np.array(strategy_data.pnl)
        returns = np.diff(pnl_series) / self.initial_capital
        
        # Basic metrics
        total_return = pnl_series[-1] / self.initial_capital
        
        # Risk-adjusted returns
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)) if np.std(returns) > 0 else 0
        
        # Drawdown analysis
        cumulative = pnl_series + self.initial_capital
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'volatility': np.std(returns) * np.sqrt(252 * 24),
            'num_trades': len(strategy_data.buy_points) + len(strategy_data.sell_points)
        }
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive visualization dashboard"""
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('🚀 Stochastic Finance Strategy Analysis Dashboard', 
                    fontsize=18, fontweight='bold', color='darkblue')
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1.2], hspace=0.3, wspace=0.3)
        
        # 1. Price Chart with Signals
        ax1 = fig.add_subplot(gs[0, :3])
        self.plot_price_signals(ax1)
        
        # 2. Performance Metrics Summary
        ax2 = fig.add_subplot(gs[0, 3])
        self.plot_performance_summary(ax2)
        
        # 3. P&L Evolution
        ax3 = fig.add_subplot(gs[1, :2])
        self.plot_pnl_evolution(ax3)
        
        # 4. Risk Analysis
        ax4 = fig.add_subplot(gs[1, 2])
        self.plot_risk_analysis(ax4)
        
        # 5. Trading Statistics
        ax5 = fig.add_subplot(gs[1, 3])
        self.plot_trading_stats(ax5)
        
        # 6. Strategy Comparison Table
        ax6 = fig.add_subplot(gs[2, :])
        self.plot_comparison_table(ax6)
        
        plt.tight_layout()
        return fig
    
    def plot_price_signals(self, ax):
        """Plot price chart with trading signals"""
        if not self.strategies:
            return
        
        # Plot market price
        timestamps = list(self.strategies.values())[0].timestamps
        prices = list(self.strategies.values())[0].prices
        
        ax.plot(timestamps, prices, 'k-', linewidth=1.5, alpha=0.8, label='Market Price')
        
        # Plot signals for each strategy
        for strategy_name, strategy_data in self.strategies.items():
            config = self.strategy_configs[strategy_name]
            
            # Buy signals
            if strategy_data.buy_points:
                buy_idx, buy_prices = zip(*strategy_data.buy_points)
                buy_times = [timestamps[i] for i in buy_idx]
                ax.scatter(buy_times, buy_prices, color=config['color'], 
                          marker='^', s=60, alpha=0.8, label=f'{strategy_name} Buy')
            
            # Sell signals
            if strategy_data.sell_points:
                sell_idx, sell_prices = zip(*strategy_data.sell_points)
                sell_times = [timestamps[i] for i in sell_idx]
                ax.scatter(sell_times, sell_prices, color=config['color'], 
                          marker='v', s=60, alpha=0.8, label=f'{strategy_name} Sell')
        
        ax.set_title('📈 Market Price & Trading Signals', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    def plot_performance_summary(self, ax):
        """Plot performance metrics summary"""
        strategies = list(self.strategies.keys())
        returns = [self.strategies[s].performance_metrics['total_return'] * 100 for s in strategies]
        sharpes = [self.strategies[s].performance_metrics['sharpe_ratio'] for s in strategies]
        
        # Create bar chart
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, returns, width, label='Total Return (%)', 
                      color=[self.strategy_configs[s]['color'] for s in strategies], alpha=0.7)
        
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, sharpes, width, label='Sharpe Ratio', 
                       color=[self.strategy_configs[s]['color'] for s in strategies], alpha=0.5)
        
        ax.set_title('📊 Performance Summary', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('LogNormalJumpMeanReversion', 'LNJMR')[:8] for s in strategies], 
                          rotation=45, fontsize=9)
        ax.set_ylabel('Return (%)', fontsize=10)
        ax2.set_ylabel('Sharpe Ratio', fontsize=10)
        
        # Add value labels on bars
        for bar, val in zip(bars1, returns):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    def plot_pnl_evolution(self, ax):
        """Plot P&L evolution over time"""
        timestamps = list(self.strategies.values())[0].timestamps
        
        for strategy_name, strategy_data in self.strategies.items():
            config = self.strategy_configs[strategy_name]
            ax.plot(timestamps, strategy_data.pnl, color=config['color'], 
                   linewidth=2.5, linestyle=config['linestyle'], 
                   label=strategy_name.replace('LogNormalJumpMeanReversion', 'LNJMR'))
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('💰 Profit & Loss Evolution', fontsize=12, fontweight='bold')
        ax.set_ylabel('P&L ($)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show values in K
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    def plot_risk_analysis(self, ax):
        """Plot risk analysis metrics"""
        strategies = list(self.strategies.keys())
        max_dd = [self.strategies[s].performance_metrics['max_drawdown'] * 100 for s in strategies]
        volatility = [self.strategies[s].performance_metrics['volatility'] * 100 for s in strategies]
        
        # Scatter plot: Risk vs Return
        returns = [self.strategies[s].performance_metrics['total_return'] * 100 for s in strategies]
        
        for i, strategy in enumerate(strategies):
            config = self.strategy_configs[strategy]
            ax.scatter(volatility[i], returns[i], color=config['color'], s=150, alpha=0.7)
            ax.annotate(strategy.replace('LogNormalJumpMeanReversion', 'LNJMR')[:8], 
                       (volatility[i], returns[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_title('⚖️ Risk vs Return', fontsize=12, fontweight='bold')
        ax.set_xlabel('Volatility (%)', fontsize=10)
        ax.set_ylabel('Return (%)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def plot_trading_stats(self, ax):
        """Plot trading statistics"""
        strategies = list(self.strategies.keys())
        win_rates = [self.strategies[s].performance_metrics['win_rate'] * 100 for s in strategies]
        num_trades = [self.strategies[s].performance_metrics['num_trades'] for s in strategies]
        
        # Bubble chart: Win Rate vs Number of Trades
        returns = [self.strategies[s].performance_metrics['total_return'] * 100 for s in strategies]
        
        for i, strategy in enumerate(strategies):
            config = self.strategy_configs[strategy]
            # Size represents return magnitude
            size = max(50, abs(returns[i]) * 5)
            ax.scatter(num_trades[i], win_rates[i], color=config['color'], 
                      s=size, alpha=0.7, edgecolors='black', linewidth=1)
            ax.annotate(strategy.replace('LogNormalJumpMeanReversion', 'LNJMR')[:8], 
                       (num_trades[i], win_rates[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_title('🎯 Trading Statistics', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Trades', fontsize=10)
        ax.set_ylabel('Win Rate (%)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def plot_comparison_table(self, ax):
        """Create detailed comparison table"""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        columns = ['Strategy', 'Return (%)', 'Sharpe', 'Max DD (%)', 'Win Rate (%)', 'Trades', 'Calmar']
        table_data = []
        
        for strategy_name, strategy_data in self.strategies.items():
            m = strategy_data.performance_metrics
            name_short = strategy_name.replace('LogNormalJumpMeanReversion', 'LNJMR')
            row = [
                name_short,
                f"{m['total_return']*100:.1f}%",
                f"{m['sharpe_ratio']:.2f}",
                f"{m['max_drawdown']*100:.1f}%",
                f"{m['win_rate']*100:.1f}%",
                f"{m['num_trades']}",
                f"{m['calmar_ratio']:.2f}"
            ]
            table_data.append(row)
        
        # Create table with enhanced formatting
        table = ax.table(cellText=table_data, colLabels=columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Enhanced styling
        for i, (strategy_name, _) in enumerate(self.strategies.items()):
            config = self.strategy_configs[strategy_name]
            for j in range(len(columns)):
                cell = table[(i+1, j)]
                cell.set_facecolor(config['color'])
                cell.set_alpha(0.2)
                cell.set_text_props(weight='bold')
        
        # Header styling
        for j in range(len(columns)):
            cell = table[(0, j)]
            cell.set_facecolor('lightgray')
            cell.set_text_props(weight='bold', color='darkblue')
    
    def run_enhanced_analysis(self, days: int = 100):
        """Run enhanced visual analysis"""
        print("🚀 Enhanced Visual Backtester Analysis")
        print("="*50)
        
        # Generate market data
        print("📊 Generating enhanced market data...")
        self.market_data = self.generate_market_data(days=days)
        print(f"✅ Generated {len(self.market_data)} data points over {days} days")
        
        # Run strategies
        available_strategies = 0
        for strategy_name in self.strategy_configs.keys():
            if os.path.exists(self.strategy_configs[strategy_name]['executable']):
                self.strategies[strategy_name] = self.run_strategy_simulation(strategy_name, self.market_data)
                available_strategies += 1
            else:
                print(f"⚠️  Strategy not available: {strategy_name}")
        
        if available_strategies == 0:
            print("❌ No strategies available! Build them first: .\\build_strategies.ps1")
            return
        
        print(f"✅ Successfully analyzed {available_strategies} strategies")
        
        # Performance summary
        print("\n📈 STRATEGY PERFORMANCE DASHBOARD")
        print("=" * 80)
        print(f"{'Strategy':<20} {'Return':<8} {'Sharpe':<7} {'Max DD':<8} {'Win%':<6} {'Trades':<7} {'Calmar':<7}")
        print("-" * 80)
        
        best_return = max(self.strategies.values(), key=lambda x: x.performance_metrics['total_return'])
        best_sharpe = max(self.strategies.values(), key=lambda x: x.performance_metrics['sharpe_ratio'])
        
        for strategy_name, strategy_data in self.strategies.items():
            m = strategy_data.performance_metrics
            name_short = strategy_name.replace('LogNormalJumpMeanReversion', 'LNJMR')
            
            # Highlight best performers
            return_str = f"{m['total_return']*100:>6.1f}%"
            sharpe_str = f"{m['sharpe_ratio']:>6.2f}"
            
            if strategy_data == best_return:
                return_str = f"👑{return_str}"
            if strategy_data == best_sharpe:
                sharpe_str = f"⭐{sharpe_str}"
            
            print(f"{name_short:<20} {return_str:<8} {sharpe_str:<7} "
                  f"{m['max_drawdown']*100:>6.1f}% {m['win_rate']*100:>5.1f}% "
                  f"{m['num_trades']:>6} {m['calmar_ratio']:>6.2f}")
        
        print("-" * 80)
        
        # Create and show dashboard
        print("\n🎨 Creating enhanced visualization dashboard...")
        fig = self.create_comprehensive_dashboard()
        
        # Save the plot
        plt.savefig('strategy_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print("💾 Dashboard saved as 'strategy_analysis_dashboard.png'")
        
        plt.show()
        print("\n🎉 Enhanced analysis complete!")

if __name__ == "__main__":
    # Run enhanced backtester
    backtester = EnhancedVisualBacktester(initial_capital=100000.0)
    backtester.run_enhanced_analysis(days=100)
