#include "../jumpdiffusion.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>

// Backtest framework for NASDAQ ITCH data
class BacktestEngine {
private:
    std::vector<MarketData> market_data;
    std::vector<Order> orders;
    double initial_capital;
    double current_capital;
    double position;
    std::vector<double> pnl_series;
    
public:
    BacktestEngine(double capital = 100000.0) 
        : initial_capital(capital), current_capital(capital), position(0.0) {}
    
    void load_market_data(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        
        // Skip header
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string token;
            
            MarketData data;
            std::getline(iss, token, ','); // timestamp
            std::getline(iss, token, ','); data.bid = std::stod(token);
            std::getline(iss, token, ','); data.ask = std::stod(token);
            std::getline(iss, token, ','); data.last_price = std::stod(token);
            std::getline(iss, token, ','); data.volume = std::stoull(token);
            
            data.timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            market_data.push_back(data);
        }
    }
    
    template<typename StrategyType>
    void run_backtest(StrategyType& strategy) {
        orders.clear();
        pnl_series.clear();
        current_capital = initial_capital;
        position = 0.0;
        
        // Calibration period (first 20% of data)
        size_t calibration_size = market_data.size() / 5;
        std::vector<double> calibration_prices;
        
        for (size_t i = 0; i < calibration_size; ++i) {
            calibration_prices.push_back(market_data[i].mid_price());
        }
        
        Eigen::VectorXd prices = Eigen::Map<Eigen::VectorXd>(
            calibration_prices.data(), calibration_prices.size());
        strategy.calibrate(prices);
        
        // Trading period
        for (size_t i = calibration_size; i < market_data.size(); ++i) {
            MarketData data = market_data[i];
            Order order = strategy.generate_order(std::move(data));
            
            if (order.side != Order::Side::HOLD) {
                execute_order(order, data);
                orders.push_back(order);
            }
            
            // Calculate P&L
            double mark_to_market = position * data.mid_price();
            double total_value = current_capital + mark_to_market;
            pnl_series.push_back(total_value - initial_capital);
        }
    }
    
    void execute_order(const Order& order, const MarketData& data) {
        double price = (order.side == Order::Side::BUY) ? data.ask : data.bid;
        double quantity = (order.side == Order::Side::BUY) ? order.quantity : -order.quantity;
        
        // Transaction costs (5 bps)
        double transaction_cost = std::abs(quantity * price * 0.0005);
        
        current_capital -= quantity * price + transaction_cost;
        position += quantity;
    }
    
    // Performance metrics
    double calculate_sharpe_ratio() const {
        if (pnl_series.empty()) return 0.0;
        
        std::vector<double> returns;
        for (size_t i = 1; i < pnl_series.size(); ++i) {
            returns.push_back((pnl_series[i] - pnl_series[i-1]) / initial_capital);
        }
        
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        variance /= returns.size();
        
        return mean_return / std::sqrt(variance) * std::sqrt(252);  // Annualized
    }
    
    double calculate_max_drawdown() const {
        if (pnl_series.empty()) return 0.0;
        
        double peak = pnl_series[0];
        double max_dd = 0.0;
        
        for (double pnl : pnl_series) {
            if (pnl > peak) peak = pnl;
            double drawdown = (peak - pnl) / initial_capital;
            if (drawdown > max_dd) max_dd = drawdown;
        }
        
        return max_dd;
    }
    
    double calculate_calmar_ratio() const {
        double total_return = (pnl_series.back() / initial_capital);
        double max_dd = calculate_max_drawdown();
        return (max_dd > 0) ? total_return / max_dd : 0.0;
    }
    
    void print_results() const {
        std::cout << "\n=== Backtest Results ===" << std::endl;
        std::cout << "Total Orders: " << orders.size() << std::endl;
        std::cout << "Final P&L: $" << pnl_series.back() << std::endl;
        std::cout << "Total Return: " << (pnl_series.back() / initial_capital * 100) << "%" << std::endl;
        std::cout << "Sharpe Ratio: " << calculate_sharpe_ratio() << std::endl;
        std::cout << "Max Drawdown: " << (calculate_max_drawdown() * 100) << "%" << std::endl;
        std::cout << "Calmar Ratio: " << calculate_calmar_ratio() << std::endl;
        
        // Validation
        bool passes_validation = true;
        if (calculate_max_drawdown() > 0.15) {
            std::cout << "❌ FAILED: Max Drawdown > 15%" << std::endl;
            passes_validation = false;
        }
        if (calculate_calmar_ratio() < 2.0) {
            std::cout << "❌ FAILED: Calmar Ratio < 2.0" << std::endl;
            passes_validation = false;
        }
        if (calculate_sharpe_ratio() < 1.5) {
            std::cout << "❌ FAILED: Sharpe Ratio < 1.5" << std::endl;
            passes_validation = false;
        }
        
        if (passes_validation) {
            std::cout << "✅ PASSED: All performance thresholds met" << std::endl;
        }
    }
};

int main() {
    JumpDiffusionStrategy<> strategy;
    BacktestEngine engine;
    
    // Load market data
    // engine.load_market_data("nasdaq_itch_data.csv");
    
    // Generate synthetic data for demo
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> price_dist(100.0, 2.0);
    
    std::vector<MarketData> synthetic_data;
    for (int i = 0; i < 10000; ++i) {
        MarketData data;
        data.last_price = price_dist(gen);
        data.bid = data.last_price - 0.01;
        data.ask = data.last_price + 0.01;
        data.volume = 1000;
        data.timestamp_ns = i * 1000000;  // 1ms intervals
        synthetic_data.push_back(data);
    }
    
    // Run backtest
    engine.run_backtest(strategy);
    engine.print_results();
    
    return 0;
}
