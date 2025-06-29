#include "jumpdiffusion.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Example usage and testing
int main() {
    JumpDiffusionStrategy<> strategy;
    
    // Generate synthetic market data for testing
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> price_dist(100.0, 2.0);
    std::normal_distribution<double> spread_dist(0.01, 0.005);
    
    std::vector<double> prices;
    for (int i = 0; i < 1000; ++i) {
        prices.push_back(price_dist(gen));
    }
    
    // Calibrate strategy
    Eigen::VectorXd price_vector = Eigen::Map<Eigen::VectorXd>(prices.data(), prices.size());
    strategy.calibrate(price_vector);
    
    std::cout << "Jump Diffusion Strategy calibrated with parameters:" << std::endl;
    std::cout << "Lambda (jump intensity): " << strategy.get_lambda() << std::endl;
    std::cout << "Mu_j (mean jump size): " << strategy.get_mu_j() << std::endl;
    std::cout << "Sigma_j (jump volatility): " << strategy.get_sigma_j() << std::endl;
    std::cout << "Mu (drift): " << strategy.get_mu() << std::endl;
    std::cout << "Sigma (volatility): " << strategy.get_sigma() << std::endl;
    
    // Test signal generation
    int buy_signals = 0, sell_signals = 0, hold_signals = 0;
    
    for (int i = 0; i < 10000; ++i) {
        MarketData data;
        data.last_price = price_dist(gen);
        double spread = std::abs(spread_dist(gen));
        data.bid = data.last_price - spread/2;
        data.ask = data.last_price + spread/2;
        data.volume = 1000;
        data.timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        Order order = strategy.generate_order(std::move(data));
        
        switch (order.side) {
            case Order::Side::BUY: buy_signals++; break;
            case Order::Side::SELL: sell_signals++; break;
            case Order::Side::HOLD: hold_signals++; break;
        }
    }
    
    std::cout << "\nJump Diffusion Signal distribution:" << std::endl;
    std::cout << "Buy signals: " << buy_signals << std::endl;
    std::cout << "Sell signals: " << sell_signals << std::endl;
    std::cout << "Hold signals: " << hold_signals << std::endl;
    std::cout << "Total jumps detected: " << strategy.get_jump_count() << std::endl;
    
    std::cout << "\nPerformance metrics:" << std::endl;
    std::cout << "Average latency: " << strategy.average_latency_us() << " μs" << std::endl;
    std::cout << "Meets latency constraint: " << (strategy.meets_latency_constraint() ? "YES" : "NO") << std::endl;
    
    return 0;
}
