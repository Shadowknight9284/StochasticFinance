#pragma once

#include <atomic>
#include <array>
#include <memory>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

// Lock-free ring buffer for price data
template<size_t N>
class RingBuffer {
private:
    std::array<double, N> buffer;
    std::atomic<size_t> head{0};
    std::atomic<size_t> tail{0};
    
public:
    [[gnu::always_inline]]
    void push(double value) noexcept {
        size_t next_head = (head.load() + 1) % N;
        buffer[head.load()] = value;
        head.store(next_head);
    }
    
    [[gnu::always_inline]]
    double back() const noexcept {
        return buffer[(head.load() - 1 + N) % N];
    }
    
    [[gnu::always_inline]]
    double operator[](size_t index) const noexcept {
        return buffer[(head.load() - 1 - index + N) % N];
    }
};

// Order structure for ultra-low latency
struct Order {
    enum class Side : uint8_t { BUY = 1, SELL = 2, HOLD = 0 };
    
    Side side;
    double price;
    uint32_t quantity;
    uint64_t timestamp_ns;
    
    Order() noexcept : side(Side::HOLD), price(0.0), quantity(0), timestamp_ns(0) {}
    Order(Side s, double p, uint32_t q) noexcept 
        : side(s), price(p), quantity(q), 
          timestamp_ns(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {}
};

// Market data structure
struct MarketData {
    double bid;
    double ask;
    double last_price;
    uint64_t volume;
    uint64_t timestamp_ns;
    
    [[gnu::always_inline]]
    double mid_price() const noexcept { return (bid + ask) * 0.5; }
    [[gnu::always_inline]]
    double spread() const noexcept { return ask - bid; }
};

// Main strategy class - Heston Stochastic Volatility Model
template <typename MarketDataType = MarketData, size_t BufferSize = 1000>
class HestonVolSurfaceStrategy {
private:
    RingBuffer<BufferSize> price_series;
    RingBuffer<BufferSize> volatility_series;
    Eigen::VectorXd params;
    std::atomic<double> current_volatility;
    std::atomic<double> spot_price;
    
    // Heston model parameters
    std::atomic<double> kappa;      // Mean reversion speed of volatility
    std::atomic<double> theta;      // Long-term volatility
    std::atomic<double> sigma_v;    // Vol of vol
    std::atomic<double> rho;        // Correlation between price and vol
    std::atomic<double> risk_free_rate;
    
    // Performance counters
    mutable std::atomic<uint64_t> signal_count{0};
    mutable std::atomic<uint64_t> total_latency_ns{0};
    
    // Random number generation (thread-local for performance)
    thread_local static std::mt19937 rng;
    thread_local static std::normal_distribution<double> normal_dist;
    
public:
    HestonVolSurfaceStrategy() noexcept 
        : params(Eigen::VectorXd::Zero(5)), current_volatility(0.2), spot_price(100.0),
          kappa(2.0), theta(0.04), sigma_v(0.3), rho(-0.7), risk_free_rate(0.05) {}
    
    // Main signal generation - Heston Model dynamics
    [[gnu::always_inline, gnu::hot]]
    Order generate_order(MarketDataType&& data) noexcept {
        auto start = std::chrono::high_resolution_clock::now();
        
        const double dt = 1.0 / 252.0 / 24.0;  // Intraday time step (hourly)
        const double current_price = data.mid_price();
        
        // Update price series
        price_series.push(current_price);
        
        // Get current state
        double S = spot_price.load();
        double v = current_volatility.load();
        
        // Generate correlated random numbers for Heston dynamics
        double z1 = normal_dist(rng);
        double z2 = rho.load() * z1 + std::sqrt(1.0 - rho.load() * rho.load()) * normal_dist(rng);
        
        // Heston volatility dynamics: dv = kappa*(theta - v)*dt + sigma_v*sqrt(v)*dW2
        double dv = kappa.load() * (theta.load() - v) * dt + 
                   sigma_v.load() * std::sqrt(std::max(v, 0.0)) * std::sqrt(dt) * z2;
        double new_v = std::max(v + dv, 0.001);  // Floor volatility to prevent negative values
        
        // Heston price dynamics: dS = r*S*dt + sqrt(v)*S*dW1
        double drift = risk_free_rate.load() * S * dt;
        double diffusion = std::sqrt(std::max(v, 0.0)) * S * std::sqrt(dt) * z1;
        double new_S = S * std::exp((risk_free_rate.load() - 0.5 * v) * dt + 
                                   std::sqrt(v) * std::sqrt(dt) * z1);
        
        // Update state
        current_volatility.store(new_v);
        spot_price.store(new_S);
        volatility_series.push(new_v);
        
        // Trading signal based on volatility surface arbitrage
        // Signal: Buy when realized vol < implied vol, Sell when realized vol > implied vol
        double implied_vol = estimate_implied_volatility(current_price);
        double realized_vol = calculate_realized_volatility();
        
        Order order;
        double vol_spread = realized_vol - implied_vol;
        
        if (vol_spread < -0.02) {  // Realized vol significantly below implied
            order = Order(Order::Side::BUY, data.ask, 100);
        } else if (vol_spread > 0.02) {  // Realized vol significantly above implied
            order = Order(Order::Side::SELL, data.bid, 100);
        }
        
        // Performance tracking
        auto end = std::chrono::high_resolution_clock::now();
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        signal_count.fetch_add(1);
        total_latency_ns.fetch_add(latency_ns);
        
        return order;
    }
    
    // Estimate implied volatility from market data
    [[gnu::always_inline]]
    double estimate_implied_volatility(double current_price) const noexcept {
        // Simplified Black-Scholes implied vol estimation
        // In practice, this would use option prices
        return current_volatility.load() * 1.1;  // Assume 10% premium
    }
    
    // Calculate realized volatility from recent price history
    [[gnu::always_inline]]
    double calculate_realized_volatility() const noexcept {
        const size_t lookback = 20;
        double sum_squared_returns = 0.0;
        
        for (size_t i = 1; i < std::min(lookback, BufferSize); ++i) {
            double return_val = std::log(price_series[i-1] / price_series[i]);
            sum_squared_returns += return_val * return_val;
        }
        
        return std::sqrt(sum_squared_returns * 252.0 / lookback);  // Annualized vol
    }
    
    // Calibration using Maximum Likelihood Estimation for Heston model
    void calibrate(const Eigen::VectorXd& prices) {
        if (prices.size() < 10) return;
        
        // Calculate log returns
        Eigen::VectorXd log_returns(prices.size() - 1);
        for (int i = 0; i < log_returns.size(); ++i) {
            log_returns(i) = std::log(prices(i + 1) / prices(i));
        }
        
        // Estimate parameters using method of moments
        double mean_return = log_returns.mean();
        double var_return = ((log_returns.array() - mean_return).square()).mean();
        
        // Estimate Heston parameters
        // kappa: mean reversion speed of volatility
        double sample_kappa = 2.0;  // Typical value
        
        // theta: long-term volatility level
        double sample_theta = var_return * 252.0;  // Annualized variance
        
        // sigma_v: volatility of volatility
        double sample_sigma_v = 0.3;  // Typical value
        
        // rho: correlation (estimated from price-volatility relationship)
        double sample_rho = -0.7;  // Typical negative correlation
        
        // Store parameters
        kappa.store(sample_kappa);
        theta.store(sample_theta);
        sigma_v.store(sample_sigma_v);
        rho.store(sample_rho);
        current_volatility.store(std::sqrt(sample_theta));
        spot_price.store(prices(prices.size() - 1));
        
        params(0) = sample_kappa;
        params(1) = sample_theta;
        params(2) = sample_sigma_v;
        params(3) = sample_rho;
        params(4) = risk_free_rate.load();
    }
    
    // Performance metrics
    [[gnu::always_inline]]
    double average_latency_us() const noexcept {
        uint64_t count = signal_count.load();
        if (count == 0) return 0.0;
        return static_cast<double>(total_latency_ns.load()) / (count * 1000.0);
    }
    
    [[gnu::always_inline]]
    bool meets_latency_constraint() const noexcept {
        return average_latency_us() < 35.0;
    }
    
    // Getters for Heston parameters
    [[gnu::always_inline]]
    double get_kappa() const noexcept { return kappa.load(); }
    
    [[gnu::always_inline]]
    double get_theta() const noexcept { return theta.load(); }
    
    [[gnu::always_inline]]
    double get_sigma_v() const noexcept { return sigma_v.load(); }
    
    [[gnu::always_inline]]
    double get_rho() const noexcept { return rho.load(); }
    
    [[gnu::always_inline]]
    double get_current_volatility() const noexcept { return current_volatility.load(); }
    
    [[gnu::always_inline]]
    void set_heston_params(double k, double t, double sv, double r) noexcept {
        kappa.store(k);
        theta.store(t);
        sigma_v.store(sv);
        rho.store(r);
    }
    
    [[gnu::always_inline]]
    const Eigen::VectorXd& get_params() const noexcept { return params; }
};

// Thread-local definitions
template <typename MarketDataType, size_t BufferSize>
thread_local std::mt19937 HestonVolSurfaceStrategy<MarketDataType, BufferSize>::rng{std::random_device{}()};

template <typename MarketDataType, size_t BufferSize>
thread_local std::normal_distribution<double> HestonVolSurfaceStrategy<MarketDataType, BufferSize>::normal_dist{0.0, 1.0};
