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

// Main strategy class - Merton Jump Diffusion Model
template <typename MarketDataType = MarketData, size_t BufferSize = 1000>
class JumpDiffusionStrategy {
private:
    RingBuffer<BufferSize> price_series;
    RingBuffer<BufferSize> jump_series;
    Eigen::VectorXd params;
    std::atomic<double> spot_price;
    std::atomic<double> volatility;
    
    // Jump diffusion parameters
    std::atomic<double> lambda;         // Jump intensity (jumps per unit time)
    std::atomic<double> mu_j;          // Mean jump size
    std::atomic<double> sigma_j;       // Jump size volatility
    std::atomic<double> mu;            // Drift rate
    std::atomic<double> sigma;         // Diffusion volatility
    std::atomic<double> risk_free_rate;
    
    // Performance counters
    mutable std::atomic<uint64_t> signal_count{0};
    mutable std::atomic<uint64_t> total_latency_ns{0};
    mutable std::atomic<uint64_t> jump_count{0};
    
    // Random number generation (thread-local for performance)
    thread_local static std::mt19937 rng;
    thread_local static std::normal_distribution<double> normal_dist;
    thread_local static std::poisson_distribution<int> poisson_dist;
    thread_local static std::exponential_distribution<double> exp_dist;
    
public:
    JumpDiffusionStrategy() noexcept 
        : params(Eigen::VectorXd::Zero(6)), spot_price(100.0), volatility(0.2),
          lambda(0.1), mu_j(-0.02), sigma_j(0.03), mu(0.05), sigma(0.2), risk_free_rate(0.05) {}
    
    // Main signal generation - Merton Jump Diffusion dynamics
    [[gnu::always_inline, gnu::hot]]
    Order generate_order(MarketDataType&& data) noexcept {
        auto start = std::chrono::high_resolution_clock::now();
        
        const double dt = 1.0 / 252.0 / 24.0;  // Intraday time step (hourly)
        const double current_price = data.mid_price();
        
        // Update price series
        price_series.push(current_price);
        
        // Get current state
        double S = spot_price.load();
        double vol = volatility.load();
        
        // Generate jump component using Poisson process
        int jump_count_dt = poisson_dist(rng);  // Number of jumps in dt
        poisson_dist.param(std::poisson_distribution<int>::param_type(lambda.load() * dt));
        
        double jump_component = 0.0;
        for (int i = 0; i < jump_count_dt; ++i) {
            // Log-normal jump sizes
            double jump_size = mu_j.load() + sigma_j.load() * normal_dist(rng);
            jump_component += std::exp(jump_size) - 1.0;
            jump_count.fetch_add(1);
        }
        
        // Store jump information
        jump_series.push(jump_component);
        
        // Merton Jump Diffusion dynamics:
        // dS = (mu - lambda*k)*S*dt + sigma*S*dW + S*sum(Y_i - 1)
        // where k = E[e^Y - 1] is the expected relative jump size
        double k = std::exp(mu_j.load() + 0.5 * sigma_j.load() * sigma_j.load()) - 1.0;
        
        // Drift component (adjusted for jump risk)
        double drift = (mu.load() - lambda.load() * k) * S * dt;
        
        // Diffusion component
        double diffusion = vol * S * std::sqrt(dt) * normal_dist(rng);
        
        // Jump component
        double jump_term = S * jump_component;
        
        // New price using exact solution
        double new_S = S * std::exp((mu.load() - lambda.load() * k - 0.5 * vol * vol) * dt + 
                                   vol * std::sqrt(dt) * normal_dist(rng)) * 
                      (1.0 + jump_component);
        
        // Update state
        spot_price.store(std::max(new_S, 0.01));  // Floor price to prevent negative values
        
        // Trading signal based on jump detection and mean reversion
        double recent_jump_activity = calculate_recent_jump_activity();
        double price_deviation = (current_price - S) / S;
        
        Order order;
        
        // Signal generation logic:
        // 1. Buy after negative jumps (oversold)
        // 2. Sell after positive jumps (overbought)
        // 3. Mean reversion when no jumps
        
        if (recent_jump_activity < -0.01) {  // Recent negative jump
            order = Order(Order::Side::BUY, data.ask, 100);
        } else if (recent_jump_activity > 0.01) {  // Recent positive jump
            order = Order(Order::Side::SELL, data.bid, 100);
        } else if (price_deviation < -0.02) {  // Mean reversion - buy low
            order = Order(Order::Side::BUY, data.ask, 50);
        } else if (price_deviation > 0.02) {  // Mean reversion - sell high
            order = Order(Order::Side::SELL, data.bid, 50);
        }
        
        // Performance tracking
        auto end = std::chrono::high_resolution_clock::now();
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        signal_count.fetch_add(1);
        total_latency_ns.fetch_add(latency_ns);
        
        return order;
    }
    
    // Calculate recent jump activity
    [[gnu::always_inline]]
    double calculate_recent_jump_activity() const noexcept {
        const size_t lookback = 5;
        double total_jump_activity = 0.0;
        
        for (size_t i = 0; i < std::min(lookback, BufferSize); ++i) {
            total_jump_activity += jump_series[i];
        }
        
        return total_jump_activity / lookback;
    }
    
    // Calibration using Maximum Likelihood Estimation for Jump Diffusion
    void calibrate(const Eigen::VectorXd& prices) {
        if (prices.size() < 20) return;
        
        // Calculate log returns
        Eigen::VectorXd log_returns(prices.size() - 1);
        for (int i = 0; i < log_returns.size(); ++i) {
            log_returns(i) = std::log(prices(i + 1) / prices(i));
        }
        
        // Estimate parameters using method of moments
        double mean_return = log_returns.mean();
        double var_return = ((log_returns.array() - mean_return).square()).mean();
        double skewness = calculate_skewness(log_returns);
        double kurtosis = calculate_kurtosis(log_returns);
        
        // Jump intensity estimation (based on excess kurtosis)
        double sample_lambda = std::max(0.0, (kurtosis - 3.0) / 10.0);  // Heuristic
        
        // Jump size parameters (based on skewness and kurtosis)
        double sample_mu_j = skewness * 0.01;  // Negative skew suggests negative jump mean
        double sample_sigma_j = std::sqrt(std::max(0.001, var_return * sample_lambda / 252.0));
        
        // Drift and volatility (adjusted for jumps)
        double k = std::exp(sample_mu_j + 0.5 * sample_sigma_j * sample_sigma_j) - 1.0;
        double sample_mu = mean_return * 252.0 + sample_lambda * k;  // Annualized
        double sample_sigma = std::sqrt(std::max(0.01, var_return * 252.0 - sample_lambda * 
                                       (sample_sigma_j * sample_sigma_j + sample_mu_j * sample_mu_j)));
        
        // Store parameters
        lambda.store(sample_lambda);
        mu_j.store(sample_mu_j);
        sigma_j.store(sample_sigma_j);
        mu.store(sample_mu);
        sigma.store(sample_sigma);
        volatility.store(sample_sigma);
        spot_price.store(prices(prices.size() - 1));
        
        params(0) = sample_lambda;
        params(1) = sample_mu_j;
        params(2) = sample_sigma_j;
        params(3) = sample_mu;
        params(4) = sample_sigma;
        params(5) = risk_free_rate.load();
    }
    
private:
    // Helper functions for statistical moments
    double calculate_skewness(const Eigen::VectorXd& data) const {
        double mean = data.mean();
        double std_dev = std::sqrt(((data.array() - mean).square()).mean());
        if (std_dev == 0) return 0;
        return ((data.array() - mean).cube()).mean() / (std_dev * std_dev * std_dev);
    }
    
    double calculate_kurtosis(const Eigen::VectorXd& data) const {
        double mean = data.mean();
        double variance = ((data.array() - mean).square()).mean();
        if (variance == 0) return 3;
        return ((data.array() - mean).pow(4)).mean() / (variance * variance);
    }
    
public:
    
    // Performance metrics
    [[gnu::always_inline]]
    double average_latency_us() const noexcept {
        uint64_t count = signal_count.load();
        if (count == 0) return 0.0;
        return static_cast<double>(total_latency_ns.load()) / (count * 1000.0);
    }
    
    [[gnu::always_inline]]
    bool meets_latency_constraint() const noexcept {
        return average_latency_us() < 50.0;
    }
    
    // Getters for Jump Diffusion parameters
    [[gnu::always_inline]]
    double get_lambda() const noexcept { return lambda.load(); }
    
    [[gnu::always_inline]]
    double get_mu_j() const noexcept { return mu_j.load(); }
    
    [[gnu::always_inline]]
    double get_sigma_j() const noexcept { return sigma_j.load(); }
    
    [[gnu::always_inline]]
    double get_mu() const noexcept { return mu.load(); }
    
    [[gnu::always_inline]]
    double get_sigma() const noexcept { return sigma.load(); }
    
    [[gnu::always_inline]]
    uint64_t get_jump_count() const noexcept { return jump_count.load(); }
    
    [[gnu::always_inline]]
    void set_jump_params(double l, double mj, double sj, double m, double s) noexcept {
        lambda.store(l);
        mu_j.store(mj);
        sigma_j.store(sj);
        mu.store(m);
        sigma.store(s);
    }
    
    [[gnu::always_inline]]
    const Eigen::VectorXd& get_params() const noexcept { return params; }
};

// Thread-local definitions
template <typename MarketDataType, size_t BufferSize>
thread_local std::mt19937 JumpDiffusionStrategy<MarketDataType, BufferSize>::rng{std::random_device{}()};

template <typename MarketDataType, size_t BufferSize>
thread_local std::normal_distribution<double> JumpDiffusionStrategy<MarketDataType, BufferSize>::normal_dist{0.0, 1.0};

template <typename MarketDataType, size_t BufferSize>
thread_local std::poisson_distribution<int> JumpDiffusionStrategy<MarketDataType, BufferSize>::poisson_dist{1};

template <typename MarketDataType, size_t BufferSize>
thread_local std::exponential_distribution<double> JumpDiffusionStrategy<MarketDataType, BufferSize>::exp_dist{1.0};
