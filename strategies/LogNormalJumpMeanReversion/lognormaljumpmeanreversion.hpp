#pragma once

#include <atomic>
#include <array>
#include <memory>
#include <chrono>
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

// Main strategy class
template <typename MarketDataType = MarketData, size_t BufferSize = 1000>
class LogNormalJumpMeanReversionStrategy {
private:
    RingBuffer<BufferSize> price_series;
    Eigen::VectorXd params;
    std::atomic<double> threshold;
    std::atomic<double> mean_estimate;
    std::atomic<double> variance_estimate;
    
    // Performance counters
    mutable std::atomic<uint64_t> signal_count{0};
    mutable std::atomic<uint64_t> total_latency_ns{0};
    
public:
    LogNormalJumpMeanReversionStrategy() noexcept 
        : params(Eigen::VectorXd::Zero(3)), threshold(1.5), 
          mean_estimate(0.0), variance_estimate(1.0) {}
    
    // Main signal generation - must be < 20μs
    [[gnu::always_inline, gnu::hot]]
    Order generate_order(MarketDataType&& data) noexcept {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Update price series
        price_series.push(data.mid_price());
        
        // Calculate z-score
        double current_price = data.mid_price();
        double mean = mean_estimate.load();
        double variance = variance_estimate.load();
        double z_score = (current_price - mean) / std::sqrt(variance);
        
        // Generate signal
        Order order;
        double thresh = threshold.load();
        
        if (z_score < -thresh) {
            order = Order(Order::Side::BUY, data.ask, 100);
        } else if (z_score > thresh) {
            order = Order(Order::Side::SELL, data.bid, 100);
        }
        
        // Performance tracking
        auto end = std::chrono::high_resolution_clock::now();
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        signal_count.fetch_add(1);
        total_latency_ns.fetch_add(latency_ns);
        
        return order;
    }
    
    // Calibration using MLE
    void calibrate(const Eigen::VectorXd& prices) {
        if (prices.size() < 2) return;
        
        // Calculate sample statistics
        double mean = prices.mean();
        double variance = ((prices.array() - mean).square()).mean();
        
        mean_estimate.store(mean);
        variance_estimate.store(variance);
        
        // Estimate mean reversion parameters
        Eigen::VectorXd returns = prices.tail(prices.size() - 1) - prices.head(prices.size() - 1);
        double kappa = -std::log(returns.array().abs().mean()) / (1.0 / 252.0);  // Daily to annual
        
        params(0) = kappa;      // Mean reversion speed
        params(1) = mean;       // Long-term mean
        params(2) = variance;   // Variance
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
        return average_latency_us() < 20.0;
    }
    
    // Getters
    [[gnu::always_inline]]
    double get_threshold() const noexcept { return threshold.load(); }
    
    [[gnu::always_inline]]
    void set_threshold(double t) noexcept { threshold.store(t); }
    
    [[gnu::always_inline]]
    const Eigen::VectorXd& get_params() const noexcept { return params; }
};
