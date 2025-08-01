#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <chrono>

namespace quantx {

enum class OrderSide {
    BUY = 1,
    SELL = -1
};

enum class OrderType {
    MARKET,
    LIMIT,
    STOP,
    STOP_LIMIT
};

enum class OrderStatus {
    PENDING,
    SUBMITTED,
    PARTIALLY_FILLED,
    FILLED,
    CANCELLED,
    REJECTED
};

enum class RiskViolationType {
    POSITION_LIMIT,
    PORTFOLIO_LIMIT,
    DAILY_LOSS_LIMIT,
    DRAWDOWN_LIMIT,
    VAR_LIMIT,
    LEVERAGE_LIMIT,
    RATE_LIMIT
};

struct Position {
    std::string symbol;
    int quantity;
    double average_price;
    double market_price;
    double unrealized_pnl;
    double realized_pnl;
    std::chrono::high_resolution_clock::time_point last_update;
    
    Position() : quantity(0), average_price(0.0), market_price(0.0), 
                unrealized_pnl(0.0), realized_pnl(0.0) {}
};

struct RiskLimits {
    double max_position_value;
    double max_portfolio_value;
    double max_daily_loss;
    double max_drawdown;
    double var_limit;
    double leverage_limit;
    int max_orders_per_second;
    int max_orders_per_minute;
    std::unordered_map<std::string, double> symbol_limits;
    
    RiskLimits() : max_position_value(100000.0), max_portfolio_value(1000000.0),
                  max_daily_loss(50000.0), max_drawdown(0.15), var_limit(30000.0),
                  leverage_limit(2.0), max_orders_per_second(10), max_orders_per_minute(100) {}
};

struct RiskMetrics {
    double portfolio_value;
    double total_pnl;
    double daily_pnl;
    double current_drawdown;
    double max_drawdown;
    double var_1day;
    double leverage;
    int positions_count;
    std::chrono::high_resolution_clock::time_point last_update;
    
    RiskMetrics() : portfolio_value(0.0), total_pnl(0.0), daily_pnl(0.0),
                   current_drawdown(0.0), max_drawdown(0.0), var_1day(0.0),
                   leverage(0.0), positions_count(0) {}
};

struct RiskViolation {
    RiskViolationType type;
    std::string symbol;
    std::string description;
    double current_value;
    double limit_value;
    std::chrono::high_resolution_clock::time_point timestamp;
};

struct Order {
    std::string order_id;
    std::string symbol;
    OrderSide side;
    int quantity;
    OrderType type;
    double price;
    OrderStatus status;
    int filled_quantity;
    double average_fill_price;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    Order() : side(OrderSide::BUY), quantity(0), type(OrderType::MARKET),
             price(0.0), status(OrderStatus::PENDING), filled_quantity(0),
             average_fill_price(0.0) {}
};

struct Fill {
    std::string order_id;
    std::string symbol;
    OrderSide side;
    int quantity;
    double price;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    Fill() : side(OrderSide::BUY), quantity(0), price(0.0) {}
};

class IRiskCallback {
public:
    virtual ~IRiskCallback() = default;
    virtual void onRiskViolation(const RiskViolation& violation) = 0;
    virtual void onPositionUpdate(const Position& position) = 0;
    virtual void onRiskMetricsUpdate(const RiskMetrics& metrics) = 0;
};

class RiskManager {
private:
    double initial_capital_;
    RiskLimits limits_;
    RiskMetrics current_metrics_;
    
    std::unordered_map<std::string, Position> positions_;
    std::vector<Order> order_history_;
    std::vector<Fill> fill_history_;
    
    std::vector<std::shared_ptr<IRiskCallback>> callbacks_;
    mutable std::mutex positions_mutex_;
    mutable std::mutex callbacks_mutex_;
    mutable std::mutex metrics_mutex_;
    
    std::atomic<bool> emergency_stop_;
    std::atomic<bool> running_;
    
    // Rate limiting
    std::vector<std::chrono::high_resolution_clock::time_point> recent_orders_;
    mutable std::mutex rate_limit_mutex_;
    
    // Performance tracking
    std::atomic<uint64_t> risk_checks_performed_;
    std::atomic<uint64_t> violations_detected_;
    
public:
    RiskManager(double initial_capital);
    ~RiskManager();
    
    void start();
    void stop();
    void emergencyStop();
    
    bool isEmergencyStop() const { return emergency_stop_.load(); }
    bool isRunning() const { return running_.load(); }
    
    // Risk limits
    void setRiskLimits(const RiskLimits& limits);
    RiskLimits getRiskLimits() const;
    
    // Order validation
    bool validateOrder(const Order& order);
    bool checkPositionLimit(const std::string& symbol, OrderSide side, int quantity, double price);
    bool checkPortfolioLimit(double additional_value);
    bool checkRateLimit();
    
    // Position management
    void updatePosition(const Fill& fill);
    void updateMarketPrice(const std::string& symbol, double price);
    Position getPosition(const std::string& symbol) const;
    std::vector<Position> getAllPositions() const;
    
    // Risk metrics
    RiskMetrics getCurrentMetrics() const;
    void calculateRiskMetrics();
    double calculateVaR(double confidence_level = 0.99) const;
    double calculatePortfolioValue() const;
    
    // Callbacks
    void addCallback(std::shared_ptr<IRiskCallback> callback);
    void removeCallback(std::shared_ptr<IRiskCallback> callback);
    
    // Performance metrics
    uint64_t getRiskChecksPerformed() const { return risk_checks_performed_.load(); }
    uint64_t getViolationsDetected() const { return violations_detected_.load(); }
    
private:
    void notifyCallbacks(std::function<void(IRiskCallback*)> func);
    void reportViolation(RiskViolationType type, const std::string& symbol,
                        const std::string& description, double current_value, double limit_value);
    
    bool checkDailyLossLimit() const;
    bool checkDrawdownLimit() const;
    bool checkLeverageLimit() const;
    
    void updateDailyPnL();
    void updateDrawdown();
};

} // namespace quantx
