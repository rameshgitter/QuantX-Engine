#include "risk_manager.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <random>

namespace quantx {

RiskManager::RiskManager(double initial_capital) 
    : initial_capital_(initial_capital), emergency_stop_(false), running_(false),
      risk_checks_performed_(0), violations_detected_(0) {
    
    current_metrics_.portfolio_value = initial_capital;
    current_metrics_.last_update = std::chrono::high_resolution_clock::now();
}

RiskManager::~RiskManager() {
    stop();
}

void RiskManager::start() {
    running_.store(true);
    emergency_stop_.store(false);
    std::cout << "Risk Manager started with initial capital: ₹" << initial_capital_ << std::endl;
}

void RiskManager::stop() {
    running_.store(false);
    std::cout << "Risk Manager stopped" << std::endl;
}

void RiskManager::emergencyStop() {
    emergency_stop_.store(true);
    std::cout << "EMERGENCY STOP ACTIVATED!" << std::endl;
    
    // Notify all callbacks
    notifyCallbacks([](IRiskCallback* cb) {
        RiskViolation violation;
        violation.type = RiskViolationType::DAILY_LOSS_LIMIT;
        violation.description = "Emergency stop activated";
        violation.timestamp = std::chrono::high_resolution_clock::now();
        cb->onRiskViolation(violation);
    });
}

void RiskManager::setRiskLimits(const RiskLimits& limits) {
    limits_ = limits;
    std::cout << "Risk limits updated" << std::endl;
}

RiskLimits RiskManager::getRiskLimits() const {
    return limits_;
}

bool RiskManager::validateOrder(const Order& order) {
    risk_checks_performed_.fetch_add(1);
    
    if (emergency_stop_.load()) {
        reportViolation(RiskViolationType::DAILY_LOSS_LIMIT, order.symbol,
                       "Emergency stop is active", 0, 0);
        return false;
    }
    
    // Check rate limits
    if (!checkRateLimit()) {
        reportViolation(RiskViolationType::RATE_LIMIT, order.symbol,
                       "Rate limit exceeded", 0, limits_.max_orders_per_second);
        return false;
    }
    
    // Check position limits
    double order_value = order.quantity * order.price;
    if (!checkPositionLimit(order.symbol, order.side, order.quantity, order.price)) {
        reportViolation(RiskViolationType::POSITION_LIMIT, order.symbol,
                       "Position limit exceeded", order_value, limits_.max_position_value);
        return false;
    }
    
    // Check portfolio limits
    if (!checkPortfolioLimit(order_value)) {
        reportViolation(RiskViolationType::PORTFOLIO_LIMIT, order.symbol,
                       "Portfolio limit exceeded", current_metrics_.portfolio_value + order_value,
                       limits_.max_portfolio_value);
        return false;
    }
    
    // Check daily loss limit
    if (!checkDailyLossLimit()) {
        reportViolation(RiskViolationType::DAILY_LOSS_LIMIT, order.symbol,
                       "Daily loss limit exceeded", current_metrics_.daily_pnl, -limits_.max_daily_loss);
        return false;
    }
    
    // Check drawdown limit
    if (!checkDrawdownLimit()) {
        reportViolation(RiskViolationType::DRAWDOWN_LIMIT, order.symbol,
                       "Drawdown limit exceeded", current_metrics_.current_drawdown, limits_.max_drawdown);
        return false;
    }
    
    // Check leverage limit
    if (!checkLeverageLimit()) {
        reportViolation(RiskViolationType::LEVERAGE_LIMIT, order.symbol,
                       "Leverage limit exceeded", current_metrics_.leverage, limits_.leverage_limit);
        return false;
    }
    
    return true;
}

bool RiskManager::checkPositionLimit(const std::string& symbol, OrderSide side, int quantity, double price) {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    
    auto it = positions_.find(symbol);
    double current_position_value = 0.0;
    
    if (it != positions_.end()) {
        current_position_value = std::abs(it->second.quantity * it->second.market_price);
    }
    
    double additional_value = quantity * price;
    double total_value = current_position_value + additional_value;
    
    // Check symbol-specific limit
    auto symbol_limit_it = limits_.symbol_limits.find(symbol);
    if (symbol_limit_it != limits_.symbol_limits.end()) {
        return total_value <= symbol_limit_it->second;
    }
    
    // Check general position limit
    return total_value <= limits_.max_position_value;
}

bool RiskManager::checkPortfolioLimit(double additional_value) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return (current_metrics_.portfolio_value + additional_value) <= limits_.max_portfolio_value;
}

bool RiskManager::checkRateLimit() {
    std::lock_guard<std::mutex> lock(rate_limit_mutex_);
    
    auto now = std::chrono::high_resolution_clock::now();
    
    // Remove orders older than 1 second
    recent_orders_.erase(
        std::remove_if(recent_orders_.begin(), recent_orders_.end(),
            [now](const auto& timestamp) {
                return std::chrono::duration_cast<std::chrono::seconds>(now - timestamp).count() >= 1;
            }),
        recent_orders_.end()
    );
    
    // Check if we're within limits
    if (static_cast<int>(recent_orders_.size()) >= limits_.max_orders_per_second) {
        return false;
    }
    
    // Add current order timestamp
    recent_orders_.push_back(now);
    return true;
}

void RiskManager::updatePosition(const Fill& fill) {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    
    auto& position = positions_[fill.symbol];
    position.symbol = fill.symbol;
    
    int side_multiplier = (fill.side == OrderSide::BUY) ? 1 : -1;
    int fill_quantity = fill.quantity * side_multiplier;
    
    if (position.quantity == 0) {
        // New position
        position.quantity = fill_quantity;
        position.average_price = fill.price;
    } else if ((position.quantity > 0 && fill_quantity > 0) || 
               (position.quantity < 0 && fill_quantity < 0)) {
        // Adding to existing position
        double total_value = position.quantity * position.average_price + fill_quantity * fill.price;
        position.quantity += fill_quantity;
        position.average_price = total_value / position.quantity;
    } else {
        // Reducing or closing position
        int abs_fill = std::abs(fill_quantity);
        int abs_position = std::abs(position.quantity);
        
        if (abs_fill >= abs_position) {
            // Closing and potentially reversing position
            double realized_pnl = position.quantity * (fill.price - position.average_price);
            position.realized_pnl += realized_pnl;
            
            position.quantity = fill_quantity - (position.quantity > 0 ? abs_position : -abs_position);
            if (position.quantity != 0) {
                position.average_price = fill.price;
            }
        } else {
            // Partial close
            double realized_pnl = fill_quantity * (fill.price - position.average_price);
            position.realized_pnl += realized_pnl;
            position.quantity += fill_quantity;
        }
    }
    
    position.last_update = std::chrono::high_resolution_clock::now();
    
    // Calculate unrealized P&L
    position.unrealized_pnl = position.quantity * (position.market_price - position.average_price);
    
    // Notify callbacks
    notifyCallbacks([&position](IRiskCallback* cb) {
        cb->onPositionUpdate(position);
    });
    
    // Update risk metrics
    calculateRiskMetrics();
}

void RiskManager::updateMarketPrice(const std::string& symbol, double price) {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    
    auto it = positions_.find(symbol);
    if (it != positions_.end()) {
        it->second.market_price = price;
        it->second.unrealized_pnl = it->second.quantity * (price - it->second.average_price);
        it->second.last_update = std::chrono::high_resolution_clock::now();
        
        // Update risk metrics
        calculateRiskMetrics();
    }
}

Position RiskManager::getPosition(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    auto it = positions_.find(symbol);
    return (it != positions_.end()) ? it->second : Position{};
}

std::vector<Position> RiskManager::getAllPositions() const {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    std::vector<Position> positions;
    for (const auto& pair : positions_) {
        if (pair.second.quantity != 0) {  // Only return non-zero positions
            positions.push_back(pair.second);
        }
    }
    return positions;
}

RiskMetrics RiskManager::getCurrentMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

void RiskManager::calculateRiskMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    std::lock_guard<std::mutex> pos_lock(positions_mutex_);
    
    double total_unrealized_pnl = 0.0;
    double total_realized_pnl = 0.0;
    double total_position_value = 0.0;
    int positions_count = 0;
    
    for (const auto& pair : positions_) {
        const auto& position = pair.second;
        if (position.quantity != 0) {
            total_unrealized_pnl += position.unrealized_pnl;
            total_realized_pnl += position.realized_pnl;
            total_position_value += std::abs(position.quantity * position.market_price);
            positions_count++;
        }
    }
    
    current_metrics_.total_pnl = total_realized_pnl + total_unrealized_pnl;
    current_metrics_.portfolio_value = initial_capital_ + current_metrics_.total_pnl;
    current_metrics_.positions_count = positions_count;
    current_metrics_.leverage = total_position_value / current_metrics_.portfolio_value;
    current_metrics_.var_1day = calculateVaR();
    current_metrics_.last_update = std::chrono::high_resolution_clock::now();
    
    updateDailyPnL();
    updateDrawdown();
    
    // Notify callbacks
    notifyCallbacks([this](IRiskCallback* cb) {
        cb->onRiskMetricsUpdate(current_metrics_);
    });
}

double RiskManager::calculateVaR(double confidence_level) const {
    // Simplified VaR calculation using historical simulation
    // In production, this would use more sophisticated methods
    
    std::vector<double> portfolio_returns;
    
    // Calculate daily returns (simplified)
    double portfolio_volatility = 0.02; // Assume 2% daily volatility
    double z_score = (confidence_level == 0.99) ? 2.33 : 1.65; // 99% or 95% confidence
    
    return current_metrics_.portfolio_value * portfolio_volatility * z_score;
}

double RiskManager::calculatePortfolioValue() const {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    
    double total_pnl = 0.0;
    for (const auto& pair : positions_) {
        const auto& position = pair.second;
        total_pnl += position.realized_pnl + position.unrealized_pnl;
    }
    
    return initial_capital_ + total_pnl;
}

void RiskManager::addCallback(std::shared_ptr<IRiskCallback> callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.push_back(callback);
}

void RiskManager::removeCallback(std::shared_ptr<IRiskCallback> callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.erase(
        std::remove_if(callbacks_.begin(), callbacks_.end(),
            [&callback](const std::weak_ptr<IRiskCallback>& weak_cb) {
                return weak_cb.lock() == callback;
            }),
        callbacks_.end()
    );
}

void RiskManager::notifyCallbacks(std::function<void(IRiskCallback*)> func) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    for (auto& weak_cb : callbacks_) {
        if (auto cb = weak_cb.lock()) {
            try {
                func(cb.get());
            } catch (const std::exception& e) {
                std::cerr << "Risk callback error: " << e.what() << std::endl;
            }
        }
    }
}

void RiskManager::reportViolation(RiskViolationType type, const std::string& symbol,
                                 const std::string& description, double current_value, double limit_value) {
    violations_detected_.fetch_add(1);
    
    RiskViolation violation;
    violation.type = type;
    violation.symbol = symbol;
    violation.description = description;
    violation.current_value = current_value;
    violation.limit_value = limit_value;
    violation.timestamp = std::chrono::high_resolution_clock::now();
    
    std::cout << "RISK VIOLATION: " << description 
              << " (Current: " << current_value << ", Limit: " << limit_value << ")" << std::endl;
    
    notifyCallbacks([&violation](IRiskCallback* cb) {
        cb->onRiskViolation(violation);
    });
}

bool RiskManager::checkDailyLossLimit() const {
    return current_metrics_.daily_pnl >= -limits_.max_daily_loss;
}

bool RiskManager::checkDrawdownLimit() const {
    return current_metrics_.current_drawdown <= limits_.max_drawdown;
}

bool RiskManager::checkLeverageLimit() const {
    return current_metrics_.leverage <= limits_.leverage_limit;
}

void RiskManager::updateDailyPnL() {
    // Simplified daily P&L calculation
    // In production, this would track P&L from market open
    current_metrics_.daily_pnl = current_metrics_.total_pnl;
}

void RiskManager::updateDrawdown() {
    static double peak_portfolio_value = initial_capital_;
    
    if (current_metrics_.portfolio_value > peak_portfolio_value) {
        peak_portfolio_value = current_metrics_.portfolio_value;
    }
    
    current_metrics_.current_drawdown = (peak_portfolio_value - current_metrics_.portfolio_value) / peak_portfolio_value;
    current_metrics_.max_drawdown = std::max(current_metrics_.max_drawdown, current_metrics_.current_drawdown);
}

} // namespace quantx
