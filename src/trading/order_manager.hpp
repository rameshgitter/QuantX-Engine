#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <thread>
#include <condition_variable>
#include "../risk/risk_manager.hpp"

namespace quantx {

class IOrderCallback {
public:
    virtual ~IOrderCallback() = default;
    virtual void onOrderUpdate(const Order& order) = 0;
    virtual void onFill(const Fill& fill) = 0;
    virtual void onOrderRejected(const Order& order, const std::string& reason) = 0;
};

class ITradingConnector {
public:
    virtual ~ITradingConnector() = default;
    virtual bool connect() = 0;
    virtual void disconnect() = 0;
    virtual bool isConnected() const = 0;
    virtual std::string submitOrder(const Order& order) = 0;
    virtual bool cancelOrder(const std::string& order_id) = 0;
    virtual bool modifyOrder(const std::string& order_id, int new_quantity, double new_price) = 0;
};

class PaperTradingConnector : public ITradingConnector {
private:
    std::atomic<bool> connected_;
    std::unordered_map<std::string, Order> active_orders_;
    std::mutex orders_mutex_;
    
    // Paper trading parameters
    double fill_probability_;
    double slippage_factor_;
    std::chrono::milliseconds fill_delay_;
    double commission_rate_;
    
    // Simulation thread
    std::thread simulation_thread_;
    std::atomic<bool> running_;
    std::queue<std::string> pending_fills_;
    std::mutex pending_mutex_;
    std::condition_variable pending_cv_;
    
    std::vector<std::shared_ptr<IOrderCallback>> callbacks_;
    std::mutex callbacks_mutex_;
    
    std::atomic<uint64_t> order_counter_;
    
public:
    PaperTradingConnector();
    ~PaperTradingConnector();
    
    bool connect() override;
    void disconnect() override;
    bool isConnected() const override { return connected_.load(); }
    
    std::string submitOrder(const Order& order) override;
    bool cancelOrder(const std::string& order_id) override;
    bool modifyOrder(const std::string& order_id, int new_quantity, double new_price) override;
    
    void addCallback(std::shared_ptr<IOrderCallback> callback);
    void removeCallback(std::shared_ptr<IOrderCallback> callback);
    
    // Paper trading configuration
    void setFillProbability(double probability) { fill_probability_ = probability; }
    void setSlippageFactor(double factor) { slippage_factor_ = factor; }
    void setFillDelay(std::chrono::milliseconds delay) { fill_delay_ = delay; }
    void setCommissionRate(double rate) { commission_rate_ = rate; }
    
private:
    void simulationLoop();
    void processPendingFill(const std::string& order_id);
    void notifyCallbacks(std::function<void(IOrderCallback*)> func);
    double calculateFillPrice(const Order& order);
    std::string generateOrderId();
};

class OrderManager {
private:
    std::shared_ptr<RiskManager> risk_manager_;
    std::shared_ptr<ITradingConnector> connector_;
    
    std::unordered_map<std::string, Order> orders_;
    std::mutex orders_mutex_;
    
    std::vector<std::shared_ptr<IOrderCallback>> callbacks_;
    std::mutex callbacks_mutex_;
    
    std::atomic<bool> running_;
    
    // Performance metrics
    std::atomic<uint64_t> orders_submitted_;
    std::atomic<uint64_t> orders_filled_;
    std::atomic<uint64_t> orders_rejected_;
    std::atomic<uint64_t> total_processing_time_us_;
    
public:
    OrderManager(std::shared_ptr<RiskManager> risk_manager, 
                std::shared_ptr<ITradingConnector> connector);
    ~OrderManager();
    
    void start();
    void stop();
    bool isRunning() const { return running_.load(); }
    
    std::string submitOrder(const std::string& symbol, OrderSide side, int quantity, 
                           OrderType type, double price = 0.0);
    bool cancelOrder(const std::string& order_id);
    bool modifyOrder(const std::string& order_id, int new_quantity, double new_price);
    
    Order getOrder(const std::string& order_id) const;
    std::vector<Order> getActiveOrders() const;
    std::vector<Order> getOrderHistory() const;
    
    void addCallback(std::shared_ptr<IOrderCallback> callback);
    void removeCallback(std::shared_ptr<IOrderCallback> callback);
    
    // Performance metrics
    uint64_t getOrdersSubmitted() const { return orders_submitted_.load(); }
    uint64_t getOrdersFilled() const { return orders_filled_.load(); }
    uint64_t getOrdersRejected() const { return orders_rejected_.load(); }
    double getAverageProcessingTimeMs() const;
    
private:
    void onOrderUpdate(const Order& order);
    void onFill(const Fill& fill);
    void onOrderRejected(const Order& order, const std::string& reason);
    
    void notifyCallbacks(std::function<void(IOrderCallback*)> func);
    std::string generateOrderId();
};

} // namespace quantx
