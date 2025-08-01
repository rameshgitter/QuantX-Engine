#include "order_manager.hpp"
#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#include <functional>
#include <memory>
#include <condition_variable>

namespace quantx {

class IOrderCallback {
public:
    virtual ~IOrderCallback() = default;
    virtual void onOrderUpdate(const Order& order) = 0;
    virtual void onFill(const Fill& fill) = 0;
    virtual void onOrderRejected(const Order& order, const std::string& reason) = 0;
};

class RiskManager {
public:
    bool validateOrder(const Order& order) {
        // Placeholder for risk validation logic
        return true;
    }
    
    void updatePosition(const Fill& fill) {
        // Placeholder for position update logic
    }
};

class ITradingConnector {
public:
    virtual ~ITradingConnector() = default;
    virtual bool connect() = 0;
    virtual void disconnect() = 0;
    virtual std::string submitOrder(const Order& order) = 0;
    virtual bool cancelOrder(const std::string& order_id) = 0;
    virtual bool modifyOrder(const std::string& order_id, int new_quantity, double new_price) = 0;
};

class OrderBook {
public:
    std::vector<Order> asks;
    std::vector<Order> bids;
    double mid_price = 0.0;
};

enum class OrderStatus {
    SUBMITTED,
    PARTIALLY_FILLED,
    FILLED,
    CANCELLED,
    PENDING,
    REJECTED
};

enum class OrderType {
    LIMIT,
    MARKET
};

enum class OrderSide {
    BUY,
    SELL
};

class Order {
public:
    std::string order_id;
    std::string symbol;
    OrderSide side;
    int quantity;
    OrderType type;
    double price;
    OrderStatus status;
    int filled_quantity = 0;
    double avg_fill_price = 0.0;
    std::chrono::high_resolution_clock::time_point timestamp;
    std::string exchange_order_id;
};

class Fill {
public:
    std::string fill_id;
    std::string order_id;
    std::string symbol;
    OrderSide side;
    int quantity;
    double price;
    double fees;
    std::chrono::high_resolution_clock::time_point timestamp;
    std::string exchange_fill_id;
};

class PaperTradingConnector : public ITradingConnector {
public:
    PaperTradingConnector() 
        : connected_(false), fill_probability_(0.95), slippage_factor_(0.0002),
          fill_delay_(std::chrono::milliseconds(100)), commission_rate_(0.0003),
          running_(false), order_counter_(0) {
    }

    ~PaperTradingConnector() {
        disconnect();
    }

    bool connect() {
        if (connected_.load()) {
            return true;
        }
        
        connected_.store(true);
        running_.store(true);
        
        // Start simulation thread
        simulation_thread_ = std::thread(&PaperTradingConnector::simulationLoop, this);
        
        std::cout << "Paper trading connector connected" << std::endl;
        return true;
    }

    void disconnect() {
        if (!connected_.load()) {
            return;
        }
        
        std::cout << "Disconnecting paper trading connector..." << std::endl;
        
        connected_.store(false);
        running_.store(false);
        
        // Wake up simulation thread
        pending_cv_.notify_all();
        
        if (simulation_thread_.joinable()) {
            simulation_thread_.join();
        }
        
        std::cout << "Paper trading connector disconnected" << std::endl;
    }

    std::string submitOrder(const Order& order) {
        if (!connected_.load()) {
            return "";
        }
        
        std::string order_id = generateOrderId();
        
        {
            std::lock_guard<std::mutex> lock(orders_mutex_);
            Order paper_order = order;
            paper_order.order_id = order_id;
            paper_order.status = OrderStatus::SUBMITTED;
            paper_order.timestamp = std::chrono::high_resolution_clock::now();
            active_orders_[order_id] = paper_order;
        }
        
        // Notify order submitted
        notifyCallbacks([&order_id, this](IOrderCallback* cb) {
            std::lock_guard<std::mutex> lock(orders_mutex_);
            auto it = active_orders_.find(order_id);
            if (it != active_orders_.end()) {
                cb->onOrderUpdate(it->second);
            }
        });
        
        // Add to pending fills queue
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_fills_.push(order_id);
        }
        pending_cv_.notify_one();
        
        return order_id;
    }

    bool cancelOrder(const std::string& order_id) {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        
        auto it = active_orders_.find(order_id);
        if (it != active_orders_.end() && it->second.status == OrderStatus::SUBMITTED) {
            it->second.status = OrderStatus::CANCELLED;
            
            notifyCallbacks([&it](IOrderCallback* cb) {
                cb->onOrderUpdate(it->second);
            });
            
            active_orders_.erase(it);
            return true;
        }
        
        return false;
    }

    bool modifyOrder(const std::string& order_id, int new_quantity, double new_price) {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        
        auto it = active_orders_.find(order_id);
        if (it != active_orders_.end() && it->second.status == OrderStatus::SUBMITTED) {
            it->second.quantity = new_quantity;
            it->second.price = new_price;
            
            notifyCallbacks([&it](IOrderCallback* cb) {
                cb->onOrderUpdate(it->second);
            });
            
            return true;
        }
        
        return false;
    }

    void addCallback(std::shared_ptr<IOrderCallback> callback) {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        callbacks_.push_back(callback);
    }

    void removeCallback(std::shared_ptr<IOrderCallback> callback) {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        callbacks_.erase(
            std::remove_if(callbacks_.begin(), callbacks_.end(),
                [&callback](const std::weak_ptr<IOrderCallback>& weak_cb) {
                    return weak_cb.lock() == callback;
                }),
            callbacks_.end()
        );
    }

    void simulationLoop() {
        std::cout << "Paper trading simulation started" << std::endl;
        
        while (running_.load()) {
            std::unique_lock<std::mutex> lock(pending_mutex_);
            pending_cv_.wait(lock, [this] { return !pending_fills_.empty() || !running_.load(); });
            
            while (!pending_fills_.empty() && running_.load()) {
                std::string order_id = pending_fills_.front();
                pending_fills_.pop();
                lock.unlock();
                
                // Simulate processing delay
                std::this_thread::sleep_for(fill_delay_);
                
                processPendingFill(order_id);
                
                lock.lock();
            }
        }
        
        std::cout << "Paper trading simulation stopped" << std::endl;
    }

    void processPendingFill(const std::string& order_id) {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        
        auto it = active_orders_.find(order_id);
        if (it == active_orders_.end() || it->second.status != OrderStatus::SUBMITTED) {
            return;
        }
        
        Order& order = it->second;
        
        // Simulate fill probability
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> fill_dist(0.0, 1.0);
        
        if (fill_dist(gen) > fill_probability_) {
            // Order not filled this time
            return;
        }
        
        // Calculate fill price with slippage
        double fill_price = calculateFillPrice(order);
        
        // Create fill
        Fill fill;
        fill.order_id = order_id;
        fill.symbol = order.symbol;
        fill.side = order.side;
        fill.quantity = order.quantity;  // Full fill for simplicity
        fill.price = fill_price;
        fill.timestamp = std::chrono::high_resolution_clock::now();
        
        // Update order status
        order.status = OrderStatus::FILLED;
        order.filled_quantity = order.quantity;
        order.average_fill_price = fill_price;
        
        // Notify callbacks
        notifyCallbacks([&order, &fill](IOrderCallback* cb) {
            cb->onOrderUpdate(order);
            cb->onFill(fill);
        });
        
        // Remove from active orders
        active_orders_.erase(it);
    }

    void notifyCallbacks(std::function<void(IOrderCallback*)> func) {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (auto& weak_cb : callbacks_) {
            if (auto cb = weak_cb.lock()) {
                try {
                    func(cb.get());
                } catch (const std::exception& e) {
                    std::cerr << "Paper trading callback error: " << e.what() << std::endl;
                }
            }
        }
    }

    double calculateFillPrice(const Order& order) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::normal_distribution<> slippage_dist(0.0, slippage_factor_);
        
        double base_price = order.price;
        if (order.type == OrderType::MARKET) {
            // For market orders, assume some reference price (in real system, use market data)
            base_price = order.price > 0 ? order.price : 18450.0;  // Default NIFTY price
        }
        
        // Add slippage
        double slippage = slippage_dist(gen);
        if (order.side == OrderSide::BUY) {
            slippage = std::abs(slippage);  // Positive slippage for buys
        } else {
            slippage = -std::abs(slippage);  // Negative slippage for sells
        }
        
        return base_price * (1.0 + slippage);
    }

    std::string generateOrderId() {
        uint64_t counter = order_counter_.fetch_add(1);
        std::stringstream ss;
        ss << "PAPER_" << std::setfill('0') << std::setw(8) << counter;
        return ss.str();
    }

private:
    std::atomic<bool> connected_;
    double fill_probability_;
    double slippage_factor_;
    std::chrono::milliseconds fill_delay_;
    double commission_rate_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> order_counter_;
    std::thread simulation_thread_;
    std::mutex orders_mutex_;
    std::mutex books_mutex_;
    std::mutex pending_mutex_;
    std::mutex callbacks_mutex_;
    std::condition_variable pending_cv_;
    std::unordered_map<std::string, Order> active_orders_;
    std::unordered_map<std::string, OrderBook> current_books_;
    std::queue<std::string> pending_fills_;
    std::vector<std::weak_ptr<IOrderCallback>> callbacks_;
};

class OrderManager : public IOrderCallback {
public:
    OrderManager(std::shared_ptr<RiskManager> risk_manager, 
                  std::shared_ptr<ITradingConnector> connector)
        : risk_manager_(risk_manager), connector_(connector), running_(false),
          orders_submitted_(0), orders_filled_(0), orders_rejected_(0), total_processing_time_us_(0) {
        
        // Set up connector callbacks if it's a PaperTradingConnector
        if (auto paper_connector = std::dynamic_pointer_cast<PaperTradingConnector>(connector_)) {
            paper_connector->addCallback(std::shared_ptr<IOrderCallback>(this, [](IOrderCallback*){}));
        }
    }

    ~OrderManager() {
        stop();
    }

    void start() {
        if (running_.load()) {
            return;
        }
        
        running_.store(true);
        
        if (!connector_->connect()) {
            std::cerr << "Failed to connect trading connector" << std::endl;
            running_.store(false);
            return;
        }
        
        std::cout << "Order Manager started" << std::endl;
    }

    void stop() {
        if (!running_.load()) {
            return;
        }
        
        running_.store(false);
        connector_->disconnect();
        
        std::cout << "Order Manager stopped" << std::endl;
    }

    std::string submitOrder(const std::string& symbol, OrderSide side, int quantity, 
                             OrderType type, double price) {
        if (!running_.load()) {
            std::cerr << "Order Manager not running" << std::endl;
            return "";
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create order
        Order order;
        order.symbol = symbol;
        order.side = side;
        order.quantity = quantity;
        order.type = type;
        order.price = price;
        order.status = OrderStatus::PENDING;
        order.timestamp = std::chrono::high_resolution_clock::now();
        
        // Validate with risk manager
        if (!risk_manager_->validateOrder(order)) {
            orders_rejected_.fetch_add(1);
            
            notifyCallbacks([&order](IOrderCallback* cb) {
                cb->onOrderRejected(order, "Risk validation failed");
            });
            
            return "";
        }
        
        // Submit to connector
        std::string order_id = connector_->submitOrder(order);
        
        if (order_id.empty()) {
            orders_rejected_.fetch_add(1);
            
            notifyCallbacks([&order](IOrderCallback* cb) {
                cb->onOrderRejected(order, "Connector submission failed");
            });
            
            return "";
        }
        
        // Store order
        order.order_id = order_id;
        {
            std::lock_guard<std::mutex> lock(orders_mutex_);
            orders_[order_id] = order;
        }
        
        orders_submitted_.fetch_add(1);
        
        // Update processing time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        total_processing_time_us_.fetch_add(processing_time.count());
        
        return order_id;
    }

    bool cancelOrder(const std::string& order_id) {
        if (!running_.load()) {
            return false;
        }
        
        return connector_->cancelOrder(order_id);
    }

    bool modifyOrder(const std::string& order_id, int new_quantity, double new_price) {
        if (!running_.load()) {
            return false;
        }
        
        return connector_->modifyOrder(order_id, new_quantity, new_price);
    }

    Order getOrder(const std::string& order_id) const {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        auto it = orders_.find(order_id);
        return (it != orders_.end()) ? it->second : Order{};
    }

    std::vector<Order> getActiveOrders() const {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        std::vector<Order> active_orders;
        
        for (const auto& pair : orders_) {
            if (pair.second.status == OrderStatus::SUBMITTED || 
                pair.second.status == OrderStatus::PARTIALLY_FILLED) {
                active_orders.push_back(pair.second);
            }
        }
        
        return active_orders;
    }

    std::vector<Order> getOrderHistory() const {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        std::vector<Order> history;
        
        for (const auto& pair : orders_) {
            history.push_back(pair.second);
        }
        
        return history;
    }

    void addCallback(std::shared_ptr<IOrderCallback> callback) {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        callbacks_.push_back(callback);
    }

    void removeCallback(std::shared_ptr<IOrderCallback> callback) {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        callbacks_.erase(
            std::remove_if(callbacks_.begin(), callbacks_.end(),
                [&callback](const std::weak_ptr<IOrderCallback>& weak_cb) {
                    return weak_cb.lock() == callback;
                }),
            callbacks_.end()
        );
    }

    double getAverageProcessingTimeMs() const {
        uint64_t total_orders = orders_submitted_.load();
        if (total_orders == 0) return 0.0;
        
        return static_cast<double>(total_processing_time_us_.load()) / (total_orders * 1000.0);
    }

    void onOrderUpdate(const Order& order) {
        {
            std::lock_guard<std::mutex> lock(orders_mutex_);
            orders_[order.order_id] = order;
        }
        
        notifyCallbacks([&order](IOrderCallback* cb) {
            cb->onOrderUpdate(order);
        });
    }

    void onFill(const Fill& fill) {
        orders_filled_.fetch_add(1);
        
        // Update risk manager
        risk_manager_->updatePosition(fill);
        
        notifyCallbacks([&fill](IOrderCallback* cb) {
            cb->onFill(fill);
        });
    }

    void onOrderRejected(const Order& order, const std::string& reason) {
        orders_rejected_.fetch_add(1);
        
        notifyCallbacks([&order, &reason](IOrderCallback* cb) {
            cb->onOrderRejected(order, reason);
        });
    }

    void notifyCallbacks(std::function<void(IOrderCallback*)> func) {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (auto& weak_cb : callbacks_) {
            if (auto cb = weak_cb.lock()) {
                try {
                    func(cb.get());
                } catch (const std::exception& e) {
                    std::cerr << "Order Manager callback error: " << e.what() << std::endl;
                }
            }
        }
    }

    std::string generateOrderId() {
        static std::atomic<uint64_t> counter{0};
        uint64_t id = counter.fetch_add(1);
        std::stringstream ss;
        ss << "ORD_" << std::setfill('0') << std::setw(8) << id;
        return ss.str();
    }

private:
    std::shared_ptr<RiskManager> risk_manager_;
    std::shared_ptr<ITradingConnector> connector_;
    std::atomic<bool> running_;
    std::mutex orders_mutex_;
    std::mutex callbacks_mutex_;
    std::unordered_map<std::string, Order> orders_;
    std::atomic<uint64_t> orders_submitted_;
    std::atomic<uint64_t> orders_filled_;
    std::atomic<uint64_t> orders_rejected_;
    std::atomic<uint64_t> total_processing_time_us_;
    std::vector<std::weak_ptr<IOrderCallback>> callbacks_;
};

} // namespace quantx
