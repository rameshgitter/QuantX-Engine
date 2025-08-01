#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <signal.h>

#include "core/market_data_handler.hpp"
#include "ml/onnx_predictor.hpp"
#include "risk/risk_manager.hpp"
#include "trading/order_manager.hpp"

using namespace quantx;

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Shutting down gracefully..." << std::endl;
    g_running.store(false);
}

class QuantXEngine : public IMarketDataCallback, public IOrderCallback, public IRiskCallback {
private:
    // Core components
    std::shared_ptr<MarketDataHandler> market_data_handler_;
    std::shared_ptr<EnsemblePredictor> predictor_;
    std::shared_ptr<RiskManager> risk_manager_;
    std::shared_ptr<OrderManager> order_manager_;
    std::shared_ptr<PaperTradingConnector> paper_connector_;
    
    // Configuration
    std::string config_file_;
    std::vector<std::string> symbols_;
    
    // Performance metrics
    std::atomic<uint64_t> signals_generated_;
    std::atomic<uint64_t> orders_placed_;
    std::atomic<double> total_pnl_;
    
    // Main trading loop
    std::thread trading_thread_;
    
public:
    QuantXEngine(const std::string& config_file = "config.json") 
        : config_file_(config_file), signals_generated_(0), orders_placed_(0), total_pnl_(0.0) {
        
        // Initialize symbols
        symbols_ = {"NIFTY50", "BANKNIFTY", "RELIANCE", "TCS", "INFY"};
        
        std::cout << "QuantX Engine initializing..." << std::endl;
        
        // Initialize components
        initializeComponents();
    }
    
    ~QuantXEngine() {
        shutdown();
    }
    
    bool start() {
        std::cout << "Starting QuantX Engine..." << std::endl;
        
        // Start all components
        if (!market_data_handler_->connect("wss://api.example.com/ws", "your_api_key")) {
            std::cerr << "Failed to connect to market data feed" << std::endl;
            return false;
        }
        
        risk_manager_->start();
        order_manager_->start();
        
        // Subscribe to symbols
        for (const auto& symbol : symbols_) {
            market_data_handler_->subscribe(symbol);
        }
        
        // Start trading loop
        trading_thread_ = std::thread(&QuantXEngine::tradingLoop, this);
        
        std::cout << "QuantX Engine started successfully!" << std::endl;
        std::cout << "Monitoring " << symbols_.size() << " symbols" << std::endl;
        std::cout << "Press Ctrl+C to stop..." << std::endl;
        
        return true;
    }
    
    void shutdown() {
        std::cout << "Shutting down QuantX Engine..." << std::endl;
        
        g_running.store(false);
        
        if (trading_thread_.joinable()) {
            trading_thread_.join();
        }
        
        if (order_manager_) {
            order_manager_->stop();
        }
        
        if (risk_manager_) {
            risk_manager_->stop();
        }
        
        if (market_data_handler_) {
            market_data_handler_->disconnect();
        }
        
        // Print final statistics
        printStatistics();
        
        std::cout << "QuantX Engine shutdown complete." << std::endl;
    }
    
    // Market data callbacks
    void onOrderBookUpdate(const OrderBook& book) override {
        try {
            // Generate prediction
            auto prediction = predictor_->predictCombined(book, {});
            signals_generated_.fetch_add(1);
            
            // Check if signal is strong enough to trade
            if (std::abs(prediction.signal_strength) > 0.3 && prediction.confidence > 0.7) {
                
                // Determine order parameters
                OrderSide side = (prediction.direction > 0) ? OrderSide::BUY : OrderSide::SELL;
                int quantity = calculateOrderSize(book.symbol, prediction);
                
                if (quantity > 0) {
                    // Submit order
                    std::string order_id = order_manager_->submitOrder(
                        book.symbol, side, quantity, OrderType::LIMIT, 
                        calculateLimitPrice(book, side)
                    );
                    
                    if (!order_id.empty()) {
                        orders_placed_.fetch_add(1);
                        
                        std::cout << "Order placed: " << order_id 
                                  << " " << book.symbol 
                                  << " " << (side == OrderSide::BUY ? "BUY" : "SELL")
                                  << " " << quantity 
                                  << " (Signal: " << std::fixed << std::setprecision(3) 
                                  << prediction.signal_strength 
                                  << ", Conf: " << prediction.confidence << ")" << std::endl;
                    }
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing order book update: " << e.what() << std::endl;
        }
    }
    
    void onTradeUpdate(const Trade& trade) override {
        // Update any trade-based analytics
    }
    
    void onMarketDataUpdate(const MarketData& data) override {
        // Update market data for risk calculations
    }
    
    void onConnectionStatus(bool connected) override {
        if (connected) {
            std::cout << "Market data connection established" << std::endl;
        } else {
            std::cout << "Market data connection lost" << std::endl;
        }
    }
    
    // Order callbacks
    void onOrderUpdate(const Order& order) override {
        std::cout << "Order update: " << order.order_id 
                  << " " << static_cast<int>(order.status) << std::endl;
    }
    
    void onFill(const Fill& fill) override {
        std::cout << "Fill: " << fill.order_id 
                  << " " << fill.quantity << "@" << fill.price << std::endl;
        
        // Update P&L tracking
        double pnl_impact = (fill.side == OrderSide::SELL ? 1 : -1) * 
                           fill.quantity * fill.price;
        total_pnl_.fetch_add(pnl_impact);
    }
    
    void onOrderRejected(const Order& order, const std::string& reason) override {
        std::cout << "Order rejected: " << order.order_id 
                  << " Reason: " << reason << std::endl;
    }
    
    // Risk callbacks
    void onRiskViolation(const RiskViolation& violation) override {
        std::cout << "RISK VIOLATION: " << violation.description << std::endl;
        
        // Take appropriate action based on violation type
        if (violation.type == RiskViolationType::DAILY_LOSS_LIMIT ||
            violation.type == RiskViolationType::DRAWDOWN_LIMIT) {
            
            std::cout << "Emergency stop triggered due to risk violation!" << std::endl;
            risk_manager_->emergencyStop();
        }
    }
    
    void onPositionUpdate(const Position& position) override {
        // Log significant position changes
        if (std::abs(position.unrealized_pnl) > 1000) {
            std::cout << "Position update: " << position.symbol 
                      << " qty=" << position.quantity 
                      << " PnL=" << position.unrealized_pnl << std::endl;
        }
    }
    
    void onRiskMetricsUpdate(const RiskMetrics& metrics) override {
        // Periodically log risk metrics
        static auto last_log = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        
        if (now - last_log > std::chrono::minutes(5)) {
            std::cout << "Risk Metrics - Portfolio: ₹" << metrics.portfolio_value 
                      << " PnL: ₹" << metrics.total_pnl 
                      << " Drawdown: " << (metrics.current_drawdown * 100) << "%" << std::endl;
            last_log = now;
        }
    }
    
private:
    void initializeComponents() {
        // Initialize market data handler
        market_data_handler_ = std::make_shared<MarketDataHandler>();
        market_data_handler_->addCallback(shared_from_this());
        
        // Initialize ML predictor
        predictor_ = std::make_shared<EnsemblePredictor>();
        if (!predictor_->initialize(
            "models/lob_lstm_predictor.onnx",
            "models/iv_surface_predictor.onnx", 
            "models/ensemble_predictor.onnx")) {
            
            std::cout << "Warning: Could not load all ML models. Using fallback logic." << std::endl;
        }
        
        // Initialize risk manager
        risk_manager_ = std::make_shared<RiskManager>(1000000.0); // 10 lakh initial capital
        risk_manager_->addCallback(shared_from_this());
        
        // Set risk limits
        RiskLimits limits;
        limits.max_position_value = 100000.0;      // 1 lakh per position
        limits.max_portfolio_value = 2000000.0;    // 20 lakh max portfolio
        limits.max_daily_loss = 50000.0;           // 50k daily loss limit
        limits.max_drawdown = 0.15;                // 15% max drawdown
        limits.var_limit = 30000.0;                // 30k VaR limit
        limits.leverage_limit = 2.0;               // 2x leverage
        limits.max_orders_per_second = 5;
        limits.max_orders_per_minute = 50;
        
        risk_manager_->setRiskLimits(limits);
        
        // Initialize paper trading connector
        paper_connector_ = std::make_shared<PaperTradingConnector>();
        paper_connector_->setFillProbability(0.95);
        paper_connector_->setSlippageFactor(0.0002); // 2 bps slippage
        paper_connector_->setFillDelay(std::chrono::milliseconds(100));
        
        // Initialize order manager
        order_manager_ = std::make_shared<OrderManager>(risk_manager_, paper_connector_);
        order_manager_->addCallback(shared_from_this());
    }
    
    void tradingLoop() {
        std::cout << "Trading loop started" << std::endl;
        
        while (g_running.load()) {
            try {
                // Perform periodic tasks
                performPeriodicTasks();
                
                // Sleep for a short interval
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
            } catch (const std::exception& e) {
                std::cerr << "Trading loop error: " << e.what() << std::endl;
            }
        }
        
        std::cout << "Trading loop stopped" << std::endl;
    }
    
    void performPeriodicTasks() {
        static auto last_stats = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        
        // Print statistics every minute
        if (now - last_stats > std::chrono::minutes(1)) {
            printPeriodicStatistics();
            last_stats = now;
        }
        
        // Check for any emergency conditions
        checkEmergencyConditions();
    }
    
    void checkEmergencyConditions() {
        // Check if market data connection is lost
        if (!market_data_handler_->isConnected()) {
            static auto last_warning = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            
            if (now - last_warning > std::chrono::minutes(1)) {
                std::cout << "Warning: Market data connection lost" << std::endl;
                last_warning = now;
            }
        }
        
        // Check system performance
        auto risk_metrics = risk_manager_->getCurrentMetrics();
        if (risk_metrics.current_drawdown > 0.10) { // 10% drawdown warning
            std::cout << "Warning: High drawdown detected: " 
                      << (risk_metrics.current_drawdown * 100) << "%" << std::endl;
        }
    }
    
    int calculateOrderSize(const std::string& symbol, const PredictionResult& prediction) {
        // Simple position sizing based on signal strength and confidence
        double base_size = 100; // Base order size
        double size_multiplier = prediction.confidence * std::abs(prediction.signal_strength);
        
        int quantity = static_cast<int>(base_size * size_multiplier);
        
        // Cap at reasonable limits
        return std::min(quantity, 500);
    }
    
    double calculateLimitPrice(const OrderBook& book, OrderSide side) {
        if (book.bids.empty() || book.asks.empty()) {
            return book.mid_price;
        }
        
        // Place limit orders slightly inside the spread
        if (side == OrderSide::BUY) {
            return book.bids[0].price + 0.05; // 5 paisa above best bid
        } else {
            return book.asks[0].price - 0.05; // 5 paisa below best ask
        }
    }
    
    void printPeriodicStatistics() {
        auto risk_metrics = risk_manager_->getCurrentMetrics();
        
        std::cout << "\n=== QuantX Engine Statistics ===" << std::endl;
        std::cout << "Signals Generated: " << signals_generated_.load() << std::endl;
        std::cout << "Orders Placed: " << orders_placed_.load() << std::endl;
        std::cout << "Portfolio Value: ₹" << std::fixed << std::setprecision(2) 
                  << risk_metrics.portfolio_value << std::endl;
        std::cout << "Total P&L: ₹" << risk_metrics.total_pnl << std::endl;
        std::cout << "Current Drawdown: " << (risk_metrics.current_drawdown * 100) << "%" << std::endl;
        std::cout << "Leverage: " << risk_metrics.leverage << "x" << std::endl;
        std::cout << "================================\n" << std::endl;
    }
    
    void printStatistics() {
        std::cout << "\n=== Final Statistics ===" << std::endl;
        std::cout << "Total Signals Generated: " << signals_generated_.load() << std::endl;
        std::cout << "Total Orders Placed: " << orders_placed_.load() << std::endl;
        std::cout << "Final P&L: ₹" << total_pnl_.load() << std::endl;
        
        if (predictor_) {
            std::cout << "ML Predictions Made: " << predictor_->getRecentPredictions(1000).size() << std::endl;
        }
        
        if (risk_manager_) {
            std::cout << "Risk Checks Performed: " << risk_manager_->getRiskChecksPerformed() << std::endl;
            std::cout << "Risk Violations: " << risk_manager_->getViolationsDetected() << std::endl;
        }
        
        if (order_manager_) {
            std::cout << "Orders Submitted: " << order_manager_->getOrdersSubmitted() << std::endl;
            std::cout << "Orders Filled: " << order_manager_->getOrdersFilled() << std::endl;
            std::cout << "Orders Rejected: " << order_manager_->getOrdersRejected() << std::endl;
            std::cout << "Avg Processing Time: " << order_manager_->getAverageProcessingTimeMs() << "ms" << std::endl;
        }
        
        std::cout << "=========================" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    std::cout << "QuantX Engine - High-Frequency Trading Platform" << std::endl;
    std::cout << "Version 1.0.0" << std::endl;
    std::cout << "================================================" << std::endl;
    
    try {
        // Create and start the engine
        auto engine = std::make_shared<QuantXEngine>();
        
        if (!engine->start()) {
            std::cerr << "Failed to start QuantX Engine" << std::endl;
            return 1;
        }
        
        // Main loop - wait for shutdown signal
        while (g_running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        // Graceful shutdown
        engine->shutdown();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "QuantX Engine terminated successfully." << std::endl;
    return 0;
}
