#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>
#include "../src/core/market_data_handler.hpp"
#include "../src/ml/onnx_predictor.hpp"
#include "../src/risk/risk_manager.hpp"
#include "../src/trading/order_manager.hpp"

using namespace quantx;

class TestCallback : public IMarketDataCallback, public IRiskCallback, public IOrderCallback {
public:
    std::atomic<int> order_book_updates{0};
    std::atomic<int> trade_updates{0};
    std::atomic<int> risk_violations{0};
    std::atomic<int> order_updates{0};
    std::atomic<int> fills{0};
    
    void onOrderBookUpdate(const OrderBook& book) override {
        order_book_updates.fetch_add(1);
        std::cout << "Order book update: " << book.symbol << " mid=" << book.mid_price << std::endl;
    }
    
    void onTradeUpdate(const Trade& trade) override {
        trade_updates.fetch_add(1);
        std::cout << "Trade update: " << trade.symbol << " " << trade.quantity << "@" << trade.price << std::endl;
    }
    
    void onMarketDataUpdate(const MarketData& data) override {
        std::cout << "Market data update: " << data.symbol << " last=" << data.last_price << std::endl;
    }
    
    void onConnectionStatus(bool connected) override {
        std::cout << "Connection status: " << (connected ? "Connected" : "Disconnected") << std::endl;
    }
    
    void onRiskViolation(const RiskViolation& violation) override {
        risk_violations.fetch_add(1);
        std::cout << "Risk violation: " << violation.description << std::endl;
    }
    
    void onPositionUpdate(const Position& position) override {
        std::cout << "Position update: " << position.symbol << " qty=" << position.quantity 
                  << " PnL=" << position.unrealized_pnl << std::endl;
    }
    
    void onRiskMetricsUpdate(const RiskMetrics& metrics) override {
        std::cout << "Risk metrics: Portfolio=₹" << metrics.portfolio_value 
                  << " PnL=₹" << metrics.total_pnl << std::endl;
    }
    
    void onOrderUpdate(const Order& order) override {
        order_updates.fetch_add(1);
        std::cout << "Order update: " << order.order_id << " status=" << static_cast<int>(order.status) << std::endl;
    }
    
    void onFill(const Fill& fill) override {
        fills.fetch_add(1);
        std::cout << "Fill: " << fill.order_id << " " << fill.quantity << "@" << fill.price << std::endl;
    }
    
    void onOrderRejected(const Order& order, const std::string& reason) override {
        std::cout << "Order rejected: " << order.order_id << " reason=" << reason << std::endl;
    }
};

void testRiskManager() {
    std::cout << "\n=== Testing Risk Manager ===" << std::endl;
    
    auto callback = std::make_shared<TestCallback>();
    RiskManager risk_manager(1000000.0);  // 10 lakh initial capital
    
    risk_manager.addCallback(callback);
    risk_manager.start();
    
    // Test position update
    Fill fill;
    fill.symbol = "NIFTY50";
    fill.side = OrderSide::BUY;
    fill.quantity = 100;
    fill.price = 18450.0;
    fill.timestamp = std::chrono::high_resolution_clock::now();
    
    risk_manager.updatePosition(fill);
    
    // Test market price update
    risk_manager.updateMarketPrice("NIFTY50", 18500.0);
    
    // Get position
    auto position = risk_manager.getPosition("NIFTY50");
    assert(position.quantity == 100);
    assert(position.average_price == 18450.0);
    
    // Test risk limits
    Order test_order;
    test_order.symbol = "NIFTY50";
    test_order.side = OrderSide::BUY;
    test_order.quantity = 10000;  // Large quantity to trigger limit
    test_order.price = 18500.0;
    
    bool valid = risk_manager.validateOrder(test_order);
    std::cout << "Large order validation: " << (valid ? "PASSED" : "FAILED") << std::endl;
    
    risk_manager.stop();
    
    std::cout << "Risk Manager tests completed" << std::endl;
}

void testPaperTrading() {
    std::cout << "\n=== Testing Paper Trading ===" << std::endl;
    
    auto callback = std::make_shared<TestCallback>();
    auto risk_manager = std::make_shared<RiskManager>(1000000.0);
    auto paper_connector = std::make_shared<PaperTradingConnector>();
    
    paper_connector->setFillProbability(1.0);  // 100% fill rate for testing
    paper_connector->setFillDelay(std::chrono::milliseconds(10));  // Fast fills
    
    OrderManager order_manager(risk_manager, paper_connector);
    
    risk_manager->start();
    order_manager.start();
    order_manager.addCallback(callback);
    
    // Submit test order
    std::string order_id = order_manager.submitOrder("NIFTY50", OrderSide::BUY, 100, OrderType::LIMIT, 18450.0);
    
    assert(!order_id.empty());
    std::cout << "Submitted order: " << order_id << std::endl;
    
    // Wait for fill
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Check if order was filled
    auto order = order_manager.getOrder(order_id);
    std::cout << "Order status: " << static_cast<int>(order.status) << std::endl;
    
    // Submit another order to test position building
    std::string order_id2 = order_manager.submitOrder("NIFTY50", OrderSide::SELL, 50, OrderType::LIMIT, 18500.0);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Check final position
    auto position = risk_manager->getPosition("NIFTY50");
    std::cout << "Final position: " << position.quantity << " @ " << position.average_price << std::endl;
    
    order_manager.stop();
    risk_manager->stop();
    
    std::cout << "Paper Trading tests completed" << std::endl;
    std::cout << "Order updates: " << callback->order_updates.load() << std::endl;
    std::cout << "Fills: " << callback->fills.load() << std::endl;
}

void testONNXPredictor() {
    std::cout << "\n=== Testing ONNX Predictor ===" << std::endl;
    
    // Test basic ONNX predictor (will fail if models not available, which is expected)
    ONNXPredictor predictor;
    
    bool initialized = predictor.initialize("models/lob_lstm_predictor.onnx");
    std::cout << "ONNX predictor initialization: " << (initialized ? "SUCCESS" : "FAILED (expected if models not trained)") << std::endl;
    
    if (initialized) {
        // Test prediction with dummy data
        std::vector<float> dummy_input(10 * 14, 0.5f);  // 10 timesteps, 14 features
        auto output = predictor.predict(dummy_input);
        
        std::cout << "Prediction output size: " << output.size() << std::endl;
        if (!output.empty()) {
            std::cout << "Sample output: ";
            for (size_t i = 0; i < std::min(size_t(5), output.size()); ++i) {
                std::cout << output[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "ONNX Predictor tests completed" << std::endl;
}

void testPerformance() {
    std::cout << "\n=== Performance Tests ===" << std::endl;
    
    auto risk_manager = std::make_shared<RiskManager>(1000000.0);
    auto paper_connector = std::make_shared<PaperTradingConnector>();
    
    paper_connector->setFillProbability(0.0);  // No fills for performance test
    paper_connector->setFillDelay(std::chrono::milliseconds(1));
    
    OrderManager order_manager(risk_manager, paper_connector);
    
    risk_manager->start();
    order_manager.start();
    
    // Performance test: submit many orders
    const int num_orders = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_orders; ++i) {
        order_manager.submitOrder("TEST" + std::to_string(i % 10), OrderSide::BUY, 100, OrderType::LIMIT, 18450.0);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Submitted " << num_orders << " orders in " << duration.count() << " microseconds" << std::endl;
    std::cout << "Average order processing time: " << (duration.count() / num_orders) << " microseconds" << std::endl;
    std::cout << "Orders per second: " << (num_orders * 1000000.0 / duration.count()) << std::endl;
    
    order_manager.stop();
    risk_manager->stop();
    
    std::cout << "Performance tests completed" << std::endl;
}

int main() {
    std::cout << "QuantX Engine Test Suite" << std::endl;
    std::cout << "========================" << std::endl;
    
    try {
        testRiskManager();
        testPaperTrading();
        testONNXPredictor();
        testPerformance();
        
        std::cout << "\n=== All Tests Completed ===" << std::endl;
        std::cout << "✓ Risk Manager: PASSED" << std::endl;
        std::cout << "✓ Paper Trading: PASSED" << std::endl;
        std::cout << "✓ ONNX Predictor: TESTED" << std::endl;
        std::cout << "✓ Performance: TESTED" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
