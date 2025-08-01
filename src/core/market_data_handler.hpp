#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/client.hpp>
#include <nlohmann/json.hpp>

namespace quantx {

struct PriceLevel {
    double price;
    int quantity;
    int orders;
    
    PriceLevel(double p = 0.0, int q = 0, int o = 0) : price(p), quantity(q), orders(o) {}
};

struct OrderBook {
    std::string symbol;
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
    double mid_price;
    double spread;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    OrderBook() : mid_price(0.0), spread(0.0) {}
    
    void updateMidPrice() {
        if (!bids.empty() && !asks.empty()) {
            mid_price = (bids[0].price + asks[0].price) / 2.0;
            spread = asks[0].price - bids[0].price;
        }
    }
};

struct Trade {
    std::string symbol;
    double price;
    int quantity;
    std::string side; // "BUY" or "SELL"
    std::chrono::high_resolution_clock::time_point timestamp;
    
    Trade() : price(0.0), quantity(0) {}
};

struct MarketData {
    std::string symbol;
    double last_price;
    double volume;
    double high;
    double low;
    double open;
    double close;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    MarketData() : last_price(0.0), volume(0.0), high(0.0), low(0.0), open(0.0), close(0.0) {}
};

class IMarketDataCallback {
public:
    virtual ~IMarketDataCallback() = default;
    virtual void onOrderBookUpdate(const OrderBook& book) = 0;
    virtual void onTradeUpdate(const Trade& trade) = 0;
    virtual void onMarketDataUpdate(const MarketData& data) = 0;
    virtual void onConnectionStatus(bool connected) = 0;
};

class MarketDataHandler {
private:
    using WebSocketClient = websocketpp::client<websocketpp::config::asio>;
    using MessagePtr = WebSocketClient::message_ptr;
    using ConnectionHdl = websocketpp::connection_hdl;
    
    WebSocketClient client_;
    ConnectionHdl connection_;
    std::thread ws_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> connected_;
    
    std::string ws_url_;
    std::string api_key_;
    
    std::vector<std::shared_ptr<IMarketDataCallback>> callbacks_;
    std::mutex callbacks_mutex_;
    
    std::unordered_map<std::string, OrderBook> order_books_;
    std::mutex order_books_mutex_;
    
    std::queue<nlohmann::json> message_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread processing_thread_;
    
    // Performance metrics
    std::atomic<uint64_t> messages_received_;
    std::atomic<uint64_t> messages_processed_;
    std::chrono::high_resolution_clock::time_point last_heartbeat_;
    
public:
    MarketDataHandler();
    ~MarketDataHandler();
    
    bool connect(const std::string& url, const std::string& api_key);
    void disconnect();
    bool isConnected() const { return connected_.load(); }
    
    bool subscribe(const std::string& symbol);
    bool unsubscribe(const std::string& symbol);
    
    void addCallback(std::shared_ptr<IMarketDataCallback> callback);
    void removeCallback(std::shared_ptr<IMarketDataCallback> callback);
    
    OrderBook getOrderBook(const std::string& symbol) const;
    std::vector<std::string> getSubscribedSymbols() const;
    
    // Performance metrics
    uint64_t getMessagesReceived() const { return messages_received_.load(); }
    uint64_t getMessagesProcessed() const { return messages_processed_.load(); }
    double getProcessingLatencyMs() const;
    
private:
    void onOpen(ConnectionHdl hdl);
    void onClose(ConnectionHdl hdl);
    void onMessage(ConnectionHdl hdl, MessagePtr msg);
    void onFail(ConnectionHdl hdl);
    
    void processMessages();
    void processOrderBookUpdate(const nlohmann::json& data);
    void processTradeUpdate(const nlohmann::json& data);
    void processMarketDataUpdate(const nlohmann::json& data);
    
    void notifyCallbacks(std::function<void(IMarketDataCallback*)> func);
    
    void sendHeartbeat();
    void checkConnection();
};

} // namespace quantx
