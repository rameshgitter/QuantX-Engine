#include "market_data_handler.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace quantx {

MarketDataHandler::MarketDataHandler() 
    : running_(false), connected_(false), messages_received_(0), messages_processed_(0) {
    
    // Configure WebSocket client
    client_.set_access_channels(websocketpp::log::alevel::all);
    client_.clear_access_channels(websocketpp::log::alevel::frame_payload);
    client_.set_error_channels(websocketpp::log::elevel::all);
    
    client_.init_asio();
    client_.set_reuse_addr(true);
    
    // Set handlers
    client_.set_open_handler([this](websocketpp::connection_hdl hdl) { onOpen(hdl); });
    client_.set_close_handler([this](websocketpp::connection_hdl hdl) { onClose(hdl); });
    client_.set_message_handler([this](websocketpp::connection_hdl hdl, MessagePtr msg) { onMessage(hdl, msg); });
    client_.set_fail_handler([this](websocketpp::connection_hdl hdl) { onFail(hdl); });
}

MarketDataHandler::~MarketDataHandler() {
    disconnect();
}

bool MarketDataHandler::connect(const std::string& url, const std::string& api_key) {
    if (connected_.load()) {
        std::cout << "Already connected to market data feed" << std::endl;
        return true;
    }
    
    ws_url_ = url;
    api_key_ = api_key;
    running_.store(true);
    
    try {
        websocketpp::lib::error_code ec;
        auto con = client_.get_connection(url, ec);
        
        if (ec) {
            std::cerr << "Connection creation error: " << ec.message() << std::endl;
            return false;
        }
        
        // Add headers for authentication
        con->append_header("Authorization", "Bearer " + api_key);
        con->append_header("User-Agent", "QuantX-Engine/1.0");
        
        connection_ = con->get_handle();
        client_.connect(con);
        
        // Start WebSocket thread
        ws_thread_ = std::thread([this]() {
            try {
                client_.run();
            } catch (const std::exception& e) {
                std::cerr << "WebSocket thread error: " << e.what() << std::endl;
            }
        });
        
        // Start message processing thread
        processing_thread_ = std::thread(&MarketDataHandler::processMessages, this);
        
        // Wait for connection (with timeout)
        auto start = std::chrono::steady_clock::now();
        while (!connected_.load() && running_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
                std::cerr << "Connection timeout" << std::endl;
                disconnect();
                return false;
            }
        }
        
        std::cout << "Connected to market data feed: " << url << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Connection error: " << e.what() << std::endl;
        return false;
    }
}

void MarketDataHandler::disconnect() {
    if (!running_.load()) {
        return;
    }
    
    std::cout << "Disconnecting from market data feed..." << std::endl;
    
    running_.store(false);
    connected_.store(false);
    
    try {
        if (connection_.lock()) {
            client_.close(connection_, websocketpp::close::status::going_away, "Client disconnect");
        }
        client_.stop();
    } catch (const std::exception& e) {
        std::cerr << "Disconnect error: " << e.what() << std::endl;
    }
    
    // Wake up processing thread
    queue_cv_.notify_all();
    
    if (ws_thread_.joinable()) {
        ws_thread_.join();
    }
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    std::cout << "Disconnected from market data feed" << std::endl;
}

bool MarketDataHandler::subscribe(const std::string& symbol) {
    if (!connected_.load()) {
        std::cerr << "Not connected to market data feed" << std::endl;
        return false;
    }
    
    try {
        nlohmann::json subscribe_msg = {
            {"action", "subscribe"},
            {"symbol", symbol},
            {"type", "orderbook"}
        };
        
        websocketpp::lib::error_code ec;
        client_.send(connection_, subscribe_msg.dump(), websocketpp::frame::opcode::text, ec);
        
        if (ec) {
            std::cerr << "Subscribe error: " << ec.message() << std::endl;
            return false;
        }
        
        std::cout << "Subscribed to " << symbol << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Subscribe error: " << e.what() << std::endl;
        return false;
    }
}

bool MarketDataHandler::unsubscribe(const std::string& symbol) {
    if (!connected_.load()) {
        return false;
    }
    
    try {
        nlohmann::json unsubscribe_msg = {
            {"action", "unsubscribe"},
            {"symbol", symbol}
        };
        
        websocketpp::lib::error_code ec;
        client_.send(connection_, unsubscribe_msg.dump(), websocketpp::frame::opcode::text, ec);
        
        if (ec) {
            std::cerr << "Unsubscribe error: " << ec.message() << std::endl;
            return false;
        }
        
        // Remove from local order books
        {
            std::lock_guard<std::mutex> lock(order_books_mutex_);
            order_books_.erase(symbol);
        }
        
        std::cout << "Unsubscribed from " << symbol << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Unsubscribe error: " << e.what() << std::endl;
        return false;
    }
}

void MarketDataHandler::addCallback(std::shared_ptr<IMarketDataCallback> callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.push_back(callback);
}

void MarketDataHandler::removeCallback(std::shared_ptr<IMarketDataCallback> callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.erase(
        std::remove_if(callbacks_.begin(), callbacks_.end(),
            [&callback](const std::weak_ptr<IMarketDataCallback>& weak_cb) {
                return weak_cb.lock() == callback;
            }),
        callbacks_.end()
    );
}

OrderBook MarketDataHandler::getOrderBook(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(order_books_mutex_);
    auto it = order_books_.find(symbol);
    return (it != order_books_.end()) ? it->second : OrderBook{};
}

std::vector<std::string> MarketDataHandler::getSubscribedSymbols() const {
    std::lock_guard<std::mutex> lock(order_books_mutex_);
    std::vector<std::string> symbols;
    for (const auto& pair : order_books_) {
        symbols.push_back(pair.first);
    }
    return symbols;
}

double MarketDataHandler::getProcessingLatencyMs() const {
    uint64_t received = messages_received_.load();
    uint64_t processed = messages_processed_.load();
    
    if (received == 0) return 0.0;
    
    // Simple approximation - in real implementation, track actual latencies
    return (received - processed) * 0.1; // Assume 0.1ms per queued message
}

void MarketDataHandler::onOpen(websocketpp::connection_hdl hdl) {
    std::cout << "WebSocket connection opened" << std::endl;
    connected_.store(true);
    last_heartbeat_ = std::chrono::high_resolution_clock::now();
    
    notifyCallbacks([](IMarketDataCallback* cb) {
        cb->onConnectionStatus(true);
    });
}

void MarketDataHandler::onClose(websocketpp::connection_hdl hdl) {
    std::cout << "WebSocket connection closed" << std::endl;
    connected_.store(false);
    
    notifyCallbacks([](IMarketDataCallback* cb) {
        cb->onConnectionStatus(false);
    });
}

void MarketDataHandler::onMessage(websocketpp::connection_hdl hdl, MessagePtr msg) {
    messages_received_.fetch_add(1);
    
    try {
        auto json_msg = nlohmann::json::parse(msg->get_payload());
        
        // Add to processing queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            message_queue_.push(json_msg);
        }
        queue_cv_.notify_one();
        
    } catch (const std::exception& e) {
        std::cerr << "Message parsing error: " << e.what() << std::endl;
    }
}

void MarketDataHandler::onFail(websocketpp::connection_hdl hdl) {
    std::cerr << "WebSocket connection failed" << std::endl;
    connected_.store(false);
    
    notifyCallbacks([](IMarketDataCallback* cb) {
        cb->onConnectionStatus(false);
    });
}

void MarketDataHandler::processMessages() {
    while (running_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return !message_queue_.empty() || !running_.load(); });
        
        while (!message_queue_.empty() && running_.load()) {
            auto msg = message_queue_.front();
            message_queue_.pop();
            lock.unlock();
            
            try {
                std::string msg_type = msg.value("type", "");
                
                if (msg_type == "orderbook") {
                    processOrderBookUpdate(msg);
                } else if (msg_type == "trade") {
                    processTradeUpdate(msg);
                } else if (msg_type == "market_data") {
                    processMarketDataUpdate(msg);
                } else if (msg_type == "heartbeat") {
                    last_heartbeat_ = std::chrono::high_resolution_clock::now();
                }
                
                messages_processed_.fetch_add(1);
                
            } catch (const std::exception& e) {
                std::cerr << "Message processing error: " << e.what() << std::endl;
            }
            
            lock.lock();
        }
    }
}

void MarketDataHandler::processOrderBookUpdate(const nlohmann::json& data) {
    try {
        std::string symbol = data["symbol"];
        OrderBook book;
        book.symbol = symbol;
        book.timestamp = std::chrono::high_resolution_clock::now();
        
        // Parse bids
        if (data.contains("bids")) {
            for (const auto& bid : data["bids"]) {
                book.bids.emplace_back(bid[0], bid[1], bid.value(2, 1));
            }
        }
        
        // Parse asks
        if (data.contains("asks")) {
            for (const auto& ask : data["asks"]) {
                book.asks.emplace_back(ask[0], ask[1], ask.value(2, 1));
            }
        }
        
        book.updateMidPrice();
        
        // Update local order book
        {
            std::lock_guard<std::mutex> lock(order_books_mutex_);
            order_books_[symbol] = book;
        }
        
        // Notify callbacks
        notifyCallbacks([&book](IMarketDataCallback* cb) {
            cb->onOrderBookUpdate(book);
        });
        
    } catch (const std::exception& e) {
        std::cerr << "Order book processing error: " << e.what() << std::endl;
    }
}

void MarketDataHandler::processTradeUpdate(const nlohmann::json& data) {
    try {
        Trade trade;
        trade.symbol = data["symbol"];
        trade.price = data["price"];
        trade.quantity = data["quantity"];
        trade.side = data["side"];
        trade.timestamp = std::chrono::high_resolution_clock::now();
        
        notifyCallbacks([&trade](IMarketDataCallback* cb) {
            cb->onTradeUpdate(trade);
        });
        
    } catch (const std::exception& e) {
        std::cerr << "Trade processing error: " << e.what() << std::endl;
    }
}

void MarketDataHandler::processMarketDataUpdate(const nlohmann::json& data) {
    try {
        MarketData market_data;
        market_data.symbol = data["symbol"];
        market_data.last_price = data["last_price"];
        market_data.volume = data.value("volume", 0.0);
        market_data.high = data.value("high", 0.0);
        market_data.low = data.value("low", 0.0);
        market_data.open = data.value("open", 0.0);
        market_data.close = data.value("close", 0.0);
        market_data.timestamp = std::chrono::high_resolution_clock::now();
        
        notifyCallbacks([&market_data](IMarketDataCallback* cb) {
            cb->onMarketDataUpdate(market_data);
        });
        
    } catch (const std::exception& e) {
        std::cerr << "Market data processing error: " << e.what() << std::endl;
    }
}

void MarketDataHandler::notifyCallbacks(std::function<void(IMarketDataCallback*)> func) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    for (auto& weak_cb : callbacks_) {
        if (auto cb = weak_cb.lock()) {
            try {
                func(cb.get());
            } catch (const std::exception& e) {
                std::cerr << "Callback error: " << e.what() << std::endl;
            }
        }
    }
}

} // namespace quantx
