-- QuantX Engine Database Schema
-- Creates tables for storing trading data, signals, and performance metrics

-- Create database
CREATE DATABASE IF NOT EXISTS quantx_engine;
USE quantx_engine;

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP(6) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    volume INT NOT NULL,
    bid_price DECIMAL(10,2),
    ask_price DECIMAL(10,2),
    spread DECIMAL(8,4),
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_timestamp (timestamp)
);

-- Order book snapshots
CREATE TABLE IF NOT EXISTS order_book_snapshots (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP(6) NOT NULL,
    level_number TINYINT NOT NULL,
    side ENUM('BID', 'ASK') NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    quantity INT NOT NULL,
    order_count INT DEFAULT 1,
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_timestamp_side (timestamp, side)
);

-- Trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP(6) NOT NULL,
    signal_type ENUM('LOB', 'IV', 'COMBINED') NOT NULL,
    signal_value DECIMAL(8,4) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    direction ENUM('BUY', 'SELL', 'NEUTRAL') NOT NULL,
    features JSON,
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_signal_type (signal_type),
    INDEX idx_timestamp (timestamp)
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side ENUM('BUY', 'SELL') NOT NULL,
    order_type ENUM('MARKET', 'LIMIT', 'STOP') NOT NULL,
    quantity INT NOT NULL,
    price DECIMAL(10,2),
    status ENUM('PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED') NOT NULL,
    created_at TIMESTAMP(6) NOT NULL,
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    filled_quantity INT DEFAULT 0,
    avg_fill_price DECIMAL(10,2),
    INDEX idx_symbol (symbol),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Order fills/executions
CREATE TABLE IF NOT EXISTS order_fills (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL,
    fill_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side ENUM('BUY', 'SELL') NOT NULL,
    quantity INT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    timestamp TIMESTAMP(6) NOT NULL,
    fees DECIMAL(10,2) DEFAULT 0,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    INDEX idx_order_id (order_id),
    INDEX idx_timestamp (timestamp)
);

-- Portfolio positions
CREATE TABLE IF NOT EXISTS positions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    quantity INT NOT NULL DEFAULT 0,
    avg_cost DECIMAL(10,2) NOT NULL DEFAULT 0,
    market_value DECIMAL(12,2),
    unrealized_pnl DECIMAL(12,2),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    INDEX idx_symbol (symbol)
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    total_pnl DECIMAL(12,2) NOT NULL,
    realized_pnl DECIMAL(12,2) NOT NULL,
    unrealized_pnl DECIMAL(12,2) NOT NULL,
    total_trades INT NOT NULL DEFAULT 0,
    winning_trades INT NOT NULL DEFAULT 0,
    losing_trades INT NOT NULL DEFAULT 0,
    max_drawdown DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    portfolio_value DECIMAL(15,2),
    INDEX idx_date (date)
);

-- System metrics for monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP(6) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(12,4) NOT NULL,
    metric_unit VARCHAR(20),
    INDEX idx_timestamp_metric (timestamp, metric_name),
    INDEX idx_metric_name (metric_name)
);

-- IV surface data
CREATE TABLE IF NOT EXISTS iv_surface (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP(6) NOT NULL,
    expiry_date DATE NOT NULL,
    strike_price DECIMAL(10,2) NOT NULL,
    implied_volatility DECIMAL(8,6) NOT NULL,
    delta_value DECIMAL(8,6),
    gamma_value DECIMAL(8,6),
    theta_value DECIMAL(8,6),
    vega_value DECIMAL(8,6),
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_expiry_strike (expiry_date, strike_price)
);

-- Backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    backtest_id VARCHAR(50) UNIQUE NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2) NOT NULL,
    final_capital DECIMAL(15,2) NOT NULL,
    total_return DECIMAL(8,4) NOT NULL,
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    win_rate DECIMAL(5,4),
    total_trades INT NOT NULL,
    avg_trade_pnl DECIMAL(10,2),
    volatility DECIMAL(8,4),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_strategy (strategy_name),
    INDEX idx_created_at (created_at)
);

-- Risk metrics
CREATE TABLE IF NOT EXISTS risk_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP(6) NOT NULL,
    portfolio_var DECIMAL(12,2),
    portfolio_cvar DECIMAL(12,2),
    beta DECIMAL(8,4),
    correlation DECIMAL(8,4),
    max_position_size DECIMAL(12,2),
    leverage DECIMAL(8,4),
    INDEX idx_timestamp (timestamp)
);

-- Create views for common queries
CREATE VIEW daily_pnl AS
SELECT 
    DATE(timestamp) as trade_date,
    SUM(CASE WHEN side = 'SELL' THEN quantity * price ELSE -quantity * price END) as daily_pnl,
    COUNT(*) as trade_count
FROM order_fills 
GROUP BY DATE(timestamp)
ORDER BY trade_date DESC;

CREATE VIEW current_positions AS
SELECT 
    p.symbol,
    p.quantity,
    p.avg_cost,
    p.market_value,
    p.unrealized_pnl,
    CASE 
        WHEN p.quantity > 0 THEN 'LONG'
        WHEN p.quantity < 0 THEN 'SHORT'
        ELSE 'FLAT'
    END as position_type
FROM positions p
WHERE p.quantity != 0;

-- Insert initial data
INSERT INTO performance_metrics (date, total_pnl, realized_pnl, unrealized_pnl, total_trades, winning_trades, losing_trades, portfolio_value)
VALUES (CURDATE(), 0, 0, 0, 0, 0, 0, 1000000);

-- Create stored procedures for common operations
DELIMITER //

CREATE PROCEDURE UpdatePosition(
    IN p_symbol VARCHAR(20),
    IN p_quantity INT,
    IN p_price DECIMAL(10,2)
)
BEGIN
    DECLARE current_qty INT DEFAULT 0;
    DECLARE current_avg_cost DECIMAL(10,2) DEFAULT 0;
    DECLARE new_qty INT;
    DECLARE new_avg_cost DECIMAL(10,2);
    
    -- Get current position
    SELECT quantity, avg_cost INTO current_qty, current_avg_cost
    FROM positions WHERE symbol = p_symbol;
    
    -- Calculate new position
    SET new_qty = current_qty + p_quantity;
    
    IF new_qty = 0 THEN
        SET new_avg_cost = 0;
    ELSE
        SET new_avg_cost = ((current_qty * current_avg_cost) + (p_quantity * p_price)) / new_qty;
    END IF;
    
    -- Update or insert position
    INSERT INTO positions (symbol, quantity, avg_cost, updated_at)
    VALUES (p_symbol, new_qty, new_avg_cost, NOW(6))
    ON DUPLICATE KEY UPDATE
        quantity = new_qty,
        avg_cost = new_avg_cost,
        updated_at = NOW(6);
END //

CREATE PROCEDURE CalculateDailyMetrics()
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE total_pnl DECIMAL(12,2) DEFAULT 0;
    DECLARE realized_pnl DECIMAL(12,2) DEFAULT 0;
    DECLARE unrealized_pnl DECIMAL(12,2) DEFAULT 0;
    DECLARE trade_count INT DEFAULT 0;
    DECLARE win_count INT DEFAULT 0;
    DECLARE loss_count INT DEFAULT 0;
    
    -- Calculate realized P&L for today
    SELECT 
        COALESCE(SUM(CASE WHEN side = 'SELL' THEN quantity * price ELSE -quantity * price END), 0),
        COUNT(*)
    INTO realized_pnl, trade_count
    FROM order_fills 
    WHERE DATE(timestamp) = CURDATE();
    
    -- Calculate unrealized P&L from current positions
    SELECT COALESCE(SUM(unrealized_pnl), 0) INTO unrealized_pnl FROM positions;
    
    -- Total P&L
    SET total_pnl = realized_pnl + unrealized_pnl;
    
    -- Count winning/losing trades (simplified)
    SELECT 
        COUNT(CASE WHEN quantity * price > 0 THEN 1 END),
        COUNT(CASE WHEN quantity * price < 0 THEN 1 END)
    INTO win_count, loss_count
    FROM order_fills 
    WHERE DATE(timestamp) = CURDATE();
    
    -- Update performance metrics
    INSERT INTO performance_metrics (
        date, total_pnl, realized_pnl, unrealized_pnl, 
        total_trades, winning_trades, losing_trades, portfolio_value
    )
    VALUES (
        CURDATE(), total_pnl, realized_pnl, unrealized_pnl,
        trade_count, win_count, loss_count, 1000000 + total_pnl
    )
    ON DUPLICATE KEY UPDATE
        total_pnl = VALUES(total_pnl),
        realized_pnl = VALUES(realized_pnl),
        unrealized_pnl = VALUES(unrealized_pnl),
        total_trades = VALUES(total_trades),
        winning_trades = VALUES(winning_trades),
        losing_trades = VALUES(losing_trades),
        portfolio_value = VALUES(portfolio_value);
END //

DELIMITER ;

-- Create indexes for performance
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_order_fills_timestamp ON order_fills(timestamp DESC);
CREATE INDEX idx_trading_signals_timestamp ON trading_signals(timestamp DESC);

COMMIT;
