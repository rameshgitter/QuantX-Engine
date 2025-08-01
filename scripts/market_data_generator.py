import asyncio
import json
import random
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class MarketDataGenerator:
    """
    High-frequency market data generator for QuantX Engine
    Simulates NSE/BSE order book data with realistic microstructure
    """
    
    def __init__(self, symbol="NIFTY50", base_price=18450.0):
        self.symbol = symbol
        self.base_price = base_price
        self.current_price = base_price
        self.tick_size = 0.05
        self.lot_size = 50
        
        # Market microstructure parameters
        self.spread_mean = 0.25
        self.spread_std = 0.15
        self.volume_mean = 500
        self.volume_std = 200
        
        # Price dynamics
        self.volatility = 0.15  # Annual volatility
        self.drift = 0.0  # No drift for intraday
        self.mean_reversion = 0.1
        
        # Order book levels
        self.book_depth = 10
        
    def generate_price_movement(self, dt=1.0):
        """Generate realistic price movement using mean-reverting process"""
        # Ornstein-Uhlenbeck process for mean reversion
        dt_scaled = dt / (252 * 6.5 * 3600)  # Convert to fraction of trading year
        
        # Mean reversion to base price
        drift_component = self.mean_reversion * (self.base_price - self.current_price) * dt_scaled
        
        # Random walk component
        random_component = self.volatility * np.sqrt(dt_scaled) * np.random.normal()
        
        # Jump component (rare large moves)
        if np.random.random() < 0.001:  # 0.1% chance of jump
            jump_component = np.random.choice([-1, 1]) * np.random.exponential(2.0)
        else:
            jump_component = 0
            
        price_change = drift_component + random_component + jump_component
        self.current_price += price_change
        
        # Round to tick size
        self.current_price = round(self.current_price / self.tick_size) * self.tick_size
        
        return self.current_price
    
    def generate_order_book(self):
        """Generate realistic order book with bid/ask levels"""
        # Generate spread
        spread = max(self.tick_size, np.random.normal(self.spread_mean, self.spread_std))
        spread = round(spread / self.tick_size) * self.tick_size
        
        mid_price = self.current_price
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # Generate bid levels
        bids = []
        for i in range(self.book_depth):
            price = best_bid - i * self.tick_size
            # Volume decreases with distance from best bid
            base_volume = max(50, np.random.normal(self.volume_mean, self.volume_std))
            volume_decay = np.exp(-i * 0.2)  # Exponential decay
            volume = int(base_volume * volume_decay)
            orders = max(1, int(volume / 100) + np.random.poisson(2))
            
            bids.append({
                'price': round(price, 2),
                'quantity': volume,
                'orders': orders
            })
        
        # Generate ask levels
        asks = []
        for i in range(self.book_depth):
            price = best_ask + i * self.tick_size
            base_volume = max(50, np.random.normal(self.volume_mean, self.volume_std))
            volume_decay = np.exp(-i * 0.2)
            volume = int(base_volume * volume_decay)
            orders = max(1, int(volume / 100) + np.random.poisson(2))
            
            asks.append({
                'price': round(price, 2),
                'quantity': volume,
                'orders': orders
            })
        
        return {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'bids': bids,
            'asks': asks,
            'spread': spread,
            'mid_price': mid_price
        }
    
    def generate_trade(self):
        """Generate a realistic trade"""
        # Trade occurs at bid or ask with some probability
        if np.random.random() < 0.5:
            # Buy trade (at ask)
            side = 'BUY'
            price = self.current_price + self.spread_mean / 2
        else:
            # Sell trade (at bid)
            side = 'SELL'
            price = self.current_price - self.spread_mean / 2
        
        # Trade size follows power law distribution
        size = int(np.random.pareto(1.5) * 100 + 50)
        size = min(size, 5000)  # Cap at reasonable size
        
        return {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'price': round(price, 2),
            'quantity': size,
            'side': side,
            'trade_id': f"T{int(time.time() * 1000)}"
        }
    
    def calculate_features(self, order_book):
        """Calculate LOB features for ML models"""
        bids = order_book['bids']
        asks = order_book['asks']
        
        if not bids or not asks:
            return {}
        
        # Basic features
        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Volume features
        bid_volume_1 = bids[0]['quantity']
        ask_volume_1 = asks[0]['quantity']
        
        # Top 5 levels volume
        bid_volume_5 = sum(level['quantity'] for level in bids[:5])
        ask_volume_5 = sum(level['quantity'] for level in asks[:5])
        
        # Imbalance features
        volume_imbalance = (bid_volume_5 - ask_volume_5) / (bid_volume_5 + ask_volume_5)
        
        # Weighted mid price (micro price)
        micro_price = (best_bid * ask_volume_1 + best_ask * bid_volume_1) / (bid_volume_1 + ask_volume_1)
        
        # Price impact estimation
        impact_bid = sum(level['quantity'] for level in bids[:3])
        impact_ask = sum(level['quantity'] for level in asks[:3])
        
        return {
            'spread': spread,
            'mid_price': mid_price,
            'micro_price': micro_price,
            'volume_imbalance': volume_imbalance,
            'bid_ask_ratio': bid_volume_1 / ask_volume_1 if ask_volume_1 > 0 else 1,
            'depth_imbalance': bid_volume_5 / ask_volume_5 if ask_volume_5 > 0 else 1,
            'price_impact_bid': impact_bid,
            'price_impact_ask': impact_ask,
            'relative_spread': spread / mid_price if mid_price > 0 else 0
        }

async def main():
    """Main function to generate and stream market data"""
    generator = MarketDataGenerator()
    
    print("Starting QuantX Market Data Generator...")
    print(f"Symbol: {generator.symbol}")
    print(f"Base Price: ₹{generator.base_price}")
    print("-" * 50)
    
    try:
        while True:
            # Generate new price
            new_price = generator.generate_price_movement(dt=1.0)
            
            # Generate order book
            order_book = generator.generate_order_book()
            
            # Calculate features
            features = generator.calculate_features(order_book)
            
            # Generate trade (10% probability each second)
            trade = None
            if np.random.random() < 0.1:
                trade = generator.generate_trade()
            
            # Output data
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] Price: ₹{new_price:.2f} | "
                  f"Spread: ₹{features.get('spread', 0):.2f} | "
                  f"Imbalance: {features.get('volume_imbalance', 0):.3f}")
            
            if trade:
                print(f"  → TRADE: {trade['side']} {trade['quantity']} @ ₹{trade['price']:.2f}")
            
            # Save to file (optional)
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'price': new_price,
                'order_book': order_book,
                'features': features,
                'trade': trade
            }
            
            # In production, this would be sent to the C++ engine via shared memory
            # For now, we'll just simulate the data generation
            
            await asyncio.sleep(1)  # 1 second intervals
            
    except KeyboardInterrupt:
        print("\nShutting down market data generator...")

if __name__ == "__main__":
    asyncio.run(main())
