import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
import asyncio
from datetime import datetime

@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str  # 'MARKET' or 'LIMIT'
    price: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class Fill:
    """Represents an order fill"""
    price: float
    quantity: int
    timestamp: datetime
    fees: float = 0.0

class CostSimulator:
    """
    Real-time cost and market impact simulator
    Estimates slippage, impact, and transaction costs
    """
    
    def __init__(self):
        # Market parameters
        self.tick_size = 0.05
        self.lot_size = 50
        
        # Cost model parameters
        self.brokerage_rate = 0.0003  # 0.03% per trade
        self.exchange_fee = 0.00002   # 0.002% per trade
        self.stt_rate = 0.001         # 0.1% on sell side
        self.gst_rate = 0.18          # 18% on brokerage + exchange fees
        
        # Market impact parameters
        self.temporary_impact_coeff = 0.1
        self.permanent_impact_coeff = 0.05
        self.liquidity_factor = 1000000  # Market liquidity in notional
        
        # Slippage parameters
        self.base_slippage = 0.0001  # 1 bps base slippage
        self.volume_slippage_coeff = 0.5
        
    def calculate_transaction_costs(self, order: Order, fill_price: float) -> Dict[str, float]:
        """Calculate all transaction costs for an order"""
        notional = order.quantity * fill_price
        
        costs = {}
        
        # Brokerage
        costs['brokerage'] = notional * self.brokerage_rate
        
        # Exchange fees
        costs['exchange_fee'] = notional * self.exchange_fee
        
        # Securities Transaction Tax (STT) - only on sell side
        if order.side == 'SELL':
            costs['stt'] = notional * self.stt_rate
        else:
            costs['stt'] = 0.0
        
        # GST on brokerage and exchange fees
        taxable_amount = costs['brokerage'] + costs['exchange_fee']
        costs['gst'] = taxable_amount * self.gst_rate
        
        # Total transaction cost
        costs['total_transaction_cost'] = sum(costs.values())
        
        return costs
    
    def estimate_market_impact(self, order: Order, order_book: Dict) -> Dict[str, float]:
        """Estimate market impact of an order"""
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return {'temporary_impact': 0, 'permanent_impact': 0, 'total_impact': 0}
        
        # Determine relevant side of book
        if order.side == 'BUY':
            book_side = order_book['asks']
        else:
            book_side = order_book['bids']
        
        if not book_side:
            return {'temporary_impact': 0, 'permanent_impact': 0, 'total_impact': 0}
        
        # Calculate order size as fraction of available liquidity
        total_liquidity = sum(level['quantity'] for level in book_side[:5])  # Top 5 levels
        if total_liquidity == 0:
            return {'temporary_impact': 0, 'permanent_impact': 0, 'total_impact': 0}
        
        participation_rate = order.quantity / total_liquidity
        
        # Temporary impact (recovers after trade)
        temporary_impact = self.temporary_impact_coeff * np.sqrt(participation_rate)
        
        # Permanent impact (persists)
        permanent_impact = self.permanent_impact_coeff * participation_rate
        
        # Total impact
        total_impact = temporary_impact + permanent_impact
        
        return {
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'total_impact': total_impact,
            'participation_rate': participation_rate
        }
    
    def estimate_slippage(self, order: Order, order_book: Dict) -> Dict[str, float]:
        """Estimate slippage for an order"""
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return {'expected_slippage': 0, 'worst_case_slippage': 0}
        
        # Get relevant side of book
        if order.side == 'BUY':
            book_side = order_book['asks']
            reference_price = order_book['asks'][0]['price'] if order_book['asks'] else 0
        else:
            book_side = order_book['bids']
            reference_price = order_book['bids'][0]['price'] if order_book['bids'] else 0
        
        if not book_side or reference_price == 0:
            return {'expected_slippage': 0, 'worst_case_slippage': 0}
        
        # Walk through the book to estimate fill prices
        remaining_quantity = order.quantity
        total_cost = 0
        levels_consumed = 0
        
        for level in book_side:
            if remaining_quantity <= 0:
                break
                
            level_quantity = level['quantity']
            level_price = level['price']
            
            fill_quantity = min(remaining_quantity, level_quantity)
            total_cost += fill_quantity * level_price
            remaining_quantity -= fill_quantity
            levels_consumed += 1
        
        if order.quantity > 0:
            average_fill_price = total_cost / (order.quantity - remaining_quantity)
            expected_slippage = abs(average_fill_price - reference_price) / reference_price
        else:
            expected_slippage = 0
        
        # Worst case slippage (if we consume more levels)
        worst_case_slippage = expected_slippage * (1 + levels_consumed * 0.1)
        
        return {
            'expected_slippage': expected_slippage,
            'worst_case_slippage': worst_case_slippage,
            'average_fill_price': average_fill_price if order.quantity > 0 else reference_price,
            'levels_consumed': levels_consumed
        }
    
    def simulate_order_execution(self, order: Order, order_book: Dict) -> Dict:
        """Simulate complete order execution with all costs"""
        
        # Get market data
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return {'error': 'Invalid order book data'}
        
        bids = order_book['bids']
        asks = order_book['asks']
        
        if not bids or not asks:
            return {'error': 'Empty order book'}
        
        # Determine reference price
        if order.side == 'BUY':
            reference_price = asks[0]['price']
            book_side = asks
        else:
            reference_price = bids[0]['price']
            book_side = bids
        
        # Estimate slippage
        slippage_analysis = self.estimate_slippage(order, order_book)
        
        # Estimate market impact
        impact_analysis = self.estimate_market_impact(order, order_book)
        
        # Calculate expected fill price
        expected_fill_price = slippage_analysis.get('average_fill_price', reference_price)
        
        # Apply market impact to fill price
        if order.side == 'BUY':
            impact_adjusted_price = expected_fill_price * (1 + impact_analysis['total_impact'])
        else:
            impact_adjusted_price = expected_fill_price * (1 - impact_analysis['total_impact'])
        
        # Calculate transaction costs
        transaction_costs = self.calculate_transaction_costs(order, impact_adjusted_price)
        
        # Calculate total cost
        notional = order.quantity * impact_adjusted_price
        total_slippage_cost = abs(impact_adjusted_price - reference_price) * order.quantity
        total_cost = transaction_costs['total_transaction_cost'] + total_slippage_cost
        
        # Cost as percentage of notional
        cost_percentage = (total_cost / notional) * 100 if notional > 0 else 0
        
        return {
            'order': {
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'order_type': order.order_type
            },
            'execution': {
                'reference_price': reference_price,
                'expected_fill_price': impact_adjusted_price,
                'notional': notional
            },
            'slippage': slippage_analysis,
            'market_impact': impact_analysis,
            'transaction_costs': transaction_costs,
            'summary': {
                'total_cost': total_cost,
                'cost_percentage': cost_percentage,
                'cost_per_share': total_cost / order.quantity if order.quantity > 0 else 0
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def optimize_order_size(self, target_notional: float, order_book: Dict, 
                           max_impact: float = 0.01) -> Dict:
        """Optimize order size to minimize costs while staying within impact limits"""
        
        if not order_book or 'asks' not in order_book:
            return {'error': 'Invalid order book'}
        
        asks = order_book['asks']
        if not asks:
            return {'error': 'No ask levels available'}
        
        reference_price = asks[0]['price']
        target_quantity = int(target_notional / reference_price)
        
        # Binary search for optimal size
        min_size = 1
        max_size = target_quantity
        optimal_size = target_quantity
        
        while min_size <= max_size:
            test_size = (min_size + max_size) // 2
            test_order = Order(
                symbol="NIFTY50",
                side="BUY",
                quantity=test_size,
                order_type="MARKET"
            )
            
            impact = self.estimate_market_impact(test_order, order_book)
            
            if impact['total_impact'] <= max_impact:
                optimal_size = test_size
                min_size = test_size + 1
            else:
                max_size = test_size - 1
        
        # Calculate execution plan
        execution_plan = []
        remaining_notional = target_notional
        
        while remaining_notional > 0 and optimal_size > 0:
            order_notional = min(remaining_notional, optimal_size * reference_price)
            order_quantity = int(order_notional / reference_price)
            
            if order_quantity == 0:
                break
            
            execution_plan.append({
                'quantity': order_quantity,
                'estimated_price': reference_price,
                'notional': order_quantity * reference_price
            })
            
            remaining_notional -= order_quantity * reference_price
        
        return {
            'target_notional': target_notional,
            'target_quantity': target_quantity,
            'optimal_size_per_order': optimal_size,
            'execution_plan': execution_plan,
            'total_orders': len(execution_plan),
            'max_impact_constraint': max_impact
        }

async def main():
    """Main cost simulation loop"""
    print("Starting QuantX Cost Simulator...")
    
    simulator = CostSimulator()
    
    # Sample order book
    sample_order_book = {
        'bids': [
            {'price': 18449.75, 'quantity': 500},
            {'price': 18449.50, 'quantity': 300},
            {'price': 18449.25, 'quantity': 200},
            {'price': 18449.00, 'quantity': 150},
            {'price': 18448.75, 'quantity': 100}
        ],
        'asks': [
            {'price': 18450.25, 'quantity': 400},
            {'price': 18450.50, 'quantity': 350},
            {'price': 18450.75, 'quantity': 250},
            {'price': 18451.00, 'quantity': 200},
            {'price': 18451.25, 'quantity': 150}
        ]
    }
    
    try:
        while True:
            # Simulate different order sizes
            order_sizes = [100, 500, 1000, 2000]
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cost Analysis:")
            print("=" * 80)
            
            for size in order_sizes:
                # Create test order
                test_order = Order(
                    symbol="NIFTY50",
                    side="BUY",
                    quantity=size,
                    order_type="MARKET"
                )
                
                # Simulate execution
                result = simulator.simulate_order_execution(test_order, sample_order_book)
                
                if 'error' not in result:
                    print(f"\nOrder Size: {size} shares")
                    print(f"  Expected Fill: ₹{result['execution']['expected_fill_price']:.2f}")
                    print(f"  Slippage: {result['slippage']['expected_slippage']*100:.3f}%")
                    print(f"  Market Impact: {result['market_impact']['total_impact']*100:.3f}%")
                    print(f"  Transaction Costs: ₹{result['transaction_costs']['total_transaction_cost']:.2f}")
                    print(f"  Total Cost: ₹{result['summary']['total_cost']:.2f} ({result['summary']['cost_percentage']:.3f}%)")
                else:
                    print(f"Order Size: {size} - Error: {result['error']}")
            
            # Optimize order size
            target_notional = 1000000  # 10 lakh
            optimization = simulator.optimize_order_size(target_notional, sample_order_book)
            
            if 'error' not in optimization:
                print(f"\nOrder Optimization for ₹{target_notional:,}:")
                print(f"  Optimal size per order: {optimization['optimal_size_per_order']} shares")
                print(f"  Total orders needed: {optimization['total_orders']}")
                print(f"  Max impact constraint: {optimization['max_impact_constraint']*100:.1f}%")
            
            # Add some randomness to order book for next iteration
            for bid in sample_order_book['bids']:
                bid['quantity'] = max(50, bid['quantity'] + np.random.randint(-50, 51))
                bid['price'] += np.random.choice([-0.25, 0, 0.25]) * 0.1
            
            for ask in sample_order_book['asks']:
                ask['quantity'] = max(50, ask['quantity'] + np.random.randint(-50, 51))
                ask['price'] += np.random.choice([-0.25, 0, 0.25]) * 0.1
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\nShutting down cost simulator...")

if __name__ == "__main__":
    asyncio.run(main())
