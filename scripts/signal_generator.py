import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import asyncio
import json
from datetime import datetime, timedelta

class LOBPredictor:
    """
    Limit Order Book predictor using machine learning
    Predicts short-term price movements based on order book features
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'spread', 'volume_imbalance', 'bid_ask_ratio', 'depth_imbalance',
            'relative_spread', 'price_impact_bid', 'price_impact_ask',
            'micro_price_diff', 'volume_ratio', 'order_flow'
        ]
        self.lookback_window = 10
        self.feature_buffer = []
        self.price_buffer = []
        
    def create_features(self, order_book_data):
        """Create features from order book data"""
        if not order_book_data:
            return None
            
        features = {}
        
        # Basic features from order book
        bids = order_book_data.get('bids', [])
        asks = order_book_data.get('asks', [])
        
        if not bids or not asks:
            return None
            
        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Volume features
        bid_vol_1 = bids[0]['quantity']
        ask_vol_1 = asks[0]['quantity']
        bid_vol_5 = sum(level['quantity'] for level in bids[:5])
        ask_vol_5 = sum(level['quantity'] for level in asks[:5])
        
        # Core features
        features['spread'] = spread
        features['volume_imbalance'] = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5) if (bid_vol_5 + ask_vol_5) > 0 else 0
        features['bid_ask_ratio'] = bid_vol_1 / ask_vol_1 if ask_vol_1 > 0 else 1
        features['depth_imbalance'] = bid_vol_5 / ask_vol_5 if ask_vol_5 > 0 else 1
        features['relative_spread'] = spread / mid_price if mid_price > 0 else 0
        
        # Price impact features
        features['price_impact_bid'] = sum(level['quantity'] for level in bids[:3])
        features['price_impact_ask'] = sum(level['quantity'] for level in asks[:3])
        
        # Micro price
        micro_price = (best_bid * ask_vol_1 + best_ask * bid_vol_1) / (bid_vol_1 + ask_vol_1) if (bid_vol_1 + ask_vol_1) > 0 else mid_price
        features['micro_price_diff'] = (micro_price - mid_price) / mid_price if mid_price > 0 else 0
        
        # Volume ratio
        features['volume_ratio'] = (bid_vol_1 + ask_vol_1) / (bid_vol_5 + ask_vol_5) if (bid_vol_5 + ask_vol_5) > 0 else 0
        
        # Order flow (simplified)
        features['order_flow'] = (len(bids) - len(asks)) / (len(bids) + len(asks)) if (len(bids) + len(asks)) > 0 else 0
        
        return features
    
    def create_temporal_features(self, feature_history):
        """Create temporal features from feature history"""
        if len(feature_history) < 2:
            return {}
            
        temporal_features = {}
        
        # Calculate changes and trends
        for feature_name in self.feature_names:
            if feature_name in feature_history[-1] and feature_name in feature_history[-2]:
                current = feature_history[-1][feature_name]
                previous = feature_history[-2][feature_name]
                
                # Feature change
                temporal_features[f'{feature_name}_change'] = current - previous
                
                # Feature momentum (if we have enough history)
                if len(feature_history) >= 3 and feature_name in feature_history[-3]:
                    prev_prev = feature_history[-3][feature_name]
                    momentum = (current - previous) - (previous - prev_prev)
                    temporal_features[f'{feature_name}_momentum'] = momentum
        
        return temporal_features
    
    def train_model(self, training_data):
        """Train the LOB prediction model"""
        print("Training LOB prediction model...")
        
        # In a real implementation, this would use historical order book data
        # For demo purposes, we'll create synthetic training data
        n_samples = 10000
        X = np.random.randn(n_samples, len(self.feature_names))
        
        # Create synthetic labels based on feature combinations
        # This is a simplified example - real labels would come from future price movements
        y = np.zeros(n_samples)
        for i in range(n_samples):
            # Simple rule: if volume imbalance > 0.1 and spread < 0.3, predict up
            if X[i, 1] > 0.1 and X[i, 0] < 0.3:  # volume_imbalance and spread
                y[i] = 1  # Up
            elif X[i, 1] < -0.1 and X[i, 0] < 0.3:
                y[i] = -1  # Down
            else:
                y[i] = 0  # Neutral
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)
        
        # Calculate feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        print("Feature Importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.3f}")
        
        print("Model training completed!")
        return self.model
    
    def predict(self, features):
        """Make prediction based on current features"""
        if self.model is None:
            return {'signal': 0, 'confidence': 0, 'probabilities': [0.33, 0.33, 0.34]}
        
        # Prepare feature vector
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        # Scale features
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Make prediction
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        
        # Calculate confidence as max probability
        confidence = np.max(probabilities)
        
        return {
            'signal': int(prediction),
            'confidence': float(confidence),
            'probabilities': probabilities.tolist()
        }

class IVPredictor:
    """
    Implied Volatility surface predictor
    Forecasts option volatility changes
    """
    
    def __init__(self):
        self.model = None
        self.current_iv_surface = {}
        
    def update_iv_surface(self, option_data):
        """Update the current IV surface with new option data"""
        # In production, this would connect to option data feed
        # For demo, we'll simulate IV surface updates
        
        strikes = range(18000, 19000, 50)
        expiries = ['1W', '2W', '1M', '2M', '3M']
        
        base_iv = 0.18 + np.sin(datetime.now().timestamp() * 0.001) * 0.02
        
        for expiry in expiries:
            self.current_iv_surface[expiry] = {}
            for strike in strikes:
                # Simulate IV with skew
                moneyness = np.log(strike / 18450)
                skew_effect = -0.1 * moneyness + 0.05 * moneyness * moneyness
                iv = base_iv + skew_effect + np.random.normal(0, 0.01)
                self.current_iv_surface[expiry][strike] = max(0.05, iv)
    
    def calculate_iv_features(self):
        """Calculate features from IV surface"""
        if not self.current_iv_surface:
            return {}
        
        features = {}
        
        # ATM IV for different expiries
        atm_strike = 18450
        for expiry in ['1W', '1M', '3M']:
            if expiry in self.current_iv_surface:
                # Find closest strike to ATM
                closest_strike = min(self.current_iv_surface[expiry].keys(), 
                                   key=lambda x: abs(x - atm_strike))
                features[f'atm_iv_{expiry}'] = self.current_iv_surface[expiry][closest_strike]
        
        # IV skew (25-delta put vs call approximation)
        if '1M' in self.current_iv_surface:
            strikes = sorted(self.current_iv_surface['1M'].keys())
            if len(strikes) >= 5:
                put_strike = strikes[2]  # OTM put
                call_strike = strikes[-3]  # OTM call
                iv_skew = self.current_iv_surface['1M'][put_strike] - self.current_iv_surface['1M'][call_strike]
                features['iv_skew'] = iv_skew
        
        # Term structure slope
        if 'atm_iv_1W' in features and 'atm_iv_3M' in features:
            features['term_structure_slope'] = features['atm_iv_3M'] - features['atm_iv_1W']
        
        return features
    
    def predict_iv_signal(self):
        """Generate IV-based trading signal"""
        features = self.calculate_iv_features()
        
        if not features:
            return {'signal': 0, 'confidence': 0}
        
        # Simple IV signal logic
        signal = 0
        confidence = 0.5
        
        # High IV skew suggests put buying pressure (bearish)
        if 'iv_skew' in features:
            if features['iv_skew'] > 0.02:
                signal = -1  # Bearish
                confidence = 0.7
            elif features['iv_skew'] < -0.02:
                signal = 1  # Bullish
                confidence = 0.7
        
        # Inverted term structure (backwardation) suggests volatility spike
        if 'term_structure_slope' in features:
            if features['term_structure_slope'] < -0.05:
                signal = -1  # Bearish (volatility spike often coincides with market stress)
                confidence = max(confidence, 0.6)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'features': features
        }

class EnsembleSignalGenerator:
    """
    Combines LOB and IV signals into ensemble prediction
    """
    
    def __init__(self):
        self.lob_predictor = LOBPredictor()
        self.iv_predictor = IVPredictor()
        
        # Signal weights
        self.lob_weight = 0.6
        self.iv_weight = 0.4
        
        # Initialize models
        self.lob_predictor.train_model(None)
    
    def generate_combined_signal(self, order_book_data, option_data=None):
        """Generate combined signal from LOB and IV predictors"""
        
        # Get LOB signal
        lob_features = self.lob_predictor.create_features(order_book_data)
        if lob_features:
            lob_prediction = self.lob_predictor.predict(lob_features)
        else:
            lob_prediction = {'signal': 0, 'confidence': 0}
        
        # Update IV surface and get IV signal
        self.iv_predictor.update_iv_surface(option_data)
        iv_prediction = self.iv_predictor.predict_iv_signal()
        
        # Combine signals
        combined_signal = (
            lob_prediction['signal'] * lob_prediction['confidence'] * self.lob_weight +
            iv_prediction['signal'] * iv_prediction['confidence'] * self.iv_weight
        )
        
        # Normalize combined signal
        max_possible = self.lob_weight + self.iv_weight
        if max_possible > 0:
            combined_signal = combined_signal / max_possible
        
        # Combined confidence
        combined_confidence = (
            lob_prediction['confidence'] * self.lob_weight +
            iv_prediction['confidence'] * self.iv_weight
        ) / (self.lob_weight + self.iv_weight)
        
        return {
            'lob_signal': lob_prediction['signal'],
            'lob_confidence': lob_prediction['confidence'],
            'iv_signal': iv_prediction['signal'],
            'iv_confidence': iv_prediction['confidence'],
            'combined_signal': combined_signal,
            'combined_confidence': combined_confidence,
            'timestamp': datetime.now().isoformat()
        }

async def main():
    """Main signal generation loop"""
    print("Starting QuantX Signal Generator...")
    
    signal_generator = EnsembleSignalGenerator()
    
    # Simulate order book data
    sample_order_book = {
        'bids': [
            {'price': 18449.75, 'quantity': 500, 'orders': 5},
            {'price': 18449.50, 'quantity': 300, 'orders': 3},
            {'price': 18449.25, 'quantity': 200, 'orders': 2}
        ],
        'asks': [
            {'price': 18450.25, 'quantity': 400, 'orders': 4},
            {'price': 18450.50, 'quantity': 350, 'orders': 3},
            {'price': 18450.75, 'quantity': 250, 'orders': 2}
        ]
    }
    
    try:
        while True:
            # Add some randomness to the order book
            for bid in sample_order_book['bids']:
                bid['quantity'] = max(50, bid['quantity'] + np.random.randint(-50, 51))
                bid['price'] += np.random.choice([-0.25, 0, 0.25]) * np.random.random()
            
            for ask in sample_order_book['asks']:
                ask['quantity'] = max(50, ask['quantity'] + np.random.randint(-50, 51))
                ask['price'] += np.random.choice([-0.25, 0, 0.25]) * np.random.random()
            
            # Generate signal
            signal_result = signal_generator.generate_combined_signal(sample_order_book)
            
            # Display results
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] Signals Generated:")
            print(f"  LOB: {signal_result['lob_signal']:+.0f} (conf: {signal_result['lob_confidence']:.2f})")
            print(f"  IV:  {signal_result['iv_signal']:+.0f} (conf: {signal_result['iv_confidence']:.2f})")
            print(f"  Combined: {signal_result['combined_signal']:+.3f} (conf: {signal_result['combined_confidence']:.2f})")
            
            # Determine action
            if abs(signal_result['combined_signal']) > 0.3 and signal_result['combined_confidence'] > 0.6:
                action = "BUY" if signal_result['combined_signal'] > 0 else "SELL"
                print(f"  → ACTION: {action} (Strength: {abs(signal_result['combined_signal']):.2f})")
            else:
                print(f"  → ACTION: HOLD (Signal too weak)")
            
            print("-" * 60)
            
            await asyncio.sleep(2)  # Generate signals every 2 seconds
            
    except KeyboardInterrupt:
        print("\nShutting down signal generator...")

if __name__ == "__main__":
    asyncio.run(main())
