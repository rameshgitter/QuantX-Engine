import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os
from datetime import datetime, timedelta

class LSTMPredictor(nn.Module):
    """LSTM model for LOB prediction"""
    
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.softmax(out)
        
        return out

class TransformerPredictor(nn.Module):
    """Transformer model for LOB prediction"""
    
    def __init__(self, input_size=14, d_model=128, nhead=8, num_layers=4, num_classes=3, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        seq_len = x.size(1)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply transformer
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        # Classification
        out = self.classifier(pooled)
        
        return out

class IVSurfacePredictor(nn.Module):
    """Neural network for IV surface prediction"""
    
    def __init__(self, input_size=10, hidden_sizes=[128, 64, 32], output_size=1, dropout=0.2):
        super(IVSurfacePredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # IV is always positive
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def generate_synthetic_lob_data(n_samples=50000, seq_len=10):
    """Generate synthetic LOB data for training"""
    
    print(f"Generating {n_samples} synthetic LOB samples...")
    
    # Feature names
    feature_names = [
        'spread', 'volume_imbalance', 'bid_ask_ratio', 'depth_imbalance',
        'relative_spread', 'price_impact_bid', 'price_impact_ask',
        'micro_price_diff', 'volume_ratio', 'order_flow',
        'spread_change', 'volume_imbalance_change',
        'spread_momentum', 'volume_imbalance_momentum'
    ]
    
    # Generate base features
    np.random.seed(42)
    
    # Create correlated features that resemble real LOB data
    data = []
    labels = []
    
    for i in range(n_samples):
        # Generate a sequence of LOB states
        sequence = []
        
        # Base market state
        base_spread = np.random.exponential(0.5) + 0.05
        base_imbalance = np.random.normal(0, 0.3)
        
        for t in range(seq_len):
            # Add some temporal dynamics
            spread = base_spread + np.random.normal(0, 0.1)
            volume_imbalance = base_imbalance + np.random.normal(0, 0.1)
            
            # Other features with realistic relationships
            bid_ask_ratio = np.exp(volume_imbalance * 0.5) + np.random.normal(0, 0.2)
            depth_imbalance = bid_ask_ratio * (1 + np.random.normal(0, 0.1))
            relative_spread = spread / (18450 + np.random.normal(0, 50))
            
            price_impact_bid = np.random.exponential(1000)
            price_impact_ask = np.random.exponential(1000)
            
            micro_price_diff = np.random.normal(0, 0.001)
            volume_ratio = np.random.beta(2, 2)
            order_flow = np.random.normal(0, 0.2)
            
            # Temporal features
            if t > 0:
                spread_change = spread - sequence[t-1][0]
                volume_imbalance_change = volume_imbalance - sequence[t-1][1]
            else:
                spread_change = 0
                volume_imbalance_change = 0
            
            if t > 1:
                spread_momentum = spread_change - (sequence[t-1][0] - sequence[t-2][0])
                volume_imbalance_momentum = volume_imbalance_change - (sequence[t-1][1] - sequence[t-2][1])
            else:
                spread_momentum = 0
                volume_imbalance_momentum = 0
            
            features = [
                spread, volume_imbalance, bid_ask_ratio, depth_imbalance,
                relative_spread, price_impact_bid, price_impact_ask,
                micro_price_diff, volume_ratio, order_flow,
                spread_change, volume_imbalance_change,
                spread_momentum, volume_imbalance_momentum
            ]
            
            sequence.append(features)
        
        data.append(sequence)
        
        # Generate label based on final state features
        final_features = sequence[-1]
        
        # Simple labeling logic
        if final_features[1] > 0.2 and final_features[0] < 0.3:  # Strong buy imbalance, tight spread
            label = 2  # Buy
        elif final_features[1] < -0.2 and final_features[0] < 0.3:  # Strong sell imbalance, tight spread
            label = 0  # Sell
        else:
            label = 1  # Neutral
        
        labels.append(label)
    
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64), feature_names

def generate_synthetic_iv_data(n_samples=20000):
    """Generate synthetic IV surface data"""
    
    print(f"Generating {n_samples} synthetic IV samples...")
    
    np.random.seed(42)
    
    data = []
    labels = []
    
    for i in range(n_samples):
        # Market features
        spot_price = 18450 + np.random.normal(0, 500)
        time_to_expiry = np.random.uniform(0.02, 1.0)  # 1 week to 1 year
        strike_price = spot_price * np.random.uniform(0.8, 1.2)
        
        # Moneyness
        moneyness = np.log(strike_price / spot_price)
        
        # Market regime features
        vix_level = np.random.uniform(10, 40)
        term_structure_slope = np.random.normal(0, 0.05)
        skew = np.random.normal(-0.02, 0.01)
        
        # Historical volatility
        hist_vol = np.random.uniform(0.1, 0.4)
        
        # Put-call ratio
        put_call_ratio = np.random.uniform(0.5, 2.0)
        
        # Volume and open interest
        volume = np.random.exponential(1000)
        open_interest = np.random.exponential(5000)
        
        features = [
            moneyness, time_to_expiry, vix_level, term_structure_slope,
            skew, hist_vol, put_call_ratio, volume, open_interest,
            np.sin(time_to_expiry * 2 * np.pi)  # Seasonal component
        ]
        
        # Generate IV based on realistic model
        base_iv = 0.2 + vix_level * 0.005
        skew_effect = skew * moneyness
        time_effect = np.sqrt(time_to_expiry) * 0.1
        
        iv = base_iv + skew_effect + time_effect + np.random.normal(0, 0.02)
        iv = max(0.05, min(1.0, iv))  # Clamp to reasonable range
        
        data.append(features)
        labels.append(iv)
    
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

def train_and_export_lstm_model():
    """Train and export LSTM model to ONNX"""
    
    print("Training LSTM model...")
    
    # Generate data
    X, y, feature_names = generate_synthetic_lob_data(n_samples=50000, seq_len=10)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create model
    model = LSTMPredictor(input_size=14, hidden_size=64, num_layers=2, num_classes=3)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # Training loop
    batch_size = 256
    num_epochs = 50
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'Test Accuracy: {accuracy:.4f}')
    
    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 10, 14)  # batch_size=1, seq_len=10, input_size=14
    
    onnx_path = "models/lob_lstm_predictor.onnx"
    os.makedirs("models", exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"LSTM model exported to {onnx_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Test ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print("ONNX model verification successful!")
    
    return feature_names

def train_and_export_transformer_model():
    """Train and export Transformer model to ONNX"""
    
    print("Training Transformer model...")
    
    # Generate data
    X, y, feature_names = generate_synthetic_lob_data(n_samples=30000, seq_len=20)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create model
    model = TransformerPredictor(input_size=14, d_model=128, nhead=8, num_layers=4, num_classes=3)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training loop
    batch_size = 128
    num_epochs = 30
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'Test Accuracy: {accuracy:.4f}')
    
    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 20, 14)  # batch_size=1, seq_len=20, input_size=14
    
    onnx_path = "models/lob_transformer_predictor.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Transformer model exported to {onnx_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print("Transformer ONNX model verification successful!")

def train_and_export_iv_model():
    """Train and export IV surface model to ONNX"""
    
    print("Training IV Surface model...")
    
    # Generate data
    X, y = generate_synthetic_iv_data(n_samples=20000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create model
    model = IVSurfacePredictor(input_size=10, hidden_sizes=[128, 64, 32], output_size=1)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    batch_size = 256
    num_epochs = 100
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        mse = criterion(test_outputs, y_test_tensor)
        rmse = torch.sqrt(mse)
        print(f'Test RMSE: {rmse:.6f}')
    
    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 10)  # batch_size=1, input_size=10
    
    onnx_path = "models/iv_surface_predictor.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"IV Surface model exported to {onnx_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print("IV Surface ONNX model verification successful!")

def train_and_export_random_forest():
    """Train and export Random Forest model using sklearn-onnx"""
    
    print("Training Random Forest model...")
    
    # Generate data (flattened for sklearn)
    X, y, feature_names = generate_synthetic_lob_data(n_samples=50000, seq_len=1)
    X = X.reshape(X.shape[0], -1)  # Flatten to 2D
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_accuracy = rf_model.score(X_train_scaled, y_train)
    test_accuracy = rf_model.score(X_test_scaled, y_test)
    
    print(f"Random Forest Train Accuracy: {train_accuracy:.4f}")
    print(f"Random Forest Test Accuracy: {test_accuracy:.4f}")
    
    # Save sklearn model and scaler
    joblib.dump(rf_model, "models/lob_random_forest.pkl")
    joblib.dump(scaler, "models/lob_scaler.pkl")
    
    # Save normalization parameters for C++
    normalization_params = {
        "means": scaler.mean_.tolist(),
        "stds": scaler.scale_.tolist(),
        "feature_names": feature_names
    }
    
    with open("models/lob_normalization_params.json", "w") as f:
        json.dump(normalization_params, f, indent=2)
    
    print("Random Forest model and normalization parameters saved!")
    
    # Convert to ONNX using skl2onnx
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, X_train_scaled.shape[1]]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(rf_model, initial_types=initial_type)
        
        # Save ONNX model
        onnx_path = "models/lob_random_forest.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"Random Forest ONNX model exported to {onnx_path}")
        
        # Verify ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: X_test_scaled[:5].astype(np.float32)}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print("Random Forest ONNX model verification successful!")
        
    except ImportError:
        print("skl2onnx not available. Install with: pip install skl2onnx")
        print("Sklearn model saved as .pkl file instead.")

def create_ensemble_model():
    """Create a simple ensemble model that combines predictions"""
    
    print("Creating ensemble model...")
    
    class EnsembleModel(nn.Module):
        def __init__(self, num_models=3, input_size=3, hidden_size=32, output_size=3):
            super(EnsembleModel, self).__init__()
            
            # Simple MLP to combine predictions from different models
            self.combiner = nn.Sequential(
                nn.Linear(num_models * input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size),
                nn.Softmax(dim=1)
            )
        
        def forward(self, x):
            # x should be concatenated predictions from all models
            return self.combiner(x)
    
    # Create and train ensemble model
    model = EnsembleModel(num_models=3, input_size=3, hidden_size=32, output_size=3)
    
    # Generate synthetic ensemble training data
    n_samples = 10000
    X_ensemble = torch.randn(n_samples, 9)  # 3 models * 3 classes each
    
    # Create synthetic labels based on ensemble logic
    y_ensemble = []
    for i in range(n_samples):
        # Average the predictions and take argmax
        pred1 = torch.softmax(X_ensemble[i, :3], dim=0)
        pred2 = torch.softmax(X_ensemble[i, 3:6], dim=0)
        pred3 = torch.softmax(X_ensemble[i, 6:9], dim=0)
        
        avg_pred = (pred1 + pred2 + pred3) / 3
        label = torch.argmax(avg_pred).item()
        y_ensemble.append(label)
    
    y_ensemble = torch.LongTensor(y_ensemble)
    
    # Train ensemble model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    batch_size = 256
    num_epochs = 50
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_ensemble), batch_size):
            batch_X = X_ensemble[i:i+batch_size]
            batch_y = y_ensemble[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f'Ensemble Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 9)  # batch_size=1, 3 models * 3 predictions
    
    onnx_path = "models/ensemble_predictor.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Ensemble model exported to {onnx_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print("Ensemble ONNX model verification successful!")

def main():
    """Main function to train and export all models"""
    
    print("=== QuantX Engine Model Training and ONNX Export ===")
    print(f"Started at: {datetime.now()}")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    try:
        # Train and export LSTM model
        print("\n" + "="*50)
        feature_names = train_and_export_lstm_model()
        
        # Train and export Transformer model
        print("\n" + "="*50)
        train_and_export_transformer_model()
        
        # Train and export IV Surface model
        print("\n" + "="*50)
        train_and_export_iv_model()
        
        # Train and export Random Forest model
        print("\n" + "="*50)
        train_and_export_random_forest()
        
        # Create ensemble model
        print("\n" + "="*50)
        create_ensemble_model()
        
        print("\n" + "="*50)
        print("All models trained and exported successfully!")
        print("\nGenerated files:")
        print("- models/lob_lstm_predictor.onnx")
        print("- models/lob_transformer_predictor.onnx")
        print("- models/iv_surface_predictor.onnx")
        print("- models/lob_random_forest.onnx (if skl2onnx available)")
        print("- models/lob_random_forest.pkl")
        print("- models/lob_scaler.pkl")
        print("- models/lob_normalization_params.json")
        print("- models/ensemble_predictor.onnx")
        
        print(f"\nCompleted at: {datetime.now()}")
        
        # Test all ONNX models
        print("\n" + "="*50)
        print("Testing all ONNX models...")
        
        test_models()
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()

def test_models():
    """Test all exported ONNX models"""
    
    model_paths = [
        "models/lob_lstm_predictor.onnx",
        "models/lob_transformer_predictor.onnx", 
        "models/iv_surface_predictor.onnx",
        "models/ensemble_predictor.onnx"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Testing {model_path}...")
                
                # Load ONNX model
                ort_session = ort.InferenceSession(model_path)
                
                # Get input shape
                input_shape = ort_session.get_inputs()[0].shape
                print(f"  Input shape: {input_shape}")
                
                # Create dummy input
                if "lstm" in model_path:
                    dummy_input = np.random.randn(1, 10, 14).astype(np.float32)
                elif "transformer" in model_path:
                    dummy_input = np.random.randn(1, 20, 14).astype(np.float32)
                elif "iv_surface" in model_path:
                    dummy_input = np.random.randn(1, 10).astype(np.float32)
                elif "ensemble" in model_path:
                    dummy_input = np.random.randn(1, 9).astype(np.float32)
                else:
                    dummy_input = np.random.randn(1, 14).astype(np.float32)
                
                # Run inference
                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
                ort_outputs = ort_session.run(None, ort_inputs)
                
                print(f"  Output shape: {ort_outputs[0].shape}")
                print(f"  Sample output: {ort_outputs[0][0][:5]}")  # First 5 values
                print(f"  ✓ Model test passed!")
                
            except Exception as e:
                print(f"  ✗ Model test failed: {e}")
        else:
            print(f"Model not found: {model_path}")
    
    # Test Random Forest if available
    rf_path = "models/lob_random_forest.onnx"
    if os.path.exists(rf_path):
        try:
            print(f"Testing {rf_path}...")
            ort_session = ort.InferenceSession(rf_path)
            dummy_input = np.random.randn(1, 14).astype(np.float32)
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            print(f"  ✓ Random Forest ONNX test passed!")
        except Exception as e:
            print(f"  ✗ Random Forest ONNX test failed: {e}")

if __name__ == "__main__":
    main()
