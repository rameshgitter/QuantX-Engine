# Multi-stage build for QuantX Engine
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libboost-all-dev \
    libssl-dev \
    nlohmann-json3-dev \
    libwebsocketpp-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for model training
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install numpy pandas scikit-learn onnx onnxruntime joblib

# Install ONNX Runtime C++
WORKDIR /tmp
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.3.tgz \
    && cp -r onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/ \
    && cp -r onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/ \
    && ldconfig

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Train and export ML models
RUN python3 scripts/export_models_to_onnx.py

# Build the application
RUN mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# Runtime stage
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libboost-system1.74.0 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy ONNX Runtime libraries
COPY --from=builder /usr/local/lib/libonnxruntime* /usr/local/lib/
RUN ldconfig

# Create app user
RUN useradd -m -s /bin/bash quantx

# Set working directory
WORKDIR /app

# Copy built application and models
COPY --from=builder /app/build/quantx_engine /app/
COPY --from=builder /app/models /app/models/

# Copy configuration files
COPY config/ /app/config/

# Set ownership
RUN chown -R quantx:quantx /app

# Switch to app user
USER quantx

# Expose ports (if needed for web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep quantx_engine || exit 1

# Run the application
CMD ["./quantx_engine"]
