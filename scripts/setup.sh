#!/bin/bash

# QuantX Engine Setup Script
# This script sets up the development environment and dependencies

set -e

echo "QuantX Engine Setup Script"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if [ -f /etc/debian_version ]; then
            DISTRO="debian"
        elif [ -f /etc/redhat-release ]; then
            DISTRO="redhat"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    log_info "Detected OS: $OS"
}

# Install system dependencies
install_system_deps() {
    log_step "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        if [[ "$DISTRO" == "debian" ]]; then
            sudo apt-get update
            sudo apt-get install -y \
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
                python3-venv
        elif [[ "$DISTRO" == "redhat" ]]; then
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                cmake \
                git \
                wget \
                curl \
                pkgconfig \
                boost-devel \
                openssl-devel \
                json-devel \
                python3 \
                python3-pip
        fi
    elif [[ "$OS" == "macos" ]]; then
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        brew install \
            cmake \
            boost \
            openssl \
            nlohmann-json \
            websocketpp \
            python3
    fi
    
    log_info "System dependencies installed successfully"
}

# Install ONNX Runtime
install_onnx_runtime() {
    log_step "Installing ONNX Runtime..."
    
    ONNX_VERSION="1.16.3"
    
    if [[ "$OS" == "linux" ]]; then
        ONNX_PACKAGE="onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_PACKAGE}"
    elif [[ "$OS" == "macos" ]]; then
        ONNX_PACKAGE="onnxruntime-osx-x86_64-${ONNX_VERSION}.tgz"
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_PACKAGE}"
    fi
    
    # Create third_party directory
    mkdir -p third_party
    cd third_party
    
    # Download and extract ONNX Runtime
    if [ ! -f "$ONNX_PACKAGE" ]; then
        log_info "Downloading ONNX Runtime..."
        wget "$ONNX_URL"
    fi
    
    if [ ! -d "onnxruntime" ]; then
        log_info "Extracting ONNX Runtime..."
        tar -xzf "$ONNX_PACKAGE"
        mv "onnxruntime-"* onnxruntime
    fi
    
    # Install to system (optional)
    if [[ "$OS" == "linux" ]]; then
        sudo cp -r onnxruntime/include/* /usr/local/include/
        sudo cp -r onnxruntime/lib/* /usr/local/lib/
        sudo ldconfig
    fi
    
    cd ..
    log_info "ONNX Runtime installed successfully"
}

# Setup Python environment
setup_python_env() {
    log_step "Setting up Python environment..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    log_info "Installing Python dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install \
        numpy \
        pandas \
        scikit-learn \
        onnx \
        onnxruntime \
        joblib \
        matplotlib \
        seaborn \
        jupyter
    
    # Try to install optional dependencies
    pip install skl2onnx || log_warn "Could not install skl2onnx (optional)"
    
    log_info "Python environment setup completed"
}

# Create project structure
create_project_structure() {
    log_step "Creating project structure..."
    
    mkdir -p {build,models,logs,data,config,monitoring/grafana/{dashboards,datasources}}
    
    # Create empty model files to prevent errors
    touch models/.gitkeep
    touch logs/.gitkeep
    touch data/.gitkeep
    
    log_info "Project structure created"
}

# Build the project
build_project() {
    log_step "Building the project..."
    
    mkdir -p build
    cd build
    
    # Configure with CMake
    cmake .. -DCMAKE_BUILD_TYPE=Release
    
    # Build
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    cd ..
    log_info "Project built successfully"
}

# Train ML models
train_models() {
    log_step "Training ML models..."
    
    # Activate Python environment
    source venv/bin/activate
    
    # Run model training script
    python scripts/export_models_to_onnx.py
    
    log_info "ML models trained and exported"
}

# Run tests
run_tests() {
    log_step "Running tests..."
    
    cd build
    ./test_quantx
    cd ..
    
    log_info "Tests completed"
}

# Main setup function
main() {
    log_info "Starting QuantX Engine setup..."
    
    detect_os
    install_system_deps
    install_onnx_runtime
    setup_python_env
    create_project_structure
    
    log_info "Training ML models (this may take a few minutes)..."
    train_models
    
    log_info "Building C++ components..."
    build_project
    
    log_info "Running tests..."
    run_tests
    
    echo ""
    echo "=================================="
    echo "✅ QuantX Engine Setup Complete!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "1. Run the engine: ./build/quantx_engine"
    echo "2. Or use Docker: docker-compose up -d"
    echo "3. Monitor at: http://localhost:3000 (Grafana)"
    echo "4. Check logs: tail -f logs/quantx_engine.log"
    echo ""
    echo "For paper trading on VPS:"
    echo "1. Edit config/config.json with your API keys"
    echo "2. Run: ./scripts/deploy.sh"
    echo ""
}

# Run main function
main "$@"
