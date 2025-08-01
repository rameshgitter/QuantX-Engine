#!/bin/bash

# QuantX Engine Deployment Script
# This script deploys the QuantX Engine to a VPS for paper trading

set -e

echo "QuantX Engine Deployment Script"
echo "==============================="

# Configuration
VPS_HOST=${VPS_HOST:-"your-vps-ip"}
VPS_USER=${VPS_USER:-"ubuntu"}
VPS_PORT=${VPS_PORT:-"22"}
DEPLOY_DIR=${DEPLOY_DIR:-"/opt/quantx"}
SERVICE_NAME="quantx-engine"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    if ! command -v ssh &> /dev/null; then
        log_error "SSH is not installed. Please install SSH client first."
        exit 1
    fi
    
    log_info "Prerequisites check passed."
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build the main application image
    docker build -t quantx-engine:latest .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully."
    else
        log_error "Failed to build Docker image."
        exit 1
    fi
}

# Test locally before deployment
test_locally() {
    log_info "Testing application locally..."
    
    # Run tests
    docker run --rm quantx-engine:latest ./test_quantx
    
    if [ $? -eq 0 ]; then
        log_info "Local tests passed."
    else
        log_error "Local tests failed."
        exit 1
    fi
}

# Deploy to VPS
deploy_to_vps() {
    log_info "Deploying to VPS: $VPS_HOST"
    
    # Create deployment directory on VPS
    ssh -p $VPS_PORT $VPS_USER@$VPS_HOST "sudo mkdir -p $DEPLOY_DIR"
    ssh -p $VPS_PORT $VPS_USER@$VPS_HOST "sudo chown $VPS_USER:$VPS_USER $DEPLOY_DIR"
    
    # Copy files to VPS
    log_info "Copying files to VPS..."
    scp -P $VPS_PORT -r . $VPS_USER@$VPS_HOST:$DEPLOY_DIR/
    
    # Install Docker on VPS if not present
    ssh -p $VPS_PORT $VPS_USER@$VPS_HOST << 'EOF'
        if ! command -v docker &> /dev/null; then
            echo "Installing Docker..."
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            echo "Installing Docker Compose..."
            sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
        fi
EOF
    
    # Build and start services on VPS
    ssh -p $VPS_PORT $VPS_USER@$VPS_HOST << EOF
        cd $DEPLOY_DIR
        
        # Stop existing services
        docker-compose down || true
        
        # Build and start services
        docker-compose up -d --build
        
        # Wait for services to start
        sleep 30
        
        # Check service status
        docker-compose ps
EOF
    
    log_info "Deployment completed."
}

# Setup systemd service
setup_systemd_service() {
    log_info "Setting up systemd service..."
    
    ssh -p $VPS_PORT $VPS_USER@$VPS_HOST << EOF
        # Create systemd service file
        sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=QuantX Engine High-Frequency Trading Platform
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$DEPLOY_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
User=$VPS_USER
Group=$VPS_USER

[Install]
WantedBy=multi-user.target
SERVICE_EOF
        
        # Enable and start service
        sudo systemctl daemon-reload
        sudo systemctl enable $SERVICE_NAME
        sudo systemctl start $SERVICE_NAME
        
        # Check service status
        sudo systemctl status $SERVICE_NAME
EOF
    
    log_info "Systemd service setup completed."
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    ssh -p $VPS_PORT $VPS_USER@$VPS_HOST << EOF
        cd $DEPLOY_DIR
        
        # Create monitoring directories
        mkdir -p monitoring/grafana/dashboards
        mkdir -p monitoring/grafana/datasources
        mkdir -p logs
        
        # Set up log rotation
        sudo tee /etc/logrotate.d/quantx-engine > /dev/null << 'LOGROTATE_EOF'
$DEPLOY_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $VPS_USER $VPS_USER
    postrotate
        docker-compose restart quantx-engine
    endscript
}
LOGROTATE_EOF
        
        # Test log rotation
        sudo logrotate -d /etc/logrotate.d/quantx-engine
EOF
    
    log_info "Monitoring setup completed."
}

# Setup firewall
setup_firewall() {
    log_info "Setting up firewall..."
    
    ssh -p $VPS_PORT $VPS_USER@$VPS_HOST << 'EOF'
        # Install ufw if not present
        if ! command -v ufw &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y ufw
        fi
        
        # Configure firewall
        sudo ufw default deny incoming
        sudo ufw default allow outgoing
        
        # Allow SSH
        sudo ufw allow ssh
