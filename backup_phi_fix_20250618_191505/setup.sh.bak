#!/bin/bash
# enhanced_setup.sh - Complete setup script for Enhanced 4-Model LLM Proxy
# Supports: Development, Docker, Production deployments

set -e

# =============================================================================
# Configuration and Colors
# =============================================================================
SCRIPT_VERSION="2.2.0"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_header() { echo -e "\n${BLUE}$1${NC}"; echo "$(printf '=%.0s' {1..50})"; }
print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# =============================================================================
# Banner and Introduction
# =============================================================================
print_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
🚀 ════════════════════════════════════════════════════════════════════════════════ 🚀
                     ENHANCED 4-MODEL LLM PROXY SETUP
                              
    🤖 AI Models:
    ├── 🧠 Phi-3.5 Reasoning     → Complex math, logic, scientific analysis
    ├── 🎨 Llama3 8B-Instruct   → Creative writing, conversations, storytelling  
    ├── ⚙️  Gemma 7B-Instruct    → Technical documentation, coding, programming
    └── ⚡ Mistral 7B           → Quick facts, summaries, efficient responses
    
    ✨ Enhanced Features:
    ├── 🧠 Smart content-based routing      ├── 📊 Real-time performance monitoring
    ├── 🔄 Intelligent caching system       ├── 🎯 Semantic intent classification  
    ├── ⚡ React dashboard interface        └── 📈 Advanced analytics dashboard
    
🚀 ════════════════════════════════════════════════════════════════════════════════ 🚀
EOF
    echo -e "${NC}"
}

# =============================================================================
# System Detection and Requirements
# =============================================================================
detect_system() {
    print_header "🔍 System Detection"
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if [ -f /etc/debian_version ]; then
            DISTRO="debian"
        elif [ -f /etc/redhat-release ]; then
            DISTRO="redhat"
        else
            DISTRO="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        DISTRO="windows"
    else
        OS="unknown"
        DISTRO="unknown"
    fi
    
    # Detect container environment
    CONTAINER_ENV="none"
    if [ -f /.dockerenv ]; then
        CONTAINER_ENV="docker"
    elif [ -n "$RUNPOD_POD_ID" ]; then
        CONTAINER_ENV="runpod"
    elif [ -n "$COLAB_GPU" ]; then
        CONTAINER_ENV="colab"
    fi
    
    print_status "Operating System: $OS ($DISTRO)"
    print_status "Container Environment: $CONTAINER_ENV"
    
    # Detect available resources
    if command -v free >/dev/null 2>&1; then
        TOTAL_RAM=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
        print_status "Available RAM: ${TOTAL_RAM}GB"
    fi
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
        print_status "GPU Detected: $GPU_INFO"
        HAS_GPU=true
    else
        print_info "No GPU detected (CPU mode)"
        HAS_GPU=false
    fi
}

# =============================================================================
# Dependency Installation
# =============================================================================
install_dependencies() {
    print_header "📦 Installing Dependencies"
    
    # Python
    if ! command -v python3 >/dev/null 2>&1; then
        print_info "Installing Python..."
        if [ "$OS" = "linux" ] && [ "$DISTRO" = "debian" ]; then
            sudo apt update && sudo apt install -y python3 python3-pip python3-venv
        elif [ "$OS" = "macos" ]; then
            brew install python3
        else
            print_error "Please install Python 3.8+ manually"
            exit 1
        fi
    fi
    print_status "Python $(python3 --version | cut -d' ' -f2) available"
    
    # Node.js
    if ! command -v node >/dev/null 2>&1; then
        print_info "Installing Node.js..."
        if [ "$OS" = "linux" ]; then
            curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
            sudo apt-get install -y nodejs
        elif [ "$OS" = "macos" ]; then
            brew install node
        else
            print_error "Please install Node.js 18+ manually"
            exit 1
        fi
    fi
    print_status "Node.js $(node --version) available"
    
    # Docker
    if ! command -v docker >/dev/null 2>&1; then
        print_info "Installing Docker..."
        if [ "$OS" = "linux" ]; then
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
        elif [ "$OS" = "macos" ]; then
            print_warning "Please install Docker Desktop manually from https://docker.com"
        fi
    else
        print_status "Docker $(docker --version | cut -d' ' -f3 | tr -d ',') available"
    fi
    
    # Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1; then
        print_info "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    print_status "Docker Compose available"
    
    # Ollama
    if ! command -v ollama >/dev/null 2>&1; then
        print_info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    print_status "Ollama available"
}

# =============================================================================
# Project Structure Setup
# =============================================================================
setup_project_structure() {
    print_header "📁 Setting Up Project Structure"
    
    # Create directories
    directories=(
        "data/cache"
        "data/logs" 
        "data/models"
        "logs"
        "cache"
        "models"
        "static"
        "tests"
        "scripts"
        "backups"
        "frontend/build"
        "services"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        touch "$dir/.gitkeep" 2>/dev/null || true
    done
    
    print_status "Project directories created"
    
    # Set permissions
    chmod 755 data logs cache models static 2>/dev/null || true
    chmod +x scripts/*.sh 2>/dev/null || true
    
    print_status "Permissions set"
}

# =============================================================================
# Environment Configuration
# =============================================================================
setup_environment() {
    print_header "⚙️ Configuring Environment"
    
    # Detect optimal configuration based on system
    if [ "$HAS_GPU" = true ] && [ "$TOTAL_RAM" -gt 16 ]; then
        CONFIG_PROFILE="high-performance"
        MAX_MEMORY=20480
        MODEL_MEMORY=12288
        CACHE_MEMORY=4096
        ENABLE_SEMANTIC=true
        MAX_MODELS=4
    elif [ "$TOTAL_RAM" -gt 8 ]; then
        CONFIG_PROFILE="standard"
        MAX_MEMORY=12288
        MODEL_MEMORY=8192
        CACHE_MEMORY=2048
        ENABLE_SEMANTIC=true
        MAX_MODELS=3
    else
        CONFIG_PROFILE="minimal"
        MAX_MEMORY=6144
        MODEL_MEMORY=4096
        CACHE_MEMORY=1024
        ENABLE_SEMANTIC=false
        MAX_MODELS=2
    fi
    
    print_status "Configuration profile: $CONFIG_PROFILE"
    
    # Create .env file
    cat > .env << EOF
# Enhanced 4-Model LLM Proxy Configuration
# Generated by setup script v$SCRIPT_VERSION
# Profile: $CONFIG_PROFILE

# =============================================================================
# Core Application Settings
# =============================================================================
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8001
LOG_LEVEL=INFO
DEBUG=false

# =============================================================================
# 4-Model System Configuration
# =============================================================================
DEFAULT_MODEL=mistral:7b-instruct-q4_0
ENABLE_4_MODEL_ROUTING=true
PHI_MODEL=phi:3.5
MISTRAL_MODEL=mistral:7b-instruct-q4_0
GEMMA_MODEL=gemma:7b-instruct
LLAMA_MODEL=llama3:8b-instruct-q4_0

# =============================================================================
# Ollama Configuration
# =============================================================================
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_NUM_PARALLEL=4
OLLAMA_MAX_LOADED_MODELS=$MAX_MODELS
OLLAMA_GPU_OVERHEAD=0
OLLAMA_DEBUG=INFO

# =============================================================================
# Memory Management (Optimized for $CONFIG_PROFILE)
# =============================================================================
MAX_MEMORY_MB=$MAX_MEMORY
CACHE_MEMORY_LIMIT_MB=$CACHE_MEMORY
MODEL_MEMORY_LIMIT_MB=$MODEL_MEMORY
SEMANTIC_MODEL_MAX_MEMORY_MB=1024

# =============================================================================
# Enhanced Features
# =============================================================================
ENABLE_SEMANTIC_CLASSIFICATION=$ENABLE_SEMANTIC
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_DETAILED_METRICS=true
ENABLE_DASHBOARD=true
ENABLE_REACT_DASHBOARD=true
ENABLE_WEBSOCKET_DASHBOARD=true
ENABLE_ENHANCED_ROUTING=true

# =============================================================================
# Performance & Caching
# =============================================================================
ENABLE_REDIS_CACHE=true
REDIS_URL=redis://localhost:6379
ENABLE_SEMANTIC_CACHE=true
SEMANTIC_SIMILARITY_THRESHOLD=0.85
ENABLE_CIRCUIT_BREAKER=true
ENABLE_CONNECTION_POOLING=true
ENABLE_PERFORMANCE_MONITORING=true

# =============================================================================
# Security & CORS
# =============================================================================
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-dev-key-change-in-production
CORS_ORIGINS=["http://localhost:3000","http://localhost:8001","*"]
CORS_ALLOW_CREDENTIALS=true

# =============================================================================
# Dashboard Configuration  
# =============================================================================
DASHBOARD_PATH=/app/static
ENABLE_WEBSOCKET=true

# =============================================================================
# Auto-download Models (for Docker)
# =============================================================================
AUTO_DOWNLOAD_MODELS=false
EOF
    
    print_status "Environment configuration created (.env)"
    
    # Create frontend environment
    if [ -d "frontend" ] || [ -f "frontend/package.json" ]; then
        mkdir -p frontend
        cat > frontend/.env << EOF
# Frontend Environment Configuration
REACT_APP_API_BASE_URL=http://localhost:8001
REACT_APP_WS_URL=ws://localhost:8001
REACT_APP_ENVIRONMENT=development
GENERATE_SOURCEMAP=false
REACT_APP_VERSION=$SCRIPT_VERSION
EOF
        print_status "Frontend environment configured"
    fi
}

# =============================================================================
# Python Dependencies Installation
# =============================================================================
install_python_dependencies() {
    print_header "🐍 Installing Python Dependencies"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate || true
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install base requirements
    if [ -f "requirements.txt" ]; then
        print_info "Installing base requirements..."
        pip install -r requirements.txt
        print_status "Base requirements installed"
    fi
    
    # Install enhanced features if enabled
    if [ "$ENABLE_SEMANTIC" = true ]; then
        print_info "Installing enhanced features..."
        pip install \
            sentence-transformers \
            faiss-cpu \
            sse-starlette \
            redis \
            aioredis \
            prometheus-client \
            numpy \
            scikit-learn \
            || print_warning "Some enhanced features may not be available"
        print_status "Enhanced features installed"
    fi
    
    print_status "Python dependencies installation complete"
}

# =============================================================================
# Frontend Setup
# =============================================================================
setup_frontend() {
    print_header "⚛️ Setting Up Frontend"
    
    if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
        cd frontend
        
        print_info "Installing frontend dependencies..."
        npm install --legacy-peer-deps || npm install
        
        print_info "Building frontend..."
        npm run build || print_warning "Frontend build failed, but continuing..."
        
        if [ -f "build/index.html" ]; then
            print_status "Frontend build successful"
        else
            print_warning "Frontend build may have issues"
        fi
        
        cd ..
    else
        print_info "Frontend directory not found, skipping frontend setup"
    fi
}

# =============================================================================
# Docker Setup
# =============================================================================
setup_docker() {
    print_header "🐳 Setting Up Docker Configuration"
    
    # Create optimized docker-compose.yml
    cat > docker-compose.yml << EOF
version: '3.8'

services:
  # Enhanced 4-Model LLM Proxy
  llm-proxy:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: llm-proxy-enhanced
    ports:
      - "8001:8001"
    environment:
      - HOST=0.0.0.0
      - PORT=8001
      - REDIS_URL=redis://redis:6379
      - AUTO_DOWNLOAD_MODELS=false
      - ENABLE_4_MODEL_ROUTING=true
      - MAX_MEMORY_MB=$MAX_MEMORY
      - ENABLE_SEMANTIC_CLASSIFICATION=$ENABLE_SEMANTIC
    volumes:
      - ollama_data:/root/.ollama
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 15s
      retries: 3
      start_period: 60s

  # Ollama Service
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-service
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_HOST=0.0.0.0
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
$(if [ "$HAS_GPU" = true ]; then cat << 'GPUEOF'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
GPUEOF
fi)

  # Redis for Caching
  redis:
    image: redis:7-alpine
    container_name: llm-proxy-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory ${CACHE_MEMORY}mb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 5s
      retries: 3

volumes:
  ollama_data:
    driver: local
  redis_data:
    driver: local

networks:
  default:
    name: llm-proxy-network
EOF
    
    print_status "Docker Compose configuration created"
    
    # Create .dockerignore if it doesn't exist
    if [ ! -f ".dockerignore" ]; then
        cat > .dockerignore << 'EOF'
# Development files
.git
.gitignore
*.md
.env
.venv
venv/
__pycache__/
*.pyc
node_modules/
.npm

# Logs and data
logs/
data/
*.log

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Test files
tests/
.pytest_cache/
coverage/

# Backup files
backups/
*.bak
*.backup
EOF
        print_status ".dockerignore created"
    fi
}

# =============================================================================
# Create Helper Scripts
# =============================================================================
create_helper_scripts() {
    print_header "📜 Creating Helper Scripts"
    
    # Development start script
    cat > start_dev.sh << 'EOF'
#!/bin/bash
# start_dev.sh - Start development environment

set -e

echo "🚀 Starting Enhanced 4-Model LLM Proxy (Development)"
echo "====================================================="

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start services
echo "📡 Starting Ollama..."
if ! pgrep -f "ollama serve" > /dev/null; then
    ollama serve > logs/ollama.log 2>&1 &
    sleep 5
fi

echo "📦 Starting Redis..."
if ! pgrep -f "redis-server" > /dev/null; then
    redis-server --daemonize yes --port 6379 > logs/redis.log 2>&1 || echo "Redis not available"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the application
echo "🌐 Starting FastAPI application..."
if [ -f "main_master.py" ]; then
    python main_master.py
elif [ -f "main.py" ]; then
    python main.py
else
    echo "❌ No main file found!"
    exit 1
fi
EOF

    # Docker start script
    cat > start_docker.sh << 'EOF'
#!/bin/bash
# start_docker.sh - Start with Docker

set -e

echo "🐳 Starting Enhanced 4-Model LLM Proxy (Docker)"
echo "==============================================="

# Build and start
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🏥 Performing health check..."
if curl -f http://localhost:8001/health >/dev/null 2>&1; then
    echo "✅ All services are healthy!"
    echo ""
    echo "🌐 Access points:"
    echo "  • Dashboard: http://localhost:8001/app"
    echo "  • API Docs:  http://localhost:8001/docs" 
    echo "  • Health:    http://localhost:8001/health"
    echo ""
    echo "📊 View logs: docker-compose logs -f"
else
    echo "❌ Health check failed"
    docker-compose logs
    exit 1
fi
EOF

    # Model download script
    cat > download_models.sh << 'EOF'
#!/bin/bash
# download_models.sh - Download the 4 models

set -e

echo "📦 Downloading 4 Models for Enhanced LLM Proxy"
echo "==============================================="

# Start Ollama if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🤖 Starting Ollama..."
    ollama serve > logs/ollama.log 2>&1 &
    sleep 10
fi

# Models to download
models=(
    "phi:3.5:🧠 Phi-3.5 (Reasoning)"
    "mistral:7b-instruct-q4_0:⚡ Mistral 7B (General)"
    "gemma:7b-instruct:⚙️ Gemma 7B (Technical)"
    "llama3:8b-instruct-q4_0:🎨 Llama3 8B (Creative)"
)

echo "Starting downloads..."
for model_info in "${models[@]}"; do
    IFS=':' read -r model desc <<< "$model_info"
    echo "📥 Downloading $desc..."
    ollama pull "$model" || echo "⚠️ Failed to download $model"
done

echo ""
echo "✅ Download complete!"
echo "📊 Available models:"
ollama list
EOF

    # Test script
    cat > test_system.sh << 'EOF'
#!/bin/bash
# test_system.sh - Test the complete system

set -e

BASE_URL="http://localhost:8001"

echo "🧪 Testing Enhanced 4-Model LLM Proxy"
echo "======================================"

# Test health
echo "1. Testing health endpoint..."
curl -f "$BASE_URL/health" && echo "✅ Health OK" || echo "❌ Health failed"

# Test models
echo "2. Testing models endpoint..."
curl -f "$BASE_URL/models" >/dev/null && echo "✅ Models OK" || echo "❌ Models failed"

# Test dashboard
echo "3. Testing dashboard..."
curl -f "$BASE_URL/app" >/dev/null && echo "✅ Dashboard OK" || echo "❌ Dashboard failed"

# Test API
echo "4. Testing API..."
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello"}],"max_tokens":5}' \
  >/dev/null && echo "✅ API OK" || echo "❌ API failed"

echo ""
echo "🎉 System testing complete!"
EOF

    # Make scripts executable
    chmod +x start_dev.sh start_docker.sh download_models.sh test_system.sh
    
    print_status "Helper scripts created"
}

# =============================================================================
# Setup Summary and Instructions
# =============================================================================
show_setup_summary() {
    print_header "🎉 Setup Complete!"
    
    echo "📋 What was configured:"
    echo "├── ✅ System dependencies installed"
    echo "├── ✅ Project structure created"
    echo "├── ✅ Environment configured ($CONFIG_PROFILE profile)"
    echo "├── ✅ Python dependencies installed"
    echo "├── ✅ Docker configuration created"
    echo "└── ✅ Helper scripts created"
    echo ""
    
    echo "🚀 Next Steps:"
    echo "=============="
    echo "1. Download the 4 models:"
    echo "   ./download_models.sh"
    echo ""
    echo "2. Start the system:"
    echo "   Development: ./start_dev.sh"
    echo "   Docker:      ./start_docker.sh"
    echo ""
    echo "3. Test the system:"
    echo "   ./test_system.sh"
    echo ""
    
    echo "🌐 Access Points (after startup):"
    echo "================================="
    echo "├── 📊 Dashboard:    http://localhost:8001/app"
    echo "├── 📚 API Docs:     http://localhost:8001/docs"
    echo "├── 🏥 Health:       http://localhost:8001/health"
    echo "├── 📈 Metrics:      http://localhost:8001/metrics"
    echo "└── 🔗 WebSocket:    ws://localhost:8001/ws"
    echo ""
    
    echo "🎯 Your 4 models will route as:"
    echo "==============================="
    echo "├── Math/Logic queries    → Phi-3.5"
    echo "├── Coding/Technical      → Gemma 7B"
    echo "├── Creative/Writing      → Llama3 8B"
    echo "└── General/Quick facts   → Mistral 7B"
    echo ""
    
    print_status "Enhanced 4-Model LLM Proxy setup completed successfully!"
    
    # Show resource recommendations
    echo ""
    echo "💡 Resource Recommendations:"
    echo "============================"
    if [ "$HAS_GPU" = true ]; then
        echo "├── ✅ GPU detected - optimal performance expected"
    else
        echo "├── ⚠️  No GPU - CPU mode (slower but functional)"
    fi
    
    if [ "$TOTAL_RAM" -gt 16 ]; then
        echo "├── ✅ ${TOTAL_RAM}GB RAM - excellent for all 4 models"
    elif [ "$TOTAL_RAM" -gt 8 ]; then
        echo "├── ✅ ${TOTAL_RAM}GB RAM - good for 3-4 models"
    else
        echo "├── ⚠️  ${TOTAL_RAM}GB RAM - may limit concurrent models"
    fi
    
    echo "└── 📊 Configuration: $CONFIG_PROFILE profile"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    print_banner
    
    echo "Enhanced 4-Model LLM Proxy Setup Script v$SCRIPT_VERSION"
    echo "========================================================="
    
    # Interactive mode check
    if [ "$1" = "--auto" ]; then
        AUTO_MODE=true
        print_info "Running in automatic mode"
    else
        echo ""
        read -p "Continue with setup? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
            echo "Setup cancelled."
            exit 0
        fi
        AUTO_MODE=false
    fi
    
    # Run setup steps
    detect_system
    install_dependencies
    setup_project_structure
    setup_environment
    install_python_dependencies
    setup_frontend
    setup_docker
    create_helper_scripts
    show_setup_summary
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "Run './start_dev.sh' to begin, or './start_docker.sh' for Docker deployment."
}

# Run main function with all arguments
main "$@"
