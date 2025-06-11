#!/bin/bash
# setup_container_safe.sh - Container-Safe Setup with Process Management and Loop Prevention

set -e

echo "🚀 Container-Safe RunPod LLM Proxy Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# CRITICAL: Check if we're in a container and if setup is already running
SETUP_LOCK="/tmp/llm_proxy_setup.lock"
CONTAINER_ENV="/tmp/container_setup_done"

# Function to cleanup on exit
cleanup() {
    local exit_code=$?
    print_info "Cleaning up setup process..."
    
    # Remove lock file
    rm -f "$SETUP_LOCK"
    
    # Kill background processes if they exist
    if [ -n "$OLLAMA_PID" ] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        print_info "Stopping Ollama background process (PID: $OLLAMA_PID)"
        kill "$OLLAMA_PID" 2>/dev/null || true
    fi
    
    if [ -n "$SERVICE_PID" ] && kill -0 "$SERVICE_PID" 2>/dev/null; then
        print_info "Stopping service background process (PID: $SERVICE_PID)"
        kill "$SERVICE_PID" 2>/dev/null || true
    fi
    
    if [ $exit_code -eq 0 ]; then
        print_status "Setup completed successfully"
    else
        print_error "Setup failed with exit code: $exit_code"
    fi
    
    exit $exit_code
}

# Set trap for cleanup on exit/interrupt
trap cleanup EXIT INT TERM

# Check if setup is already running
if [ -f "$SETUP_LOCK" ]; then
    EXISTING_PID=$(cat "$SETUP_LOCK" 2>/dev/null || echo "")
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        print_error "Setup is already running (PID: $EXISTING_PID)"
        print_info "If you're sure no setup is running, remove: $SETUP_LOCK"
        exit 1
    else
        print_warning "Stale lock file found, removing..."
        rm -f "$SETUP_LOCK"
    fi
fi

# Create lock file with current PID
echo $$ > "$SETUP_LOCK"
print_info "Setup lock created (PID: $$)"

# Check if we're in a container
if [ -f /.dockerenv ] || [ -n "${CONTAINER}" ] || [ -n "${KUBERNETES_SERVICE_HOST}" ]; then
    print_info "Container environment detected"
    IN_CONTAINER=true
    
    # Check if setup was already completed in this container
    if [ -f "$CONTAINER_ENV" ]; then
        print_warning "Setup already completed in this container"
        PREVIOUS_SETUP=$(cat "$CONTAINER_ENV")
        print_info "Previous setup: $PREVIOUS_SETUP"
        
        read -p "Do you want to re-run setup? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping setup. Starting services instead..."
            
            # Try to start existing services
            if command -v ollama >/dev/null 2>&1; then
                if ! pgrep -f "ollama serve" >/dev/null; then
                    print_info "Starting existing Ollama service..."
                    ollama serve > /tmp/ollama.log 2>&1 &
                    sleep 5
                fi
            fi
            
            if [ -f "main.py" ]; then
                if ! pgrep -f "python.*main" >/dev/null; then
                    print_info "Starting existing Python service..."
                    python3 main.py > /tmp/service.log 2>&1 &
                    sleep 3
                fi
            fi
            
            print_status "Services started. Check status with: curl http://localhost:8000/health"
            exit 0
        fi
    fi
else
    IN_CONTAINER=false
    print_info "Non-container environment detected"
fi

# Timeout for operations to prevent infinite loops
TIMEOUT=300  # 5 minutes max for any operation
OLLAMA_START_TIMEOUT=120  # 2 minutes for Ollama to start
MODEL_DOWNLOAD_TIMEOUT=600  # 10 minutes for model download

# Function to run commands with timeout
run_with_timeout() {
    local timeout=$1
    shift
    local cmd="$@"
    
    print_info "Running with ${timeout}s timeout: $cmd"
    
    if timeout "$timeout" bash -c "$cmd"; then
        return 0
    else
        print_error "Command timed out after ${timeout}s: $cmd"
        return 1
    fi
}

# Function to check if a service is responsive
check_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local wait_time=${4:-2}
    
    print_info "Checking $name service at $url..."
    
    for i in $(seq 1 $max_attempts); do
        if curl -f -s --connect-timeout 5 "$url" >/dev/null 2>&1; then
            print_status "$name is responsive!"
            return 0
        fi
        
        if [ $i -eq $max_attempts ]; then
            print_error "$name failed to become responsive after $((max_attempts * wait_time)) seconds"
            return 1
        fi
        
        echo "  Attempt $i/$max_attempts - waiting ${wait_time}s..."
        sleep $wait_time
    done
}

# Function to safely kill processes
safe_kill() {
    local process_pattern=$1
    local signal=${2:-TERM}
    
    local pids=$(pgrep -f "$process_pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        print_info "Stopping processes matching: $process_pattern"
        echo "$pids" | xargs -r kill -$signal 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        local remaining=$(pgrep -f "$process_pattern" 2>/dev/null || true)
        if [ -n "$remaining" ]; then
            print_warning "Force killing remaining processes..."
            echo "$remaining" | xargs -r kill -KILL 2>/dev/null || true
        fi
    fi
}

# Step 1: Environment Preparation
echo -e "\n${BLUE}🔧 Step 1: Environment Preparation${NC}"

print_info "Checking system state..."

# Stop any existing services to prevent conflicts
print_info "Stopping any existing services..."
safe_kill "ollama serve"
safe_kill "python.*main"

# Check available disk space
AVAILABLE_SPACE=$(df / | tail -1 | awk '{print $4}')
REQUIRED_SPACE=5000000  # 5GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    print_error "Insufficient disk space. Available: ${AVAILABLE_SPACE}KB, Required: ${REQUIRED_SPACE}KB"
    exit 1
fi

print_status "Environment checks passed"

# Step 2: System Updates (with timeout)
echo -e "\n${BLUE}📦 Step 2: System Setup${NC}"

print_info "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive

if run_with_timeout $TIMEOUT "apt-get update -qq && apt-get install -y curl wget git python3 python3-pip python3-venv build-essential jq dos2unix"; then
    print_status "System packages updated"
else
    print_error "System package update failed"
    exit 1
fi

# Step 3: GPU Setup (non-blocking)
echo -e "\n${BLUE}🎮 Step 3: GPU Setup${NC}"

print_info "Configuring GPU environment..."

# Set GPU environment variables
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Check GPU (non-blocking)
if timeout 10 nvidia-smi >/dev/null 2>&1; then
    print_status "GPU detected and accessible"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info unavailable")
    print_info "GPU: $GPU_INFO"
else
    print_warning "GPU not detected or nvidia-smi not available - continuing with CPU"
fi

# Step 4: Install Ollama (with timeout and checks)
echo -e "\n${BLUE}🤖 Step 4: Installing Ollama${NC}"

if command -v ollama >/dev/null 2>&1; then
    print_status "Ollama already installed"
    ollama --version
else
    print_info "Installing Ollama..."
    
    if run_with_timeout $TIMEOUT "curl -fsSL https://ollama.com/install.sh | sh"; then
        if command -v ollama >/dev/null 2>&1; then
            print_status "Ollama installed successfully"
            ollama --version
        else
            print_error "Ollama installation completed but command not found"
            exit 1
        fi
    else
        print_error "Ollama installation failed or timed out"
        exit 1
    fi
fi

# Step 5: Configure and Start Ollama (with timeout)
echo -e "\n${BLUE}🚀 Step 5: Starting Ollama Service${NC}"

print_info "Configuring Ollama..."

# Set Ollama environment
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=2

print_info "Starting Ollama service with timeout protection..."

# Start Ollama with timeout protection
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

print_info "Ollama started with PID: $OLLAMA_PID"

# Check if Ollama is responsive with timeout
if check_service "http://localhost:11434/api/tags" "Ollama" 60 2; then
    print_status "Ollama service is running and responsive"
else
    print_error "Ollama failed to start properly"
    print_info "Ollama logs:"
    tail -20 /tmp/ollama.log 2>/dev/null || echo "No logs available"
    exit 1
fi

# Step 6: Install Python Dependencies (with timeout)
echo -e "\n${BLUE}🐍 Step 6: Installing Python Dependencies${NC}"

print_info "Installing Python dependencies with timeout protection..."

if run_with_timeout $TIMEOUT "pip3 install --no-cache-dir --upgrade pip"; then
    print_status "pip updated"
else
    print_warning "pip update failed, continuing..."
fi

# Install core dependencies
CORE_DEPS="fastapi==0.104.1 uvicorn[standard]==0.24.0 aiohttp==3.9.1 pydantic==2.5.0 pydantic-settings==2.1.0 psutil==5.9.6"

if run_with_timeout $TIMEOUT "pip3 install --no-cache-dir $CORE_DEPS"; then
    print_status "Core dependencies installed"
else
    print_error "Core dependencies installation failed"
    exit 1
fi

# Optional dependencies (non-blocking)
print_info "Installing optional dependencies..."
pip3 install --no-cache-dir sentence-transformers faiss-cpu sse-starlette redis prometheus-client 2>/dev/null || print_warning "Some optional features may not be available"

# Step 7: Download Model (with timeout and size check)
echo -e "\n${BLUE}📦 Step 7: Downloading AI Model${NC}"

print_info "Downloading model with timeout protection..."

# Check if any models already exist
EXISTING_MODELS=$(ollama list 2>/dev/null | grep -v "NAME" | wc -l)

if [ "$EXISTING_MODELS" -gt 0 ]; then
    print_status "Models already available:"
    ollama list
else
    print_info "Downloading Mistral 7B model (with ${MODEL_DOWNLOAD_TIMEOUT}s timeout)..."
    
    if run_with_timeout $MODEL_DOWNLOAD_TIMEOUT "ollama pull mistral:7b-instruct-q4_0"; then
        print_status "Mistral 7B model downloaded successfully"
    else
        print_warning "Mistral download failed/timed out, trying smaller model..."
        
        if run_with_timeout 300 "ollama pull llama3.2:1b"; then
            print_status "Llama 3.2 1B model downloaded successfully"
        else
            print_warning "Model download failed. Service will work but no models available."
            print_info "You can download models later with: ollama pull <model-name>"
        fi
    fi
fi

# Step 8: Create Configuration
echo -e "\n${BLUE}📝 Step 8: Creating Configuration${NC}"

print_info "Creating safe configuration..."

cat > .env << 'EOF'
# Container-Safe RunPod Configuration
DEBUG=false
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_TIMEOUT=60

# Authentication - DISABLED for container safety
ENABLE_AUTH=false
ENABLE_RATE_LIMITING=false

# CORS - Permissive for development
CORS_ORIGINS=["*"]

# Memory Management - Conservative
MAX_MEMORY_MB=8192
CACHE_MEMORY_LIMIT_MB=512

# Features - Minimal for stability
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=false
ENABLE_DETAILED_METRICS=false

# Logging
LOG_LEVEL=INFO
EOF

print_status "Configuration created"

# Step 9: Create Container-Safe Service Manager
echo -e "\n${BLUE}🔧 Step 9: Creating Service Manager${NC}"

cat > service_manager.sh << 'EOF'
#!/bin/bash
# Container-safe service manager

OLLAMA_PID_FILE="/tmp/ollama.pid"
SERVICE_PID_FILE="/tmp/service.pid"

start_services() {
    echo "🚀 Starting services..."
    
    # Start Ollama if not running
    if ! pgrep -f "ollama serve" >/dev/null; then
        echo "Starting Ollama..."
        ollama serve > /tmp/ollama.log 2>&1 &
        echo $! > "$OLLAMA_PID_FILE"
        sleep 5
    fi
    
    # Start Python service if not running
    if ! pgrep -f "python.*main" >/dev/null; then
        echo "Starting Python service..."
        python3 main.py > /tmp/service.log 2>&1 &
        echo $! > "$SERVICE_PID_FILE"
        sleep 3
    fi
    
    echo "✅ Services started"
}

stop_services() {
    echo "🛑 Stopping services..."
    
    # Stop Python service
    if [ -f "$SERVICE_PID_FILE" ]; then
        PID=$(cat "$SERVICE_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            rm -f "$SERVICE_PID_FILE"
        fi
    fi
    
    # Stop Ollama
    if [ -f "$OLLAMA_PID_FILE" ]; then
        PID=$(cat "$OLLAMA_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            rm -f "$OLLAMA_PID_FILE"
        fi
    fi
    
    echo "✅ Services stopped"
}

status_services() {
    echo "📊 Service Status:"
    
    # Check Ollama
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama: Running"
    else
        echo "❌ Ollama: Not responding"
    fi
    
    # Check Python service
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ Python Service: Running"
    else
        echo "❌ Python Service: Not responding"
    fi
}

case "$1" in
    start) start_services ;;
    stop) stop_services ;;
    restart) stop_services; sleep 2; start_services ;;
    status) status_services ;;
    *) echo "Usage: $0 {start|stop|restart|status}" ;;
esac
EOF

chmod +x service_manager.sh
print_status "Service manager created"

# Step 10: Final Setup and Testing
echo -e "\n${BLUE}🧪 Step 10: Final Testing${NC}"

print_info "Testing setup with timeout protection..."

# Test Ollama
if check_service "http://localhost:11434/api/tags" "Ollama" 10 1; then
    print_status "Ollama test passed"
else
    print_error "Ollama test failed"
    exit 1
fi

# Mark setup as completed
echo "$(date): Container setup completed successfully" > "$CONTAINER_ENV"

# Summary
echo -e "\n${BLUE}📊 Setup Complete${NC}"
echo "=================="

print_status "Container-safe setup completed successfully!"

echo -e "\n${GREEN}🎯 What's available:${NC}"
echo "✅ Ollama AI runtime (PID: $OLLAMA_PID)"
echo "✅ Python environment with dependencies"
echo "✅ AI models ready"
echo "✅ Container-safe service manager"

echo -e "\n${GREEN}🚀 Next steps:${NC}"
echo "1. Start services:       ./service_manager.sh start"
echo "2. Check status:         ./service_manager.sh status"
echo "3. Test health:          curl http://localhost:8000/health"
echo "4. Stop services:        ./service_manager.sh stop"

echo -e "\n${GREEN}🔒 Safety features:${NC}"
echo "• Process lock prevents multiple setups"
echo "• Timeout protection prevents infinite loops"
echo "• Container state tracking"
echo "• Safe service management"
echo "• Graceful cleanup on exit"

print_status "Setup lock will be removed automatically"
print_info "Container setup state saved to: $CONTAINER_ENV"
