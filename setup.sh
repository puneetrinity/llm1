#!/bin/bash
# setup_noninteractive.sh - Fully Non-Interactive Container Setup for RunPod

set -e

echo "🚀 Non-Interactive RunPod LLM Proxy Setup"
echo "========================================="

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

# CRITICAL: Non-interactive configuration
SETUP_LOCK="/tmp/llm_proxy_setup.lock"
CONTAINER_ENV="/tmp/container_setup_done"
FORCE_SETUP="${FORCE_SETUP:-false}"  # Can be set via environment variable

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

# Check container environment - NO INTERACTIVE PROMPTS
if [ -f /.dockerenv ] || [ -n "${CONTAINER}" ] || [ -n "${KUBERNETES_SERVICE_HOST}" ]; then
    print_info "Container environment detected"
    IN_CONTAINER=true
    
    # Check if setup was already completed
    if [ -f "$CONTAINER_ENV" ] && [ "$FORCE_SETUP" != "true" ]; then
        print_info "Setup already completed in this container"
        PREVIOUS_SETUP=$(cat "$CONTAINER_ENV")
        print_info "Previous setup: $PREVIOUS_SETUP"
        
        # NON-INTERACTIVE: Always start services instead of re-running setup
        print_info "Starting existing services (non-interactive mode)..."
        
        # Try to start existing services
        if command -v ollama >/dev/null 2>&1; then
            if ! pgrep -f "ollama serve" >/dev/null; then
                print_info "Starting existing Ollama service..."
                export OLLAMA_HOST=0.0.0.0:11434
                export CUDA_VISIBLE_DEVICES=0
                ollama serve > /tmp/ollama.log 2>&1 &
                sleep 5
                
                # Wait for Ollama to be ready
                for i in {1..30}; do
                    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
                        print_status "Ollama service started successfully"
                        break
                    fi
                    sleep 2
                done
            else
                print_status "Ollama already running"
            fi
        fi
        
        if [ -f "main.py" ]; then
            if ! pgrep -f "python.*main" >/dev/null; then
                print_info "Starting existing Python service..."
                python3 main.py > /tmp/service.log 2>&1 &
                sleep 3
                
                # Wait for service to be ready
                for i in {1..20}; do
                    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
                        print_status "Python service started successfully"
                        break
                    fi
                    sleep 2
                done
            else
                print_status "Python service already running"
            fi
        fi
        
        # Test services
        print_info "Testing services..."
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            print_status "✅ Services are running and healthy!"
            print_info "Health check: curl http://localhost:8000/health"
            print_info "API docs: http://localhost:8000/docs"
            print_info "Ollama API: http://localhost:11434"
        else
            print_warning "Services started but health check failed"
            print_info "Check logs: tail -f /tmp/service.log /tmp/ollama.log"
        fi
        
        exit 0
    else
        if [ "$FORCE_SETUP" = "true" ]; then
            print_warning "FORCE_SETUP=true, re-running setup despite previous completion"
        fi
    fi
else
    IN_CONTAINER=false
    print_info "Non-container environment detected"
fi

# Timeout for operations
TIMEOUT=300
OLLAMA_START_TIMEOUT=120
MODEL_DOWNLOAD_TIMEOUT=600

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

print_info "Preparing environment for setup..."

# Stop any existing services
print_info "Stopping any existing services..."
safe_kill "ollama serve"
safe_kill "python.*main"

# Check disk space
AVAILABLE_SPACE=$(df / | tail -1 | awk '{print $4}')
REQUIRED_SPACE=5000000  # 5GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    print_error "Insufficient disk space. Available: ${AVAILABLE_SPACE}KB, Required: ${REQUIRED_SPACE}KB"
    exit 1
fi

print_status "Environment preparation completed"

# Step 2: System Updates
echo -e "\n${BLUE}📦 Step 2: System Setup${NC}"

print_info "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive

if run_with_timeout $TIMEOUT "apt-get update -qq && apt-get install -y curl wget git python3 python3-pip python3-venv build-essential jq dos2unix"; then
    print_status "System packages updated"
else
    print_error "System package update failed"
    exit 1
fi

# Step 3: GPU Setup
echo -e "\n${BLUE}🎮 Step 3: GPU Setup${NC}"

print_info "Configuring GPU environment..."

# Set GPU environment variables
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make environment persistent
cat >> ~/.bashrc << 'EOF'
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF

# Check GPU
if timeout 10 nvidia-smi >/dev/null 2>&1; then
    print_status "GPU detected and accessible"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info unavailable")
    print_info "GPU: $GPU_INFO"
else
    print_warning "GPU not detected - continuing with CPU mode"
fi

# Step 4: Install Ollama
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

# Step 5: Start Ollama Service
echo -e "\n${BLUE}🚀 Step 5: Starting Ollama Service${NC}"

print_info "Configuring and starting Ollama..."

# Set Ollama environment
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=2

# Make Ollama environment persistent
cat >> ~/.bashrc << 'EOF'
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=2
EOF

print_info "Starting Ollama service..."

# Start Ollama
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

print_info "Ollama started with PID: $OLLAMA_PID"

# Check if Ollama is responsive
if check_service "http://localhost:11434/api/tags" "Ollama" 60 2; then
    print_status "Ollama service is running and responsive"
else
    print_error "Ollama failed to start properly"
    print_info "Ollama logs:"
    tail -20 /tmp/ollama.log 2>/dev/null || echo "No logs available"
    exit 1
fi

# Step 6: Install Python Dependencies
echo -e "\n${BLUE}🐍 Step 6: Installing Python Dependencies${NC}"

print_info "Installing Python dependencies..."

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

# Optional dependencies
print_info "Installing optional dependencies..."
pip3 install --no-cache-dir sentence-transformers faiss-cpu sse-starlette redis prometheus-client 2>/dev/null || print_warning "Some optional features may not be available"

# Step 7: Download Model
echo -e "\n${BLUE}📦 Step 7: Downloading AI Model${NC}"

# Check existing models
EXISTING_MODELS=$(ollama list 2>/dev/null | grep -v "NAME" | wc -l)

if [ "$EXISTING_MODELS" -gt 0 ]; then
    print_status "Models already available:"
    ollama list
else
    print_info "Downloading model..."
    
    # Try Mistral first, fallback to smaller model
    if run_with_timeout $MODEL_DOWNLOAD_TIMEOUT "ollama pull mistral:7b-instruct-q4_0"; then
        print_status "Mistral 7B model downloaded successfully"
    else
        print_warning "Mistral download failed, trying smaller model..."
        
        if run_with_timeout 300 "ollama pull llama3.2:1b"; then
            print_status "Llama 3.2 1B model downloaded successfully"
        else
            print_warning "Model download failed. Service will work but responses may fail."
        fi
    fi
fi

# Step 8: Create Configuration
echo -e "\n${BLUE}📝 Step 8: Creating Configuration${NC}"

print_info "Creating configuration files..."

cat > .env << 'EOF'
# Non-Interactive RunPod Configuration
DEBUG=false
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_TIMEOUT=60

# Authentication - DISABLED for simplicity
ENABLE_AUTH=false
ENABLE_RATE_LIMITING=false

# CORS - Open for development
CORS_ORIGINS=["*"]

# Memory - Conservative
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

# Step 9: Create Service Management Scripts
echo -e "\n${BLUE}🔧 Step 9: Creating Management Scripts${NC}"

# Non-interactive service manager
cat > run_services.sh << 'EOF'
#!/bin/bash
# Non-interactive service runner

echo "🚀 Starting LLM Proxy Services..."

# Set environment
export OLLAMA_HOST=0.0.0.0:11434
export CUDA_VISIBLE_DEVICES=0

# Start Ollama if not running
if ! pgrep -f "ollama serve" >/dev/null; then
    echo "Starting Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
    
    # Wait for Ollama
    for i in {1..30}; do
        if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama ready"
            break
        fi
        sleep 2
    done
fi

# Start Python service if not running
if ! pgrep -f "python.*main" >/dev/null; then
    echo "Starting Python service..."
    python3 main.py > /tmp/service.log 2>&1 &
    sleep 3
    
    # Wait for service
    for i in {1..20}; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            echo "✅ Service ready"
            break
        fi
        sleep 2
    done
fi

echo "🎯 Services Status:"
curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health

echo ""
echo "📡 Endpoints:"
echo "• Health: http://localhost:8000/health"
echo "• Docs: http://localhost:8000/docs"
echo "• Chat: http://localhost:8000/v1/chat/completions"
EOF

chmod +x run_services.sh

# Quick test script
cat > test_api.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing LLM Proxy API..."

echo "1. Health Check:"
curl -s http://localhost:8000/health | jq .

echo -e "\n2. Models:"
curl -s http://localhost:8000/models | jq '.data[].id'

echo -e "\n3. Chat Test:"
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Say TEST"}],
    "max_tokens": 5
  }' | jq '.choices[0].message.content'

echo -e "\n✅ API test completed"
EOF

chmod +x test_api.sh

print_status "Management scripts created"

# Step 10: Mark Setup Complete and Start Services
echo -e "\n${BLUE}🎯 Step 10: Completing Setup${NC}"

# Mark setup as completed
echo "$(date): Non-interactive container setup completed successfully" > "$CONTAINER_ENV"

print_info "Starting services automatically..."

# Start the service automatically
if [ -f "main.py" ]; then
    print_info "Starting Python service..."
    python3 main.py > /tmp/service.log 2>&1 &
    SERVICE_PID=$!
    
    print_info "Service started with PID: $SERVICE_PID"
    
    # Wait for service to be ready
    if check_service "http://localhost:8000/health" "Python Service" 20 3; then
        print_status "Python service is ready!"
    else
        print_warning "Service started but may not be fully ready"
    fi
else
    print_warning "main.py not found - service not started"
fi

# Final Summary
echo -e "\n${BLUE}📊 Setup Complete${NC}"
echo "=================="

print_status "Non-interactive setup completed successfully!"

echo -e "\n${GREEN}🎯 What's Running:${NC}"
echo "✅ Ollama AI runtime (PID: $OLLAMA_PID)"
if [ -n "$SERVICE_PID" ]; then
    echo "✅ Python LLM Proxy (PID: $SERVICE_PID)"
fi

echo -e "\n${GREEN}🌐 Endpoints:${NC}"
echo "• Health:     http://localhost:8000/health"
echo "• API Docs:   http://localhost:8000/docs"
echo "• Chat API:   http://localhost:8000/v1/chat/completions"
echo "• Ollama:     http://localhost:11434"

echo -e "\n${GREEN}🔧 Commands:${NC}"
echo "• Start services: ./run_services.sh"
echo "• Test API:       ./test_api.sh"
echo "• View logs:      tail -f /tmp/service.log /tmp/ollama.log"

echo -e "\n${GREEN}🧪 Quick Test:${NC}"
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    print_status "✅ Service is healthy and ready!"
    HEALTH_STATUS=$(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "unknown")
    print_info "Status: $HEALTH_STATUS"
else
    print_warning "⚠️  Service may still be starting..."
    print_info "Wait a moment and check: curl http://localhost:8000/health"
fi

print_status "Setup lock will be removed automatically on exit"
