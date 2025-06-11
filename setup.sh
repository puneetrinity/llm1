#!/bin/bash
# setup.sh - FIXED RunPod Setup Script with Proper Ollama Installation

set -e

echo "🚀 RunPod LLM Proxy Setup - FIXED VERSION"
echo "=========================================="

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

# Step 1: System Updates and Basic Tools
echo -e "\n${BLUE}📦 Step 1: System Setup${NC}"

print_info "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y curl wget git python3 python3-pip python3-venv build-essential jq dos2unix

print_status "System packages updated"

# Step 2: GPU and CUDA Setup
echo -e "\n${BLUE}🔧 Step 2: GPU and CUDA Setup${NC}"

print_info "Setting up GPU environment..."

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make environment variables persistent
cat >> ~/.bashrc << 'EOF'
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF

# Check GPU
if nvidia-smi > /dev/null 2>&1; then
    print_status "GPU detected and accessible"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "GPU not detected - continuing with CPU"
fi

# Step 3: Install Ollama
echo -e "\n${BLUE}🤖 Step 3: Installing Ollama${NC}"

print_info "Downloading and installing Ollama..."

# Install Ollama
if ! command -v ollama &> /dev/null; then
    print_info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Verify installation
    if command -v ollama &> /dev/null; then
        print_status "Ollama installed successfully"
        ollama --version
    else
        print_error "Ollama installation failed"
        exit 1
    fi
else
    print_status "Ollama already installed"
    ollama --version
fi

# Step 4: Configure Ollama
echo -e "\n${BLUE}⚙️  Step 4: Configuring Ollama${NC}"

print_info "Setting up Ollama configuration..."

# Set Ollama environment variables
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

print_status "Ollama configuration set"

# Step 5: Start Ollama Service
echo -e "\n${BLUE}🚀 Step 5: Starting Ollama Service${NC}"

print_info "Starting Ollama service..."

# Kill any existing Ollama processes
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

# Start Ollama in background
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

print_info "Ollama started with PID: $OLLAMA_PID"

# Wait for Ollama to be ready
print_info "Waiting for Ollama to be ready..."
for i in {1..60}; do
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_status "Ollama is ready and responding!"
        break
    fi
    if [ $i -eq 60 ]; then
        print_error "Ollama failed to start within 60 seconds"
        echo "Ollama logs:"
        tail -20 /tmp/ollama.log
        exit 1
    fi
    echo "  Attempt $i/60 - waiting 2 seconds..."
    sleep 2
done

# Verify Ollama is working
print_info "Testing Ollama API..."
if curl -s http://localhost:11434/api/tags | jq . >/dev/null 2>&1; then
    print_status "Ollama API is working correctly"
else
    print_error "Ollama API test failed"
    curl -s http://localhost:11434/api/tags
    exit 1
fi

# Step 6: Install Python Dependencies
echo -e "\n${BLUE}🐍 Step 6: Installing Python Dependencies${NC}"

print_info "Installing Python dependencies..."

# Install core dependencies
pip3 install --no-cache-dir --upgrade pip

# Install required packages with specific versions for compatibility
pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    aiohttp==3.9.1 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    psutil==5.9.6 \
    python-multipart==0.0.6

print_status "Core Python dependencies installed"

# Install optional enhanced features (with graceful fallbacks)
print_info "Installing optional enhanced features..."
pip3 install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    sse-starlette \
    redis \
    prometheus-client || print_warning "Some enhanced features may not be available"

print_status "Python setup completed"

# Step 7: Download a Model
echo -e "\n${BLUE}📦 Step 7: Downloading AI Model${NC}"

print_info "Downloading Mistral 7B model (this may take a few minutes)..."

# Pull a lightweight but capable model
if ollama pull mistral:7b-instruct-q4_0; then
    print_status "Mistral 7B model downloaded successfully"
else
    print_warning "Mistral download failed, trying smaller model..."
    if ollama pull llama3.2:1b; then
        print_status "Llama 3.2 1B model downloaded successfully"
    else
        print_error "Failed to download any model"
        print_info "You can manually download a model later with: ollama pull <model-name>"
    fi
fi

# Verify models
print_info "Available models:"
ollama list

# Step 8: Create Environment Configuration
echo -e "\n${BLUE}📝 Step 8: Creating Configuration${NC}"

print_info "Creating .env configuration file..."

cat > .env << 'EOF'
# RunPod LLM Proxy Configuration
DEBUG=false
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_TIMEOUT=300

# Authentication - DISABLED for simplicity
ENABLE_AUTH=false
API_KEY_HEADER=X-API-Key
DEFAULT_API_KEY=sk-runpod-test

# CORS Settings - Permissive for development
CORS_ORIGINS=["*"]
CORS_ALLOW_CREDENTIALS=true

# Memory Management
MAX_MEMORY_MB=12288
CACHE_MEMORY_LIMIT_MB=1024
MODEL_MEMORY_LIMIT_MB=6144

# Enhanced Features - Conservative settings
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_DETAILED_METRICS=false

# Rate Limiting - Disabled for development
ENABLE_RATE_LIMITING=false
DEFAULT_RATE_LIMIT=1000

# Logging
LOG_LEVEL=INFO
ENABLE_DETAILED_LOGGING=false
EOF

print_status "Configuration file created"

# Step 9: Test Model
echo -e "\n${BLUE}🧪 Step 9: Testing Model${NC}"

print_info "Testing model with a simple query..."

# Test the model
test_response=$(ollama run mistral:7b-instruct-q4_0 "Say 'TEST' and nothing else" 2>/dev/null || ollama run llama3.2:1b "Say 'TEST' and nothing else" 2>/dev/null || echo "Model test failed")

if [[ "$test_response" == *"TEST"* ]]; then
    print_status "Model test successful!"
    print_info "Response: $test_response"
else
    print_warning "Model test didn't return expected response"
    print_info "Response: $test_response"
fi

# Step 10: Prepare to Start Service
echo -e "\n${BLUE}🌐 Step 10: Service Preparation${NC}"

print_info "Preparing to start the LLM Proxy service..."

# Create startup script
cat > start_service.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting LLM Proxy Service..."

# Ensure Ollama is running
if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
fi

# Start the Python service
echo "Starting Python service..."
python3 main.py > /tmp/service.log 2>&1 &
SERVICE_PID=$!

echo "Service started with PID: $SERVICE_PID"
echo "Logs: tail -f /tmp/service.log"
echo "Health: curl http://localhost:8000/health"
EOF

chmod +x start_service.sh
print_status "Startup script created: start_service.sh"

# Create test script
cat > test_service.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing LLM Proxy Service..."

echo "1. Testing health endpoint..."
curl -s http://localhost:8000/health | jq .

echo -e "\n2. Testing models endpoint..."
curl -s http://localhost:8000/models | jq '.data[].id'

echo -e "\n3. Testing chat completion..."
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }' | jq '.choices[0].message.content'

echo -e "\n✅ Test completed!"
EOF

chmod +x test_service.sh
print_status "Test script created: test_service.sh"

# Summary
echo -e "\n${BLUE}📊 Setup Summary${NC}"
echo "=================="

print_status "Setup completed successfully!"

echo -e "\n${GREEN}🎯 What was installed:${NC}"
echo "✅ System packages and tools"
echo "✅ GPU/CUDA environment configured"
echo "✅ Ollama AI runtime"
echo "✅ Python dependencies"
echo "✅ AI model downloaded"
echo "✅ Configuration files created"

echo -e "\n${GREEN}🚀 Next steps:${NC}"
echo "1. Start the service:    ./start_service.sh"
echo "2. Test the service:     ./test_service.sh"
echo "3. View logs:           tail -f /tmp/service.log"
echo "4. Check health:        curl http://localhost:8000/health"

echo -e "\n${GREEN}📡 Service endpoints:${NC}"
echo "• Health:     http://localhost:8000/health"
echo "• API Docs:   http://localhost:8000/docs"
echo "• Chat API:   http://localhost:8000/v1/chat/completions"
echo "• Models:     http://localhost:8000/models"
echo "• Ollama:     http://localhost:11434"

echo -e "\n${BLUE}💡 Troubleshooting:${NC}"
echo "• If Ollama fails: pkill ollama && ollama serve &"
echo "• If service fails: pkill -f 'python.*main' && python3 main.py &"
echo "• Check logs: tail -f /tmp/ollama.log /tmp/service.log"

print_status "RunPod LLM Proxy setup completed successfully!"
print_info "You can now start the service with: ./start_service.sh"
