#!/bin/bash

# build_and_deploy.sh - Build Clean Enhanced LLM Proxy Image for RunPod

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

echo "🚀 Building Clean Enhanced LLM Proxy for RunPod"
echo "=============================================="

# Step 1: Verify files exist
print_info "Step 1: Verifying required files..."

required_files=(
    "main.py"
    "config.py" 
    "requirements.txt"
    "Dockerfile.clean"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    print_error "Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

print_status "All required files found"

# Step 2: Create requirements.txt if minimal
print_info "Step 2: Ensuring complete requirements.txt..."

if [ ! -s requirements.txt ] || [ $(wc -l < requirements.txt) -lt 5 ]; then
    print_warning "requirements.txt appears minimal, creating comprehensive version..."
    
    cat > requirements.txt << 'EOF'
# Core FastAPI dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
aiofiles==23.2.1

# HTTP and async
aiohttp==3.9.1
httpx==0.25.2
requests==2.31.0

# Data handling
pydantic==2.5.0
pydantic-settings==2.1.0

# Enhanced features
sentence-transformers==2.2.2
scikit-learn==1.3.2
sse-starlette==1.8.2

# System monitoring
psutil==5.9.6

# Optional Redis support
redis==5.0.1

# Development (optional)
pytest==7.4.3
pytest-asyncio==0.21.1
EOF
    
    print_status "Created comprehensive requirements.txt"
fi

# Step 3: Build Docker image
print_info "Step 3: Building Docker image..."

IMAGE_NAME="enhanced-llm-proxy"
IMAGE_TAG="clean-$(date +%Y%m%d-%H%M%S)"

print_info "Building: $IMAGE_NAME:$IMAGE_TAG"

if docker build -f Dockerfile.clean -t "$IMAGE_NAME:$IMAGE_TAG" -t "$IMAGE_NAME:latest" .; then
    print_status "Docker image built successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Step 4: Test image locally (optional)
print_info "Step 4: Testing image locally..."

# Kill any existing containers
docker rm -f enhanced-llm-proxy-test 2>/dev/null || true

# Start test container
print_info "Starting test container..."
docker run -d \
    --name enhanced-llm-proxy-test \
    -p 8001:8001 \
    "$IMAGE_NAME:latest" || {
    print_error "Failed to start test container"
    exit 1
}

# Wait for startup
print_info "Waiting for service to start..."
sleep 30

# Test health endpoint
if curl -s -f http://localhost:8001/health >/dev/null; then
    print_status "Health check passed"
else
    print_warning "Health check failed - checking logs..."
    docker logs enhanced-llm-proxy-test | tail -20
fi

# Cleanup test container
docker rm -f enhanced-llm-proxy-test

# Step 5: Create RunPod deployment instructions
print_info "Step 5: Creating RunPod deployment files..."

cat > runpod-deploy.md << 'EOF'
# RunPod Deployment Instructions

## Option 1: Docker Hub Deployment (Recommended)

### 1. Push to Docker Hub
```bash
# Tag for Docker Hub
docker tag enhanced-llm-proxy:latest your-dockerhub-username/enhanced-llm-proxy:latest

# Push to Docker Hub
docker push your-dockerhub-username/enhanced-llm-proxy:latest
```

### 2. Deploy on RunPod
1. Go to RunPod.io
2. Create new pod
3. Use custom Docker image: `your-dockerhub-username/enhanced-llm-proxy:latest`
4. Set GPU: RTX A5000 or similar (24GB+ VRAM recommended)
5. Set ports: 8001
6. Start pod

## Option 2: Direct File Upload

### 1. Create deployment package
```bash
# Create archive with essential files
tar -czf enhanced-llm-proxy.tar.gz \
    main.py \
    config.py \
    config_enhanced.py \
    requirements.txt \
    services/ \
    utils/ \
    models/
```

### 2. RunPod Setup Commands
```bash
# Extract files
tar -xzf enhanced-llm-proxy.tar.gz

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies  
pip install -r requirements.txt

# Install enhanced dependencies
pip install sentence-transformers scikit-learn sse-starlette

# Start services
ollama serve &
sleep 15

# Download models
ollama pull phi3.5 &
ollama pull mistral:7b-instruct-q4_0 &
ollama pull gemma:7b-instruct &
ollama pull llama3:8b-instruct-q4_0 &
wait

# Start application
python main.py
```

## Access Points

Once deployed, access via:
- **Main API**: `https://your-runpod-id-8001.proxy.runpod.net/`
- **Health**: `https://your-runpod-id-8001.proxy.runpod.net/health`
- **API Docs**: `https://your-runpod-id-8001.proxy.runpod.net/docs`
- **Chat API**: `https://your-runpod-id-8001.proxy.runpod.net/v1/chat/completions`

## Test Commands

```bash
# Health check
curl https://your-runpod-id-8001.proxy.runpod.net/health

# Test chat (math query - should route to Phi3.5)
curl -X POST https://your-runpod-id-8001.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10}'

# Test chat (creative query - should route to Llama3)  
curl -X POST https://your-runpod-id-8001.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Write a short story"}],"max_tokens":50}'
```
EOF

cat > docker-compose.runpod.yml << 'EOF'
version: '3.8'

services:
  enhanced-llm-proxy:
    image: enhanced-llm-proxy:latest
    container_name: enhanced-llm-proxy
    ports:
      - "8001:8001"
    environment:
      - HOST=0.0.0.0
      - PORT=8001
      - OLLAMA_BASE_URL=http://localhost:11434
      - ENABLE_4_MODEL_ROUTING=true
      - ENABLE_SEMANTIC_CLASSIFICATION=true
      - MAX_MEMORY_MB=16384
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 15s
      retries: 3
      start_period: 120s
EOF

print_status "Created RunPod deployment files"

# Summary
echo ""
echo "🎉 BUILD COMPLETE!"
echo "=================="
echo ""
print_status "Docker image: $IMAGE_NAME:$IMAGE_TAG"
print_status "Files created:"
echo "  • runpod-deploy.md - Deployment instructions"
echo "  • docker-compose.runpod.yml - Compose file for RunPod"
echo ""
print_info "Next steps:"
echo "1. Push to Docker Hub: docker push your-username/$IMAGE_NAME:latest"
echo "2. Deploy on RunPod using the image or follow runpod-deploy.md"
echo "3. Test the deployment using the provided curl commands"
echo ""
print_status "Ready for RunPod deployment! 🚀"
