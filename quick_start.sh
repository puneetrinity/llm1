#!/bin/bash
# quick_start.sh - Complete Setup and Testing Script

set -e

echo "🚀 Complete LLM Proxy - Quick Start Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended for development"
fi

# Step 1: Environment Setup
echo -e "\n${BLUE}📋 Step 1: Environment Setup${NC}"

# Create necessary directories
print_info "Creating directory structure..."
mkdir -p data/{cache,logs,models}
mkdir -p security
touch data/{cache,logs,models}/.gitkeep

# Check if .env exists
if [ ! -f .env ]; then
    print_info "Creating .env from template..."
    if [ -f .env.template ]; then
        cp .env.template .env
        print_status ".env created from template"
        print_warning "Please review and customize .env file before proceeding"
    else
        print_error ".env.template not found!"
        exit 1
    fi
else
    print_status ".env file already exists"
fi

# Step 2: Python Dependencies
echo -e "\n${BLUE}📦 Step 2: Installing Dependencies${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi

print_status "Python version: $python_version"

# Install core dependencies
print_info "Installing core dependencies..."
pip3 install -r requirements.txt

# Ask about enhanced features
echo -e "\n${YELLOW}🎯 Enhanced Features Setup${NC}"
echo "Would you like to install enhanced features?"
echo "This includes:"
echo "  • Semantic classification (requires ~500MB additional memory)"
echo "  • Advanced streaming capabilities"
echo "  • FAISS-based similarity search"

read -p "Install enhanced features? (y/N): " install_enhanced

if [[ $install_enhanced =~ ^[Yy]$ ]]; then
    print_info "Installing enhanced dependencies..."
    pip3 install sentence-transformers faiss-cpu sse-starlette
    
    # Update .env to enable features
    if grep -q "ENABLE_SEMANTIC_CLASSIFICATION=false" .env; then
        sed -i 's/ENABLE_SEMANTIC_CLASSIFICATION=false/ENABLE_SEMANTIC_CLASSIFICATION=true/' .env
        print_status "Enabled semantic classification in .env"
    fi
    
    print_status "Enhanced features installed"
else
    print_info "Skipping enhanced features (can be installed later)"
fi

# Step 3: Security Configuration
echo -e "\n${BLUE}🔒 Step 3: Security Configuration${NC}"

# Check environment setting
env_setting=$(grep "ENVIRONMENT=" .env | cut -d'=' -f2)
if [ "$env_setting" = "production" ]; then
    print_warning "Production environment detected!"
    
    # Check for secure API key
    api_key=$(grep "DEFAULT_API_KEY=" .env | cut -d'=' -f2)
    if [[ $api_key == *"insecure"* ]] || [[ $api_key == *"change-me"* ]] || [ ${#api_key} -lt 32 ]; then
        print_error "Insecure API key detected in production environment!"
        echo "Generate a secure API key with:"
        echo "python3 -c \"import secrets; print(f'sk-{secrets.token_urlsafe(32)}')\""
        
        read -p "Generate secure API key now? (Y/n): " generate_key
        if [[ ! $generate_key =~ ^[Nn]$ ]]; then
            new_key=$(python3 -c "import secrets; print(f'sk-{secrets.token_urlsafe(32)}')")
            sed -i "s/DEFAULT_API_KEY=.*/DEFAULT_API_KEY=$new_key/" .env
            print_status "Secure API key generated and saved to .env"
            print_warning "Save this key securely: $new_key"
        fi
    fi
    
    # Enable authentication for production
    if grep -q "ENABLE_AUTH=false" .env; then
        sed -i 's/ENABLE_AUTH=false/ENABLE_AUTH=true/' .env
        print_status "Authentication enabled for production"
    fi
else
    print_info "Development environment - using default security settings"
fi

# Step 4: Docker Setup (Optional)
echo -e "\n${BLUE}🐳 Step 4: Docker Setup (Optional)${NC}"

if command -v docker &> /dev/null; then
    print_status "Docker is available"
    
    read -p "Build Docker image? (y/N): " build_docker
    if [[ $build_docker =~ ^[Yy]$ ]]; then
        print_info "Building Docker image..."
        docker build -t llm-proxy-complete -f Dockerfile.enhanced .
        print_status "Docker image built successfully"
        
        echo "To run with Docker:"
        echo "docker run -d --name llm-proxy --gpus all -p 8000:8000 -p 11434:11434 -v ./data:/app/data llm-proxy-complete"
    fi
else
    print_warning "Docker not found - skipping Docker setup"
fi

# Step 5: Ollama Setup Check
echo -e "\n${BLUE}🤖 Step 5: Ollama Setup Check${NC}"

if command -v ollama &> /dev/null; then
    print_status "Ollama is installed"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        print_status "Ollama service is running"
        
        # List available models
        models=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "")
        if [ -n "$models" ]; then
            print_status "Available models:"
            echo "$models" | while read -r model; do
                echo "  • $model"
            done
        else
            print_warning "No models found. You may want to pull a model:"
            echo "ollama pull mistral:7b-instruct-q4_0"
        fi
    else
        print_warning "Ollama service not running. Start with: ollama serve"
    fi
else
    print_warning "Ollama not found. Install from: https://ollama.com"
    echo "After installation, pull a model: ollama pull mistral:7b-instruct-q4_0"
fi

# Step 6: Start Application
echo -e "\n${BLUE}🚀 Step 6: Starting Application${NC}"

read -p "Start the LLM Proxy now? (Y/n): " start_app
if [[ ! $start_app =~ ^[Nn]$ ]]; then
    print_info "Starting LLM Proxy..."
    
    # Start in background for testing
    python3 main.py &
    APP_PID=$!
    
    print_info "Waiting for service to start..."
    sleep 5
    
    # Test health endpoint
    echo -e "\n${BLUE}🧪 Testing Service${NC}"
    
    max_attempts=12
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health &> /dev/null; then
            print_status "Service is healthy!"
            break
        else
            print_info "Waiting for service... (attempt $((attempt + 1))/$max_attempts)"
            sleep 5
            attempt=$((attempt + 1))
        fi
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "Service failed to start properly"
        kill $APP_PID 2>/dev/null || true
        exit 1
    fi
    
    # Test basic functionality
    print_info "Testing basic functionality..."
    
    # Test health check
    health_response=$(curl -s http://localhost:8000/health)
    if echo "$health_response" | grep -q "healthy"; then
        print_status "Health check passed"
    else
        print_warning "Health check failed"
    fi
    
    # Test models endpoint
    models_response=$(curl -s http://localhost:8000/models)
    if echo "$models_response" | grep -q "data"; then
        print_status "Models endpoint working"
    else
        print_warning "Models endpoint failed"
    fi
    
    # Test metrics endpoint
    metrics_response=$(curl -s http://localhost:8000/metrics)
    if echo "$metrics_response" | grep -q "features"; then
        print_status "Metrics endpoint working"
    else
        print_warning "Metrics endpoint failed"
    fi
    
    # Stop the test instance
    kill $APP_PID 2>/dev/null || true
    
    echo -e "\n${GREEN}🎉 Setup Complete!${NC}"
    echo "=========================================="
    echo "Service URL: http://localhost:8000"
    echo "API Documentation: http://localhost:8000/docs"
    echo "Health Check: http://localhost:8000/health"
    echo "Metrics: http://localhost:8000/metrics"
    echo ""
    echo "To start the service:"
    echo "  python3 main.py"
    echo ""
    echo "To start with Docker:"
    echo "  docker-compose up -d"
    echo ""
    
    if [ "$env_setting" = "production" ]; then
        echo -e "${YELLOW}Production Deployment Reminders:${NC}"
        echo "• Set up HTTPS with reverse proxy"
        echo "• Configure firewall rules"
        echo "• Set up monitoring and alerting"
        echo "• Review CORS origins"
        echo "• Set up log rotation"
    fi
    
else
    print_info "Skipping application start"
    echo -e "\n${GREEN}Setup Complete!${NC}"
    echo "Start the service with: python3 main.py"
fi

echo -e "\n${BLUE}📚 Additional Resources:${NC}"
echo "• Documentation: Check QUICK_START.md"
echo "• Security: Review security/config.py"
echo "• Memory Management: /admin/memory endpoint"
echo "• Feature Status: /admin/status endpoint"
