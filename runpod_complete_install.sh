#!/bin/bash
# install.sh - Fixed Installation Script for Complete LLM Proxy

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions for colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo "🚀 Complete LLM Proxy - Installation Script"
echo "=========================================="

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended for development"
fi

# Step 1: Check Python version
echo -e "\n${BLUE}📋 Step 1: Checking Python version${NC}"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi

print_status "Python version: $python_version"

# Step 2: Create project directory structure
echo -e "\n${BLUE}📁 Step 2: Creating project structure${NC}"

# Create necessary directories
mkdir -p app/{services,models,middleware,utils,tests,data/{cache,logs,models},security}
touch app/data/{cache,logs,models}/.gitkeep

print_status "Directory structure created"

# Step 3: Create virtual environment FIRST (before trying to activate it!)
echo -e "\n${BLUE}🐍 Step 3: Creating Python virtual environment${NC}"

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
else
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Now activate the virtual environment AFTER it's created
echo "Activating virtual environment..."
source venv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    print_status "Virtual environment activated: $VIRTUAL_ENV"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Step 4: Upgrade pip
echo -e "\n${BLUE}📦 Step 4: Upgrading pip${NC}"
pip install --upgrade pip

# Step 5: Create requirements.txt
echo -e "\n${BLUE}📋 Step 5: Creating requirements.txt${NC}"

cat > app/requirements.txt << 'EOF'
# Core Dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
aiohttp==3.9.1
psutil==5.9.6
python-multipart==0.0.6
numpy>=1.21.0,<1.25.0

# Optional Enhanced Features (install what you need)
# Uncomment to enable:

# Redis for advanced caching
# redis>=4.5.0
# aioredis>=2.0.0

# Semantic similarity 
# sentence-transformers>=2.2.0
# faiss-cpu==1.7.4

# Streaming support
# sse-starlette==1.6.5

# Monitoring
# prometheus-client>=0.19.0
EOF

print_status "requirements.txt created"

# Step 6: Install core dependencies
echo -e "\n${BLUE}📦 Step 6: Installing core dependencies${NC}"

cd app
pip install -r requirements.txt

cd ..
print_status "Core dependencies installed"

# Step 7: Create basic configuration files
echo -e "\n${BLUE}⚙️  Step 7: Creating configuration files${NC}"

# Create .env.template
cat > app/.env.template << 'EOF'
# Environment Configuration
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300

# Authentication (set to true for production)
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-dev-change-me

# Memory Management
MAX_MEMORY_MB=8192
CACHE_MEMORY_LIMIT_MB=1024
MODEL_MEMORY_LIMIT_MB=4096

# Enhanced Features (disabled by default)
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
EOF

# Copy to .env if it doesn't exist
if [ ! -f "app/.env" ]; then
    cp app/.env.template app/.env
    print_status ".env file created from template"
else
    print_warning ".env file already exists"
fi

# Step 8: Create a basic test script
echo -e "\n${BLUE}🧪 Step 8: Creating test script${NC}"

cat > app/test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test if installation was successful"""

import sys

def test_imports():
    """Test if all core imports work"""
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
        
        import uvicorn
        print("✅ Uvicorn imported successfully")
        
        import pydantic
        print("✅ Pydantic imported successfully")
        
        import aiohttp
        print("✅ Aiohttp imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_optional_imports():
    """Test optional imports"""
    print("\n📦 Testing optional features:")
    
    # Redis
    try:
        import redis
        print("✅ Redis available")
    except ImportError:
        print("⏸️  Redis not installed (optional)")
    
    # Sentence transformers
    try:
        import sentence_transformers
        print("✅ Sentence transformers available")
    except ImportError:
        print("⏸️  Sentence transformers not installed (optional)")
    
    # SSE
    try:
        import sse_starlette
        print("✅ SSE streaming available")
    except ImportError:
        print("⏸️  SSE not installed (optional)")

if __name__ == "__main__":
    print("🧪 Testing LLM Proxy installation...")
    print("=" * 40)
    
    if test_imports():
        print("\n✅ Core installation successful!")
        test_optional_imports()
        print("\n🎉 Installation test completed!")
        sys.exit(0)
    else:
        print("\n❌ Installation test failed!")
        sys.exit(1)
EOF

chmod +x app/test_installation.py

# Step 9: Run installation test
echo -e "\n${BLUE}🧪 Step 9: Running installation test${NC}"

cd app
python test_installation.py
cd ..

# Step 10: Create start script
echo -e "\n${BLUE}🚀 Step 10: Creating start script${NC}"

cat > start.sh << 'EOF'
#!/bin/bash
# Start script for LLM Proxy

# Activate virtual environment
source venv/bin/activate

# Change to app directory
cd app

# Start the application
echo "🚀 Starting LLM Proxy..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
EOF

chmod +x start.sh
print_status "Start script created"

# Step 11: Optional features installation
echo -e "\n${BLUE}🎯 Step 11: Optional Features${NC}"

echo "Would you like to install optional enhanced features?"
echo "This includes:"
echo "  • Redis for advanced caching"
echo "  • Semantic similarity search"  
echo "  • Streaming support"
echo ""
read -p "Install enhanced features? (y/N): " install_enhanced

if [[ $install_enhanced =~ ^[Yy]$ ]]; then
    print_info "Installing enhanced features..."
    
    cd app
    pip install redis>=4.5.0 aioredis>=2.0.0 sse-starlette==1.6.5
    
    # Try to install sentence-transformers (may fail on some systems)
    pip install sentence-transformers>=2.2.0 faiss-cpu==1.7.4 || {
        print_warning "Sentence transformers installation failed (this is okay)"
        print_info "Semantic features will be disabled"
    }
    
    cd ..
    print_status "Enhanced features installed (where available)"
else
    print_info "Skipping enhanced features"
fi

# Final summary
echo -e "\n${GREEN}🎉 Installation Complete!${NC}"
echo "=========================================="
echo ""
echo "📁 Project structure created in: $(pwd)/app"
echo "🐍 Virtual environment created in: $(pwd)/venv"
echo ""
echo "📋 Next steps:"
echo "1. Copy your LLM proxy files to the app/ directory"
echo "2. Edit app/.env with your configuration"
echo "3. Run: ./start.sh"
echo ""
echo "🧪 To test again: "
echo "   source venv/bin/activate"
echo "   cd app && python test_installation.py"
echo ""
echo "📚 For more help, check the documentation"

# Create a simple README
cat > README_INSTALLATION.md << 'EOF'
# LLM Proxy Installation

## Installation completed successfully! 

### Project Structure
```
.
├── app/                    # Your application directory
│   ├── services/          # Service modules
│   ├── models/            # Data models
│   ├── middleware/        # Middleware components
│   ├── utils/             # Utilities
│   ├── tests/             # Test files
│   ├── data/              # Data directories
│   │   ├── cache/         # Cache storage
│   │   ├── logs/          # Log files
│   │   └── models/        # Model storage
│   ├── .env               # Configuration file
│   └── requirements.txt   # Python dependencies
├── venv/                  # Python virtual environment
├── start.sh               # Start script
└── install.sh             # This installation script
```

### Starting the Application
1. Copy your LLM proxy Python files to the `app/` directory
2. Edit `app/.env` with your configuration
3. Run: `./start.sh`

### Manual Start
```bash
source venv/bin/activate
cd app
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Adding Dependencies
```bash
source venv/bin/activate
pip install <package-name>
pip freeze > app/requirements.txt
```
EOF

print_status "README_INSTALLATION.md created"
echo ""
echo "Happy coding! 🚀"
