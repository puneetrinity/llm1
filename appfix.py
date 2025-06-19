#!/usr/bin/env python3
"""
Setup script to create all necessary directories and files for the Enhanced LLM Proxy
"""

import os
import sys
from pathlib import Path


def create_project_structure():
    """Create all necessary directories and files"""
    
    # Define directory structure
    directories = [
        "models",
        "services",
        "middleware",
        "utils",
        "data",
        "data/logs",
        "data/cache",
        "static",
        "static/dashboard",
        "frontend",
        "frontend/build",
        "frontend/src",
        "frontend/public",
        "tests"
    ]
    
    # Create directories
    print("🚀 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    # Create __init__.py files
    print("\n📄 Creating __init__.py files...")
    init_dirs = ["models", "services", "middleware", "utils", "tests"]
    for directory in init_dirs:
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"  ✅ Created: {init_file}")
    
    # Create .gitkeep files for empty directories
    print("\n📌 Creating .gitkeep files...")
    gitkeep_dirs = ["data/logs", "data/cache", "static/dashboard", "frontend/build"]
    for directory in gitkeep_dirs:
        gitkeep_file = Path(directory) / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            print(f"  ✅ Created: {gitkeep_file}")
    
    # Create basic .env file if it doesn't exist
    if not Path(".env").exists():
        print("\n🔧 Creating .env file...")
        env_content = """# Enhanced LLM Proxy Configuration
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8001
DEBUG=true
LOG_LEVEL=INFO

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300

# Features
ENABLE_AUTH=false
ENABLE_CACHE=true
ENABLE_STREAMING=true
ENABLE_MODEL_ROUTING=true
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_DASHBOARD=true

# Models
PHI_MODEL=phi3.5
MISTRAL_MODEL=mistral:7b-instruct-q4_0
GEMMA_MODEL=gemma:7b-instruct
LLAMA_MODEL=llama3:8b-instruct-q4_0

# Memory Limits
MAX_MEMORY_MB=8192
MODEL_MEMORY_LIMIT_MB=4096
CACHE_MEMORY_LIMIT_MB=1024
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("  ✅ Created: .env")
    
    # Create .gitignore if it doesn't exist
    if not Path(".gitignore").exists():
        print("\n📝 Creating .gitignore file...")
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Environment
.env
.env.*

# Logs
logs/
*.log

# Data
data/
!data/.gitkeep

# Models
models/*.bin
models/*.gguf

# Cache
cache/
.cache/

# Test
.pytest_cache/
.coverage
htmlcov/

# Ollama
.ollama/
"""
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("  ✅ Created: .gitignore")
    
    print("\n✨ Project structure created successfully!")
    
    # Create README if it doesn't exist
    if not Path("README.md").exists():
        print("\n📖 Creating README.md...")
        readme_content = """# Enhanced LLM Proxy

A production-ready OpenAI-compatible API proxy with intelligent 4-model routing, caching, and enhanced features.

## Features

- 🧠 **4-Model Intelligent Routing**
  - Phi 3.5 for math/reasoning
  - Mistral 7B for general queries
  - Gemma 7B for coding/technical
  - Llama3 8B for creative writing

- ⚡ **Performance**
  - Advanced caching with TTL
  - Response streaming
  - Model warmup service
  - Circuit breaker pattern

- 🔒 **Security & Management**
  - API key authentication
  - Rate limiting
  - CORS configuration
  - Request tracking

- 📊 **Monitoring**
  - Comprehensive metrics
  - Health checks
  - Performance tracking
  - Dashboard UI

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run the application:
```bash
python main.py
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /metrics` - Performance metrics
- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Chat completions
- `GET /dashboard` - Web dashboard

## Docker Deployment

```bash
docker-compose up -d
```

## Documentation

See the `/docs` endpoint when running in debug mode.
"""
        with open("README.md", "w") as f:
            f.write(readme_content)
        print("  ✅ Created: README.md")
    
    print("\n" + "="*50)
    print("🎉 Setup complete!")
    print("="*50)
    print("\n📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Configure environment: edit .env file")
    print("3. Run the application: python main.py")
    print("\n💡 For Docker deployment: docker-compose up -d")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_modules = [
        "fastapi",
        "uvicorn",
        "aiohttp",
        "pydantic",
        "psutil"
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} (missing)")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main setup function"""
    print("🚀 Enhanced LLM Proxy Setup")
    print("="*50)
    
    # Create project structure
    create_project_structure()
    
    # Check dependencies
    dependencies_ok = check_dependencies()
    
    # Check for Ollama
    print("\n🔍 Checking for Ollama...")
    try:
        import subprocess
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Ollama installed: {result.stdout.strip()}")
        else:
            print("  ❌ Ollama not found")
            print("  Install from: https://ollama.ai")
    except FileNotFoundError:
        print("  ❌ Ollama not found")
        print("  Install from: https://ollama.ai")
    
    print("\n✅ Setup script completed!")
    
    if not dependencies_ok:
        print("\n⚠️  Please install missing dependencies before running the application.")
        sys.exit(1)


if __name__ == "__main__":
    main()
