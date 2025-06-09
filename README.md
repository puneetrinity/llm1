# Complete LLM Proxy 🚀

**Production-ready LLM routing proxy with enhanced features, security, and monitoring**

[![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)](https://github.com/your-repo/llm-proxy)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Features

### ✅ **Core Features**
- **OpenAI-compatible API** endpoints (`/v1/chat/completions`, `/v1/completions`)
- **Intelligent model routing** with automatic fallbacks
- **Comprehensive health monitoring** and metrics collection
- **Production-ready security** with API key authentication
- **Memory management** with automatic allocation tracking
- **Standardized error handling** with detailed logging

### 🚀 **Enhanced Features** (Auto-detected)
- **Semantic classification** for intelligent routing (requires `sentence-transformers`)
- **Real-time streaming** responses (requires `sse-starlette`) 
- **Model warmup service** for reduced latency
- **Advanced caching** with semantic similarity
- **Performance monitoring** and optimization recommendations

### 🔒 **Security Features**
- **Strong API key validation** with configurable requirements
- **Environment-based security** (development/staging/production)
- **Rate limiting** and request size validation
- **Security headers** and CORS protection
- **Production deployment validation**

## 🚀 Quick Start

### Method 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd llm-proxy

# Run the automated setup script
chmod +x quick_start.sh
./quick_start.sh
```

The script will:
- Set up directory structure
- Install dependencies (core + optional enhanced features)
- Configure security settings
- Test the installation
- Provide deployment guidance

### Method 2: Manual Setup

```bash
# 1. Create environment
cp .env.template .env
# Edit .env with your configuration

# 2. Install dependencies
pip install -r requirements.txt

# 3. Optional: Install enhanced features
pip install sentence-transformers faiss-cpu sse-starlette

# 4. Start the service
python main.py
```

## 📋 Requirements

### Core Requirements
- **Python 3.8+**
- **Ollama** running on `localhost:11434` (or configured URL)
- At least **4GB RAM** (8GB+ recommended)

### Optional Enhanced Features
```bash
# For semantic classification and advanced routing
pip install sentence-transformers faiss-cpu

# For streaming responses
pip install sse-starlette
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Environment (development/staging/production)
ENVIRONMENT=development

# Security
ENABLE_AUTH=true
DEFAULT_API_KEY=sk-your-secure-key-here

# Memory Management (MB)
MAX_MEMORY_MB=8192
MODEL_MEMORY_LIMIT_MB=4096
CACHE_MEMORY_LIMIT_MB=1024

# Enhanced Features
ENABLE_SEMANTIC_CLASSIFICATION=true
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
```

### Hardware-Specific Configurations

#### RunPod A5000 (24GB VRAM)
```bash
MAX_MEMORY_MB=16384
MODEL_MEMORY_LIMIT_MB=8192
OLLAMA_MAX_LOADED_MODELS=3
ENABLE_SEMANTIC_CLASSIFICATION=true
GPU_MEMORY_FRACTION=0.9
```

#### Smaller Systems (8GB RAM)
```bash
MAX_MEMORY_MB=6144
MODEL_MEMORY_LIMIT_MB=3072
OLLAMA_MAX_LOADED_MODELS=1
ENABLE_SEMANTIC_CLASSIFICATION=false
```

## 🐳 Docker Deployment

### Basic Deployment
```bash
# Build image
docker build -t llm-proxy-complete -f Dockerfile.enhanced .

# Run container
docker run -d \
  --name llm-proxy \
  --gpus all \
  -p 8000:8000 \
  -p 11434:11434 \
  -v ./data:/app/data \
  -v ./.env:/app/.env \
  llm-proxy-complete
```

### Docker Compose (Recommended)
```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f llm-proxy

# Stop services
docker-compose down
```

## 📊 API Usage

### Basic Chat Completion
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ]
  }'
```

### Streaming Response
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Write a story"}
    ],
    "stream": true
  }'
```

### Python Client Example
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": "your-api-key"
    },
    json={
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ]
    }
)

print(response.json())
```

## 🔍 Monitoring & Administration

### Health Check
```bash
curl http://localhost:8000/health
```

### Comprehensive Metrics
```bash
curl http://localhost:8000/metrics
```

### Memory Status
```bash
curl http://localhost:8000/admin/memory
```

### Manual Model Warmup
```bash
curl -X POST "http://localhost:8000/admin/warmup/mistral:7b-instruct-q4_0" \
  -H "X-API-Key: your-api-key"
```

### Admin Dashboard
Visit `http://localhost:8000/docs` for the interactive API documentation.

## 🛡️ Security

### Development vs Production

#### Development
```bash
ENVIRONMENT=development
ENABLE_AUTH=false  # Optional
DEFAULT_API_KEY=sk-dev-key  # Weak keys allowed
CORS_ORIGINS=["*"]  # Permissive CORS
```

#### Production
```bash
ENVIRONMENT=production
ENABLE_AUTH=true  # Required
DEFAULT_API_KEY=sk-very-long-secure-key  # Strong keys required
CORS_ORIGINS=["https://yourdomain.com"]  # Specific origins
```

### Generate Secure API Key
```bash
python -c "import secrets; print(f'sk-{secrets.token_urlsafe(32)}')"
```

### Security Validation
```bash
# Run security check
python -m security.config

# This will validate:
# - API key strength
# - Environment configuration
# - CORS settings
# - Security headers
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check logs
python main.py

# Common fixes:
# - Verify Ollama is running: curl http://localhost:11434/api/tags
# - Check memory limits in .env
# - Verify Python dependencies: pip install -r requirements.txt
```

#### 2. Enhanced Features Not Working
```bash
# Check feature status
curl http://localhost:8000/admin/status

# Install enhanced dependencies
pip install sentence-transformers faiss-cpu sse-starlette

# Enable in .env
ENABLE_SEMANTIC_CLASSIFICATION=true
```

#### 3. Memory Issues
```bash
# Check memory status
curl http://localhost:8000/admin/memory

# Trigger cleanup
curl -X POST http://localhost:8000/admin/memory/cleanup

# Reduce memory limits in .env
MAX_MEMORY_MB=4096
MODEL_MEMORY_LIMIT_MB=2048
```

#### 4. Authentication Errors
```bash
# Check API key format
echo $DEFAULT_API_KEY | wc -c  # Should be > 32 characters

# Generate new key
python -c "import secrets; print(f'sk-{secrets.token_urlsafe(32)}')"
```

### Debug Mode
```bash
# Enable debug logging
DEBUG=true python main.py

# Or in .env
DEBUG=true
LOG_LEVEL=DEBUG
```

## 📈 Performance Optimization

### Memory Optimization
1. **Start with basic features** and enable enhanced ones gradually
2. **Monitor memory usage** via `/admin/memory`
3. **Adjust limits** based on your hardware
4. **Use model warmup** for consistent performance

### Response Time Optimization
1. **Enable model warmup** for popular models
2. **Use appropriate routing** for different query types
3. **Monitor via metrics** endpoint
4. **Consider semantic caching** for similar queries

### Cost Optimization
1. **Route simple queries** to smaller/cheaper models
2. **Enable caching** to reduce redundant requests
3. **Monitor usage patterns** via metrics
4. **Use rate limiting** to control costs

## 🏗️ Architecture

### Component Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│  Enhanced Router │────│  Ollama Client  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
    ┌────▼────┐           ┌──────▼──────┐         ┌──────▼──────┐
    │Security │           │ Semantic    │         │   Model     │
    │Manager  │           │Classifier   │         │  Warmup     │
    └─────────┘           └─────────────┘         └─────────────┘
         │                       │                       │
    ┌────▼────┐           ┌──────▼──────┐         ┌──────▼──────┐
    │Memory   │           │ Streaming   │         │   Health    │
    │Manager  │           │ Service     │         │  Monitor    │
    └─────────┘           └─────────────┘         └─────────────┘
```

### Key Components

- **Enhanced Router**: Intelligent model selection with semantic classification
- **Memory Manager**: Centralized memory allocation and monitoring
- **Security Manager**: Authentication, rate limiting, and security headers
- **Error Handler**: Standardized error handling and logging
- **Import Manager**: Safe loading of optional dependencies
- **Streaming Service**: Real-time response streaming
- **Warmup Service**: Model preloading for reduced latency

## 📝 Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=. tests/
```

### Code Style
```bash
# Install development dependencies
pip install black isort flake8

# Format code
black .
isort .

# Lint code
flake8 .
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/v1/completions` | POST | OpenAI-compatible text completions |
| `/health` | GET | Service health status |
| `/metrics` | GET | Comprehensive metrics |
| `/models` | GET | Available models list |
| `/admin/status` | GET | Admin status and configuration |
| `/admin/memory` | GET | Memory usage statistics |
| `/admin/warmup/{model}` | POST | Manual model warmup |
| `/docs` | GET | Interactive API documentation |

### Response Formats

All responses follow OpenAI API format with additional metadata:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "mistral:7b-instruct-q4_0",
  "choices": [...],
  "usage": {...},
  "cache_hit": false,
  "processing_time": 1.23,
  "selected_model": "mistral:7b-instruct-q4_0"
}
```

## 🆘 Support

### Getting Help
1. Check the [troubleshooting section](#troubleshooting)
2. Review logs for error details
3. Verify configuration with `/admin/status`
4. Check memory usage with `/admin/memory`

### Reporting Issues
When reporting issues, include:
- Service version (`/admin/status`)
- Configuration (`.env` file, without API keys)
- Error logs
- System specifications
- Steps to reproduce

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Ollama](https://ollama.com/) for local LLM serving
- [Sentence Transformers](https://www.sbert.net/) for semantic embeddings
- [FAISS](https://faiss.ai/) for efficient similarity search

---

**Ready to deploy your own LLM proxy? Start with the [Quick Start](#quick-start) guide!** 🚀
