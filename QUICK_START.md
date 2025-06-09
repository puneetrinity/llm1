# Enhanced LLM Proxy - Quick Start Guide

This guide will help you get the Enhanced LLM Proxy up and running quickly with the fixed configuration.

## 🚀 Quick Deployment (Fixed Version)

### 1. Prepare Environment

```bash
# Clone or prepare your project directory
cd your-llm-proxy-directory

# Copy environment template
cp .env.template .env

# Edit .env file to match your needs (optional for basic testing)
nano .env
```

### 2. Build and Run with Docker

```bash
# Build the enhanced image
docker build -t llm-proxy-enhanced -f Dockerfile.enhanced .

# Run with basic configuration
docker run -d \
  --name llm-proxy \
  --gpus all \
  -p 8000:8000 \
  -p 11434:11434 \
  -v ./models:/root/.ollama \
  -v ./cache:/app/cache \
  -v ./logs:/app/logs \
  -e ENABLE_SEMANTIC_CLASSIFICATION=false \
  -e ENABLE_STREAMING=true \
  -e ENABLE_MODEL_WARMUP=true \
  -e MAX_MEMORY_MB=8192 \
  --restart unless-stopped \
  llm-proxy-enhanced
```

### 3. Wait for Startup and Test

```bash
# Wait for service to start (2-3 minutes)
sleep 180

# Check health
curl http://localhost:8000/health

# Test basic completion
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'
```

## 🔧 Configuration Options

### Basic Configuration (.env)

```bash
# Memory limits (adjust for your hardware)
MAX_MEMORY_MB=8192                    # Total memory limit
MODEL_MEMORY_LIMIT_MB=4096           # Memory for models
CACHE_MEMORY_LIMIT_MB=1024           # Memory for caching

# Feature toggles (start with conservative settings)
ENABLE_SEMANTIC_CLASSIFICATION=false # Requires additional 500MB
ENABLE_STREAMING=true                # Safe to enable
ENABLE_MODEL_WARMUP=true            # Recommended
ENABLE_DETAILED_METRICS=true        # Safe to enable

# Authentication (enable for production)
ENABLE_AUTH=false                    # Set to true for production
DEFAULT_API_KEY=sk-your-key-here    # Change this!
```

### Hardware-Specific Settings

#### For RunPod A5000 (24GB VRAM):
```bash
MAX_MEMORY_MB=16384
MODEL_MEMORY_LIMIT_MB=8192
OLLAMA_MAX_LOADED_MODELS=3
GPU_MEMORY_FRACTION=0.9
ENABLE_SEMANTIC_CLASSIFICATION=true  # Can enable with 24GB
```

#### For Smaller Systems (8GB RAM):
```bash
MAX_MEMORY_MB=6144
MODEL_MEMORY_LIMIT_MB=3072
OLLAMA_MAX_LOADED_MODELS=1
ENABLE_SEMANTIC_CLASSIFICATION=false # Disable to save memory
```

## 📊 Monitoring and Admin

### Health Check
```bash
curl http://localhost:8000/health
```

### Metrics
```bash
curl http://localhost:8000/metrics
```

### Available Models
```bash
curl http://localhost:8000/models
```

### Admin Status
```bash
curl http://localhost:8000/admin/status
```

### Manual Model Warmup
```bash
curl -X POST http://localhost:8000/admin/warmup/mistral:7b-instruct-q4_0
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Service Won't Start
```bash
# Check logs
docker logs llm-proxy

# Common fixes:
# - Reduce memory limits in .env
# - Disable semantic classification
# - Check Ollama service status
```

#### 2. Out of Memory Errors
```bash
# Edit .env file:
ENABLE_SEMANTIC_CLASSIFICATION=false
MAX_MEMORY_MB=4096
MODEL_MEMORY_LIMIT_MB=2048

# Restart container
docker restart llm-proxy
```

#### 3. Models Not Loading
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Pull models manually
docker exec llm-proxy ollama pull mistral:7b-instruct-q4_0
```

#### 4. Slow Response Times
```bash
# Enable model warmup
curl -X POST http://localhost:8000/admin/warmup/mistral:7b-instruct-q4_0

# Check if models are loaded
curl http://localhost:8000/metrics
```

## 🚦 Gradual Feature Enablement

Start with basic features and gradually enable advanced ones:

### Phase 1: Basic Functionality
```bash
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
```

### Phase 2: Add Intelligence (if memory allows)
```bash
ENABLE_SEMANTIC_CLASSIFICATION=true
SEMANTIC_MODEL_MAX_MEMORY_MB=500
```

### Phase 3: Production Features
```bash
ENABLE_AUTH=true
ENABLE_RATE_LIMITING=true
ENABLE_DETAILED_METRICS=true
```

## 📈 Performance Optimization

### Memory Optimization
- Start with semantic classification disabled
- Monitor memory usage via `/metrics`
- Gradually increase limits based on actual usage

### Response Time Optimization
- Enable model warmup
- Use appropriate model routing
- Monitor via `/admin/status`

### Cost Optimization
- Route simple queries to smaller models
- Enable caching
- Monitor usage patterns

## 🔗 API Usage Examples

### Basic Chat Completion
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
  }'
```

### Streaming Response
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Write a short story"}
    ],
    "stream": true
  }'
```

### With Authentication (if enabled)
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-your-key-here" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## 🆘 Getting Help

If you encounter issues:

1. Check the logs: `docker logs llm-proxy`
2. Verify health: `curl http://localhost:8000/health`
3. Check metrics: `curl http://localhost:8000/metrics`
4. Try reducing memory limits in `.env`
5. Disable advanced features temporarily

## 🎯 Success Indicators

Your deployment is successful when:

- ✅ Health check returns `{"healthy": true}`
- ✅ `/models` shows available models
- ✅ Basic chat completion works
- ✅ Response times are < 10 seconds
- ✅ No memory errors in logs

Ready to use your Enhanced LLM Proxy! 🎉
