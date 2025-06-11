# manual_test_commands.sh - Individual test commands for manual debugging
# Copy and paste these commands one by one for targeted testing

# ===========================================
# 1. BASIC SYSTEM CHECKS
# ===========================================

# Check running processes
echo "=== Running Processes ==="
ps aux | grep -E "(ollama|uvicorn|python.*main)" | grep -v grep

# Check open ports
echo "=== Open Ports ==="
ss -tuln | grep -E "(8000|11434)"
netstat -tuln | grep -E "(8000|11434)"

# Check system resources
echo "=== System Resources ==="
free -h
df -h
top -bn1 | head -10

# Check GPU (if available)
echo "=== GPU Status ==="
nvidia-smi || echo "No GPU detected"

# ===========================================
# 2. OLLAMA SERVICE TESTS
# ===========================================

# Check Ollama health
echo "=== Ollama Health ==="
curl -s http://localhost:11434/api/tags | jq '.' || curl -s http://localhost:11434/api/tags

# List available models
echo "=== Available Models ==="
curl -s http://localhost:11434/api/tags | jq '.models[].name' || echo "Failed to get models"

# Test model generation (replace 'mistral:7b-instruct-q4_0' with your model)
echo "=== Test Model Generation ==="
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b-instruct-q4_0",
    "prompt": "Hello, respond with just one word",
    "stream": false,
    "options": {"num_predict": 1}
  }' | jq '.response' || echo "Model generation failed"

# ===========================================
# 3. LLM PROXY SERVICE TESTS  
# ===========================================

# Check LLM Proxy health
echo "=== LLM Proxy Health ==="
curl -s http://localhost:8000/health | jq '.' || curl -s http://localhost:8000/health

# Check available models through proxy
echo "=== Models via Proxy ==="
curl -s http://localhost:8000/models | jq '.' || curl -s http://localhost:8000/models

# Check metrics
echo "=== Proxy Metrics ==="
curl -s http://localhost:8000/metrics | jq '.' || curl -s http://localhost:8000/metrics

# ===========================================
# 4. API FUNCTIONALITY TESTS
# ===========================================

# Test chat completions API (basic)
echo "=== Chat Completions Test ==="
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-default" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Say hello in exactly 2 words"}
    ],
    "max_tokens": 5,
    "temperature": 0.1
  }' | jq '.' || echo "Chat completion failed"

# Test completions API
echo "=== Completions Test ==="
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-default" \
  -d '{
    "model": "gpt-3.5-turbo", 
    "prompt": "The sky is",
    "max_tokens": 3
  }' | jq '.' || echo "Completion failed"

# Test streaming (shows first few chunks)
echo "=== Streaming Test ==="
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-default" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Count from 1 to 3"}],
    "stream": true,
    "max_tokens": 20
  }' --no-buffer | head -n 5

# ===========================================
# 5. ENHANCED FEATURES TESTS
# ===========================================

# Check admin status
echo "=== Admin Status ==="
curl -s http://localhost:8000/admin/status | jq '.' || echo "Admin endpoint not available"

# Check memory usage
echo "=== Memory Management ==="
curl -s http://localhost:8000/admin/memory | jq '.' || echo "Memory endpoint not available"

# Check circuit breakers
echo "=== Circuit Breakers ==="
curl -s http://localhost:8000/admin/circuit-breakers | jq '.' || echo "Circuit breaker endpoint not available"

# Check cache stats
echo "=== Cache Statistics ==="
curl -s http://localhost:8000/admin/cache/stats | jq '.' || echo "Cache endpoint not available"

# ===========================================
# 6. AUTHENTICATION TESTS
# ===========================================

# Test without API key (should fail if auth enabled)
echo "=== Test Without API Key ==="
curl -s -w "%{http_code}" http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}]}' \
  -o /tmp/no_auth_response.json
echo "Response code: $(cat /tmp/no_auth_response.json)"

# Test with wrong API key
echo "=== Test Wrong API Key ==="
curl -s -w "%{http_code}" http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: wrong-key" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}]}' \
  -o /tmp/wrong_auth_response.json
echo "Response code: $(cat /tmp/wrong_auth_response.json)"

# ===========================================
# 7. PERFORMANCE TESTS
# ===========================================

# Response time test
echo "=== Response Time Test ==="
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-default" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 1
  }' > /dev/null

# Multiple concurrent requests test
echo "=== Concurrent Requests Test ==="
for i in {1..3}; do
  echo "Starting request $i..."
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "X-API-Key: sk-default" \
    -d "{\"model\":\"gpt-3.5-turbo\",\"messages\":[{\"role\":\"user\",\"content\":\"Test $i\"}],\"max_tokens\":1}" \
    > /tmp/concurrent_$i.json &
done

echo "Waiting for concurrent requests to complete..."
wait
echo "Results:"
for i in {1..3}; do
  if [ -f /tmp/concurrent_$i.json ]; then
    echo "Request $i: $(jq -r '.choices[0].message.content // "Failed"' /tmp/concurrent_$i.json 2>/dev/null || echo "Failed")"
  fi
done

# ===========================================
# 8. TROUBLESHOOTING COMMANDS
# ===========================================

# Check logs (adjust paths as needed)
echo "=== Recent Logs ==="
if [ -f "app.log" ]; then
    echo "Application logs (last 20 lines):"
    tail -20 app.log
elif [ -d "logs" ]; then
    echo "Log directory contents:"
    ls -la logs/
    if [ -f "logs/app.log" ]; then
        tail -20 logs/app.log
    fi
elif [ -d "data/logs" ]; then
    echo "Data log directory contents:"
    ls -la data/logs/
fi

# Check Docker logs (if running in Docker)
echo "=== Docker Logs ==="
if docker ps --format "{{.Names}}" | grep -E "(llm-proxy|ollama)" >/dev/null 2>&1; then
    echo "Docker containers found:"
    docker ps --format "table {{.Names}}\t{{.Status}}"
    
    # Get logs from LLM proxy container
    proxy_container=$(docker ps --format "{{.Names}}" | grep llm-proxy | head -1)
    if [ -n "$proxy_container" ]; then
        echo "LLM Proxy container logs (last 20 lines):"
        docker logs --tail 20 "$proxy_container"
    fi
    
    # Get logs from Ollama container  
    ollama_container=$(docker ps --format "{{.Names}}" | grep ollama | head -1)
    if [ -n "$ollama_container" ]; then
        echo "Ollama container logs (last 20 lines):"
        docker logs --tail 20 "$ollama_container"
    fi
else
    echo "No relevant Docker containers running"
fi

# Check process status in detail
echo "=== Process Details ==="
if pgrep -f "uvicorn\|python.*main" > /dev/null; then
    echo "LLM Proxy process details:"
    ps aux | grep -E "(uvicorn|python.*main)" | grep -v grep
    
    # Check what port it's actually using
    proxy_pid=$(pgrep -f "uvicorn\|python.*main" | head -1)
    if [ -n "$proxy_pid" ]; then
        echo "Ports used by LLM Proxy (PID $proxy_pid):"
        ss -tuln | grep ":8000\|:8080\|:5000" || echo "No standard ports detected"
    fi
fi

if pgrep -f "ollama" > /dev/null; then
    echo "Ollama process details:"
    ps aux | grep ollama | grep -v grep
    
    ollama_pid=$(pgrep -f "ollama" | head -1)
    if [ -n "$ollama_pid" ]; then
        echo "Ports used by Ollama (PID $ollama_pid):"
        ss -tuln | grep ":11434" || echo "Port 11434 not detected"
    fi
fi

# Check configuration files
echo "=== Configuration Check ==="
if [ -f ".env" ]; then
    echo ".env file found. Key settings:"
    grep -E "^(HOST|PORT|OLLAMA|ENABLE_|DEBUG)" .env 2>/dev/null || echo "No key settings found"
else
    echo "No .env file found"
fi

if [ -f "config.py" ]; then
    echo "config.py found"
fi

if [ -f "config_enhanced.py" ]; then
    echo "config_enhanced.py found"
fi

# Check Python environment
echo "=== Python Environment ==="
echo "Python version: $(python3 --version)"
echo "Python path: $(which python3)"

echo "Checking key dependencies:"
for dep in fastapi uvicorn aiohttp pydantic; do
    if python3 -c "import $dep; print(f'$dep: {$dep.__version__}')" 2>/dev/null; then
        echo "✅ $dep installed"
    else
        echo "❌ $dep missing"
    fi
done

echo "Checking enhanced dependencies:"
for dep in sentence_transformers redis numpy torch; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo "✅ $dep available"
    else
        echo "⚠️  $dep not available (optional)"
    fi
done

# Check file permissions
echo "=== File Permissions ==="
if [ -f "main.py" ]; then
    echo "main.py permissions: $(ls -l main.py)"
else
    echo "main.py not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Directory contents:"
    ls -la | head -10
fi

# Check for startup scripts
echo "=== Startup Scripts ==="
for script in start.sh enhanced_start.sh setup.sh; do
    if [ -f "$script" ]; then
        echo "$script found ($(ls -l $script | awk '{print $1}'))"
    fi
done

# ===========================================
# 9. NETWORK DIAGNOSTICS
# ===========================================

echo "=== Network Diagnostics ==="

# Check if services are binding to correct interfaces
echo "Services listening on ports:"
ss -tuln | grep -E "(8000|11434)" | while read line; do
    echo "  $line"
done

# Test localhost connectivity
echo "Testing localhost connectivity:"
if ping -c 1 localhost >/dev/null 2>&1; then
    echo "✅ localhost ping successful"
else
    echo "❌ localhost ping failed"
fi

# Test specific endpoints with verbose output
echo "Testing endpoint accessibility:"
for port in 8000 11434; do
    if timeout 5 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
        echo "✅ Port $port is accessible"
    else
        echo "❌ Port $port is not accessible"
    fi
done

# ===========================================
# 10. COMMON FIXES
# ===========================================

echo "=== Common Fixes ==="
echo "If services aren't running:"
echo "  1. Start Ollama: ollama serve &"
echo "  2. Start LLM Proxy: python3 main.py"
echo "  3. Or use Docker: docker-compose up -d"

echo ""
echo "If models aren't available:"
echo "  1. Pull a model: ollama pull mistral:7b-instruct-q4_0"
echo "  2. List models: ollama list"
echo "  3. Check model status: curl http://localhost:11434/api/tags"

echo ""
echo "If API calls fail:"
echo "  1. Check auth: Remove X-API-Key header or use correct key"
echo "  2. Check model name: Use model names from /models endpoint"
echo "  3. Check request format: Ensure JSON is valid"

echo ""
echo "If ports are in use:"
echo "  1. Find process: lsof -i :8000 or lsof -i :11434"
echo "  2. Kill process: kill <PID>"
echo "  3. Or change ports in configuration"

echo ""
echo "If memory issues:"
echo "  1. Check usage: free -h"
echo "  2. Restart services: pkill -f ollama && pkill -f uvicorn"
echo "  3. Reduce model size or disable features"

echo ""
echo "For Docker issues:"
echo "  1. Check containers: docker ps -a"
echo "  2. View logs: docker logs <container_name>"
echo "  3. Restart: docker-compose restart"
echo "  4. Rebuild: docker-compose build --no-cache"

# ===========================================
# 11. QUICK VERIFICATION COMMANDS
# ===========================================

echo "=== Quick Verification ==="
echo "Run these commands to verify everything is working:"

echo ""
echo "1. Basic health checks:"
echo "   curl http://localhost:8000/health"
echo "   curl http://localhost:11434/api/tags"

echo ""
echo "2. Simple API test:"
echo '   curl -X POST http://localhost:8000/v1/chat/completions \'
echo '     -H "Content-Type: application/json" \'
echo '     -H "X-API-Key: sk-default" \'
echo '     -d '"'"'{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello"}],"max_tokens":5}'"'"

echo ""
echo "3. Model availability:"
echo "   curl http://localhost:8000/models | jq '.data[].id'"

echo ""
echo "4. System metrics:"
echo "   curl http://localhost:8000/metrics | jq '.'"

echo ""
echo "5. Full system test:"
echo "   bash complete_system_test.sh"

echo ""
echo "=== Debugging Tips ==="
echo "• Check logs regularly: tail -f *.log"
echo "• Monitor resources: htop or top"
echo "• Test incrementally: Start with basic endpoints first"  
echo "• Use verbose curl: Add -v flag to see full request/response"
echo "• Check firewall: Ensure ports 8000 and 11434 are open"
echo "• Verify GPU: nvidia-smi (if using GPU acceleration)"

echo ""
echo "For more detailed testing, run: ./complete_system_test.sh"
