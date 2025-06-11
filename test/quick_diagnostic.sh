#!/bin/bash
# quick_diagnostic.sh - Fast system diagnostic for immediate feedback
# Run this first for quick status check

echo "🔍 QUICK DIAGNOSTIC - LLM PROXY SYSTEM"
echo "======================================="

# Check if services are running
echo "1. Service Status:"
if pgrep -f "uvicorn\|python.*main" > /dev/null; then
    echo "✅ LLM Proxy service is running"
    echo "   PID: $(pgrep -f "uvicorn\|python.*main")"
else
    echo "❌ LLM Proxy service is NOT running"
    echo "   Try: python main.py or ./start.sh"
fi

if pgrep -f "ollama" > /dev/null; then
    echo "✅ Ollama service is running" 
    echo "   PID: $(pgrep -f "ollama")"
else
    echo "❌ Ollama service is NOT running"
    echo "   Try: ollama serve"
fi

# Quick port check
echo -e "\n2. Port Check:"
if ss -tuln | grep -q ":8000 "; then
    echo "✅ Port 8000 (LLM Proxy) is open"
else
    echo "❌ Port 8000 is not accessible"
fi

if ss -tuln | grep -q ":11434 "; then
    echo "✅ Port 11434 (Ollama) is open"
else
    echo "❌ Port 11434 is not accessible"
fi

# Quick API test
echo -e "\n3. Quick API Test:"
if curl -s http://localhost:8000/health | grep -q "healthy\|true"; then
    echo "✅ LLM Proxy API is responding"
else
    echo "❌ LLM Proxy API is not responding"
    echo "   Check: curl http://localhost:8000/health"
fi

if curl -s http://localhost:11434/api/tags | grep -q "models"; then
    echo "✅ Ollama API is responding"
    
    # Count models
    model_count=$(curl -s http://localhost:11434/api/tags | jq '.models | length' 2>/dev/null || echo "0")
    echo "   Available models: $model_count"
    
    if [ "$model_count" -eq 0 ]; then
        echo "   ⚠️  No models found. Run: ollama pull mistral:7b-instruct-q4_0"
    fi
else
    echo "❌ Ollama API is not responding"
    echo "   Check: curl http://localhost:11434/api/tags"
fi

# Resource check
echo -e "\n4. Resource Usage:"
echo "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "   Memory: $(free | grep Mem | awk '{printf("%.1f%%\n", $3/$2 * 100.0)}')"
echo "   Disk: $(df -h / | awk 'NR==2{printf "%s", $5}')"

# GPU check
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1)
    echo "   GPU: $gpu_info"
else
    echo "   GPU: Not detected"
fi

echo -e "\n5. Quick Fixes:"
echo "   Start LLM Proxy: python main.py"
echo "   Start Ollama: ollama serve &"
echo "   Pull a model: ollama pull mistral:7b-instruct-q4_0"
echo "   View logs: tail -f *.log or docker logs <container>"
echo "   Full test: ./complete_system_test.sh"

echo -e "\nRun './complete_system_test.sh' for detailed testing"
