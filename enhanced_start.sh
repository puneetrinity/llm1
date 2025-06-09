#!/bin/bash

# Enhanced startup script with model preloading and warmup

echo "🚀 Starting Enhanced LLM Proxy Service..."

# Start Ollama in background
echo "📡 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
for i in {1..30}; do
  if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is ready!"
    break
  fi
  echo "   Attempt $i/30 - waiting 2 seconds..."
  sleep 2
done

# Check if Ollama started successfully
if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "❌ Failed to start Ollama service"
  exit 1
fi

# Pull and preload models in priority order
echo "📦 Pulling and preloading models..."

# Priority 1: Most frequently used (Mistral)
echo "   🔄 Pulling Mistral 7B (Priority 1)..."
ollama pull mistral:7b-instruct-q4_0 &
MISTRAL_PID=$!

# Priority 2: Analysis models
echo "   🔄 Pulling DeepSeek V2 7B (Priority 2)..."
ollama pull deepseek-v2:7b-q4_0 &
DEEPSEEK_PID=$!

echo "   🔄 Pulling LLaMA3 8B (Priority 2)..."
ollama pull llama3:8b-instruct-q4_0 &
LLAMA_PID=$!

# Wait for priority 1 model (Mistral) to complete first
echo "   ⏳ Waiting for priority model (Mistral)..."
wait $MISTRAL_PID
echo "   ✅ Mistral 7B ready!"

# Warm up the priority model immediately
echo "   🔥 Warming up Mistral..."
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b-instruct-q4_0",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false,
    "options": {"num_predict": 5}
  }' >/dev/null 2>&1

echo "   ✅ Mistral warmed up and ready for requests!"

# Wait for other models in background
echo "   ⏳ Waiting for remaining models..."
wait $DEEPSEEK_PID && echo "   ✅ DeepSeek V2 7B ready!"
wait $LLAMA_PID && echo "   ✅ LLaMA3 8B ready!"

echo "🎯 All models loaded successfully!"

# Start the FastAPI application
echo "🌐 Starting FastAPI application..."
python3 main_enhanced.py

# Cleanup function
cleanup() {
  echo "🛑 Shutting down services..."
  kill $OLLAMA_PID 2>/dev/null
  exit
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Keep the script running
wait