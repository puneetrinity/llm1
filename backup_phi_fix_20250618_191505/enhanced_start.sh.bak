#!/bin/bash

# Add Ollama to PATH (if installed in custom location)
export PATH=/workspace/ollama/bin:$PATH

# Enhanced startup script with 4-model preloading and warmup
echo "🚀 Starting Enhanced 4-Model LLM Proxy Service..."

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

# Pull and preload 4 models in priority order
echo "📦 Pulling and preloading 4 models..."

# Priority 1: Phi for reasoning
echo "   🔄 Pulling Phi-3.5 (Reasoning - Priority 1)..."
ollama pull phi:3.5 &
PHI_PID=$!

# Priority 2: Mistral for general
echo "   🔄 Pulling Mistral 7B (General - Priority 2)..."
ollama pull mistral:7b-instruct-q4_0 &
MISTRAL_PID=$!

# Priority 2: Gemma for coding  
echo "   🔄 Pulling Gemma 7B (Technical - Priority 2)..."
ollama pull gemma:7b-instruct &
GEMMA_PID=$!

# Priority 3: Llama3 for creative
echo "   🔄 Pulling Llama3 8B (Creative - Priority 3)..."
ollama pull llama3:8b-instruct-q4_0 &
LLAMA_PID=$!

# Wait for priority 1 model (Phi) to complete first
echo "   ⏳ Waiting for priority model (Phi-3.5)..."
wait $PHI_PID
echo "   ✅ Phi-3.5 ready!"

# Warm up the priority model immediately
echo "   🔥 Warming up Phi-3.5..."
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi:3.5",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": false,
    "options": {"num_predict": 5}
  }' >/dev/null 2>&1

echo "   ✅ Phi-3.5 warmed up and ready for reasoning tasks!"

# Wait for other models in background
echo "   ⏳ Waiting for remaining models..."
wait $MISTRAL_PID && echo "   ✅ Mistral 7B ready!"
wait $GEMMA_PID && echo "   ✅ Gemma 7B ready!"
wait $LLAMA_PID && echo "   ✅ Llama3 8B ready!"

echo "🎯 All 4 models loaded successfully!"
echo ""
echo "🎯 Model Routing:"
echo "├── 🧠 Math/Logic/Reasoning    → Phi-3.5"
echo "├── ⚙️  Coding/Technical        → Gemma 7B"
echo "├── 🎨 Creative/Storytelling   → Llama3 8B"
echo "└── ⚡ General/Quick Facts     → Mistral 7B"

# Start the FastAPI application
echo "🌐 Starting FastAPI application..."
python3 main_master.py

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
