#!/bin/bash
# download_4_models.sh - Download the correct 4 models

set -e

echo "📦 Downloading 4 Models for LLM Proxy"
echo "====================================="
echo ""
echo "🎯 Target Models:"
echo "├── 🧠 Phi-3.5 (Reasoning)"
echo "├── ⚡ Mistral 7B (General)"  
echo "├── ⚙️  Gemma 7B (Technical)"
echo "└── 🎨 Llama3 8B (Creative)"
echo ""

# Start Ollama if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🤖 Starting Ollama..."
    ollama serve &
    sleep 10
fi

echo "📥 Downloading models in priority order..."

# Priority 1: Phi for reasoning
echo "🧠 Downloading Phi-3.5 (Reasoning model)..."
ollama pull phi:3.5 &
PHI_PID=$!

# Priority 2: Mistral for general use
echo "⚡ Downloading Mistral 7B (General model)..."
ollama pull mistral:7b-instruct-q4_0 &
MISTRAL_PID=$!

# Priority 2: Gemma for coding
echo "⚙️  Downloading Gemma 7B (Technical model)..."
ollama pull gemma:7b-instruct &
GEMMA_PID=$!

# Priority 3: Llama3 for creative
echo "🎨 Downloading Llama3 8B (Creative model)..."
ollama pull llama3:8b-instruct-q4_0 &
LLAMA_PID=$!

echo "⏳ Waiting for downloads to complete..."

# Wait for all downloads
wait $PHI_PID && echo "✅ Phi-3.5 ready"
wait $MISTRAL_PID && echo "✅ Mistral 7B ready"  
wait $GEMMA_PID && echo "✅ Gemma 7B ready"
wait $LLAMA_PID && echo "✅ Llama3 8B ready"

echo ""
echo "🎉 All 4 models downloaded successfully!"
echo ""
echo "📊 Verify with: ollama list"
ollama list

echo ""
echo "🚀 Ready to start your 4-model LLM proxy:"
echo "   python main_master.py"
