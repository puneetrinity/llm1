#!/bin/bash

# start.sh - LLM Proxy Server Startup Script

set -e  # Exit on any error

echo "🚀 Starting Complete LLM Proxy Server..."
echo "=============================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    netstat -tulpn 2>/dev/null | grep ":$1 " >/dev/null
}

# Environment setup
export PYTHONPATH="/app:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Default environment variables (can be overridden)
export ENABLE_AUTH=${ENABLE_AUTH:-false}
export ENABLE_WEBSOCKET_DASHBOARD=${ENABLE_WEBSOCKET_DASHBOARD:-false}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8001}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "📋 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Auth Enabled: $ENABLE_AUTH"
echo "   WebSocket Dashboard: $ENABLE_WEBSOCKET_DASHBOARD"
echo "   Log Level: $LOG_LEVEL"

# Check if Python is available
if ! command_exists python3; then
    echo "❌ Python3 is not installed!"
    exit 1
fi

# Check if required files exist
if [ ! -f "/app/main_master.py" ] && [ ! -f "/app/main.py" ]; then
    echo "❌ No main application file found (main_master.py or main.py)!"
    exit 1
fi

# Determine which main file to use
MAIN_FILE=""
if [ -f "/app/main_master.py" ]; then
    MAIN_FILE="main_master"
    echo "✅ Using main_master.py"
elif [ -f "/app/main.py" ]; then
    MAIN_FILE="main"
    echo "✅ Using main.py"
fi

# Check if port is already in use
if port_in_use $PORT; then
    echo "⚠️  Port $PORT is already in use. Trying to kill existing processes..."
    pkill -f "uvicorn.*$PORT" || true
    sleep 2
    
    if port_in_use $PORT; then
        echo "❌ Port $PORT is still in use after cleanup. Exiting..."
        exit 1
    fi
fi

# Create required directories
mkdir -p /app/logs /app/cache /app/models /app/data /app/static

# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "🤖 Starting Ollama..."
    ollama serve > /app/logs/ollama.log 2>&1 &
    sleep 5
fi

# Test Python import
echo "🔍 Testing Python imports..."
cd /app
if ! python3 -c "import $MAIN_FILE" 2>/dev/null; then
    echo "❌ Failed to import $MAIN_FILE. Checking for errors..."
    python3 -c "import $MAIN_FILE" || {
        echo "❌ Import failed. Check the logs above for errors."
        exit 1
    }
fi
echo "✅ Python imports successful"

# Check if dashboard files exist
if [ -f "/app/frontend/build/index.html" ]; then
    echo "✅ Dashboard files found"
else
    echo "⚠️  Dashboard files not found, will use fallback"
fi

echo "🚀 Starting uvicorn server..."
echo "📍 Server will be available at: http://$HOST:$PORT"
echo "📊 Dashboard will be available at: http://$HOST:$PORT/app"
echo "📚 API docs will be available at: http://$HOST:$PORT/docs"
echo "=============================================="

# Start the server with proper error handling
exec python3 -m uvicorn ${MAIN_FILE}:app \
    --host $HOST \
    --port $PORT \
    --log-level $(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]') \
    --access-log \
    --reload \
    --reload-dir /app \
    --reload-exclude "logs/*" \
    --reload-exclude "cache/*" \
    --reload-exclude "models/*" \
    --timeout-keep-alive 30
