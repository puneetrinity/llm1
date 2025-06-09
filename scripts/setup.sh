#!/bin/bash
# scripts/setup.sh - Setup script for development

set -e

echo "🚀 Setting up Enhanced LLM Proxy development environment"

# Create necessary directories
mkdir -p data/{cache,logs,models}
mkdir -p tests

# Create .env from template if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env from template..."
    cp .env.template .env
    echo "✅ Please edit .env file with your configuration"
fi

# Create placeholder files for data directories
touch data/cache/.gitkeep
touch data/logs/.gitkeep  
touch data/models/.gitkeep

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: docker-compose up --build"
echo "3. Test: curl http://localhost:8000/health"
