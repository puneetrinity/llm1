#!/bin/bash
# scripts/deploy.sh - Production deployment script

set -e

echo "🚀 Deploying Enhanced LLM Proxy to production"

# Build image
docker build -t llm-proxy-enhanced:latest -f Dockerfile.enhanced .

# Deploy with docker-compose
docker-compose down
docker-compose up -d

# Wait for service
echo "⏳ Waiting for service to start..."
sleep 60

# Health check
echo "🏥 Performing health check..."
curl -f http://localhost:8000/health || {
    echo "❌ Health check failed"
    docker-compose logs llm-proxy
    exit 1
}

echo "✅ Deployment successful!"
echo "📊 Service running at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
