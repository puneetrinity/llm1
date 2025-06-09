#!/bin/bash
# scripts/test.sh - Quick test script

set -e

BASE_URL=${1:-"http://localhost:8000"}

echo "🧪 Testing Enhanced LLM Proxy at $BASE_URL"

# Test health endpoint
echo "Testing health check..."
curl -s "$BASE_URL/health" | grep -q "healthy" && echo "✅ Health check passed" || echo "❌ Health check failed"

# Test models endpoint
echo "Testing models endpoint..."
curl -s "$BASE_URL/models" | grep -q "data" && echo "✅ Models endpoint working" || echo "❌ Models endpoint failed"

# Test basic completion
echo "Testing basic completion..."
response=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say hello"}]
    }')

if echo "$response" | grep -q "choices"; then
    echo "✅ Basic completion working"
else
    echo "❌ Basic completion failed"
    echo "Response: $response"
fi

echo "🎉 Testing complete!"
