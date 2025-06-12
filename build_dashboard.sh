#!/bin/bash
set -e
echo "🔨 Building React Dashboard..."

# Create build lock to prevent concurrent builds
if [ -f ".dashboard_building" ]; then
    echo "❌ Build already in progress, cleaning up..."
    rm -f ".dashboard_building"
fi
echo "$(date): Build started" > .dashboard_building

# Cleanup function
cleanup() {
    rm -f ".dashboard_building"
    echo "🔓 Build lock removed"
}
trap cleanup EXIT

cd /app/frontend

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found in /app/frontend/"
    exit 1
fi

# Install dependencies with timeout
echo "📦 Installing React dependencies..."
timeout 300 npm install --silent --legacy-peer-deps || {
    echo "❌ npm install failed or timed out"
    exit 1
}

# Build React app with timeout
echo "🏗️ Building React app..."
export NODE_ENV=production
export CI=false
timeout 180 npm run build || {
    echo "❌ React build failed or timed out"
    exit 1
}

# Copy to static directory
echo "📋 Copying to static directory..."
mkdir -p ../static
cp -r build/* ../static/

# Verify deployment
if [ ! -f "../static/index.html" ]; then
    echo "❌ Deployment failed - index.html not found"
    exit 1
fi

echo "✅ Dashboard built successfully!"
ls -la ../static/
