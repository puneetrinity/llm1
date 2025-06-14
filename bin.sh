#!/bin/bash
# test_build.sh - Test the frontend build locally

set -e

echo "🧪 Testing Frontend Build"
echo "========================="

cd frontend

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build dist node_modules/.cache 2>/dev/null || true

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the project
echo "🏗️ Building project..."
npm run build

# Check results
echo "🔍 Checking build results..."

if [ -f "build/index.html" ]; then
    echo "✅ SUCCESS: build/index.html exists"
    echo "📊 Build stats:"
    echo "   Directory: build/"
    echo "   Files: $(find build -type f | wc -l)"
    echo "   Size: $(du -sh build | cut -f1)"
    echo ""
    echo "📁 Build contents:"
    ls -la build/
    echo ""
    echo "🎉 Build is ready for Docker!"
elif [ -f "dist/index.html" ]; then
    echo "⚠️  Found dist/index.html instead of build/"
    echo "🔧 Moving to correct location..."
    mv dist build
    echo "✅ Fixed: Now in build/ directory"
    ls -la build/
    echo ""
    echo "⚠️  Your project built to 'dist/' instead of 'build/'. Please check your build configuration."
    echo "🎉 Build is ready for Docker after adjustment!"
else
    echo "❌ ERROR: No index.html found in build/ or dist/"
    echo "💡 Check your build configuration and ensure it outputs to either directory"
    exit 1
fi
