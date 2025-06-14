#!/bin/bash
# fix_css_frontend.sh - Fix CSS and Frontend Build Issues

set -e

echo "🎨 Fixing CSS and Frontend Build Issues..."

# 1. Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Frontend directory not found!"
    exit 1
fi

cd frontend

# 2. Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found!"
    exit 1
fi

# 3. Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ node_modules/.cache/ .parcel-cache/

# 4. Install dependencies
echo "📦 Installing dependencies..."
npm install --silent

# 5. Build the React app
echo "🔨 Building React app..."
npm run build

# 6. Verify build output
if [ -d "build" ] && [ -f "build/index.html" ]; then
    echo "✅ Build successful!"
    echo "📁 Build contents:"
    ls -la build/
    
    # Check CSS files
    if ls build/static/css/*.css 1> /dev/null 2>&1; then
        echo "✅ CSS files found:"
        ls -la build/static/css/
    else
        echo "⚠️ No CSS files found in build"
    fi
    
    # Check JS files
    if ls build/static/js/*.js 1> /dev/null 2>&1; then
        echo "✅ JS files found:"
        ls -la build/static/js/
    else
        echo "⚠️ No JS files found in build"
    fi
else
    echo "❌ Build failed!"
    exit 1
fi

# 7. Copy build to static directory (if needed)
cd ..
if [ ! -d "static" ]; then
    mkdir -p static
fi

echo "📋 Copying build files to static directory..."
cp -r frontend/build/* static/

# 8. Verify static files
if [ -f "static/index.html" ]; then
    echo "✅ Static files copied successfully!"
    echo "📊 Dashboard should be available at: http://localhost:8001/app/"
else
    echo "❌ Failed to copy static files"
    exit 1
fi

echo "🎉 CSS and frontend build issues fixed!"
echo ""
echo "Next steps:"
echo "1. Restart your server: python main_master.py"
echo "2. Visit: http://localhost:8001/app/"
echo "3. Check browser console for any remaining errors"
