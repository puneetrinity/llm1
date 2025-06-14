#!/bin/bash
# diagnostic.sh - Comprehensive Dashboard and Auth Diagnostic

set -e

echo "🔍 LLM Proxy Dashboard & Auth Diagnostic"
echo "========================================"
echo ""

# 1. Check project structure
echo "📁 Project Structure:"
echo "--------------------"
for dir in frontend static; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/ exists"
        if [ "$dir" = "frontend" ]; then
            [ -f "frontend/package.json" ] && echo "  ✅ package.json found" || echo "  ❌ package.json missing"
            [ -d "frontend/build" ] && echo "  ✅ build/ directory exists" || echo "  ❌ build/ directory missing"
            [ -f "frontend/build/index.html" ] && echo "  ✅ build/index.html exists" || echo "  ❌ build/index.html missing"
        fi
    else
        echo "❌ $dir/ missing"
    fi
done
echo ""

# 2. Check configuration files
echo "⚙️ Configuration Files:"
echo "----------------------"
for file in .env config.py main_master.py; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done
echo ""

# 3. Check environment variables
echo "🔧 Environment Configuration:"
echo "----------------------------"
if [ -f ".env" ]; then
    echo "Current .env settings:"
    grep -E "(ENABLE_AUTH|DEFAULT_API_KEY|ENABLE_WEBSOCKET|HOST|PORT)" .env | while read line; do
        echo "  $line"
    done
else
    echo "❌ .env file not found"
fi
echo ""

# 4. Test Python configuration
echo "🐍 Python Configuration Test:"
echo "-----------------------------"
python3 -c "
import sys
import os
sys.path.append('.')

try:
    # Try to import settings
    try:
        from config import settings
        print('✅ config.py imported successfully')
        print(f'   HOST: {settings.HOST}')
        print(f'   PORT: {settings.PORT}')
        print(f'   ENABLE_AUTH: {settings.ENABLE_AUTH}')
        print(f'   DEFAULT_API_KEY: {settings.DEFAULT_API_KEY[:8]}...' if hasattr(settings, 'DEFAULT_API_KEY') else '   DEFAULT_API_KEY: Not set')
    except ImportError as e:
        print(f'⚠️ config.py import failed: {e}')
        print('   Trying alternative configuration...')
        
        # Try environment variables directly
        print(f'   ENABLE_AUTH: {os.getenv(\"ENABLE_AUTH\", \"Not set\")}')
        print(f'   DEFAULT_API_KEY: {os.getenv(\"DEFAULT_API_KEY\", \"Not set\")[:8]}...' if os.getenv('DEFAULT_API_KEY') else '   DEFAULT_API_KEY: Not set')
    
    # Check if main_master.py can be imported
    try:
        import main_master
        print('✅ main_master.py can be imported')
    except ImportError as e:
        print(f'❌ main_master.py import failed: {e}')
    except Exception as e:
        print(f'⚠️ main_master.py import error: {e}')

except Exception as e:
    print(f'❌ Python configuration test failed: {e}')
"
echo ""

# 5. Check server status
echo "🌐 Server Status Check:"
echo "---------------------"
if curl -s --connect-timeout 5 http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Server is running on port 8001"
    
    # Test health endpoint
    health_response=$(curl -s http://localhost:8001/health)
    echo "  Health response: $health_response"
    
    # Test dashboard endpoint
    if curl -s --connect-timeout 5 http://localhost:8001/app/ > /dev/null 2>&1; then
        echo "✅ Dashboard endpoint accessible"
    else
        echo "❌ Dashboard endpoint not accessible"
    fi
    
    # Test API key authentication
    api_key=$(grep "DEFAULT_API_KEY=" .env 2>/dev/null | cut -d'=' -f2 || echo "")
    if [ -n "$api_key" ]; then
        auth_test=$(curl -s -w "%{http_code}" -o /dev/null -X POST http://localhost:8001/auth/websocket-session \
                   -H "Content-Type: application/json" \
                   -H "X-API-Key: $api_key")
        if [ "$auth_test" = "200" ]; then
            echo "✅ API key authentication working"
        elif [ "$auth_test" = "401" ]; then
            echo "❌ API key authentication failed (401 Unauthorized)"
            echo "  This is likely the cause of your WebSocket session error"
        else
            echo "⚠️ API key authentication returned: $auth_test"
        fi
    else
        echo "⚠️ No API key found to test"
    fi
    
else
    echo "❌ Server not running on port 8001"
    echo "  Start server with: python main_master.py"
fi
echo ""

# 6. Frontend diagnostic
echo "⚛️ Frontend Diagnostic:"
echo "----------------------"
if [ -d "frontend" ]; then
    cd frontend
    
    # Check Node.js and npm
    if command -v node > /dev/null 2>&1; then
        echo "✅ Node.js: $(node --version)"
    else
        echo "❌ Node.js not installed"
    fi
    
    if command -v npm > /dev/null 2>&1; then
        echo "✅ npm: $(npm --version)"
    else
        echo "❌ npm not installed"
    fi
    
    # Check dependencies
    if [ -f "package.json" ] && [ -d "node_modules" ]; then
        echo "✅ Dependencies installed"
    elif [ -f "package.json" ]; then
        echo "⚠️ package.json exists but node_modules missing"
        echo "  Run: cd frontend && npm install"
    else
        echo "❌ package.json missing"
    fi
    
    # Check build
    if [ -d "build" ] && [ -f "build/index.html" ]; then
        echo "✅ Build directory exists"
        build_size=$(du -sh build 2>/dev/null | cut -f1)
        echo "  Build size: $build_size"
        
        # Check for CSS files
        if ls build/static/css/*.css 1> /dev/null 2>&1; then
            css_count=$(ls build/static/css/*.css | wc -l)
            echo "✅ CSS files: $css_count"
        else
            echo "❌ No CSS files found in build"
            echo "  This could be causing your styling issues"
        fi
        
        # Check for JS files
        if ls build/static/js/*.js 1> /dev/null 2>&1; then
            js_count=$(ls build/static/js/*.js | wc -l)
            echo "✅ JS files: $js_count"
        else
            echo "❌ No JS files found in build"
        fi
    else
        echo "❌ Build directory missing or incomplete"
        echo "  Run: cd frontend && npm run build"
    fi
    
    cd ..
else
    echo "❌ Frontend directory not found"
fi
echo ""

# 7. Summary and recommendations
echo "📋 Summary & Recommendations:"
echo "=============================="

# CSS Issue Analysis
if [ ! -d "frontend/build" ] || [ ! -f "frontend/build/index.html" ]; then
    echo "🎨 CSS Issue: Frontend not built properly"
    echo "   Fix: Run ./fix_css_frontend.sh or manually:"
    echo "   cd frontend && npm install && npm run build"
fi

# Auth Issue Analysis
enable_auth=$(grep "ENABLE_AUTH=" .env 2>/dev/null | cut -d'=' -f2)
if [ "$enable_auth" = "true" ]; then
    echo "🔒 Auth Issue: Authentication enabled but may have problems"
    echo "   Fix: Run ./fix_auth_websocket.sh"
    echo "   Or disable auth: ENABLE_AUTH=false in .env"
fi

# Server Issue Analysis
if ! curl -s --connect-timeout 5 http://localhost:8001/health > /dev/null 2>&1; then
    echo "🚫 Server Issue: Server not running"
    echo "   Fix: python main_master.py"
fi

echo ""
echo "🎯 Quick Fixes:"
echo "1. For CSS issues: chmod +x fix_css_frontend.sh && ./fix_css_frontend.sh"
echo "2. For 401 auth issues: chmod +x fix_auth_websocket.sh && ./fix_auth_websocket.sh"
echo "3. Restart server: python main_master.py"
echo "4. Test dashboard: http://localhost:8001/app/"
