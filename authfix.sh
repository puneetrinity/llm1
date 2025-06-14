#!/bin/bash
# fix_auth_websocket.sh - Fix 401 Authentication and WebSocket Issues

set -e

echo "🔐 Fixing Authentication and WebSocket Issues..."

# 1. Check current .env configuration
if [ -f ".env" ]; then
    echo "📄 Current .env configuration:"
    echo "ENABLE_AUTH=$(grep ENABLE_AUTH .env | cut -d'=' -f2)"
    echo "DEFAULT_API_KEY=$(grep DEFAULT_API_KEY .env | cut -d'=' -f2)"
    echo "API_KEY_HEADER=$(grep API_KEY_HEADER .env | cut -d'=' -f2)"
    echo ""
else
    echo "❌ .env file not found! Creating from template..."
    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo "✅ Created .env from template"
    else
        echo "❌ .env.template also not found!"
        exit 1
    fi
fi

# 2. Get current auth settings
ENABLE_AUTH=$(grep "ENABLE_AUTH=" .env | cut -d'=' -f2 | tr -d ' ')
DEFAULT_API_KEY=$(grep "DEFAULT_API_KEY=" .env | cut -d'=' -f2 | tr -d ' ')

echo "Current settings:"
echo "  ENABLE_AUTH: $ENABLE_AUTH"
echo "  DEFAULT_API_KEY: $DEFAULT_API_KEY"
echo ""

# 3. Provide options to fix the issue
echo "🛠️ Choose a solution:"
echo "1. Disable authentication (for development)"
echo "2. Set a proper API key and keep authentication enabled"
echo "3. Generate a new secure API key"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "🔓 Disabling authentication..."
        sed -i 's/ENABLE_AUTH=true/ENABLE_AUTH=false/' .env
        sed -i 's/ENABLE_WEBSOCKET_DASHBOARD=true/ENABLE_WEBSOCKET_DASHBOARD=false/' .env
        sed -i 's/ENABLE_WEBSOCKET=true/ENABLE_WEBSOCKET=false/' .env
        echo "✅ Authentication disabled"
        echo "⚠️ This is only recommended for development!"
        ;;
    2)
        echo "🔑 Setting up API key authentication..."
        read -p "Enter your API key (must start with 'sk-'): " api_key
        if [[ $api_key == sk-* ]] && [[ ${#api_key} -gt 10 ]]; then
            sed -i "s/DEFAULT_API_KEY=.*/DEFAULT_API_KEY=$api_key/" .env
            sed -i 's/ENABLE_AUTH=false/ENABLE_AUTH=true/' .env
            echo "✅ API key set: ${api_key:0:8}..."
        else
            echo "❌ Invalid API key format. Must start with 'sk-' and be longer than 10 characters"
            exit 1
        fi
        ;;
    3)
        echo "🎲 Generating secure API key..."
        new_key=$(python3 -c "import secrets; print(f'sk-{secrets.token_urlsafe(32)}')" 2>/dev/null || echo "sk-$(openssl rand -hex 32)")
        sed -i "s/DEFAULT_API_KEY=.*/DEFAULT_API_KEY=$new_key/" .env
        sed -i 's/ENABLE_AUTH=false/ENABLE_AUTH=true/' .env
        echo "✅ Generated secure API key: ${new_key:0:8}..."
        echo "🔐 Full API key: $new_key"
        echo "⚠️ Save this key securely! You'll need it to access the dashboard."
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# 4. Update frontend environment if it exists
if [ -f "frontend/.env.development" ]; then
    echo "🔧 Updating frontend environment..."
    if [ "$choice" = "1" ]; then
        sed -i 's/VITE_AUTO_AUTHENTICATE=true/VITE_AUTO_AUTHENTICATE=false/' frontend/.env.development
    else
        # Get the new API key for frontend
        CURRENT_API_KEY=$(grep "DEFAULT_API_KEY=" .env | cut -d'=' -f2)
        sed -i "s/VITE_API_KEY=.*/VITE_API_KEY=$CURRENT_API_KEY/" frontend/.env.development
        sed -i 's/VITE_AUTO_AUTHENTICATE=false/VITE_AUTO_AUTHENTICATE=true/' frontend/.env.development
    fi
    echo "✅ Frontend environment updated"
fi

# 5. Show final configuration
echo ""
echo "📋 Final Configuration:"
echo "======================"
grep -E "(ENABLE_AUTH|DEFAULT_API_KEY|API_KEY_HEADER|ENABLE_WEBSOCKET)" .env
echo ""

# 6. Test the configuration
echo "🧪 Testing configuration..."
if python3 -c "
import sys
sys.path.append('.')
try:
    from config import settings
    print(f'✅ Settings loaded successfully')
    print(f'   ENABLE_AUTH: {settings.ENABLE_AUTH}')
    print(f'   API_KEY_HEADER: {settings.API_KEY_HEADER}')
    print(f'   DEFAULT_API_KEY: {settings.DEFAULT_API_KEY[:8]}...')
except Exception as e:
    print(f'❌ Error loading settings: {e}')
    sys.exit(1)
" 2>/dev/null; then
    echo "✅ Configuration test passed"
else
    echo "⚠️ Could not test configuration (config.py may not exist)"
fi

echo ""
echo "🎉 Authentication and WebSocket issues fixed!"
echo ""
echo "Next steps:"
echo "1. Restart your server: python main_master.py"
echo "2. Visit: http://localhost:8001/app/"
if [ "$choice" != "1" ]; then
    echo "3. Use API key: $(grep DEFAULT_API_KEY .env | cut -d'=' -f2)"
    echo "4. The dashboard should auto-authenticate if VITE_AUTO_AUTHENTICATE=true"
fi
echo ""
echo "If you still get 401 errors:"
echo "- Check the browser console for detailed error messages"
echo "- Verify the API key is being sent in the X-API-Key header"
echo "- Try manually entering the API key in the dashboard"
