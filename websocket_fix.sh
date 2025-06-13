#!/bin/bash

# websocket_fix.sh - Quick fix for WebSocket authentication issues

echo "🔧 Fixing WebSocket authentication issues..."

# Create backup of middleware file
if [ -f "middleware/auth.py" ]; then
    cp middleware/auth.py middleware/auth.py.backup
    echo "✅ Created backup of middleware/auth.py"
fi

# Option 1: Quick fix - exclude WebSocket from auth middleware
echo "Applying quick fix to exclude WebSocket from authentication..."

# Add WebSocket path to public endpoints
if [ -f "middleware/auth.py" ]; then
    # Check if websocket path is already excluded
    if ! grep -q '"/ws/dashboard"' middleware/auth.py; then
        # Add WebSocket path to public endpoints
        sed -i 's|"/favicon.ico"|"/favicon.ico",\n            "/ws/dashboard"  # WebSocket endpoints|' middleware/auth.py
        echo "✅ Added WebSocket path to public endpoints"
    else
        echo "ℹ️  WebSocket path already excluded from authentication"
    fi
else
    echo "⚠️  middleware/auth.py not found"
fi

# Option 2: Update main.py WebSocket endpoint with query parameter auth
echo "Checking main.py for WebSocket endpoint..."

if [ -f "main.py" ]; then
    # Check if WebSocket endpoint needs updating
    if grep -q "@app.websocket(\"/ws/dashboard\")" main.py; then
        echo "ℹ️  Found WebSocket endpoint in main.py"
        echo "📝 Consider updating the WebSocket endpoint to handle query parameter authentication"
        echo "   Add 'api_key: str = Query(None)' parameter to the function"
    fi
fi

# Create test script
cat > test_websocket.py << 'EOF'
#!/usr/bin/env python3
"""
Quick WebSocket test script
"""
import asyncio
import websockets
import json
import sys

async def test_websocket():
    uri = "ws://localhost:8000/ws/dashboard"
    
    try:
        print(f"🔌 Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully!")
            
            # Send ping
            ping_message = {"type": "ping"}
            await websocket.send(json.dumps(ping_message))
            print("📤 Sent ping message")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📥 Received: {response}")
            except asyncio.TimeoutError:
                print("⏰ No response received within 5 seconds")
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing WebSocket connection...")
    success = asyncio.run(test_websocket())
    sys.exit(0 if success else 1)
EOF

chmod +x test_websocket.py
echo "✅ Created test_websocket.py script"

# Create environment check script
cat > check_websocket_config.py << 'EOF'
#!/usr/bin/env python3
"""
Check WebSocket configuration
"""
import os
from pathlib import Path

def check_config():
    print("🔍 Checking WebSocket configuration...")
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            content = f.read()
            if "ENABLE_AUTH=true" in content:
                print("🔐 Authentication is ENABLED")
                print("   WebSocket connections need authentication")
            else:
                print("🔓 Authentication is DISABLED or not set")
                print("   WebSocket connections should work without auth")
    
    # Check for auth middleware
    auth_middleware = Path("middleware/auth.py")
    if auth_middleware.exists():
        with open(auth_middleware) as f:
            content = f.read()
            if '"/ws/dashboard"' in content:
                print("✅ WebSocket path excluded from authentication")
            else:
                print("⚠️  WebSocket path NOT excluded from authentication")
                print("   This may cause 403 errors")
    
    print("\n📋 Recommendations:")
    print("1. Restart your server after making changes")
    print("2. Test WebSocket connection with: python test_websocket.py")
    print("3. Check server logs for WebSocket connection attempts")

if __name__ == "__main__":
    check_config()
EOF

chmod +x check_websocket_config.py
echo "✅ Created check_websocket_config.py script"

echo ""
echo "🎉 WebSocket fix applied!"
echo ""
echo "📋 Next steps:"
echo "1. Restart your server: python main.py"
echo "2. Check configuration: python check_websocket_config.py"
echo "3. Test WebSocket: python test_websocket.py"
echo "4. Monitor logs for any remaining issues"
echo ""
echo "💡 If issues persist, check the detailed solutions in the artifacts above"
