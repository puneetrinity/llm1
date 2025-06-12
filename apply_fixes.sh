#!/bin/bash
echo "🔧 Applying all compatibility fixes..."

# Fix 1: Connection pool tcp_nodelay compatibility
echo "Fixing connection pool..."
if [ -f "utils/connection_pool.py" ]; then
    sed -i 's/tcp_nodelay=self.config.tcp_nodelay,/# tcp_nodelay=self.config.tcp_nodelay,  # Compatibility fix/' utils/connection_pool.py
    echo "✅ Connection pool fixed"
fi

# Fix 2: WebSocket dashboard constructor
echo "Fixing WebSocket dashboard..."
if [ -f "utils/websocket_dashboard.py" ]; then
    sed -i 's/def __init__(self, enhanced_dashboard):/def __init__(self, enhanced_dashboard, metrics_collector=None, performance_monitor=None):/' utils/websocket_dashboard.py
    sed -i '/self.dashboard = enhanced_dashboard/a\        self.metrics_collector = metrics_collector\n        self.performance_monitor = performance_monitor' utils/websocket_dashboard.py
    echo "✅ WebSocket dashboard fixed"
fi

# Fix 3: Add missing config fields
echo "Fixing enhanced config..."
if [ -f "config_enhanced.py" ] && ! grep -q "ENABLE_DASHBOARD" config_enhanced.py; then
    cat >> config_enhanced.py << 'CONFIGEOF'

    # Dashboard Settings (Added for compatibility)
    ENABLE_DASHBOARD: bool = Field(default=True, description="Enable dashboard")
    ENABLE_WEBSOCKET_DASHBOARD: bool = Field(default=True, description="Enable WebSocket dashboard")
    DASHBOARD_UPDATE_INTERVAL: int = Field(default=10, description="Dashboard update interval")
CONFIGEOF
    echo "✅ Enhanced config fixed"
fi

# Fix 4: Update requirements for compatibility
echo "Fixing requirements..."
if [ -f "requirements.txt" ]; then
    sed -i 's/aiohttp==3.9.1/aiohttp==3.8.6/' requirements.txt
    echo "✅ Requirements fixed"
fi

# Fix 5: Install compatible version
echo "Installing compatible aiohttp version..."
pip install aiohttp==3.8.6 --upgrade

echo ""
echo "🎉 All compatibility fixes applied!"
echo "💡 Now restart your service with: ./enhanced_start.sh"
echo ""
