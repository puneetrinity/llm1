#!/bin/bash
# safety_check.sh - Prevent installation loops and conflicts

set -e

echo "🔒 Running Installation Safety Checks"
echo "====================================="

# Check 1: Prevent duplicate dashboard routes
check_duplicate_routes() {
    echo "🔍 Checking for duplicate dashboard routes..."
    
    if [ -f "main.py" ]; then
        dashboard_count=$(grep -c "DashboardWebSocketManager" main.py || echo "0")
        websocket_count=$(grep -c "@app.websocket(\"/ws/dashboard\")" main.py || echo "0")
        static_mount_count=$(grep -c "app.mount.*static" main.py || echo "0")
        
        if [ "$dashboard_count" -gt 1 ]; then
            echo "❌ LOOP RISK: Multiple DashboardWebSocketManager found"
            echo "   Removing duplicates..."
            # Create backup and remove duplicates
            cp main.py main.py.safety_backup
            # Keep only the first occurrence
            sed '0,/DashboardWebSocketManager/b; /DashboardWebSocketManager/,/dashboard_ws_manager = DashboardWebSocketManager()/d' main.py > main.py.tmp
            mv main.py.tmp main.py
        fi
        
        if [ "$websocket_count" -gt 1 ]; then
            echo "❌ LOOP RISK: Multiple WebSocket routes found"
            return 1
        fi
        
        if [ "$static_mount_count" -gt 1 ]; then
            echo "❌ LOOP RISK: Multiple static mounts found"
            return 1
        fi
        
        echo "✅ No duplicate routes found"
    else
        echo "⚠️  main.py not found"
    fi
}

# Check 2: Prevent build loops
check_build_state() {
    echo "🔍 Checking build state to prevent loops..."
    
    # Check if already building
    if [ -f ".dashboard_building" ]; then
        echo "❌ LOOP RISK: Build already in progress"
        echo "   Removing stale build lock..."
        rm -f ".dashboard_building"
    fi
    
    # Check if Node.js processes are stuck
    node_processes=$(pgrep -f "node" | wc -l || echo "0")
    if [ "$node_processes" -gt 5 ]; then
        echo "⚠️  Multiple Node.js processes detected: $node_processes"
        echo "   This might indicate a stuck build process"
    fi
    
    echo "✅ Build state clean"
}

# Check 3: Prevent service loops
check_service_state() {
    echo "🔍 Checking service state..."
    
    python_processes=$(pgrep -f "python.*main.py" | wc -l || echo "0")
    if [ "$python_processes" -gt 1 ]; then
        echo "❌ LOOP RISK: Multiple Python services running: $python_processes"
        echo "   Killing duplicate processes..."
        pkill -f "python.*main.py"
        sleep 2
        echo "   Cleaned up duplicate services"
    fi
    
    echo "✅ Service state clean"
}

# Check 4: File system safety
check_filesystem() {
    echo "🔍 Checking filesystem safety..."
    
    # Check for circular symlinks
    if [ -d "static" ]; then
        circular_links=$(find static -type l -exec test -e {} \; -print 2>/dev/null | wc -l || echo "0")
        if [ "$circular_links" -gt 0 ]; then
            echo "⚠️  Circular symlinks detected in static directory"
        fi
    fi
    
    # Check disk space (prevent out-of-space loops)
    available_space=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 1000000 ]; then  # Less than ~1GB
        echo "⚠️  Low disk space: ${available_space}KB available"
        echo "   Consider cleaning up before proceeding"
    fi
    
    echo "✅ Filesystem checks passed"
}

# Check 5: Port conflicts
check_port_conflicts() {
    echo "🔍 Checking for port conflicts..."
    
    # Check if port 8000 is in use by multiple processes
    port_8000_count=$(netstat -tlnp 2>/dev/null | grep ":8000 " | wc -l || echo "0")
    if [ "$port_8000_count" -gt 1 ]; then
        echo "❌ LOOP RISK: Multiple services on port 8000"
        netstat -tlnp | grep ":8000 "
        return 1
    fi
    
    echo "✅ No port conflicts"
}

# Check 6: Memory safety
check_memory() {
    echo "🔍 Checking memory usage..."
    
    # Get available memory in MB
    available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_mem" -lt 1024 ]; then  # Less than 1GB available
        echo "⚠️  Low memory: ${available_mem}MB available"
        echo "   Consider freeing memory before building dashboard"
    fi
    
    echo "✅ Memory check passed"
}

# Main safety check function
run_safety_checks() {
    echo "🛡️  Starting comprehensive safety checks..."
    
    check_duplicate_routes
    check_build_state  
    check_service_state
    check_filesystem
    check_port_conflicts
    check_memory
    
    echo ""
    echo "✅ All safety checks passed!"
    echo "🚀 Safe to proceed with installation"
}

# Create build lock to prevent concurrent builds
create_build_lock() {
    echo "$(date): Build started" > .dashboard_building
    echo "🔒 Build lock created"
}

# Remove build lock
remove_build_lock() {
    rm -f .dashboard_building
    echo "🔓 Build lock removed"
}

# Enhanced build function with loop prevention
safe_build_dashboard() {
    echo "🔨 Starting safe dashboard build..."
    
    # Check if already built
    if [ -f "static/index.html" ] && [ ! "$1" = "--force" ]; then
        echo "✅ Dashboard already built. Use --force to rebuild."
        return 0
    fi
    
    create_build_lock
    
    trap remove_build_lock EXIT
    
    cd frontend
    
    # Timeout for npm install (prevent hanging)
    timeout 300 npm install --silent || {
        echo "❌ npm install timed out or failed"
        remove_build_lock
        exit 1
    }
    
    # Timeout for build process
    timeout 180 npm run build || {
        echo "❌ npm build timed out or failed"
        remove_build_lock
        exit 1
    }
    
    cd ..
    
    # Copy with verification
    if [ -d "frontend/build" ]; then
        mkdir -p static
        cp -r frontend/build/* static/
        echo "✅ Dashboard built successfully"
    else
        echo "❌ Build directory not found"
        remove_build_lock
        exit 1
    fi
    
    remove_build_lock
}

# Run the appropriate function based on argument
case "${1:-check}" in
    "check")
        run_safety_checks
        ;;
    "build")
        run_safety_checks
        safe_build_dashboard "$2"
        ;;
    "clean")
        echo "🧹 Cleaning up potential loops..."
        pkill -f "node.*build" || echo "No node build processes"
        pkill -f "npm.*build" || echo "No npm build processes" 
        rm -f .dashboard_building
        echo "✅ Cleanup complete"
        ;;
    *)
        echo "Usage: $0 {check|build|clean}"
        echo "  check: Run safety checks only"
        echo "  build: Run safety checks and build dashboard"
        echo "  clean: Clean up stuck processes and locks"
        ;;
esac
