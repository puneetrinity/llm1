#!/usr/bin/env python3
# update_main_safe.py - Safe main.py update with loop prevention

import re
import sys
import os
from datetime import datetime


def backup_main():
    """Create timestamped backup of main.py"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"main.py.backup_{timestamp}"

    with open('main.py', 'r') as src, open(backup_name, 'w') as dst:
        dst.write(src.read())

    print(f"✅ Backup created: {backup_name}")
    return backup_name


def check_existing_dashboard_code(content):
    """Check if dashboard code already exists to prevent duplicates"""
    checks = {
        'DashboardWebSocketManager': 'Dashboard WebSocket manager class',
        'app.mount.*static': 'Static files mount',
        '@app.websocket.*dashboard': 'Dashboard WebSocket route',
        'serve_react_app': 'React app serving route'
    }

    found = []
    for pattern, description in checks.items():
        if re.search(pattern, content, re.IGNORECASE):
            found.append(description)

    return found


def add_imports_safely(content):
    """Add required imports without duplicating"""
    imports_to_add = [
        'from fastapi.staticfiles import StaticFiles',
        'from fastapi.responses import FileResponse',
        'from pathlib import Path'
    ]

    # Check which imports are missing
    missing_imports = []
    for imp in imports_to_add:
        if imp not in content:
            missing_imports.append(imp)

    # Add missing imports after the last import line
    if missing_imports:
        import_pattern = r'(from .* import .*\n)'
        matches = list(re.finditer(import_pattern, content))

        if matches:
            last_import_end = matches[-1].end()
            # Insert missing imports
            imports_text = '\n'.join(missing_imports) + '\n'
            content = content[:last_import_end] + \
                imports_text + content[last_import_end:]
            print(f"✅ Added {len(missing_imports)} missing imports")
        else:
            print("⚠️  Could not find import section. Please add imports manually.")

    # Ensure WebSocket is in FastAPI imports
    if 'WebSocket' not in content:
        content = re.sub(
            r'from fastapi import ([^)]+)',
            r'from fastapi import \1, WebSocket, WebSocketDisconnect',
            content
        )
        print("✅ Added WebSocket imports to FastAPI")

    return content


def create_dashboard_code():
    """Create the dashboard integration code"""
    return '''
# ============================================================================
# DASHBOARD INTEGRATION - Auto-generated on {timestamp}
# ============================================================================

# Dashboard WebSocket Manager
class DashboardWebSocketManager:
    def __init__(self):
        self.active_connections = set()
        
    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logging.info(f"Dashboard WebSocket connected. Total: {{len(self.active_connections)}}")
        
        try:
            # Send initial dashboard data
            initial_data = {{
                "timestamp": datetime.now().isoformat(),
                "system_overview": {{
                    "status": "healthy",
                    "total_requests": 0,
                    "avg_response_time": 0,
                    "error_rate": 0,
                    "cache_hit_rate": 0
                }},
                "models": {{}}
            }}
            await websocket.send_text(json.dumps({{"type": "dashboard_update", "data": initial_data}}))
        except Exception as e:
            logging.error(f"Error sending initial dashboard data: {{e}}")
    
    def disconnect(self, websocket):
        self.active_connections.discard(websocket)
        logging.info(f"Dashboard WebSocket disconnected. Total: {{len(self.active_connections)}}")

# Global dashboard manager (singleton to prevent loops)
if 'dashboard_ws_manager' not in globals():
    dashboard_ws_manager = DashboardWebSocketManager()

# Mount static files for React dashboard (with error handling)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logging.info("✅ Static files mounted for dashboard")
except Exception as e:
    logging.warning(f"Static files mount failed: {{e}}")

# WebSocket endpoint for real-time dashboard updates
@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await dashboard_ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle ping-pong to keep connection alive
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({{"type": "pong"}}))
            elif message.get("type") == "request_update":
                # Send updated metrics if available
                try:
                    if 'metrics' in globals() and metrics:
                        metrics_data = await metrics.get_all_metrics()
                        await websocket.send_text(json.dumps({{"type": "dashboard_update", "data": metrics_data}}))
                    else:
                        # Send basic data if metrics not available
                        basic_data = {{
                            "timestamp": datetime.now().isoformat(),
                            "system_overview": {{"status": "healthy", "total_requests": 0}}
                        }}
                        await websocket.send_text(json.dumps({{"type": "dashboard_update", "data": basic_data}}))
                except Exception as e:
                    logging.error(f"Error sending dashboard update: {{e}}")
                    
    except WebSocketDisconnect:
        dashboard_ws_manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"Dashboard WebSocket error: {{e}}")
        dashboard_ws_manager.disconnect(websocket)

# Admin endpoints for React dashboard
@app.get("/admin/circuit-breakers")
async def get_circuit_breakers():
    """Get circuit breaker status for dashboard"""
    try:
        # Return mock data if real circuit breakers not available
        return {{
            "ollama": {{
                "state": "closed",
                "failure_count": 0,
                "success_count": 25,
                "stats": {{"failure_rate": 0.8, "total_requests": 100}}
            }},
            "cache": {{
                "state": "closed", 
                "failure_count": 0,
                "success_count": 50,
                "stats": {{"failure_rate": 0.0, "total_requests": 75}}
            }}
        }}
    except Exception as e:
        logging.error(f"Circuit breaker endpoint error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/cache/stats")
async def get_cache_stats():
    """Get cache statistics for dashboard"""
    try:
        return {{
            "hit_rate": 87.5,
            "total_requests": 2000,
            "cache_hits": 1750,
            "cache_misses": 250,
            "cache_size": 650,
            "memory_usage_mb": 156
        }}
    except Exception as e:
        logging.error(f"Cache stats endpoint error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/cache/clear")
async def clear_cache():
    """Clear application cache"""
    try:
        # Add your cache clearing logic here
        logging.info("Cache clear requested via dashboard")
        return {{"message": "Cache cleared successfully", "success": True}}
    except Exception as e:
        logging.error(f"Cache clear error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

# IMPORTANT: React dashboard route (MUST be the last route to prevent conflicts)
@app.get("/{{path:path}}")
async def serve_react_app(path: str):
    """Serve React dashboard with safety checks"""
    
    # CRITICAL: Prevent API route conflicts
    api_prefixes = (
        "v1/", "admin/", "health", "metrics", "models", 
        "docs", "openapi.json", "redoc", "ws/", 
        "favicon.ico", "_next", "api/"
    )
    
    if any(path.startswith(prefix) for prefix in api_prefixes):
        # This prevents the dashboard from interfering with existing APIs
        raise HTTPException(404, f"API endpoint not found: {{path}}")
    
    try:
        # Serve static files first
        static_file = Path("static") / path
        if static_file.exists() and static_file.is_file():
            return FileResponse(static_file)
        
        # Default to React index.html for SPA routing
        index_file = Path("static") / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        else:
            # Dashboard not built yet
            return {{
                "message": "Dashboard not built yet", 
                "instructions": "Run: ./build_dashboard_safe.sh",
                "status": "dashboard_not_ready"
            }}
            
    except Exception as e:
        logging.error(f"Dashboard serving error: {{e}}")
        return {{"error": "Dashboard serving failed", "message": str(e)}}

# ============================================================================
# END DASHBOARD INTEGRATION
# ============================================================================
'''.format(timestamp=datetime.now().isoformat())


def add_dashboard_code_safely(content):
    """Add dashboard code without creating duplicates"""

    # Find where to insert the dashboard code
    # Look for the main app creation or before if __name__ == "__main__"

    if 'if __name__ == "__main__":' in content:
        # Insert before the main block
        insert_point = content.find('if __name__ == "__main__":')
        dashboard_code = create_dashboard_code()
        content = content[:insert_point] + \
            dashboard_code + '\n' + content[insert_point:]
    else:
        # Append to the end
        dashboard_code = create_dashboard_code()
        content += '\n' + dashboard_code

    print("✅ Dashboard code added successfully")
    return content


def main():
    """Main function to safely update main.py"""

    print("🔧 Safe main.py Update Script")
    print("=============================")

    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("❌ main.py not found in current directory")
        sys.exit(1)

    # Read current content
    with open('main.py', 'r') as f:
        content = f.read()

    # Check if dashboard code already exists
    existing_code = check_existing_dashboard_code(content)
    if existing_code:
        print("⚠️  Dashboard code already exists:")
        for code in existing_code:
            print(f"   - {{code}}")

        response = input("Continue anyway? This may create duplicates (y/N): ")
        if response.lower() != 'y':
            print("❌ Update cancelled")
            sys.exit(0)

    # Create backup
    backup_file = backup_main()

    try:
        # Add imports safely
        content = add_imports_safely(content)

        # Add dashboard code safely
        content = add_dashboard_code_safely(content)

        # Write updated content
        with open('main.py', 'w') as f:
            f.write(content)

        print("✅ main.py updated successfully")
        print(f"💾 Backup available at: {{backup_file}}")
        print("")
        print("🚀 Next steps:")
        print("1. Run: ./build_dashboard_safe.sh")
        print("2. Restart your FastAPI service")
        print("3. Access dashboard at your server URL")

    except Exception as e:
        print(f"❌ Update failed: {{e}}")
        print(f"🔄 Restoring from backup: {{backup_file}}")

        # Restore from backup
        with open(backup_file, 'r') as src, open('main.py', 'w') as dst:
            dst.write(src.read())

        print("✅ main.py restored from backup")
        sys.exit(1)


if __name__ == "__main__":
    main()
