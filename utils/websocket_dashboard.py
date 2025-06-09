# WebSocket dashboard for real-time updates
# utils/websocket_dashboard.py - Real-time WebSocket Dashboard
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from typing import Set

class WebSocketDashboard:
    def __init__(self, enhanced_dashboard):
        self.dashboard = enhanced_dashboard
        self.active_connections: Set[WebSocket] = set()
        self.broadcast_task = None
        
    async def connect(self, websocket: WebSocket):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Send initial dashboard data
        try:
            dashboard_data = await self.dashboard.get_comprehensive_dashboard()
            await websocket.send_text(json.dumps({
                "type": "dashboard_update",
                "data": dashboard_data
            }))
        except Exception as e:
            print(f"Error sending initial data: {e}")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        self.active_connections.discard(websocket)
    
    async def start_broadcasting(self):
        """Start broadcasting dashboard updates"""
        self.broadcast_task = asyncio.create_task(self._broadcast_loop())
    
    async def stop_broadcasting(self):
        """Stop broadcasting dashboard updates"""
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
    
    async def _broadcast_loop(self):
        """Broadcast dashboard updates every 10 seconds"""
        
        while True:
            try:
                if self.active_connections:
                    dashboard_data = await self.dashboard.get_comprehensive_dashboard()
                    
                    message = json.dumps({
                        "type": "dashboard_update",
                        "data": dashboard_data,
                        "timestamp": dashboard_data["timestamp"]
                    })
                    
                    # Send to all connected clients
                    disconnected = set()
                    for websocket in self.active_connections:
                        try:
                            await websocket.send_text(message)
                        except WebSocketDisconnect:
                            disconnected.add(websocket)
                        except Exception as e:
                            print(f"Error sending to websocket: {e}")
                            disconnected.add(websocket)
                    
                    # Remove disconnected clients
                    self.active_connections -= disconnected
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in broadcast loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error

# Add WebSocket endpoint to main app
# Enhanced WebSocket endpoints for main.py
@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    
    dashboard = EnhancedDashboard(
        metrics, 
        performance_monitor,  # This would need to be added to main.py
        cache_service, 
        warmup_service, 
        llm_router.semantic_classifier
    )
    
    ws_dashboard = WebSocketDashboard(dashboard)
    
    try:
        await ws_dashboard.connect(websocket)
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages (could be commands)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "get_dashboard":
                    # Send immediate dashboard update
                    dashboard_data = await dashboard.get_comprehensive_dashboard()
                    await websocket.send_text(json.dumps({
                        "type": "dashboard_update",
                        "data": dashboard_data
                    }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                break
                
    finally:
        ws_dashboard.disconnect(websocket)