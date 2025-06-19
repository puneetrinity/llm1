from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from typing import Set


class WebSocketDashboard:
    def __init__(
        self, enhanced_dashboard, metrics_collector=None, performance_monitor=None
    ):
        self.dashboard = enhanced_dashboard
        self.metrics_collector = metrics_collector
        self.performance_monitor = performance_monitor
        self.active_connections: Set[WebSocket] = set()
        self.broadcast_task = None

    async def connect(self, websocket: WebSocket):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections.add(websocket)

        try:
            dashboard_data = await self.dashboard.get_comprehensive_dashboard()
            await websocket.send_text(
                json.dumps({"type": "dashboard_update", "data": dashboard_data})
            )
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

                    message = json.dumps(
                        {
                            "type": "dashboard_update",
                            "data": dashboard_data,
                            "timestamp": dashboard_data.get("timestamp", ""),
                        }
                    )

                    disconnected = set()
                    for websocket in self.active_connections:
                        try:
                            await websocket.send_text(message)
                        except (WebSocketDisconnect, Exception) as e:
                            disconnected.add(websocket)

                    self.active_connections -= disconnected

                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in broadcast loop: {e}")
                await asyncio.sleep(30)
