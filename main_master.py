# main_master.py - Master FastAPI Application (Single Source of Truth)
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import sys
import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# Import our master configuration
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Response models
class HealthResponse(BaseModel):
    status: str
    healthy: bool
    timestamp: str
    version: str
    services: Dict[str, bool]

class StatusResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    features: Dict[str, bool]
    timestamp: str

# Global service state
services_state = {
    "ollama_connected": False,
    "dashboard_available": False,
    "initialization_complete": False
}

# WebSocket session storage (simple in-memory for development)
websocket_sessions = {}

async def initialize_services():
    """Initialize all services with proper error handling"""
    global services_state
    
    try:
        logging.info("🚀 Initializing services...")
        
        # Basic initialization - start simple
        services_state["initialization_complete"] = True
        
        # Check if React dashboard is available
        react_build_dir = Path(__file__).parent / "frontend" / "build"
        services_state["dashboard_available"] = (
            react_build_dir.exists() and 
            (react_build_dir / "index.html").exists()
        )
        
        # TODO: Add Ollama connection check here later
        # For now, assume not connected to start simple
        services_state["ollama_connected"] = False
        
        logging.info("✅ Basic services initialized successfully")
        
    except Exception as e:
        logging.error(f"❌ Service initialization failed: {e}")
        services_state["initialization_complete"] = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events with proper async handling"""
    # Startup
    logging.info("🌟 Starting Consolidated Enhanced LLM Proxy...")
    await initialize_services()
    
    # Log startup summary
    logging.info("=" * 60)
    logging.info(f"🎯 Server starting on {settings.HOST}:{settings.PORT}")
    logging.info(f"📊 Services: {services_state}")
    logging.info(f"🔧 Features enabled: Enhanced={settings.ENABLE_ENHANCED_FEATURES}, Dashboard={settings.ENABLE_DASHBOARD}")
    logging.info("=" * 60)
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down Consolidated LLM Proxy...")

# Create FastAPI app with proper configuration
app = FastAPI(
    title="Consolidated Enhanced LLM Proxy",
    description="Clean, consolidated FastAPI + React LLM proxy application",
    version="3.0.0-consolidated",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# FIXED: CORS middleware with proper React development support
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# FIXED: React Dashboard Integration - Proper Static File Serving
if settings.ENABLE_DASHBOARD:
    react_build_dir = Path(__file__).parent / "frontend" / "build"
    
    if react_build_dir.exists() and (react_build_dir / "index.html").exists():
        # Mount static assets (CSS, JS, images)
        static_dir = react_build_dir / "static"
        if static_dir.exists():
            app.mount("/app/static", StaticFiles(directory=static_dir), name="react_static")
        
        # Serve React SPA with proper routing
        @app.get("/app/{path:path}")
        async def serve_react_app(path: str = ""):
            """Serve React SPA with proper client-side routing support"""
            
            # Handle specific file requests
            if path and "." in path:
                file_path = react_build_dir / path
                if file_path.exists() and file_path.is_file():
                    return FileResponse(file_path)
            
            # For all other requests (including empty path), serve index.html
            # This enables React Router to handle client-side routing
            index_path = react_build_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            else:
                return JSONResponse(
                    status_code=404,
                    content={"error": "Dashboard index.html not found"}
                )
        
        logging.info("✅ React dashboard mounted at /app")
        services_state["dashboard_available"] = True
        
    else:
        # Dashboard not built - provide helpful instructions
        @app.get("/app")
        async def dashboard_not_built():
            return JSONResponse({
                "message": "React dashboard not built yet",
                "instructions": [
                    "1. cd frontend",
                    "2. npm install",
                    "3. npm run build",
                    "4. Restart this server"
                ],
                "build_directory": str(react_build_dir),
                "exists": react_build_dir.exists()
            })
        
        logging.warning(f"⚠️ React dashboard not built at {react_build_dir}")
        services_state["dashboard_available"] = False

# Authentication dependency (simplified for now)
async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Simple authentication check"""
    if not settings.ENABLE_AUTH:
        return {"user_id": "anonymous", "permissions": ["read", "write"]}
    
    api_key = request.headers.get(settings.API_KEY_HEADER)
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "API key required",
                "message": f"Please provide API key in {settings.API_KEY_HEADER} header"
            }
        )
    
    if api_key != settings.DEFAULT_API_KEY:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Invalid API key",
                "message": "The provided API key is not valid"
            }
        )
    
    return {"user_id": "authenticated", "permissions": ["read", "write"]}

# OPTION 3: Session-based WebSocket authentication
@app.post("/auth/websocket-session")
async def create_websocket_session(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a session token for WebSocket authentication"""
    try:
        # Create session using your auth service
        if 'auth_service' in globals() and auth_service:
            session_token = auth_service.create_session(
                user_id=current_user["user_id"],
                expires_in_hours=24
            )
            return {
                "session_token": session_token,
                "expires_in": 24 * 3600,  # 24 hours in seconds
                "websocket_url": f"ws://localhost:8000/ws/dashboard?session={session_token}"
            }
        else:
            # Fallback for basic setups
            session_token = secrets.token_urlsafe(32)
            # Store session in a simple in-memory dict (use Redis in production)
            websocket_sessions[session_token] = {
                "user_id": current_user["user_id"],
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(hours=24)
            }
            
            return {
                "session_token": session_token,
                "expires_in": 24 * 3600,
                "websocket_url": f"ws://{settings.HOST}:{settings.PORT}/ws/dashboard?session={session_token}"
            }
    except Exception as e:
        logging.error(f"Failed to create WebSocket session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket, session: str = Query(None)):
    """WebSocket endpoint with session-based authentication"""
    
    # Handle authentication if enabled
    if settings.ENABLE_AUTH:
        if not session:
            await websocket.close(code=1008, reason="Session token required")
            return
            
        # Validate session token
        try:
            user_info = None
            
            # Try auth service first
            if 'auth_service' in globals() and auth_service:
                session_info = auth_service.validate_session(session)
                if session_info:
                    user_info = session_info
            else:
                # Fallback to simple session store
                if session in websocket_sessions:
                    session_data = websocket_sessions[session]
                    if session_data["expires_at"] > datetime.now():
                        user_info = {
                            "user_id": session_data["user_id"],
                            "permissions": ["read", "write"]
                        }
                    else:
                        # Session expired
                        del websocket_sessions[session]
            
            if not user_info:
                await websocket.close(code=1008, reason="Invalid or expired session")
                return
                
        except Exception as e:
            logging.error(f"WebSocket session validation error: {e}")
            await websocket.close(code=1011, reason="Session validation failed")
            return
    
    # Rest of your WebSocket logic remains the same...
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "request_update":
                try:
                    basic_data = {
                        "timestamp": datetime.now().isoformat(),
                        "system_overview": {"status": "healthy", "total_requests": 0},
                        "services": services_state
                    }
                    await websocket.send_text(json.dumps({
                        "type": "dashboard_update", 
                        "data": basic_data
                    }))
                except Exception as e:
                    logging.error(f"Error sending dashboard update: {e}")
                    
    except WebSocketDisconnect:
        logging.info("WebSocket client disconnected")
    except Exception as e:
        logging.error(f"Dashboard WebSocket error: {e}")
        await websocket.close()

# Core API Endpoints
@app.get("/")
async def root():
    """Root endpoint with helpful information"""
    return {
        "message": "Consolidated Enhanced LLM Proxy",
        "version": "3.0.0-consolidated",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "status": "/api/status",
            "dashboard": "/app",
            "docs": "/docs",
            "api_docs": "/redoc",
            "websocket_auth": "/auth/websocket-session"
        },
        "services": services_state
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy" if services_state["initialization_complete"] else "initializing",
        healthy=services_state["initialization_complete"],
        timestamp=datetime.now().isoformat(),
        version="3.0.0-consolidated",
        services=services_state
    )

@app.get("/api/status", response_model=StatusResponse)
async def api_status():
    """Detailed status endpoint for React dashboard"""
    return StatusResponse(
        status="online" if services_state["initialization_complete"] else "starting",
        services={
            "ollama": services_state["ollama_connected"],
            "dashboard": services_state["dashboard_available"],
            "initialization": services_state["initialization_complete"]
        },
        features={
            "enhanced_features": settings.ENABLE_ENHANCED_FEATURES,
            "authentication": settings.ENABLE_AUTH,
            "dashboard": settings.ENABLE_DASHBOARD,
            "websocket": settings.ENABLE_WEBSOCKET
        },
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/config")
async def get_config(user: Dict[str, Any] = Depends(get_current_user)):
    """Get public configuration (requires auth if enabled)"""
    return {
        "server": {
            "version": "3.0.0-consolidated",
            "debug": settings.DEBUG,
            "host": settings.HOST,
            "port": settings.PORT
        },
        "features": {
            "enhanced_features": settings.ENABLE_ENHANCED_FEATURES,
            "dashboard": settings.ENABLE_DASHBOARD,
            "authentication": settings.ENABLE_AUTH,
            "websocket": settings.ENABLE_WEBSOCKET
        },
        "limits": {
            "max_memory_mb": settings.MAX_MEMORY_MB,
            "cache_memory_limit_mb": settings.CACHE_MEMORY_LIMIT_MB
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced error handling"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": exc.status_code,
                "detail": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler"""
    logging.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "status_code": 500,
                "detail": "Internal server error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )

# Main execution
if __name__ == "__main__":
    logging.info(f"🚀 Starting Consolidated LLM Proxy Server")
    logging.info(f"📍 Server will be available at: http://{settings.HOST}:{settings.PORT}")
    logging.info(f"📊 Dashboard will be available at: http://{settings.HOST}:{settings.PORT}/app")
    logging.info(f"🔌 WebSocket will be available at: ws://{settings.HOST}:{settings.PORT}/ws/dashboard")
    
    uvicorn.run(
        "main_master:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
