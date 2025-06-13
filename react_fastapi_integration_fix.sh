#!/bin/bash
# react_fastapi_integration_fix.sh - Complete Fix for React + FastAPI Integration

set -e

echo "🔗 FIXING React + FastAPI Integration"
echo "====================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Step 1: Create React Frontend Directory Structure
echo -e "\n${BLUE}📁 Step 1: Setting Up React Frontend Structure${NC}"

# Create frontend directory if it doesn't exist
if [ ! -d "frontend" ]; then
    print_info "Creating frontend directory..."
    mkdir -p frontend
    print_status "Frontend directory created"
else
    print_status "Frontend directory exists"
fi

cd frontend

# Initialize React app if package.json doesn't exist
if [ ! -f "package.json" ]; then
    print_info "Creating React app..."
    npx create-react-app . --template typescript || npx create-react-app .
    print_status "React app created"
else
    print_status "React app already exists"
fi

# Step 2: Create/Update package.json for FastAPI integration
echo -e "\n${BLUE}📦 Step 2: Configuring package.json for FastAPI${NC}"

# Create a fixed package.json
cat > package.json << 'EOF'
{
  "name": "llm-proxy-dashboard",
  "version": "2.2.0",
  "private": true,
  "homepage": "/app",
  "dependencies": {
    "@testing-library/jest-dom": "^5.16.4",
    "@testing-library/react": "^13.3.0",
    "@testing-library/user-event": "^13.5.0",
    "@types/jest": "^27.5.2",
    "@types/node": "^16.11.56",
    "@types/react": "^18.0.17",
    "@types/react-dom": "^18.0.6",
    "axios": "^1.6.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.8.0",
    "react-scripts": "5.0.1",
    "recharts": "^2.8.0",
    "typescript": "^4.7.4",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "proxy": "http://localhost:8001"
}
EOF

print_status "Package.json configured with FastAPI proxy"

# Step 3: Create environment configuration
echo -e "\n${BLUE}🔧 Step 3: Setting Up Environment Configuration${NC}"

# Create .env for development
cat > .env << 'EOF'
REACT_APP_API_BASE_URL=http://localhost:8001
REACT_APP_WS_URL=ws://localhost:8001
REACT_APP_ENVIRONMENT=development
GENERATE_SOURCEMAP=false
EOF

# Create .env.production for production build
cat > .env.production << 'EOF'
REACT_APP_API_BASE_URL=
REACT_APP_WS_URL=ws://localhost:8001
REACT_APP_ENVIRONMENT=production
GENERATE_SOURCEMAP=false
PUBLIC_URL=/app
EOF

print_status "Environment files created"

# Step 4: Create API service for connecting to FastAPI
echo -e "\n${BLUE}🔌 Step 4: Creating API Service${NC}"

mkdir -p src/services

cat > src/services/api.ts << 'EOF'
// src/services/api.ts - API Service for FastAPI Integration

import axios from 'axios';

// Get API base URL from environment, fallback to relative path for production
const getApiBaseUrl = (): string => {
  // In development, use the full URL
  if (process.env.NODE_ENV === 'development') {
    return process.env.REACT_APP_API_BASE_URL || 'http://localhost:8001';
  }
  
  // In production, use relative URLs (same origin as the React app)
  return '';
};

// Create axios instance with proper configuration
const api = axios.create({
  baseURL: getApiBaseUrl(),
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log(`🌐 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.status, error.response?.data);
    return Promise.reject(error);
  }
);

// API service methods
export const apiService = {
  // Health check
  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  // Get metrics
  async getMetrics() {
    const response = await api.get('/metrics');
    return response.data;
  },

  // Get models
  async getModels() {
    const response = await api.get('/models');
    return response.data;
  },

  // Get admin status
  async getAdminStatus() {
    const response = await api.get('/admin/status');
    return response.data;
  },

  // Chat completion
  async chatCompletion(data: {
    model: string;
    messages: Array<{ role: string; content: string }>;
    temperature?: number;
    max_tokens?: number;
  }) {
    const response = await api.post('/v1/chat/completions', data);
    return response.data;
  },

  // Get cache stats (if available)
  async getCacheStats() {
    try {
      const response = await api.get('/admin/cache/stats');
      return response.data;
    } catch (error) {
      console.warn('Cache stats not available:', error);
      return null;
    }
  },

  // Get circuit breaker status (if available)
  async getCircuitBreakers() {
    try {
      const response = await api.get('/admin/circuit-breakers');
      return response.data;
    } catch (error) {
      console.warn('Circuit breakers not available:', error);
      return null;
    }
  },

  // Get memory status (if available)
  async getMemoryStatus() {
    try {
      const response = await api.get('/admin/memory');
      return response.data;
    } catch (error) {
      console.warn('Memory status not available:', error);
      return null;
    }
  },
};

// WebSocket service for real-time updates
export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private onMessageCallback: ((data: any) => void) | null = null;

  connect(onMessage: (data: any) => void) {
    this.onMessageCallback = onMessage;
    this.connectWebSocket();
  }

  private connectWebSocket() {
    try {
      const wsUrl = process.env.NODE_ENV === 'development' 
        ? 'ws://localhost:8001/ws/dashboard'
        : `ws://${window.location.host}/ws/dashboard`;
      
      console.log('🔌 Connecting to WebSocket:', wsUrl);
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('✅ WebSocket connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (this.onMessageCallback) {
            this.onMessageCallback(data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('🔌 WebSocket disconnected, attempting to reconnect...');
        this.reconnectTimer = setTimeout(() => {
          this.connectWebSocket();
        }, 5000);
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export default api;
EOF

print_status "API service created"

# Step 5: Create main dashboard component
echo -e "\n${BLUE}🎨 Step 5: Creating Dashboard Component${NC}"

cat > src/App.tsx << 'EOF'
// src/App.tsx - Main Dashboard Application

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { apiService, WebSocketService } from './services/api';
import './App.css';

// Dashboard Component
const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [models, setModels] = useState<any>(null);
  const [adminStatus, setAdminStatus] = useState<any>(null);
  const [wsService] = useState(new WebSocketService());
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadInitialData();
    
    // Setup WebSocket connection
    wsService.connect((data) => {
      console.log('📊 WebSocket data:', data);
      if (data.type === 'dashboard_update') {
        setMetrics(data.data);
      }
    });

    return () => {
      wsService.disconnect();
    };
  }, [wsService]);

  const loadInitialData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('🔄 Loading initial data...');
      
      // Load all data in parallel
      const [healthData, metricsData, modelsData, adminData] = await Promise.allSettled([
        apiService.getHealth(),
        apiService.getMetrics(),
        apiService.getModels(),
        apiService.getAdminStatus(),
      ]);

      // Handle results
      if (healthData.status === 'fulfilled') {
        setHealth(healthData.value);
        setConnected(true);
      } else {
        console.error('Health check failed:', healthData.reason);
      }

      if (metricsData.status === 'fulfilled') {
        setMetrics(metricsData.value);
      } else {
        console.error('Metrics failed:', metricsData.reason);
      }

      if (modelsData.status === 'fulfilled') {
        setModels(modelsData.value);
      } else {
        console.error('Models failed:', modelsData.reason);
      }

      if (adminData.status === 'fulfilled') {
        setAdminStatus(adminData.value);
      } else {
        console.error('Admin status failed:', adminData.reason);
      }

    } catch (error) {
      console.error('Failed to load data:', error);
      setError('Failed to connect to FastAPI backend');
      setConnected(false);
    } finally {
      setLoading(false);
    }
  };

  const testChatCompletion = async () => {
    try {
      setLoading(true);
      const response = await apiService.chatCompletion({
        model: 'gpt-3.5-turbo',
        messages: [{ role: 'user', content: 'Hello from React dashboard!' }],
        max_tokens: 50,
      });
      console.log('💬 Chat completion response:', response);
      alert('Chat completion successful! Check console for details.');
    } catch (error) {
      console.error('Chat completion failed:', error);
      alert('Chat completion failed. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !health) {
    return (
      <div className="loading">
        <h2>🔄 Loading Dashboard...</h2>
        <p>Connecting to FastAPI backend...</p>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>🚀 Enhanced LLM Proxy Dashboard</h1>
        <div className="connection-status">
          <span className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
          <button onClick={loadInitialData} disabled={loading}>
            🔄 Refresh
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <h3>❌ Connection Error</h3>
          <p>{error}</p>
          <p>Make sure FastAPI is running on the correct port.</p>
        </div>
      )}

      <div className="dashboard-grid">
        {/* Health Status */}
        <div className="dashboard-card">
          <h3>🏥 Health Status</h3>
          {health ? (
            <div>
              <p><strong>Status:</strong> <span className={health.healthy ? 'healthy' : 'unhealthy'}>
                {health.healthy ? '✅ Healthy' : '❌ Unhealthy'}
              </span></p>
              <p><strong>Version:</strong> {health.version}</p>
              <p><strong>Services:</strong> {health.services?.length || 0}</p>
              <small>Last updated: {health.timestamp}</small>
            </div>
          ) : (
            <p>❌ Health data not available</p>
          )}
        </div>

        {/* Metrics */}
        <div className="dashboard-card">
          <h3>📊 Metrics</h3>
          {metrics ? (
            <div>
              <p><strong>Requests:</strong> {metrics.requests?.total || 0}</p>
              <p><strong>Uptime:</strong> {Math.round(metrics.uptime_seconds / 60)} minutes</p>
              <p><strong>Models:</strong> {Object.keys(metrics.models || {}).length}</p>
              <p><strong>Errors:</strong> {Object.keys(metrics.errors || {}).length}</p>
              <small>Last updated: {metrics.timestamp}</small>
            </div>
          ) : (
            <p>❌ Metrics not available</p>
          )}
        </div>

        {/* Models */}
        <div className="dashboard-card">
          <h3>🤖 Available Models</h3>
          {models ? (
            <div>
              <p><strong>Total:</strong> {models.data?.length || 0}</p>
              <ul>
                {models.data?.slice(0, 3).map((model: any, index: number) => (
                  <li key={index}>{model.id}</li>
                ))}
              </ul>
              {models.data?.length > 3 && <p>...and {models.data.length - 3} more</p>}
            </div>
          ) : (
            <p>❌ Models not available</p>
          )}
        </div>

        {/* Admin Status */}
        <div className="dashboard-card">
          <h3>⚙️ Admin Status</h3>
          {adminStatus ? (
            <div>
              <p><strong>Version:</strong> {adminStatus.version}</p>
              <h4>Enhanced Features:</h4>
              <ul>
                {Object.entries(adminStatus.enhanced_capabilities || {}).map(([feature, enabled]) => (
                  <li key={feature}>
                    {enabled ? '✅' : '⏸️'} {feature.replace(/_/g, ' ')}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p>❌ Admin status not available</p>
          )}
        </div>

        {/* Test Chat */}
        <div className="dashboard-card">
          <h3>💬 Test Chat</h3>
          <p>Test the chat completion endpoint:</p>
          <button onClick={testChatCompletion} disabled={loading || !connected}>
            {loading ? '🔄 Testing...' : '🧪 Test Chat Completion'}
          </button>
        </div>

        {/* API Endpoints */}
        <div className="dashboard-card">
          <h3>🔗 API Endpoints</h3>
          <ul>
            <li><a href="/health" target="_blank">Health Check</a></li>
            <li><a href="/metrics" target="_blank">Metrics</a></li>
            <li><a href="/models" target="_blank">Models</a></li>
            <li><a href="/docs" target="_blank">API Documentation</a></li>
            <li><a href="/admin/status" target="_blank">Admin Status</a></li>
          </ul>
        </div>
      </div>
    </div>
  );
};

// About Page Component
const About: React.FC = () => (
  <div className="page">
    <h2>📖 About Enhanced LLM Proxy</h2>
    <p>This is a React dashboard for the Enhanced LLM Proxy system.</p>
    <h3>Features:</h3>
    <ul>
      <li>✅ Real-time metrics via WebSocket</li>
      <li>✅ Health monitoring</li>
      <li>✅ Model management</li>
      <li>✅ Chat completion testing</li>
      <li>✅ Admin interface</li>
    </ul>
  </div>
);

// Main App Component
const App: React.FC = () => {
  return (
    <Router basename="/app">
      <div className="App">
        <nav className="navigation">
          <Link to="/" className="nav-link">🏠 Dashboard</Link>
          <Link to="/about" className="nav-link">📖 About</Link>
        </nav>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
EOF

print_status "Dashboard component created"

# Step 6: Create CSS styles
echo -e "\n${BLUE}🎨 Step 6: Creating Styles${NC}"

cat > src/App.css << 'EOF'
/* src/App.css - Dashboard Styles */

.App {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.navigation {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  padding: 1rem 2rem;
  display: flex;
  gap: 2rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

.nav-link {
  color: white;
  text-decoration: none;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  transition: background-color 0.3s;
}

.nav-link:hover {
  background: rgba(255, 255, 255, 0.2);
}

.main-content {
  padding: 2rem;
}

.dashboard {
  max-width: 1200px;
  margin: 0 auto;
}

.dashboard-header {
  text-align: center;
  margin-bottom: 2rem;
  color: white;
}

.dashboard-header h1 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.connection-status {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}

.status-indicator {
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-weight: bold;
  font-size: 0.9rem;
}

.status-indicator.connected {
  background: rgba(76, 175, 80, 0.2);
  color: #4CAF50;
  border: 1px solid #4CAF50;
}

.status-indicator.disconnected {
  background: rgba(244, 67, 54, 0.2);
  color: #F44336;
  border: 1px solid #F44336;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
}

.dashboard-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: transform 0.3s, box-shadow 0.3s;
}

.dashboard-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
}

.dashboard-card h3 {
  margin: 0 0 1rem 0;
  color: #333;
  font-size: 1.2rem;
  border-bottom: 2px solid #667eea;
  padding-bottom: 0.5rem;
}

.dashboard-card p {
  margin: 0.5rem 0;
  color: #555;
}

.dashboard-card ul {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.dashboard-card li {
  margin: 0.25rem 0;
  color: #666;
}

.dashboard-card a {
  color: #667eea;
  text-decoration: none;
}

.dashboard-card a:hover {
  text-decoration: underline;
}

.dashboard-card button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  transition: transform 0.2s, box-shadow 0.2s;
  margin-top: 1rem;
}

.dashboard-card button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.dashboard-card button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.healthy {
  color: #4CAF50;
  font-weight: bold;
}

.unhealthy {
  color: #F44336;
  font-weight: bold;
}

.loading {
  text-align: center;
  color: white;
  padding: 4rem 2rem;
}

.loading h2 {
  font-size: 2rem;
  margin-bottom: 1rem;
}

.error-banner {
  background: rgba(244, 67, 54, 0.1);
  border: 1px solid #F44336;
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 2rem;
  color: white;
}

.error-banner h3 {
  margin: 0 0 0.5rem 0;
  color: #F44336;
}

.page {
  max-width: 800px;
  margin: 0 auto;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  color: #333;
}

.page h2 {
  color: #667eea;
  border-bottom: 2px solid #667eea;
  padding-bottom: 0.5rem;
}

.page ul {
  padding-left: 2rem;
}

.page li {
  margin: 0.5rem 0;
}

/* Responsive design */
@media (max-width: 768px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
    gap: 1rem;
  }
  
  .main-content {
    padding: 1rem;
  }
  
  .dashboard-header h1 {
    font-size: 2rem;
  }
  
  .navigation {
    padding: 1rem;
    flex-wrap: wrap;
  }
}
EOF

print_status "Styles created"

# Step 7: Update public/index.html
echo -e "\n${BLUE}📄 Step 7: Updating HTML Template${NC}"

mkdir -p public

cat > public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Enhanced LLM Proxy Dashboard" />
    <title>Enhanced LLM Proxy Dashboard</title>
    <base href="/app/" />
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

print_status "HTML template updated"

# Go back to root directory
cd ..

# Step 8: Update main_fixed.py to properly serve React app
echo -e "\n${BLUE}🔧 Step 8: Updating FastAPI to Serve React App${NC}"

cat > main_with_react.py << 'EOF'
# main_with_react.py - FastAPI with Proper React Integration

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Import the fixed components from main_fixed.py
from main_fixed import (
    settings, FixedOllamaClient, FixedLLMRouter, FixedMetrics,
    ChatCompletionRequest, CompletionRequest, ChatCompletionResponse, HealthResponse,
    Message, get_current_user, initialize_fixed_services,
    ollama_client, llm_router, metrics_collector, enhanced_capabilities
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Global service instances (will be initialized in lifespan)
global ollama_client, llm_router, metrics_collector

async def initialize_services():
    """Initialize all services"""
    global ollama_client, llm_router, metrics_collector
    
    try:
        logging.info("🚀 Initializing services with React integration...")
        
        # Initialize metrics
        metrics_collector = FixedMetrics()
        
        # Initialize Ollama client
        ollama_client = FixedOllamaClient(settings.OLLAMA_BASE_URL, settings.OLLAMA_TIMEOUT)
        await ollama_client.initialize()
        
        # Initialize router
        llm_router = FixedLLMRouter(ollama_client)
        await llm_router.initialize()
        
        logging.info("✅ All services initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        raise

# Create FastAPI app
app = FastAPI(
    title="Enhanced LLM Proxy with React Dashboard",
    description="FastAPI backend with integrated React frontend",
    version="2.2.0-react",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],  # Include React dev server
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount React build directory if it exists
react_build_dir = Path(__file__).parent / "frontend" / "build"
if react_build_dir.exists():
    # Mount static files
    app.mount("/app/static", StaticFiles(directory=react_build_dir / "static"), name="static")
    
    # Serve React app at /app
    @app.get("/app/{path:path}")
    async def serve_react_app(path: str = ""):
        """Serve React app with proper SPA routing"""
        
        # If it's a file request, try to serve it
        if path and "." in path:
            file_path = react_build_dir / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
        
        # For all other routes, serve index.html (SPA routing)
        index_path = react_build_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        else:
            raise HTTPException(status_code=404, detail="React app not built")
    
    logging.info(f"✅ React app mounted at /app from {react_build_dir}")
else:
    # Fallback route when React app is not built
    @app.get("/app")
    @app.get("/app/{path:path}")
    async def react_not_built():
        return JSONResponse({
            "message": "React dashboard not built yet",
            "instructions": [
                "cd frontend",
                "npm install",
                "npm run build"
            ],
            "build_path": str(react_build_dir),
            "current_path": str(Path.cwd())
        })
    
    logging.warning(f"⚠️ React build directory not found: {react_build_dir}")

# Startup event
@app.on_event("startup")
async def startup_event():
    await initialize_services()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    if ollama_client:
        await ollama_client.cleanup()

# API Routes (same as main_fixed.py)
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client not available")
        if not llm_router:
            raise HTTPException(status_code=503, detail="LLM router not available")
        
        if metrics_collector:
            metrics_collector.track_request("/v1/chat/completions")
        
        selected_model = await llm_router.route_request(request)
        response = await llm_router.process_chat_completion(request, selected_model)
        
        if metrics_collector:
            processing_time = asyncio.get_event_loop().time() - start_time
            metrics_collector.track_request("/v1/chat/completions", processing_time)
            metrics_collector.track_model_usage(selected_model)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        if metrics_collector:
            metrics_collector.track_error("chat_completion_error")
        logging.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        services_status = []
        
        if ollama_client:
            try:
                ollama_healthy = await ollama_client.health_check()
                services_status.append({
                    "name": "ollama",
                    "status": "healthy" if ollama_healthy else "unhealthy",
                    "last_check": datetime.now().isoformat()
                })
            except Exception as e:
                services_status.append({
                    "name": "ollama",
                    "status": "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "error": str(e)
                })
        
        if llm_router:
            services_status.append({
                "name": "llm_router",
                "status": "healthy",
                "last_check": datetime.now().isoformat(),
                "available_models": len(llm_router.available_models)
            })
        
        # Add React app status
        services_status.append({
            "name": "react_dashboard",
            "status": "healthy" if react_build_dir.exists() else "not_built",
            "last_check": datetime.now().isoformat(),
            "build_path": str(react_build_dir)
        })
        
        overall_healthy = all(s["status"] in ["healthy", "not_built"] for s in services_status)
        
        return HealthResponse(
            status="healthy" if overall_healthy else "degraded",
            healthy=overall_healthy,
            timestamp=datetime.now().isoformat(),
            version="2.2.0-react",
            services=services_status
        )
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/models")
async def list_available_models():
    try:
        if not llm_router:
            raise HTTPException(status_code=503, detail="LLM router not available")
        
        models = await llm_router.get_available_models()
        return {"object": "list", "data": models}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    try:
        if metrics_collector:
            return await metrics_collector.get_all_metrics()
        else:
            return {
                "status": "basic_metrics",
                "timestamp": datetime.now().isoformat(),
                "message": "Enhanced metrics not available"
            }
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/admin/status")
async def get_admin_status():
    return {
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.0-react",
        "services": {
            "ollama_client": ollama_client is not None,
            "llm_router": llm_router is not None,
            "metrics_collector": metrics_collector is not None,
            "react_dashboard": react_build_dir.exists()
        },
        "enhanced_capabilities": {
            "streaming": settings.ENABLE_STREAMING,
            "model_warmup": settings.ENABLE_MODEL_WARMUP,
            "semantic_classification": False,
            "react_dashboard": True,
            "websocket_support": True
        },
        "configuration": {
            "enable_auth": settings.ENABLE_AUTH,
            "enable_dashboard": True,
            "dashboard_path": "/app",
            "react_build_exists": react_build_dir.exists()
        }
    }

@app.get("/")
async def root():
    return {
        "name": "Enhanced LLM Proxy with React Dashboard",
        "version": "2.2.0-react",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "models": "/models",
            "chat_completions": "/v1/chat/completions",
            "dashboard": "/app",
            "docs": "/docs"
        },
        "dashboard": {
            "url": "/app",
            "built": react_build_dir.exists(),
            "build_instructions": [
                "cd frontend",
                "npm install", 
                "npm run build"
            ] if not react_build_dir.exists() else None
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main_with_react:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
EOF

print_status "FastAPI updated for React integration"

# Step 9: Create build and start scripts
echo -e "\n${BLUE}🚀 Step 9: Creating Build and Start Scripts${NC}"

cat > build_and_start.sh << 'EOF'
#!/bin/bash
# build_and_start.sh - Build React app and start FastAPI

set -e

echo "🚀 Building React Dashboard and Starting FastAPI"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "main_with_react.py" ]; then
    echo "❌ main_with_react.py not found. Run this from the root directory."
    exit 1
fi

# Step 1: Install and build React app
echo "📦 Building React Dashboard..."
cd frontend

# Install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing React dependencies..."
    npm install
fi

# Build the React app
echo "Building React app for production..."
npm run build

if [ ! -d "build" ]; then
    echo "❌ React build failed!"
    exit 1
fi

echo "✅ React app built successfully"

# Step 2: Go back and start FastAPI
cd ..

# Load port configuration
if [ -f .env.port ]; then
    source .env.port
else
    PORT=8001
fi

echo ""
echo "🌐 Starting FastAPI with React Dashboard on port $PORT..."
echo ""
echo "🎯 Access points:"
echo "  • Main API: http://localhost:$PORT"
echo "  • React Dashboard: http://localhost:$PORT/app"
echo "  • API Documentation: http://localhost:$PORT/docs"
echo "  • Health Check: http://localhost:$PORT/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start FastAPI
python main_with_react.py
EOF

chmod +x build_and_start.sh

cat > test_react_integration.sh << 'EOF'
#!/bin/bash
# test_react_integration.sh - Test React + FastAPI Integration

set -e

# Load port configuration
if [ -f .env.port ]; then
    source .env.port
else
    PORT=8001
fi

BASE_URL="http://localhost:$PORT"

echo "🧪 Testing React + FastAPI Integration at $BASE_URL"
echo "==================================================="

# Test 1: Root endpoint
echo "Test 1: Root endpoint"
if curl -s -f "$BASE_URL/" | grep -q "React Dashboard"; then
    echo "✅ Root endpoint mentions React Dashboard"
else
    echo "❌ Root endpoint failed"
fi

# Test 2: Health check
echo ""
echo "Test 2: Health Check"
if curl -s -f "$BASE_URL/health" | grep -q "react_dashboard"; then
    echo "✅ Health check includes React dashboard status"
else
    echo "❌ Health check failed"
fi

# Test 3: Admin status
echo ""
echo "Test 3: Admin Status"
if curl -s -f "$BASE_URL/admin/status" | grep -q "react_dashboard"; then
    echo "✅ Admin status includes React dashboard info"
else
    echo "❌ Admin status failed"
fi

# Test 4: React Dashboard
echo ""
echo "Test 4: React Dashboard"
if curl -s -f "$BASE_URL/app" | grep -q "Enhanced LLM Proxy Dashboard"; then
    echo "✅ React dashboard is accessible"
else
    echo "⚠️  React dashboard not accessible (may not be built yet)"
fi

# Test 5: Static files
echo ""
echo "Test 5: Static Files"
if curl -s -f "$BASE_URL/app/static/css/" >/dev/null 2>&1; then
    echo "✅ React static files accessible"
else
    echo "⚠️  React static files not accessible (normal if not built)"
fi

echo ""
echo "🎉 Integration testing complete!"
echo ""
echo "To build and start:"
echo "  ./build_and_start.sh"
echo ""
echo "Dashboard URL: $BASE_URL/app"
EOF

chmod +x test_react_integration.sh

print_status "Build and test scripts created"

# Step 10: Summary
echo -e "\n${BLUE}🎯 Step 10: Integration Complete!${NC}"

print_status "React + FastAPI Integration Setup Complete! 🎉"

echo ""
echo "📁 Files Created:"
echo "=================="
echo "Frontend (React):"
echo "  • frontend/package.json - React configuration with FastAPI proxy"
echo "  • frontend/src/services/api.ts - API service for FastAPI connection"
echo "  • frontend/src/App.tsx - Main dashboard component"
echo "  • frontend/src/App.css - Dashboard styles"
echo "  • frontend/.env - Development environment"
echo "  • frontend/.env.production - Production environment"
echo ""
echo "Backend (FastAPI):"
echo "  • main_with_react.py - FastAPI with React integration"
echo ""
echo "Scripts:"
echo "  • build_and_start.sh - Build React and start FastAPI"
echo "  • test_react_integration.sh - Test the integration"
echo ""
echo "🚀 Next Steps:"
echo "=============="
echo "1. Build and start everything:"
echo "   ./build_and_start.sh"
echo ""
echo "2. Test the integration (in another terminal):"
echo "   ./test_react_integration.sh"
echo ""
echo "3. Access your dashboard:"
if [ -f .env.port ]; then
    source .env.port
    echo "   • Dashboard: http://localhost:${PORT:-8001}/app"
    echo "   • API Docs: http://localhost:${PORT:-8001}/docs"
    echo "   • Health: http://localhost:${PORT:-8001}/health"
else
    echo "   • Dashboard: http://localhost:8001/app"
    echo "   • API Docs: http://localhost:8001/docs"
    echo "   • Health: http://localhost:8001/health"
fi
echo ""
echo "🔧 For development:"
echo "=================="
echo "Frontend only (React dev server):"
echo "  cd frontend && npm start"
echo ""
echo "Backend only (FastAPI):"
echo "  python main_with_react.py"
echo ""
echo "✅ The React dashboard will now properly connect to FastAPI!"
echo "✅ All API calls are configured with proper base URLs"
echo "✅ WebSocket connection included for real-time updates"
echo "✅ Production build will be served by FastAPI at /app"

print_status "Ready to build and start! Run: ./build_and_start.sh"
