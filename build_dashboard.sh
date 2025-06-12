#!/bin/bash
# build_dashboard.sh - React Dashboard Build Script

set -e

echo "🔨 Building React Dashboard for LLM Proxy..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "📁 Creating basic frontend structure..."
    mkdir -p frontend/src frontend/public
    
    # Create basic package.json
    cat > frontend/package.json << 'EOF'
{
  "name": "llm-proxy-dashboard",
  "version": "2.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "recharts": "^2.8.0",
    "lucide-react": "^0.263.1",
    "axios": "^1.5.0"
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
  }
}
EOF

    # Create basic public/index.html
    cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="LLM Proxy Dashboard" />
    <title>LLM Proxy Dashboard</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

    # Create basic src/App.js
    cat > frontend/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const healthRes = await fetch('/health');
        const healthData = await healthRes.json();
        setHealth(healthData);

        const metricsRes = await fetch('/metrics');
        const metricsData = await metricsRes.json();
        setMetrics(metricsData);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>🚀 LLM Proxy Dashboard</h1>
          <p>Loading...</p>
        </header>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🚀 LLM Proxy Dashboard</h1>
        
        <div className="dashboard-grid">
          <div className="card">
            <h2>🏥 Health Status</h2>
            <div className={`status ${health?.healthy ? 'healthy' : 'unhealthy'}`}>
              {health?.healthy ? '✅ Healthy' : '❌ Unhealthy'}
            </div>
            <p>Version: {health?.version || 'Unknown'}</p>
            <p>Uptime: {health?.uptime_seconds ? Math.round(health.uptime_seconds / 60) + ' minutes' : 'Unknown'}</p>
          </div>

          <div className="card">
            <h2>📊 Quick Metrics</h2>
            {metrics?.overview ? (
              <div>
                <p>Total Requests: {metrics.overview.total_requests || 0}</p>
                <p>Error Rate: {(metrics.overview.error_rate || 0).toFixed(1)}%</p>
                <p>Avg Response Time: {(metrics.overview.avg_response_time || 0).toFixed(2)}s</p>
                <p>Cache Hit Rate: {((metrics.overview.cache_hit_rate || 0) * 100).toFixed(1)}%</p>
              </div>
            ) : (
              <p>No metrics available</p>
            )}
          </div>

          <div className="card">
            <h2>🔗 Quick Links</h2>
            <div className="links">
              <a href="/docs" target="_blank" rel="noopener noreferrer">📚 API Documentation</a>
              <a href="/health" target="_blank" rel="noopener noreferrer">🏥 Health Check</a>
              <a href="/metrics" target="_blank" rel="noopener noreferrer">📈 Raw Metrics</a>
              <a href="/models" target="_blank" rel="noopener noreferrer">🤖 Available Models</a>
            </div>
          </div>

          <div className="card">
            <h2>ℹ️ System Info</h2>
            <p>Enhanced LLM Proxy v2.0</p>
            <p>Features: Streaming, Caching, Monitoring</p>
            <p>Dashboard: React-based Real-time UI</p>
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
EOF

    # Create basic src/App.css
    cat > frontend/src/App.css << 'EOF'
.App {
  text-align: center;
  background-color: #0a0e1a;
  color: white;
  min-height: 100vh;
}

.App-header {
  padding: 20px;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 20px;
  max-width: 1200px;
  width: 100%;
}

.card {
  background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 100%);
  border-radius: 12px;
  padding: 24px;
  border: 1px solid #3a4a6b;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
}

.card h2 {
  margin-top: 0;
  color: #64b5f6;
  font-size: 1.2em;
}

.status {
  font-size: 1.1em;
  font-weight: bold;
  padding: 8px 16px;
  border-radius: 20px;
  margin: 10px 0;
  display: inline-block;
}

.status.healthy {
  background-color: #1b5e20;
  color: #4caf50;
}

.status.unhealthy {
  background-color: #b71c1c;
  color: #f44336;
}

.links {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.links a {
  color: #64b5f6;
  text-decoration: none;
  padding: 8px 12px;
  border: 1px solid #64b5f6;
  border-radius: 6px;
  transition: all 0.2s ease;
}

.links a:hover {
  background-color: #64b5f6;
  color: #0a0e1a;
}

h1 {
  background: linear-gradient(45deg, #64b5f6, #42a5f5, #2196f3);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-size: 2.5em;
  margin-bottom: 10px;
}

p {
  margin: 8px 0;
  line-height: 1.5;
}
EOF

    # Create basic src/index.js
    cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

    # Create basic src/index.css
    cat > frontend/src/index.css << 'EOF'
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: #0a0e1a;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}

* {
  box-sizing: border-box;
}
EOF

    print_success "Created basic frontend structure"
fi

# Change to frontend directory
cd frontend

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Dashboard build failed."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Dashboard build failed."
    exit 1
fi

print_success "Node.js and npm are available"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install --silent || {
        print_error "Failed to install dependencies"
        exit 1
    }
    print_success "Dependencies installed"
else
    print_success "Dependencies already installed"
fi

# Build the React app
echo "🔨 Building React application..."
npm run build --silent || {
    print_error "React build failed"
    exit 1
}

print_success "React build completed"

# Create static directory and copy build files
cd ..
mkdir -p static

if [ -d "frontend/build" ]; then
    echo "📁 Copying build files to static directory..."
    cp -r frontend/build/* static/
    print_success "Build files copied to /app/static/"
else
    print_error "Build directory not found"
    exit 1
fi

# Verify the build
if [ -f "static/index.html" ]; then
    print_success "Dashboard build completed successfully!"
    echo "📊 Dashboard will be available at: http://localhost:8000/"
else
    print_error "Dashboard build verification failed"
    exit 1
fi

# Create a simple health check for the dashboard
cat > static/health.json << 'EOF'
{
  "dashboard": "ready",
  "version": "2.0.0",
  "build_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo ""
echo "🎉 Dashboard Build Summary:"
echo "=========================="
echo "✅ React app built successfully"
echo "✅ Static files copied to /app/static/"
echo "✅ Dashboard health check created"
echo "📊 Access at: http://localhost:8000/"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
