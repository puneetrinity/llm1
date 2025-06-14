// frontend/src/App.js - Improved App with modular structure and all your features

import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer
} from 'recharts';

// Import our modular components and utilities
import { CONFIG, API_BASE } from './utils/config.js';
import { useAuth } from './hooks/useAuth.js';
import { useWebSocket } from './hooks/useWebSocket.js';
import { ErrorBoundary } from './components/ErrorBoundary.js';
import './App.css';

// Component: Metric Card
function MetricCard({ title, value, icon, color }) {
  return (
    <div className={`metric-card ${color}`}>
      <div className="metric-icon">{icon}</div>
      <div className="metric-content">
        <div className="metric-title">{title}</div>
        <div className="metric-value">{value}</div>
      </div>
    </div>
  );
}

// Component: Header
function Header({ isConnected, health, auth }) {
  const healthStatus = health?.healthy ? 'healthy' : 'unhealthy';
  const connectionStatus = isConnected ? 'connected' : 'disconnected';

  return (
    <header className="header">
      <div className="header-left">
        <h1>🚀 LLM Proxy Dashboard</h1>
        <span className="version">v2.2.0</span>
      </div>
      <div className="header-right">
        <div className={`status-indicator ${connectionStatus}`}>
          <span className="dot"></span>
          WebSocket: {connectionStatus}
        </div>
        <div className={`status-indicator ${healthStatus}`}>
          <span className="dot"></span>
          Health: {healthStatus}
        </div>
        <div className="auth-status">
          {auth.isAuthenticated ? (
            <>
              <span>🔑 {auth.apiKey.substring(0, 8)}...</span>
              <button onClick={auth.logout} className="logout-btn">
                Logout
              </button>
            </>
          ) : (
            <span>🔒 Not authenticated</span>
          )}
        </div>
      </div>
    </header>
  );
}

// Component: Authentication Section
function AuthSection({ auth }) {
  const [inputKey, setInputKey] = useState(auth.apiKey || '');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (inputKey.trim()) {
      await auth.authenticate(inputKey.trim());
    }
  };

  return (
    <div className="auth-required">
      <div className="auth-required-content">
        <h2>🔐 Authentication Required</h2>
        <p>Please enter your API key to access the dashboard.</p>
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-row">
            <input
              type="password"
              value={inputKey}
              onChange={(e) => setInputKey(e.target.value)}
              placeholder="Enter API Key (sk-...)"
              className="api-key-input"
              disabled={auth.isAuthenticating}
            />
            <button 
              type="submit" 
              disabled={auth.isAuthenticating || !inputKey.trim()}
              className="auth-submit-btn"
            >
              {auth.isAuthenticating ? '🔄 Connecting...' : '🔑 Connect'}
            </button>
          </div>
          
          {auth.authError && (
            <div className="auth-error">
              ❌ {auth.authError}
            </div>
          )}
          
          <div className="auth-help">
            💡 Default dev key: <code>sk-default</code>
            {CONFIG.debug && (
              <div style={{ marginTop: '0.5rem', fontSize: '0.8em', color: '#666' }}>
                Debug: Connecting to {CONFIG.apiBase}
              </div>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

// Component: Chat Interface
function ChatInterface({ auth }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || !auth.isAuthenticated || isLoading) return;
    
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': auth.apiKey,
        },
        body: JSON.stringify({
          messages: [...messages, userMessage],
          model: 'gpt-3.5-turbo',
          temperature: 0.7
        })
      });

      if (response.ok) {
        const data = await response.json();
        const assistantMessage = {
          role: 'assistant',
          content: data.choices[0].message.content
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Error: Failed to get response from LLM'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="chat-interface">
      <div className="chat-controls">
        <button onClick={clearChat} className="clear-btn">
          🗑️ Clear Chat
        </button>
      </div>
      
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-role">
              {msg.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message assistant loading">
            <div className="message-role">🤖</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
          disabled={!auth.isAuthenticated || isLoading}
        />
        <button 
          onClick={sendMessage} 
          disabled={!auth.isAuthenticated || isLoading || !input.trim()}
        >
          {isLoading ? '⏳' : '➤'}
        </button>
      </div>
    </div>
  );
}

// Component: Dashboard
function Dashboard({ metrics, health }) {
  if (!metrics) {
    return (
      <div className="dashboard">
        <div className="loading">📊 Loading dashboard data...</div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="metrics-grid">
        <MetricCard 
          title="Total Requests" 
          value={metrics.total_requests || 0} 
          icon="📊" 
          color="blue" 
        />
        <MetricCard 
          title="Active Connections" 
          value={metrics.active_connections || 0} 
          icon="🔗" 
          color="green" 
        />
        <MetricCard 
          title="Error Rate" 
          value={`${((metrics.errors || 0) / Math.max(metrics.total_requests || 1, 1) * 100).toFixed(1)}%`} 
          icon="⚠️" 
          color="red" 
        />
        <MetricCard 
          title="Avg Response Time" 
          value={`${metrics.avg_response_time || 0}ms`} 
          icon="⚡" 
          color="purple" 
        />
      </div>
      
      <div className="charts-grid">
        <div className="chart-container">
          <h3>System Health</h3>
          <div className="health-overview">
            <div className="health-item">
              <span className="service-name">API Server</span>
              <span className={`status-badge ${health?.healthy ? 'healthy' : 'unhealthy'}`}>
                {health?.healthy ? 'Healthy' : 'Unhealthy'}
              </span>
            </div>
            {health?.services && Object.entries(health.services).map(([name, status]) => (
              <div key={name} className="health-item">
                <span className="service-name">{name}</span>
                <span className={`status-badge ${status ? 'healthy' : 'unhealthy'}`}>
                  {status ? 'Healthy' : 'Unhealthy'}
                </span>
              </div>
            ))}
          </div>
        </div>
        
        {/* WebSocket Status */}
        <div className="chart-container">
          <h3>Connection Status</h3>
          <div className="connection-overview">
            <div className="connection-item">
              <span>WebSocket Connection</span>
              <span className={`connection-status ${metrics.websocket_connected ? 'connected' : 'disconnected'}`}>
                {metrics.websocket_connected ? '🟢 Connected' : '🔴 Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Main App Component
function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  
  // Use our custom hooks
  const auth = useAuth();
  
  // WebSocket with message handler
  const ws = useWebSocket(auth, (message) => {
    if (CONFIG.debug) {
      console.log('📨 WebSocket message:', message);
    }
    
    if (message.type === 'dashboard_update') {
      if (message.data.system_overview) {
        setMetrics(prev => ({ ...prev, ...message.data.system_overview, websocket_connected: true }));
      }
      if (message.data.healthy !== undefined) {
        setHealth(message.data);
      }
    }
  });

  // Load initial data when authenticated
  useEffect(() => {
    if (auth.isAuthenticated) {
      const loadInitialData = async () => {
        try {
          const headers = {
            'X-API-Key': auth.apiKey,
            'Content-Type': 'application/json'
          };

          // Load metrics
          try {
            const metricsRes = await fetch(`${API_BASE}/metrics`, { headers });
            if (metricsRes.ok) {
              const metricsData = await metricsRes.json();
              setMetrics(prev => ({ ...prev, ...metricsData, websocket_connected: ws.isConnected }));
            }
          } catch (e) {
            console.warn('Failed to load metrics:', e);
          }

          // Load health
          try {
            const healthRes = await fetch(`${API_BASE}/health`, { headers });
            if (healthRes.ok) {
              const healthData = await healthRes.json();
              setHealth(healthData);
            }
          } catch (e) {
            console.warn('Failed to load health:', e);
          }
        } catch (error) {
          console.error('Failed to load initial data:', error);
        }
      };

      loadInitialData();
    }
  }, [auth.isAuthenticated, auth.apiKey]);

  // Update WebSocket connection status in metrics
  useEffect(() => {
    setMetrics(prev => prev ? { ...prev, websocket_connected: ws.isConnected } : null);
  }, [ws.isConnected]);

  return (
    <ErrorBoundary>
      <div className="app">
        <Header isConnected={ws.isConnected} health={health} auth={auth} />
        
        {/* WebSocket Status */}
        {ws.connectionError && (
          <div className="connection-error-banner">
            ⚠️ {ws.connectionError}
            {ws.reconnectAttempts < CONFIG.wsMaxReconnectAttempts && (
              <button onClick={ws.forceReconnect} style={{ marginLeft: '1rem' }}>
                🔄 Retry Now
              </button>
            )}
          </div>
        )}
        
        <nav className="tab-nav">
          <button 
            className={activeTab === 'chat' ? 'active' : ''} 
            onClick={() => setActiveTab('chat')}
          >
            💬 Chat Interface
          </button>
          <button 
            className={activeTab === 'dashboard' ? 'active' : ''} 
            onClick={() => setActiveTab('dashboard')}
          >
            📊 Dashboard
          </button>
        </nav>

        <main className="main-content">
          {!auth.isAuthenticated ? (
            <AuthSection auth={auth} />
          ) : (
            <>
              {activeTab === 'chat' && <ChatInterface auth={auth} />}
              {activeTab === 'dashboard' && <Dashboard metrics={metrics} health={health} />}
            </>
          )}
        </main>
      </div>
    </ErrorBoundary>
  );
}

export default App;
