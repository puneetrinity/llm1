import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer
} from 'recharts';
import './App.css'; // Make sure to import your CSS

// Environment configuration
const API_BASE = import.meta.env.VITE_BACKEND_URL || window.location.origin;
const WS_BASE = API_BASE.replace('http://', 'ws://').replace('https://', 'wss://');

// Safe localStorage wrapper
const safeLocalStorage = {
  getItem: (key) => {
    try {
      return localStorage.getItem(key);
    } catch (error) {
      console.warn('localStorage access failed:', error);
      return null;
    }
  },
  setItem: (key, value) => {
    try {
      localStorage.setItem(key, value);
    } catch (error) {
      console.warn('localStorage access failed:', error);
    }
  },
  removeItem: (key) => {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.warn('localStorage access failed:', error);
    }
  }
};

// Authentication hook - FIXED
function useAuth() {
  const [apiKey, setApiKey] = useState(() => 
    safeLocalStorage.getItem('llm_proxy_api_key') || 'sk-default');
  const [sessionToken, setSessionToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authError, setAuthError] = useState(null);
  const [isAuthenticating, setIsAuthenticating] = useState(false);

  const authenticate = async (key) => {
    if (!key?.trim()) {
      setAuthError('API key is required');
      return false;
    }

    setIsAuthenticating(true);
    setAuthError(null);

    try {
      const response = await fetch(`${API_BASE}/auth/websocket-session`, {
        method: 'POST',
        headers: {
          'X-API-Key': key,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        setSessionToken(data.session_token);
        setApiKey(key);
        setIsAuthenticated(true);
        safeLocalStorage.setItem('llm_proxy_api_key', key);
        console.log('✅ Authentication successful');
        return true;
      } else {
        const error = await response.json().catch(() => ({}));
        setAuthError(error.detail?.message || `Authentication failed (${response.status})`);
        return false;
      }
    } catch (error) {
      console.error('Auth error:', error);
      setAuthError('Connection failed. Check your network and backend.');
      return false;
    } finally {
      setIsAuthenticating(false);
    }
  };

  const logout = () => {
    setApiKey('');
    setSessionToken(null);
    setIsAuthenticated(false);
    setAuthError(null);
    safeLocalStorage.removeItem('llm_proxy_api_key');
  };

  // Auto-authenticate on mount if API key exists
  useEffect(() => {
    if (apiKey && !sessionToken && !isAuthenticated) {
      authenticate(apiKey);
    }
  }, []);

  return {
    apiKey,
    sessionToken,
    isAuthenticated,
    authError,
    isAuthenticating,
    authenticate,
    logout
  };
}

// Error Boundary - IMPROVED
class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null, errorInfo: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-content">
            <h2>🚨 Something went wrong</h2>
            <p>{this.state.error?.message || 'An unexpected error occurred'}</p>
            <div style={{ marginTop: '1rem' }}>
              <button onClick={() => window.location.reload()}>
                🔄 Reload Page
              </button>
              <button 
                onClick={() => this.setState({ hasError: false, error: null })}
                style={{ marginLeft: '0.5rem' }}
              >
                🔄 Try Again
              </button>
            </div>
            {process.env.NODE_ENV === 'development' && (
              <details style={{ marginTop: '1rem', fontSize: '0.8rem' }}>
                <summary>Error Details (Dev Mode)</summary>
                <pre>{this.state.error?.stack}</pre>
              </details>
            )}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// Metric Card Component
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

// Header Component - IMPROVED
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

// Authentication Component - NEW
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
          </div>
        </form>
      </div>
    </div>
  );
}

// Chat Interface - IMPROVED
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

// Dashboard Component - IMPROVED
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
      </div>
    </div>
  );
}

// Main App Component - IMPROVED
function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  const auth = useAuth();

  // WebSocket connection with better error handling
  useEffect(() => {
    if (auth.isAuthenticated && auth.sessionToken) {
      console.log('🔌 Setting up WebSocket connection...');
      
      const ws = new WebSocket(`${WS_BASE}/ws/dashboard?session=${auth.sessionToken}`);
      
      ws.onopen = () => {
        console.log('🔌 WebSocket connected');
        setIsConnected(true);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('📨 WebSocket message:', data);
          
          if (data.type === 'dashboard_update') {
            if (data.data.system_overview) {
              setMetrics(data.data.system_overview);
            }
            if (data.data.healthy !== undefined) {
              setHealth(data.data);
            }
          }
        } catch (error) {
          console.error('WebSocket message parsing error:', error);
        }
      };
      
      ws.onclose = (event) => {
        console.log('🔌 WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
      };
      
      ws.onerror = (error) => {
        console.error('🔌 WebSocket error:', error);
        setIsConnected(false);
      };
      
      wsRef.current = ws;
      
      return () => {
        console.log('🔌 Cleaning up WebSocket');
        ws.close();
      };
    } else {
      setIsConnected(false);
    }
  }, [auth.isAuthenticated, auth.sessionToken]);

  // Load initial data
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
              setMetrics(metricsData);
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

  return (
    <ErrorBoundary>
      <div className="app">
        <Header isConnected={isConnected} health={health} auth={auth} />
        
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
