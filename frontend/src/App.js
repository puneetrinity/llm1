// frontend/src/App.js - Complete React Dashboard with Authentication
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer 
} from 'recharts';
import './App.css';

const API_BASE = window.location.origin;
const WS_BASE = window.location.origin.replace('http', 'ws');

// Authentication Hook
const useAuth = () => {
  const [apiKey, setApiKey] = useState(() => {
    return localStorage.getItem('llm_proxy_api_key') || 
           import.meta?.env?.VITE_API_KEY || 
           process.env?.REACT_APP_API_KEY ||
           'sk-default';
  });
  
  const [sessionToken, setSessionToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authError, setAuthError] = useState(null);
  const [isAuthenticating, setIsAuthenticating] = useState(false);

  // Auto-authenticate on component mount
  useEffect(() => {
    if (apiKey && !sessionToken) {
      authenticate(apiKey);
    }
  }, [apiKey]);

  const authenticate = async (key) => {
    setIsAuthenticating(true);
    setAuthError(null);

    try {
      // Test API key by getting session token
      const response = await fetch('/auth/websocket-session', {
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
        localStorage.setItem('llm_proxy_api_key', key);
        console.log('✅ Authentication successful');
        return true;
      } else {
        const errorData = await response.json();
        const errorMsg = errorData.detail?.message || errorData.message || 'Invalid API key';
        setAuthError(errorMsg);
        setIsAuthenticated(false);
        return false;
      }
    } catch (error) {
      console.error('Authentication error:', error);
      setAuthError('Connection failed. Please check your connection.');
      setIsAuthenticated(false);
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
    localStorage.removeItem('llm_proxy_api_key');
  };

  const updateApiKey = async (newKey) => {
    return await authenticate(newKey);
  };

  return {
    apiKey,
    sessionToken,
    isAuthenticated,
    authError,
    isAuthenticating,
    authenticate,
    logout,
    updateApiKey
  };
};

// Authentication UI Component
const AuthSection = ({ auth, className = '' }) => {
  const [showAuthForm, setShowAuthForm] = useState(!auth.isAuthenticated);
  const [inputKey, setInputKey] = useState(auth.apiKey || '');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const success = await auth.updateApiKey(inputKey);
    if (success) {
      setShowAuthForm(false);
    }
  };

  const handleLogout = () => {
    auth.logout();
    setShowAuthForm(true);
    setInputKey('');
  };

  if (auth.isAuthenticated && !showAuthForm) {
    return (
      <div className={`auth-status authenticated ${className}`}>
        <span className="auth-indicator">
          🔑 <span className="api-key-preview">{auth.apiKey.substring(0, 8)}...</span>
        </span>
        <button onClick={() => setShowAuthForm(true)} className="change-key-btn">
          Change Key
        </button>
        <button onClick={handleLogout} className="logout-btn">
          Logout
        </button>
      </div>
    );
  }

  return (
    <div className={`auth-form ${className}`}>
      <form onSubmit={handleSubmit} className="api-key-form">
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
            {auth.isAuthenticating ? '🔄' : '🔑 Connect'}
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
  );
};

// Main App Component
function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [circuitBreakers, setCircuitBreakers] = useState(null);
  const [cacheStats, setCacheStats] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  
  // Authentication
  const auth = useAuth();

  // FIXED: Wrapped connectWebSocket in useCallback with auth dependency
  const connectWebSocket = useCallback(() => {
    if (!auth.isAuthenticated || !auth.sessionToken) {
      console.log('Cannot connect WebSocket: not authenticated');
      return;
    }

    try {
      // Connect with session token authentication
      const wsUrl = `${WS_BASE}/ws/dashboard?session=${auth.sessionToken}`;
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        console.log('🔌 WebSocket connected with authentication');
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'dashboard_update') {
          updateDashboardData(data.data);
        }
      };
      
      wsRef.current.onclose = () => {
        setIsConnected(false);
        console.log('🔌 WebSocket disconnected');
        // Reconnect after 5 seconds if still authenticated
        if (auth.isAuthenticated) {
          setTimeout(connectWebSocket, 5000);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setIsConnected(false);
      };
    } catch (error) {
      console.error('WebSocket connection failed:', error);
    }
  }, [auth.isAuthenticated, auth.sessionToken]);

  // Initialize WebSocket for real-time updates
  useEffect(() => {
    if (auth.isAuthenticated && auth.sessionToken) {
      connectWebSocket();
      loadInitialData();
    } else {
      // Disconnect WebSocket if not authenticated
      if (wsRef.current) {
        wsRef.current.close();
        setIsConnected(false);
      }
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [auth.isAuthenticated, auth.sessionToken, connectWebSocket]);

  const loadInitialData = async () => {
    if (!auth.isAuthenticated) return;

    try {
      const headers = {
        'X-API-Key': auth.apiKey,
        'Content-Type': 'application/json'
      };

      const [metricsRes, healthRes] = await Promise.all([
        fetch(`${API_BASE}/metrics`, { headers }),
        fetch(`${API_BASE}/health`, { headers })
      ]);
      
      if (metricsRes.ok) {
        const metricsData = await metricsRes.json();
        setMetrics(metricsData);
      }
      
      if (healthRes.ok) {
        const healthData = await healthRes.json();
        setHealth(healthData);
      }

      // Try to load enhanced features data
      try {
        const cbRes = await fetch(`${API_BASE}/admin/circuit-breakers`, { headers });
        if (cbRes.ok) {
          setCircuitBreakers(await cbRes.json());
        }
      } catch (e) { 
        console.log('Circuit breakers not available');
      }

      try {
        const cacheRes = await fetch(`${API_BASE}/admin/cache/stats`, { headers });
        if (cacheRes.ok) {
          setCacheStats(await cacheRes.json());
        }
      } catch (e) { 
        console.log('Cache stats not available');
      }

    } catch (error) {
      console.error('Failed to load initial data:', error);
      
      // Handle authentication errors
      if (error.message?.includes('401') || error.message?.includes('403')) {
        console.log('Authentication failed, logging out...');
        auth.logout();
      }
    }
  };

  const updateDashboardData = (data) => {
    if (data.system_overview) {
      setMetrics(data);
    }
    if (data.healthy !== undefined) {
      setHealth(data);
    }
  };

  return (
    <div className="app">
      <Header 
        isConnected={isConnected} 
        health={health} 
        auth={auth}
      />
      
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
        <button 
          className={activeTab === 'performance' ? 'active' : ''} 
          onClick={() => setActiveTab('performance')}
        >
          ⚡ Performance
        </button>
        <button 
          className={activeTab === 'admin' ? 'active' : ''} 
          onClick={() => setActiveTab('admin')}
        >
          🔧 Admin
        </button>
      </nav>

      <main className="main-content">
        {!auth.isAuthenticated ? (
          <div className="auth-required">
            <div className="auth-required-content">
              <h2>🔐 Authentication Required</h2>
              <p>Please enter your API key to access the LLM Proxy interface.</p>
              <AuthSection auth={auth} />
            </div>
          </div>
        ) : (
          <>
            {activeTab === 'chat' && <ChatInterface auth={auth} />}
            {activeTab === 'dashboard' && <Dashboard metrics={metrics} health={health} />}
            {activeTab === 'performance' && <Performance metrics={metrics} circuitBreakers={circuitBreakers} />}
            {activeTab === 'admin' && <Admin cacheStats={cacheStats} auth={auth} />}
          </>
        )}
      </main>
    </div>
  );
}

// Header Component with Authentication
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
        <AuthSection auth={auth} className="header-auth" />
        <div className={`status-indicator ${connectionStatus}`}>
          <span className="dot"></span>
          WebSocket: {connectionStatus}
        </div>
        <div className={`status-indicator ${healthStatus}`}>
          <span className="dot"></span>
          Health: {healthStatus}
        </div>
      </div>
    </header>
  );
}

// Chat Interface Component with Authentication
function ChatInterface({ auth }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt-3.5-turbo');
  const [temperature, setTemperature] = useState(0.7);
  const [streamEnabled, setStreamEnabled] = useState(true);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading || !auth.isAuthenticated) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': auth.apiKey, // ✅ AUTHENTICATION HEADER
        },
        body: JSON.stringify({
          model: selectedModel,
          messages: [...messages, userMessage],
          temperature: temperature,
          stream: streamEnabled
        })
      });

      if (!response.ok) {
        // Handle authentication errors
        if (response.status === 401 || response.status === 403) {
          const errorData = await response.json();
          throw new Error(errorData.detail?.message || 'Authentication failed');
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (streamEnabled) {
        await handleStreamingResponse(response);
      } else {
        const data = await response.json();
        const assistantMessage = {
          role: 'assistant',
          content: data.choices[0].message.content
        };
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      console.error('Chat error:', error);
      
      let errorMessage = '❌ Error: Failed to get response from LLM';
      
      if (error.message.includes('Authentication') || error.message.includes('401') || error.message.includes('403')) {
        errorMessage = '🔑 Authentication Error: Please check your API key';
        // Trigger re-authentication
        auth.logout();
      } else if (error.message.includes('rate limit') || error.message.includes('429')) {
        errorMessage = '⏳ Rate Limit: Please wait before sending another message';
      } else if (error.message.includes('500')) {
        errorMessage = '🛠️ Server Error: The service is experiencing issues';
      }
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: errorMessage
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStreamingResponse = async (response) => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantMessage = { role: 'assistant', content: '' };
    
    setMessages(prev => [...prev, assistantMessage]);

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') return;

            try {
              const parsed = JSON.parse(data);
              const content = parsed.choices?.[0]?.delta?.content || '';
              if (content) {
                assistantMessage.content += content;
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1] = { ...assistantMessage };
                  return newMessages;
                });
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="chat-interface">
      <div className="chat-controls">
        <div className="control-group">
          <label>Model:</label>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
            <option value="gpt-4">GPT-4</option>
            <option value="mistral:7b-instruct-q4_0">Mistral 7B</option>
            <option value="llama3:8b-instruct-q4_0">Llama3 8B</option>
          </select>
        </div>
        
        <div className="control-group">
          <label>Temperature: {temperature}</label>
          <input 
            type="range" 
            min="0" 
            max="2" 
            step="0.1" 
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
          />
        </div>
        
        <div className="control-group">
          <label>
            <input 
              type="checkbox" 
              checked={streamEnabled}
              onChange={(e) => setStreamEnabled(e.target.checked)}
            />
            Streaming
          </label>
        </div>
        
        <button onClick={clearChat} className="clear-btn">🗑️ Clear</button>
      </div>

      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-role">{message.role === 'user' ? '👤' : '🤖'}</div>
            <div className="message-content">{message.content}</div>
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
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder={auth.isAuthenticated ? "Type your message..." : "Please authenticate first..."}
          disabled={isLoading || !auth.isAuthenticated}
        />
        <button 
          onClick={sendMessage} 
          disabled={isLoading || !input.trim() || !auth.isAuthenticated}
        >
          {isLoading ? '⏳' : '➤'}
        </button>
      </div>
    </div>
  );
}

// Dashboard Component (unchanged)
function Dashboard({ metrics, health }) {
  if (!metrics) {
    return <div className="loading">Loading dashboard data...</div>;
  }

  // Prepare chart data
  const responseTimeData = metrics.response_times?.recent?.map((time, index) => ({
    request: index + 1,
    time: time
  })) || [];

  const modelUsageData = Object.entries(metrics.models || {}).map(([model, data]) => ({
    model: model.split(':')[0], // Simplified name
    requests: data.requests || 0,
    tokens: data.tokens || 0
  }));

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <div className="dashboard">
      <div className="metrics-grid">
        <MetricCard 
          title="Total Requests" 
          value={metrics.overview?.total_requests || 0}
          icon="📊"
          color="blue"
        />
        <MetricCard 
          title="Average Response Time" 
          value={`${(metrics.overview?.avg_response_time || 0).toFixed(2)}s`}
          icon="⚡"
          color="green"
        />
        <MetricCard 
          title="Error Rate" 
          value={`${(metrics.overview?.error_rate || 0).toFixed(1)}%`}
          icon="⚠️"
          color="red"
        />
        <MetricCard 
          title="Cache Hit Rate" 
          value={`${(metrics.overview?.cache_hit_rate * 100 || 0).toFixed(1)}%`}
          icon="💾"
          color="purple"
        />
      </div>

      <div className="charts-grid">
        <div className="chart-container">
          <h3>Response Times (Recent)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={responseTimeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="request" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="time" stroke="#8884d8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>Model Usage</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelUsageData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="requests" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>Request Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={modelUsageData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({model, requests}) => `${model}: ${requests}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="requests"
              >
                {modelUsageData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>System Health</h3>
          <div className="health-overview">
            {health?.services?.map((service, index) => (
              <div key={index} className={`health-item ${service.status}`}>
                <span className="service-name">{service.name}</span>
                <span className={`status-badge ${service.status}`}>
                  {service.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// Performance Component (unchanged)
function Performance({ metrics, circuitBreakers }) {
  return (
    <div className="performance">
      <h2>Performance Monitoring</h2>
      
      {/* Circuit Breakers */}
      {circuitBreakers && (
        <div className="section">
          <h3>Circuit Breakers</h3>
          <div className="circuit-breakers-grid">
            {Object.entries(circuitBreakers).map(([name, cb]) => (
              <div key={name} className={`circuit-breaker ${cb.state}`}>
                <h4>{name}</h4>
                <div className="cb-stats">
                  <div>State: <span className={`state ${cb.state}`}>{cb.state}</span></div>
                  <div>Failures: {cb.failure_count}</div>
                  <div>Success Rate: {(100 - cb.stats.failure_rate).toFixed(1)}%</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Performance Metrics */}
      {metrics && (
        <div className="section">
          <h3>Performance Metrics</h3>
          <div className="perf-metrics">
            <div className="metric">
              <h4>Response Time Percentiles</h4>
              <div>P50: {metrics.performance_metrics?.response_times?.p50 || 'N/A'}s</div>
              <div>P95: {metrics.performance_metrics?.response_times?.p95 || 'N/A'}s</div>
              <div>P99: {metrics.performance_metrics?.response_times?.p99 || 'N/A'}s</div>
            </div>
            <div className="metric">
              <h4>Throughput</h4>
              <div>Req/min: {metrics.performance_metrics?.throughput?.requests_per_minute || 'N/A'}</div>
              <div>Tokens/sec: {metrics.performance_metrics?.throughput?.tokens_per_second || 'N/A'}</div>
            </div>
            <div className="metric">
              <h4>System Resources</h4>
              <div>CPU: {metrics.performance_metrics?.system_resources?.cpu_usage || 'N/A'}%</div>
              <div>Memory: {metrics.performance_metrics?.system_resources?.memory_usage || 'N/A'}%</div>
              <div>GPU: {metrics.performance_metrics?.system_resources?.gpu_usage?.utilization || 'N/A'}%</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Admin Component with Authentication
function Admin({ cacheStats, auth }) {
  const clearCache = async () => {
    if (!auth.isAuthenticated) return;

    try {
      const response = await fetch(`${API_BASE}/admin/cache/clear`, { 
        method: 'POST',
        headers: {
          'X-API-Key': auth.apiKey,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        alert('Cache cleared successfully');
      } else if (response.status === 401 || response.status === 403) {
        alert('Authentication failed. Please check your API key.');
        auth.logout();
      } else {
        alert('Failed to clear cache');
      }
    } catch (error) {
      console.error('Clear cache error:', error);
      alert('Failed to clear cache');
    }
  };

  const restartService = async () => {
    if (!auth.isAuthenticated) return;

    if (window.confirm('Are you sure you want to restart the service?')) {
      try {
        const response = await fetch(`${API_BASE}/admin/restart`, { 
          method: 'POST',
          headers: {
            'X-API-Key': auth.apiKey,
            'Content-Type': 'application/json'
          }
        });

        if (response.ok) {
          alert('Service restart initiated');
        } else if (response.status === 401 || response.status === 403) {
          alert('Authentication failed. Please check your API key.');
          auth.logout();
        } else {
          alert('Failed to restart service');
        }
      } catch (error) {
        console.error('Restart service error:', error);
        alert('Failed to restart service');
      }
    }
  };

  return (
    <div className="admin">
      <h2>Administration</h2>
      
      <div className="admin-actions">
        <button onClick={clearCache} className="admin-btn warning">
          🗑️ Clear Cache
        </button>
        <button onClick={restartService} className="admin-btn danger">
          🔄 Restart Service
        </button>
      </div>

      {/* Cache Statistics */}
      {cacheStats && (
        <div className="section">
          <h3>Cache Statistics</h3>
          <div className="cache-stats">
            <div>Hit Rate: {(cacheStats.hit_rate || 0).toFixed(1)}%</div>
            <div>Total Requests: {cacheStats.total_requests || 0}</div>
            <div>Cache Size: {cacheStats.cache_size || 0} entries</div>
            <div>Memory Usage: {(cacheStats.memory_usage_mb || 0).toFixed(1)} MB</div>
          </div>
        </div>
      )}

      {/* System Information */}
      <div className="section">
        <h3>System Information</h3>
        <div className="system-info">
          <div>Version: 2.2.0</div>
          <div>Uptime: {new Date().toLocaleString()}</div>
          <div>Environment: Production</div>
          <div>Authenticated User: {auth.apiKey.substring(0, 8)}...</div>
        </div>
      </div>
    </div>
  );
}

// Metric Card Component (unchanged)
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

export default App;
