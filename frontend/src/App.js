// frontend/src/App.js - Fixed React Dashboard with proper error handling
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer 
} from 'recharts';
import './App.css';

// Environment-based configuration
const getApiBase = () => {
  return import.meta.env.VITE_BACKEND_URL || window.location.origin;
};

const getWsBase = () => {
  const apiBase = getApiBase();
  return apiBase.replace('http://', 'ws://').replace('https://', 'wss://');
};

const API_BASE = getApiBase();
const WS_BASE = getWsBase();

// Safe localStorage wrapper
const safeLocalStorage = {
  getItem: (key) => {
    try {
      return localStorage.getItem(key);
    } catch (error) {
      console.warn('localStorage.getItem failed:', error);
      return null;
    }
  },
  setItem: (key, value) => {
    try {
      localStorage.setItem(key, value);
    } catch (error) {
      console.warn('localStorage.setItem failed:', error);
    }
  },
  removeItem: (key) => {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.warn('localStorage.removeItem failed:', error);
    }
  }
};

// Enhanced Authentication Hook
const useAuth = () => {
  const [apiKey, setApiKey] = useState(() => {
    return safeLocalStorage.getItem('llm_proxy_api_key') || 
           import.meta.env.VITE_API_KEY || 
           'sk-default';
  });
  
  const [sessionToken, setSessionToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authError, setAuthError] = useState(null);
  const [isAuthenticating, setIsAuthenticating] = useState(false);

  // Auto-authenticate on component mount if auto-auth is enabled
  useEffect(() => {
    const shouldAutoAuth = import.meta.env.VITE_AUTO_AUTHENTICATE === 'true';
    if (apiKey && !sessionToken && shouldAutoAuth) {
      authenticate(apiKey);
    }
  }, [apiKey, sessionToken]);

  const authenticate = async (key) => {
    if (!key || key.trim() === '') {
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
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.detail?.message || 
                        errorData.message || 
                        `HTTP ${response.status}: Invalid API key`;
        setAuthError(errorMsg);
        setIsAuthenticated(false);
        return false;
      }
    } catch (error) {
      console.error('Authentication error:', error);
      let errorMessage = 'Connection failed. Please check your connection.';
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage = 'Cannot connect to server. Is the backend running?';
      }
      
      setAuthError(errorMessage);
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
    safeLocalStorage.removeItem('llm_proxy_api_key');
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

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    console.error('React Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-content">
            <h2>🚨 Something went wrong</h2>
            <p>The application encountered an unexpected error.</p>
            <details style={{ whiteSpace: 'pre-wrap', marginTop: '1rem' }}>
              <summary>Error Details</summary>
              <pre>{this.state.error && this.state.error.toString()}</pre>
              <pre>{this.state.errorInfo.componentStack}</pre>
            </details>
            <button 
              onClick={() => window.location.reload()} 
              style={{ marginTop: '1rem', padding: '0.5rem 1rem' }}
            >
              🔄 Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Authentication UI Component (unchanged but with better error handling)
const AuthSection = ({ auth, className = '' }) => {
  const [showAuthForm, setShowAuthForm] = useState(!auth.isAuthenticated);
  const [inputKey, setInputKey] = useState(auth.apiKey || '');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputKey.trim()) {
      return;
    }
    
    const success = await auth.updateApiKey(inputKey.trim());
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
            autoComplete="off"
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

// Rest of the components remain the same...
// (MetricCard, Header, ChatInterface, Dashboard, Performance, Admin components)

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

// Header Component
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

// Enhanced Chat Interface with better error handling
function ChatInterface({ auth }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt-3.5-turbo');
  const [temperature, setTemperature] = useState(0.7);
  const [streamEnabled, setStreamEnabled] = useState(true);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || isLoading || !auth.isAuthenticated) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Create abort controller for this request
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${API_BASE}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': auth.apiKey,
        },
        body: JSON.stringify({
          model: selectedModel,
          messages: [...messages, userMessage],
          temperature: temperature,
          stream: streamEnabled
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          const errorData = await response.json().catch(() => ({}));
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
      
      if (error.name === 'AbortError') {
        return; // Request was cancelled
      }
      
      let errorMessage = '❌ Error: Failed to get response from LLM';
      
      if (error.message.includes('Authentication') || error.message.includes('401') || error.message.includes('403')) {
        errorMessage = '🔑 Authentication Error: Please check your API key';
        auth.logout();
      } else if (error.message.includes('rate limit') || error.message.includes('429')) {
        errorMessage = '⏳ Rate Limit: Please wait before sending another message';
      } else if (error.message.includes('500')) {
        errorMessage = '🛠️ Server Error: The service is experiencing issues';
      } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage = '🌐 Network Error: Cannot connect to server';
      }
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: errorMessage
      }]);
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
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
            if (data.trim() === '[DONE]') return;

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
      if (error.name !== 'AbortError') {
        console.error('Streaming error:', error);
      }
    }
  };

  const clearChat = () => {
    setMessages([]);
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
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

// ... (Rest of the components remain the same - Dashboard, Performance, Admin)

// Main App Component with Error Boundary
function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [circuitBreakers, setCircuitBreakers] = useState(null);
  const [cacheStats, setCacheStats] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  
  const auth = useAuth();

  // Enhanced WebSocket connection with better error handling
  const connectWebSocket = useCallback(() => {
    if (!auth.isAuthenticated || !auth.sessionToken) {
      console.log('Cannot connect WebSocket: not authenticated');
      return;
    }

    try {
      const wsUrl = `${WS_BASE}/ws/dashboard?session=${auth.sessionToken}`;
      console.log('Connecting to WebSocket:', wsUrl);
      
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        console.log('🔌 WebSocket connected with authentication');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'dashboard_update') {
            updateDashboardData(data.data);
          }
        } catch (error) {
          console.error('WebSocket message parsing error:', error);
        }
      };
      
      wsRef.current.onclose = (event) => {
        setIsConnected(false);
        console.log('🔌 WebSocket disconnected', event.code, event.reason);
        
        // Reconnect if authenticated and not a normal closure
        if (auth.isAuthenticated && event.code !== 1000) {
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

  const loadInitialData = useCallback(async () => {
    if (!auth.isAuthenticated) return;

    try {
      const headers = {
        'X-API-Key': auth.apiKey,
        'Content-Type': 'application/json'
      };

      const [metricsRes, healthRes] = await Promise.all([
        fetch(`${API_BASE}/metrics`, { headers }).catch(e => ({ ok: false, error: e })),
        fetch(`${API_BASE}/health`, { headers }).catch(e => ({ ok: false, error: e }))
      ]);
      
      if (metricsRes.ok) {
        const metricsData = await metricsRes.json();
        setMetrics(metricsData);
      }
      
      if (healthRes.ok) {
        const healthData = await healthRes.json();
        setHealth(healthData);
      }

      // Optional endpoints - don't fail if they don't exist
      try {
        const cbRes = await fetch(`${API_BASE}/admin/circuit-breakers`, { headers });
        if (cbRes.ok) setCircuitBreakers(await cbRes.json());
      } catch (e) { 
        console.log('Circuit breakers endpoint not available'); 
      }

      try {
        const cacheRes = await fetch(`${API_BASE}/admin/cache/stats`, { headers });
        if (cacheRes.ok) setCacheStats(await cacheRes.json());
      } catch (e) { 
        console.log('Cache stats endpoint not available'); 
      }

    } catch (error) {
      console.error('Failed to load initial data:', error);
      if (error.message?.includes('401') || error.message?.includes('403')) {
        auth.logout();
      }
    }
  }, [auth]);

  useEffect(() => {
    if (auth.isAuthenticated && auth.sessionToken) {
      connectWebSocket();
      loadInitialData();
    } else {
      if (wsRef.current) {
        wsRef.current.close(1000, 'User logged out');
        setIsConnected(false);
      }
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting');
      }
    };
  }, [auth.isAuthenticated, auth.sessionToken, connectWebSocket, loadInitialData]);

  const updateDashboardData = (data) => {
    if (data.system_overview) setMetrics(data);
    if (data.healthy !== undefined) setHealth(data);
  };

  return (
    <ErrorBoundary>
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
    </ErrorBoundary>
  );
}

export default App;
