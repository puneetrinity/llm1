// frontend/src/App.js - Complete React Dashboard
import React, { useState, useEffect, useRef } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer 
} from 'recharts';
import './App.css';

const API_BASE = window.location.origin;
const WS_BASE = window.location.origin.replace('http', 'ws');

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [circuitBreakers, setCircuitBreakers] = useState(null);
  const [cacheStats, setCacheStats] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  // Initialize WebSocket for real-time updates
  useEffect(() => {
    connectWebSocket();
    loadInitialData();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    try {
      wsRef.current = new WebSocket(`${WS_BASE}/ws/dashboard`);
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        console.log('WebSocket connected');
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'dashboard_update') {
          updateDashboardData(data.data);
        }
      };
      
      wsRef.current.onclose = () => {
        setIsConnected(false);
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
    } catch (error) {
      console.error('WebSocket connection failed:', error);
    }
  };

  const loadInitialData = async () => {
    try {
      const [metricsRes, healthRes] = await Promise.all([
        fetch(`${API_BASE}/metrics`),
        fetch(`${API_BASE}/health`)
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
        const cbRes = await fetch(`${API_BASE}/admin/circuit-breakers`);
        if (cbRes.ok) {
          setCircuitBreakers(await cbRes.json());
        }
      } catch (e) { /* Enhanced features not available */ }

      try {
        const cacheRes = await fetch(`${API_BASE}/admin/cache/stats`);
        if (cacheRes.ok) {
          setCacheStats(await cacheRes.json());
        }
      } catch (e) { /* Enhanced features not available */ }

    } catch (error) {
      console.error('Failed to load initial data:', error);
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
      <Header isConnected={isConnected} health={health} />
      
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
        {activeTab === 'chat' && <ChatInterface />}
        {activeTab === 'dashboard' && <Dashboard metrics={metrics} health={health} />}
        {activeTab === 'performance' && <Performance metrics={metrics} circuitBreakers={circuitBreakers} />}
        {activeTab === 'admin' && <Admin cacheStats={cacheStats} />}
      </main>
    </div>
  );
}

// Header Component
function Header({ isConnected, health }) {
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
      </div>
    </header>
  );
}

// Chat Interface Component
function ChatInterface() {
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
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: selectedModel,
          messages: [...messages, userMessage],
          temperature: temperature,
          stream: streamEnabled
        })
      });

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
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Error: Failed to get response from LLM'
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
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button onClick={sendMessage} disabled={isLoading || !input.trim()}>
          {isLoading ? '⏳' : '➤'}
        </button>
      </div>
    </div>
  );
}

// Dashboard Component
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

// Performance Component
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

// Admin Component
function Admin({ cacheStats }) {
  const [logs, setLogs] = useState([]);

  const clearCache = async () => {
    try {
      await fetch(`${API_BASE}/admin/cache/clear`, { method: 'POST' });
      alert('Cache cleared successfully');
    } catch (error) {
      alert('Failed to clear cache');
    }
  };

  const restartService = async () => {
    if (confirm('Are you sure you want to restart the service?')) {
      try {
        await fetch(`${API_BASE}/admin/restart`, { method: 'POST' });
        alert('Service restart initiated');
      } catch (error) {
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
        </div>
      </div>
    </div>
  );
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

export default App;
