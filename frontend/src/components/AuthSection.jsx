// frontend/src/components/AuthSection.jsx - Authentication UI for existing chat
import React, { useState } from 'react';

export const AuthSection = ({ auth, className = '' }) => {
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

// Enhanced WebSocket Hook with Authentication
export const useAuthenticatedWebSocket = (auth) => {
  const [ws, setWs] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [wsError, setWsError] = useState(null);

  useEffect(() => {
    if (auth.isAuthenticated && auth.sessionToken) {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }

    return () => disconnectWebSocket();
  }, [auth.isAuthenticated, auth.sessionToken]);

  const connectWebSocket = () => {
    try {
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/dashboard?session=${auth.sessionToken}`;
      
      const websocket = new WebSocket(wsUrl);
      
      websocket.onopen = () => {
        console.log('🔌 WebSocket connected with authentication');
        setIsConnected(true);
        setWsError(null);
      };

      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('📥 WebSocket message:', data);
          // Handle real-time updates here
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      websocket.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        setIsConnected(false);
        // Auto-reconnect if still authenticated
        if (auth.isAuthenticated) {
          setTimeout(connectWebSocket, 5000);
        }
      };

      websocket.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setWsError('WebSocket connection failed');
      };

      setWs(websocket);
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setWsError('Failed to establish WebSocket connection');
    }
  };

  const disconnectWebSocket = () => {
    if (ws) {
      ws.close();
      setWs(null);
      setIsConnected(false);
    }
  };

  const sendMessage = (message) => {
    if (ws && isConnected) {
      ws.send(JSON.stringify(message));
    }
  };

  return { ws, isConnected, wsError, sendMessage };
};
