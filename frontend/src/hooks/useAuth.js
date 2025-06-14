// frontend/src/hooks/useAuth.js - Authentication Hook for Chat Interface
import { useState, useEffect } from 'react';

export const useAuth = () => {
  const [apiKey, setApiKey] = useState(() => {
    return localStorage.getItem('llm_proxy_api_key') || 
           import.meta.env.VITE_API_KEY || 
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
