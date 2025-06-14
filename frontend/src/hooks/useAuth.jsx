// frontend/src/hooks/useAuth.js - Enhanced authentication hook with better error handling

import { useState, useEffect, useCallback } from 'react';
import { CONFIG } from '../utils/config.js';
import { secureStorage } from '../utils/secureStorage.js';

/**
 * Enhanced authentication hook with proper state management and error handling
 * @returns {Object} Authentication state and methods
 */
export const useAuth = () => {
  const [apiKey, setApiKey] = useState(() => {
    return secureStorage.getItem('llm_proxy_api_key') || CONFIG.defaultApiKey;
  });
  
  const [sessionToken, setSessionToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authError, setAuthError] = useState(null);
  const [isAuthenticating, setIsAuthenticating] = useState(false);

  /**
   * Authenticate with the backend
   * @param {string} key - API key to authenticate with
   * @returns {Promise<boolean>} Success status
   */
  const authenticate = useCallback(async (key) => {
    if (!key || key.trim() === '') {
      setAuthError('API key is required');
      return false;
    }

    setIsAuthenticating(true);
    setAuthError(null);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), CONFIG.authTimeout);

      const response = await fetch(`${CONFIG.apiBase}/auth/websocket-session`, {
        method: 'POST',
        headers: {
          'X-API-Key': key,
          'Content-Type': 'application/json'
        },
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        setSessionToken(data.session_token);
        setApiKey(key);
        setIsAuthenticated(true);
        secureStorage.setItem('llm_proxy_api_key', key);
        
        if (CONFIG.debug) {
          console.log('✅ Authentication successful');
        }
        return true;
      } else {
        const errorData = await response.json().catch(() => ({}));
        let errorMsg = 'Authentication failed';
        
        switch (response.status) {
          case 401:
            errorMsg = 'Invalid API key';
            break;
          case 403:
            errorMsg = 'Access forbidden';
            break;
          case 429:
            errorMsg = 'Too many requests. Please try again later.';
            break;
          case 500:
            errorMsg = 'Server error. Please try again later.';
            break;
          default:
            errorMsg = errorData.detail?.message || errorData.message || `HTTP ${response.status}`;
        }
        
        setAuthError(errorMsg);
        setIsAuthenticated(false);
        return false;
      }
    } catch (error) {
      console.error('Authentication error:', error);
      let errorMessage = 'Connection failed';
      
      if (error.name === 'AbortError') {
        errorMessage = 'Request timeout. Please check your connection.';
      } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage = 'Cannot connect to server. Please check if the backend is running.';
      }
      
      setAuthError(errorMessage);
      setIsAuthenticated(false);
      return false;
    } finally {
      setIsAuthenticating(false);
    }
  }, []);

  /**
   * Logout and clear authentication state
   */
  const logout = useCallback(() => {
    setApiKey('');
    setSessionToken(null);
    setIsAuthenticated(false);
    setAuthError(null);
    secureStorage.removeItem('llm_proxy_api_key');
    
    if (CONFIG.debug) {
      console.log('🚪 User logged out');
    }
  }, []);

  /**
   * Update API key (alias for authenticate)
   */
  const updateApiKey = useCallback((key) => {
    return authenticate(key);
  }, [authenticate]);

  // Auto-authenticate on mount if enabled
  useEffect(() => {
    if (CONFIG.autoAuthenticate && apiKey && !sessionToken && !isAuthenticating) {
      if (CONFIG.debug) {
        console.log('🔄 Auto-authenticating...');
      }
      authenticate(apiKey);
    }
  }, []); // Only run on mount

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
