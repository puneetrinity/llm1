// frontend/src/utils/config.js - Enhanced configuration and environment handling

/**
 * Get application configuration from environment variables
 * @returns {Object} Configuration object
 */
export const getConfig = () => {
  const config = {
    apiBase: import.meta.env.VITE_BACKEND_URL || window.location.origin,
    defaultApiKey: import.meta.env.VITE_API_KEY || '',
    autoAuthenticate: import.meta.env.VITE_AUTO_AUTHENTICATE === 'true',
    debug: import.meta.env.VITE_DEBUG === 'true',
    wsReconnectInterval: parseInt(import.meta.env.VITE_WS_RECONNECT_INTERVAL) || 5000,
    wsMaxReconnectAttempts: parseInt(import.meta.env.VITE_WS_MAX_RECONNECT_ATTEMPTS) || 5,
    authTimeout: parseInt(import.meta.env.VITE_AUTH_TIMEOUT) || 10000,
    development: import.meta.env.DEV
  };

  // Validate configuration
  if (!config.apiBase) {
    console.warn('Backend URL not configured properly');
  }

  if (config.debug) {
    console.log('🔧 Application Configuration:', config);
  }

  return config;
};

/**
 * Get WebSocket base URL from API base URL
 * @returns {string} WebSocket base URL
 */
export const getWsBase = () => {
  const apiBase = getConfig().apiBase;
  return apiBase.replace('http://', 'ws://').replace('https://', 'wss://');
};

// Export singleton config instance
export const CONFIG = getConfig();

// Export constants
export const WS_BASE = getWsBase();
export const API_BASE = CONFIG.apiBase;
