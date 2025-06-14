// frontend/src/hooks/useWebSocket.js - Enhanced WebSocket connection with reconnection logic

import { useState, useRef, useEffect, useCallback } from 'react';
import { WS_BASE, CONFIG } from '../utils/config.js';

/**
 * Enhanced WebSocket hook with automatic reconnection and error handling
 * @param {Object} auth - Authentication object from useAuth hook
 * @param {Function} onMessage - Optional message handler
 * @returns {Object} WebSocket state and methods
 */
export const useWebSocket = (auth, onMessage = null) => {
  const [isConnected, setIsConnected] = useState(false);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [connectionError, setConnectionError] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const pingIntervalRef = useRef(null);

  /**
   * Connect to WebSocket server
   */
  const connectWebSocket = useCallback(() => {
    if (!auth.isAuthenticated || !auth.sessionToken) {
      if (CONFIG.debug) {
        console.log('Cannot connect WebSocket: not authenticated');
      }
      return;
    }

    // Clean up existing connection
    if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
      wsRef.current.close();
    }

    try {
      const wsUrl = `${WS_BASE}/ws/dashboard?session=${auth.sessionToken}`;
      if (CONFIG.debug) {
        console.log('Connecting to WebSocket:', wsUrl);
      }
      
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        setReconnectAttempts(0);
        setConnectionError(null);
        
        if (CONFIG.debug) {
          console.log('🔌 WebSocket connected successfully');
        }

        // Start ping to keep connection alive
        pingIntervalRef.current = setInterval(() => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000); // Ping every 30 seconds
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          
          // Handle different message types
          switch (data.type) {
            case 'dashboard_update':
              if (CONFIG.debug) {
                console.log('Dashboard update received:', data.data);
              }
              break;
            case 'pong':
              // Heartbeat response
              if (CONFIG.debug) {
                console.log('Received pong from server');
              }
              break;
            case 'error':
              console.error('WebSocket error message:', data.message);
              setConnectionError(data.message);
              break;
            default:
              if (CONFIG.debug) {
                console.log('Unknown WebSocket message type:', data.type);
              }
          }
          
          // Call custom message handler if provided
          if (onMessage && typeof onMessage === 'function') {
            onMessage(data);
          }
        } catch (error) {
          console.error('WebSocket message parsing error:', error);
          setConnectionError('Failed to parse WebSocket message');
        }
      };
      
      wsRef.current.onclose = (event) => {
        setIsConnected(false);
        
        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        
        if (CONFIG.debug) {
          console.log('🔌 WebSocket disconnected', event.code, event.reason);
        }
        
        // Only attempt reconnection for unexpected closures
        if (auth.isAuthenticated && event.code !== 1000 && event.code !== 1001) {
          if (reconnectAttempts < CONFIG.wsMaxReconnectAttempts) {
            const timeout = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000); // Exponential backoff
            setConnectionError(`Connection lost. Reconnecting in ${timeout/1000}s...`);
            
            if (CONFIG.debug) {
              console.log(`Attempting to reconnect in ${timeout}ms (attempt ${reconnectAttempts + 1}/${CONFIG.wsMaxReconnectAttempts})`);
            }
            
            reconnectTimeoutRef.current = setTimeout(() => {
              setReconnectAttempts(prev => prev + 1);
              connectWebSocket();
            }, timeout);
          } else {
            setConnectionError('Max reconnection attempts reached. Please refresh the page.');
            console.error('Max reconnection attempts reached');
          }
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setIsConnected(false);
        setConnectionError('WebSocket connection error');
      };

    } catch (error) {
      console.error('WebSocket connection failed:', error);
      setIsConnected(false);
      setConnectionError('Failed to create WebSocket connection');
    }
  }, [auth.isAuthenticated, auth.sessionToken, reconnectAttempts, onMessage]);

  /**
   * Disconnect from WebSocket server
   */
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
    }
    
    setIsConnected(false);
    setReconnectAttempts(0);
    setConnectionError(null);
    setLastMessage(null);
    
    if (CONFIG.debug) {
      console.log('🔌 WebSocket manually disconnected');
    }
  }, []);

  /**
   * Send message through WebSocket
   * @param {Object} message - Message to send
   * @returns {boolean} Success status
   */
  const sendMessage = useCallback((message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        setConnectionError('Failed to send message');
        return false;
      }
    } else {
      console.warn('WebSocket is not connected');
      setConnectionError('WebSocket is not connected');
      return false;
    }
  }, []);

  /**
   * Force reconnection
   */
  const forceReconnect = useCallback(() => {
    setReconnectAttempts(0);
    setConnectionError(null);
    disconnect();
    setTimeout(() => connectWebSocket(), 1000);
  }, [disconnect, connectWebSocket]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  // Auto-connect when authenticated
  useEffect(() => {
    if (auth.isAuthenticated && auth.sessionToken && !isConnected && reconnectAttempts === 0) {
      connectWebSocket();
    }
  }, [auth.isAuthenticated, auth.sessionToken, isConnected, reconnectAttempts, connectWebSocket]);

  return {
    isConnected,
    connectionError,
    reconnectAttempts,
    lastMessage,
    connectWebSocket,
    disconnect,
    sendMessage,
    forceReconnect
  };
};
