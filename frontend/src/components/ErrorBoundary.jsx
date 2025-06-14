// frontend/src/components/ErrorBoundary.js - Enhanced error boundary with user reporting

import React from 'react';
import { CONFIG } from '../utils/config.js';

/**
 * Enhanced Error Boundary component with better error reporting and recovery
 */
export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false, 
      error: null, 
      errorInfo: null,
      errorId: null,
      errorCount: 0
    };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { 
      hasError: true,
      errorId: Date.now().toString(36) + Math.random().toString(36).substr(2),
      errorCount: (this.state?.errorCount || 0) + 1
    };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    
    // Log error to console
    console.error('React Error Boundary caught an error:', error, errorInfo);
    
    // In production, you might want to send this to an error reporting service
    if (CONFIG.debug) {
      console.group('🐛 Error Details');
      console.error('Error:', error);
      console.error('Component Stack:', errorInfo.componentStack);
      console.error('Error Count:', this.state.errorCount);
      console.groupEnd();
    }

    // Report to error tracking service (implement as needed)
    this.reportError(error, errorInfo);
  }

  /**
   * Report error to external service (placeholder)
   * @param {Error} error - The error object
   * @param {Object} errorInfo - Additional error information
   */
  reportError(error, errorInfo) {
    // TODO: Implement error reporting to external service
    // Example: Sentry, LogRocket, Bugsnag, etc.
    if (CONFIG.debug) {
      console.log('Error reported:', {
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        errorId: this.state.errorId,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href
      });
    }
  }

  /**
   * Reset error boundary state
   */
  resetError = () => {
    this.setState({ 
      hasError: false, 
      error: null, 
      errorInfo: null,
      errorId: null 
    });
  }

  /**
   * Reload the entire page
   */
  reloadPage = () => {
    window.location.reload();
  }

  /**
   * Copy error details to clipboard
   */
  copyErrorDetails = async () => {
    const errorDetails = {
      errorId: this.state.errorId,
      message: this.state.error?.message,
      stack: this.state.error?.stack,
      componentStack: this.state.errorInfo?.componentStack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent
    };

    try {
      await navigator.clipboard.writeText(JSON.stringify(errorDetails, null, 2));
      alert('Error details copied to clipboard');
    } catch (err) {
      console.error('Failed to copy error details:', err);
      // Fallback: select text for manual copy
      const textArea = document.createElement('textarea');
      textArea.value = JSON.stringify(errorDetails, null, 2);
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      alert('Error details copied to clipboard (fallback method)');
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '2rem',
          margin: '1rem',
          border: '2px solid #ff6b6b',
          borderRadius: '8px',
          backgroundColor: '#fff5f5',
          fontFamily: 'Arial, sans-serif'
        }}>
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
            <h2 style={{ 
              color: '#d63031', 
              marginBottom: '1rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              🚨 Something went wrong
            </h2>
            
            <p style={{ marginBottom: '1rem', color: '#2d3436' }}>
              The application encountered an unexpected error. This has been logged for investigation.
            </p>
            
            <div style={{ 
              backgroundColor: '#f8f9fa', 
              padding: '1rem', 
              borderRadius: '4px',
              marginBottom: '1rem',
              fontSize: '0.9em'
            }}>
              <p><strong>Error ID:</strong> <code>{this.state.errorId}</code></p>
              <p><strong>Error Count:</strong> {this.state.errorCount}</p>
              {this.state.error && (
                <p><strong>Message:</strong> {this.state.error.message}</p>
              )}
            </div>

            {CONFIG.debug && (
              <details style={{ 
                marginBottom: '1rem',
                backgroundColor: '#f1f3f4',
                padding: '1rem',
                borderRadius: '4px'
              }}>
                <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>
                  🔍 Error Details (Debug Mode)
                </summary>
                <pre style={{ 
                  whiteSpace: 'pre-wrap', 
                  fontSize: '0.8em',
                  overflow: 'auto',
                  maxHeight: '300px',
                  marginTop: '0.5rem'
                }}>
                  {this.state.error && this.state.error.toString()}
                  {'\n\n'}
                  {this.state.errorInfo && this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}
            
            <div style={{ 
              display: 'flex', 
              gap: '0.5rem', 
              flexWrap: 'wrap',
              alignItems: 'center'
            }}>
              <button 
                onClick={this.resetError}
                style={{ 
                  padding: '0.75rem 1rem',
                  backgroundColor: '#00b894',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.9em'
                }}
              >
                🔄 Try Again
              </button>
              
              <button 
                onClick={this.reloadPage}
                style={{ 
                  padding: '0.75rem 1rem',
                  backgroundColor: '#0984e3',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.9em'
                }}
              >
                🔄 Reload Page
              </button>
              
              <button 
                onClick={this.copyErrorDetails}
                style={{ 
                  padding: '0.75rem 1rem',
                  backgroundColor: '#636e72',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.9em'
                }}
              >
                📋 Copy Error Details
              </button>
            </div>

            {this.state.errorCount > 3 && (
              <div style={{
                marginTop: '1rem',
                padding: '1rem',
                backgroundColor: '#fff5f5',
                border: '1px solid #ff7675',
                borderRadius: '4px',
                color: '#d63031'
              }}>
                <strong>⚠️ Multiple errors detected</strong>
                <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9em' }}>
                  If this problem persists, try refreshing the page or clearing your browser cache.
                </p>
              </div>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
