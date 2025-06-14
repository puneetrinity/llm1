// frontend/src/main.jsx - Updated entry point with correct imports
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'  // Updated import from .js to .jsx
import './App.css'

// Hide loading screen if it exists
const hideLoadingScreen = () => {
  const loading = document.getElementById('loading');
  if (loading) {
    loading.style.opacity = '0';
    loading.style.transition = 'opacity 0.5s ease';
    setTimeout(() => loading.remove(), 500);
  }
};

// Error boundary wrapper for the entire app
class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('App Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh',
          padding: '2rem',
          background: '#0a0e1a',
          color: 'white',
          fontFamily: 'system-ui, -apple-system, sans-serif'
        }}>
          <div style={{
            textAlign: 'center',
            background: '#1a1f3a',
            padding: '2rem',
            borderRadius: '12px',
            border: '1px solid #3a4a6b',
            maxWidth: '500px'
          }}>
            <h2 style={{ color: '#ff4757', marginBottom: '1rem' }}>
              🚨 Application Error
            </h2>
            <p style={{ marginBottom: '1rem', lineHeight: '1.6' }}>
              The LLM Proxy Dashboard encountered an error during startup.
            </p>
            <details style={{ 
              textAlign: 'left', 
              background: '#0f1419', 
              padding: '1rem', 
              borderRadius: '6px',
              margin: '1rem 0'
            }}>
              <summary style={{ cursor: 'pointer', marginBottom: '0.5rem' }}>
                Error Details
              </summary>
              <pre style={{ 
                fontSize: '0.875rem', 
                overflow: 'auto',
                whiteSpace: 'pre-wrap'
              }}>
                {this.state.error?.toString()}
              </pre>
            </details>
            <button 
              onClick={() => window.location.reload()}
              style={{
                background: '#1da1f2',
                color: 'white',
                border: 'none',
                padding: '0.75rem 1.5rem',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '0.875rem'
              }}
            >
              🔄 Reload Application
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Create root and render app with error boundary
const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </React.StrictMode>
);

// Hide loading screen after a short delay
setTimeout(hideLoadingScreen, 500);

// Development hot reload support
if (import.meta.hot) {
  import.meta.hot.accept('./App.jsx', () => {
    // Re-render the app when App.jsx changes
    root.render(
      <React.StrictMode>
        <AppErrorBoundary>
          <App />
        </AppErrorBoundary>
      </React.StrictMode>
    );
  });
}
