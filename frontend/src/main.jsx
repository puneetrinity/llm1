// frontend/src/main.jsx - Clean entry point
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.js'

// Environment validation
const validateEnvironment = () => {
  const requiredEnvVars = ['VITE_BACKEND_URL'];
  const missing = requiredEnvVars.filter(envVar => !import.meta.env[envVar]);
  
  if (missing.length > 0) {
    console.warn('Missing environment variables:', missing);
    console.warn('Using fallback values. Check your .env file.');
  }
};

// Initialize app
const initializeApp = () => {
  validateEnvironment();

  // Hide loading screen if it exists
  const hideLoadingScreen = () => {
    const loading = document.getElementById('loading');
    if (loading) {
      loading.style.opacity = '0';
      loading.style.transition = 'opacity 0.5s ease';
      setTimeout(() => loading.remove(), 500);
    }
  };

  // Create root and render app
  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );

  // Hide loading screen after render
  setTimeout(hideLoadingScreen, 500);

  // Global error handler for unhandled promises
  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    // Prevent default browser behavior
    event.preventDefault();
  });

  // Global error handler
  window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
  });
};

// Start the application
initializeApp();
