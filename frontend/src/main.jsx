// frontend/src/main.jsx - Alternative Vite entry point
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.js'

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

// Hide loading screen after a short delay
setTimeout(hideLoadingScreen, 500);
