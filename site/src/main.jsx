import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';

// A bare URL should land on the home route, not on an empty hash, so that
// every link on the site is shaped the same way.
if (!window.location.hash) window.location.replace(`${window.location.pathname}#/`);

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
