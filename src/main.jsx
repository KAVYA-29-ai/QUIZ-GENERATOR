// main.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css'; // tailwind styles
import { ThemeProvider } from 'next-themes'; // optional, if you want dark/light toggle
import { AnimatePresence } from 'framer-motion';
import { Toaster } from 'sonner'; // optional: toast notifications

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <ThemeProvider attribute="class">
      <AnimatePresence mode="wait" initial={false}>
        <App />
        <Toaster position="top-right" richColors />
      </AnimatePresence>
    </ThemeProvider>
  </React.StrictMode>
);
