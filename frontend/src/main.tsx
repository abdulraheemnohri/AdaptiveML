import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import { MainLayout } from '@/layouts/MainLayout';
import { DashboardPage } from '@/pages/Dashboard';
import '@/styles/globals.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<DashboardPage />} />
            <Route path="training/*" element={<div className="p-6"><h1 className="text-2xl text-white">Training Pages</h1></div>} />
            <Route path="serving/*" element={<div className="p-6"><h1 className="text-2xl text-white">Serving Pages</h1></div>} />
            <Route path="system/*" element={<div className="p-6"><h1 className="text-2xl text-white">System Pages</h1></div>} />
            <Route path="settings" element={<div className="p-6"><h1 className="text-2xl text-white">Settings</h1></div>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
