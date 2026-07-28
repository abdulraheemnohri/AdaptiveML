import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from '@/components/Sidebar';
import { Header } from '@/components/UI';
import { useAppStore } from '@/stores/appStore';
import { clsx } from 'clsx';

export const MainLayout: React.FC = () => {
  const { sidebarCollapsed } = useAppStore();

  return (
    <div className="min-h-screen bg-dark-950">
      <Sidebar />
      
      <main
        className={clsx(
          'transition-all duration-300 min-h-screen',
          sidebarCollapsed ? 'ml-16' : 'ml-64'
        )}
      >
        <Outlet />
      </main>
    </div>
  );
};

export const PageLayout: React.FC<{
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
}> = ({ title, subtitle, actions, children }) => {
  return (
    <div className="p-6">
      <Header title={title} subtitle={subtitle} actions={actions} />
      <div className="mt-6">{children}</div>
    </div>
  );
};
