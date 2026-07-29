import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { clsx } from 'clsx';

const navigation = {
  training: [
    { name: 'Training Dashboard', href: '/training', icon: '🧠' },
    { name: 'Data Sources', href: '/training/data-sources', icon: '📡' },
    { name: 'Datasets', href: '/training/datasets', icon: '📊' },
    { name: 'Research Centre', href: '/training/research', icon: '🔬' },
    { name: 'Knowledge Gaps', href: '/training/knowledge-gaps', icon: '🔍' },
    { name: 'Continual Learning', href: '/training/continual-learning', icon: '🔄' },
    { name: 'Anti-Forgetting', href: '/training/anti-forgetting', icon: '🛡️' },
    { name: 'Evaluation', href: '/training/evaluation', icon: '📈' },
    { name: 'Model Registry', href: '/training/models', icon: '📦' },
  ],
  serving: [
    { name: 'Chat', href: '/serving/chat', icon: '💬' },
    { name: 'Local Model', href: '/serving/local', icon: '🖥️' },
    { name: 'AI Providers', href: '/serving/providers', icon: '☁️' },
    { name: 'AI Router', href: '/serving/router', icon: '🔀' },
    { name: 'Compare Models', href: '/serving/compare', icon: '⚖️' },
  ],
  system: [
    { name: 'Dashboard', href: '/', icon: '🏠' },
    { name: 'System Control', href: '/system/control', icon: '🎛️' },
    { name: 'Monitoring', href: '/system/monitoring', icon: '📊' },
    { name: 'Agents', href: '/system/agents', icon: '🤖' },
    { name: 'Logs', href: '/system/logs', icon: '📜' },
    { name: 'Settings', href: '/settings', icon: '⚙️' },
  ],
};

import { useAppStore } from '@/stores/appStore';

export const Sidebar: React.FC = () => {
  const location = useLocation();
  const { currentMode, setCurrentMode, sidebarCollapsed, setSidebarCollapsed } = 
    useAppStore();

  const isActive = (href: string) => {
    return location.pathname === href || location.pathname.startsWith(href + '/');
  };

  return (
    <aside
      className={clsx(
        'fixed left-0 top-0 h-full bg-dark-900 border-r border-dark-700 transition-all duration-300 z-40',
        sidebarCollapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-dark-700">
        {!sidebarCollapsed && (
          <span className="text-lg font-bold text-white">Adaptive Omni ML</span>
        )}
        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="p-2 hover:bg-dark-800 rounded-lg transition-colors"
        >
          {sidebarCollapsed ? '→' : '←'}
        </button>
      </div>

      {/* Mode Switcher */}
      <div className="p-4 border-b border-dark-700">
        <div className="flex gap-2">
          <button
            onClick={() => setCurrentMode('training')}
            className={clsx(
              'flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors',
              currentMode === 'training'
                ? 'bg-primary-600 text-white'
                : 'bg-dark-800 text-dark-300 hover:bg-dark-700'
            )}
          >
            {!sidebarCollapsed && '🧠 Training'}
          </button>
          <button
            onClick={() => setCurrentMode('serving')}
            className={clsx(
              'flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors',
              currentMode === 'serving'
                ? 'bg-primary-600 text-white'
                : 'bg-dark-800 text-dark-300 hover:bg-dark-700'
            )}
          >
            {!sidebarCollapsed && '🚀 Serving'}
          </button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Training Section */}
        {currentMode === 'training' && (
          <div>
            {!sidebarCollapsed && (
              <h3 className="text-xs font-semibold text-dark-500 uppercase tracking-wider mb-3">
                Training Mode
              </h3>
            )}
            <ul className="space-y-1">
              {navigation.training.map((item) => (
                <li key={item.name}>
                  <Link
                    to={item.href}
                    className={clsx(
                      'flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors',
                      isActive(item.href)
                        ? 'bg-primary-900/50 text-primary-400'
                        : 'text-dark-300 hover:bg-dark-800 hover:text-white'
                    )}
                  >
                    <span className="text-lg">{item.icon}</span>
                    {!sidebarCollapsed && <span>{item.name}</span>}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Serving Section */}
        {currentMode === 'serving' && (
          <div>
            {!sidebarCollapsed && (
              <h3 className="text-xs font-semibold text-dark-500 uppercase tracking-wider mb-3">
                Serving Mode
              </h3>
            )}
            <ul className="space-y-1">
              {navigation.serving.map((item) => (
                <li key={item.name}>
                  <Link
                    to={item.href}
                    className={clsx(
                      'flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors',
                      isActive(item.href)
                        ? 'bg-primary-900/50 text-primary-400'
                        : 'text-dark-300 hover:bg-dark-800 hover:text-white'
                    )}
                  >
                    <span className="text-lg">{item.icon}</span>
                    {!sidebarCollapsed && <span>{item.name}</span>}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* System Section */}
        <div>
          {!sidebarCollapsed && (
            <h3 className="text-xs font-semibold text-dark-500 uppercase tracking-wider mb-3">
              System
            </h3>
          )}
          <ul className="space-y-1">
            {navigation.system.map((item) => (
              <li key={item.name}>
                <Link
                  to={item.href}
                  className={clsx(
                    'flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors',
                    isActive(item.href)
                      ? 'bg-primary-900/50 text-primary-400'
                      : 'text-dark-300 hover:bg-dark-800 hover:text-white'
                  )}
                >
                  <span className="text-lg">{item.icon}</span>
                  {!sidebarCollapsed && <span>{item.name}</span>}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </nav>

      {/* Status Indicator */}
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-dark-700">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          {!sidebarCollapsed && (
            <span className="text-xs text-dark-400">System Online</span>
          )}
        </div>
      </div>
    </aside>
  );
};
