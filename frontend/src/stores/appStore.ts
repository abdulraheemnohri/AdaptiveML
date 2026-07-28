import { create } from 'zustand';
import type { DashboardStats, TrainingJob, SystemSettings, LogEntry } from '@/types';

interface AppState {
  // Mode
  currentMode: 'training' | 'serving';
  setCurrentMode: (mode: 'training' | 'serving') => void;
  
  // Dashboard Stats
  dashboardStats: DashboardStats | null;
  setDashboardStats: (stats: DashboardStats) => void;
  
  // Settings
  settings: SystemSettings | null;
  setSettings: (settings: SystemSettings) => void;
  
  // WebSocket Connection
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;
  
  // Notifications
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  
  // Sidebar
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  
  // Command Palette
  commandPaletteOpen: boolean;
  setCommandPaletteOpen: (open: boolean) => void;
}

interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: string;
}

export const useAppStore = create<AppState>((set) => ({
  // Mode
  currentMode: 'training',
  setCurrentMode: (mode) => set({ currentMode: mode }),
  
  // Dashboard Stats
  dashboardStats: null,
  setDashboardStats: (stats) => set({ dashboardStats: stats }),
  
  // Settings
  settings: null,
  setSettings: (settings) => set({ settings }),
  
  // WebSocket Connection
  wsConnected: false,
  setWsConnected: (connected) => set({ wsConnected: connected }),
  
  // Notifications
  notifications: [],
  addNotification: (notification) => set((state) => ({
    notifications: [
      ...state.notifications,
      { ...notification, id: Math.random().toString(36).substr(2, 9), timestamp: new Date().toISOString() },
    ].slice(-10), // Keep last 10 notifications
  })),
  removeNotification: (id) => set((state) => ({
    notifications: state.notifications.filter((n) => n.id !== id),
  })),
  clearNotifications: () => set({ notifications: [] }),
  
  // Sidebar
  sidebarCollapsed: false,
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
  
  // Command Palette
  commandPaletteOpen: false,
  setCommandPaletteOpen: (open) => set({ commandPaletteOpen: open }),
}));
