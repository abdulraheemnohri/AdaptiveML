import { create } from 'zustand'

export interface SystemHealth {
  cpu: number
  gpu: number
  vram: number
  ram: number
  disk: number
  trainingSpeed: number
  modelLatency: number
}

export interface ModelInfo {
  name: string
  version: string
  status: 'draft' | 'training' | 'testing' | 'candidate' | 'approved' | 'production' | 'archived' | 'rejected'
  adapters: string[]
  lastUpdated: string
}

export interface LearningProgress {
  knowledgeGrowth: number
  skillsGrowth: number
  languagesCount: number
  newDataCount: number
  verifiedFacts: number
  learningTasks: number
  forgettingScore: number
}

export interface DashboardState {
  systemHealth: SystemHealth
  currentModel: ModelInfo
  learningProgress: LearningProgress
  activeExperiments: number
  dataPipelineStatus: 'idle' | 'collecting' | 'processing' | 'validated' | 'failed'
  notifications: Array<{ id: string; type: 'info' | 'warning' | 'error' | 'success'; message: string }>
  updateSystemHealth: (health: Partial<SystemHealth>) => void
  updateModel: (model: Partial<ModelInfo>) => void
  updateLearningProgress: (progress: Partial<LearningProgress>) => void
  addNotification: (notification: { type: 'info' | 'warning' | 'error' | 'success'; message: string }) => void
  removeNotification: (id: string) => void
}

const initialSystemHealth: SystemHealth = {
  cpu: 15,
  gpu: 0,
  vram: 2048,
  ram: 4096,
  disk: 45,
  trainingSpeed: 0,
  modelLatency: 0,
}

const initialModel: ModelInfo = {
  name: 'Qwen2.5-Omni-3B',
  version: 'v1.0.0',
  status: 'production',
  adapters: [],
  lastUpdated: new Date().toISOString(),
}

const initialLearningProgress: LearningProgress = {
  knowledgeGrowth: 0,
  skillsGrowth: 0,
  languagesCount: 1,
  newDataCount: 0,
  verifiedFacts: 0,
  learningTasks: 0,
  forgettingScore: 0,
}

export const useDashboardStore = create<DashboardState>((set) => ({
  systemHealth: initialSystemHealth,
  currentModel: initialModel,
  learningProgress: initialLearningProgress,
  activeExperiments: 0,
  dataPipelineStatus: 'idle',
  notifications: [],
  
  updateSystemHealth: (health) =>
    set((state) => ({
      systemHealth: { ...state.systemHealth, ...health },
    })),
  
  updateModel: (model) =>
    set((state) => ({
      currentModel: { ...state.currentModel, ...model, lastUpdated: new Date().toISOString() },
    })),
  
  updateLearningProgress: (progress) =>
    set((state) => ({
      learningProgress: { ...state.learningProgress, ...progress },
    })),
  
  addNotification: (notification) =>
    set((state) => ({
      notifications: [
        ...state.notifications,
        { id: Date.now().toString(), ...notification },
      ],
    })),
  
  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),
}))
