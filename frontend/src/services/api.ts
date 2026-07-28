import axios from 'axios';
import type { 
  DashboardStats, 
  TrainingJob, 
  Dataset, 
  Model, 
  AIProvider, 
  Conversation,
  SystemSettings,
  LogEntry,
  KnowledgeGap,
  AgentTask,
  EvaluationRun
} from '@/types';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Dashboard
export const dashboardService = {
  getStats: () => api.get<DashboardStats>('/dashboard'),
  getActivity: () => api.get('/dashboard/activity'),
  getMetrics: () => api.get('/dashboard/metrics'),
};

// Training
export const trainingService = {
  getJobs: () => api.get<TrainingJob[]>('/training/jobs'),
  getJob: (id: string) => api.get<TrainingJob>(`/training/jobs/${id}`),
  startJob: (config: Record<string, unknown>) => api.post<TrainingJob>('/training/jobs', config),
  pauseJob: (id: string) => api.post(`/training/jobs/${id}/pause`),
  resumeJob: (id: string) => api.post(`/training/jobs/${id}/resume`),
  stopJob: (id: string) => api.post(`/training/jobs/${id}/stop`),
  cancelJob: (id: string) => api.post(`/training/jobs/${id}/cancel`),
};

// Datasets
export const datasetService = {
  getAll: () => api.get<Dataset[]>('/datasets'),
  getById: (id: string) => api.get<Dataset>(`/datasets/${id}`),
  create: (data: Partial<Dataset>) => api.post<Dataset>('/datasets', data),
  update: (id: string, data: Partial<Dataset>) => api.put<Dataset>(`/datasets/${id}`, data),
  delete: (id: string) => api.delete(`/datasets/${id}`),
  importDataset: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post<Dataset>('/datasets/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  exportDataset: (id: string) => api.get(`/datasets/${id}/export`, { responseType: 'blob' }),
};

// Data Sources
export const dataSourceService = {
  getAll: () => api.get('/data-sources'),
  getById: (id: string) => api.get(`/data-sources/${id}`),
  create: (data: Record<string, unknown>) => api.post('/data-sources', data),
  update: (id: string, data: Record<string, unknown>) => api.put(`/data-sources/${id}`, data),
  delete: (id: string) => api.delete(`/data-sources/${id}`),
  enable: (id: string) => api.post(`/data-sources/${id}/enable`),
  disable: (id: string) => api.post(`/data-sources/${id}/disable`),
  test: (id: string) => api.post(`/data-sources/${id}/test`),
  sync: (id: string) => api.post(`/data-sources/${id}/sync`),
};

// Models
export const modelService = {
  getAll: () => api.get<Model[]>('/models'),
  getById: (id: string) => api.get<Model>(`/models/${id}`),
  getByStatus: (status: string) => api.get<Model[]>(`/models?status=${status}`),
  promote: (id: string) => api.post(`/models/${id}/promote`),
  archive: (id: string) => api.post(`/models/${id}/archive`),
  rollback: (version: string) => api.post(`/models/rollback/${version}`),
  compare: (modelIds: string[]) => api.post('/models/compare', { model_ids: modelIds }),
};

// Evaluation
export const evaluationService = {
  getRuns: () => api.get<EvaluationRun[]>('/evaluation/runs'),
  createRun: (modelId: string, testSuiteId: string) => 
    api.post<EvaluationRun>('/evaluation/runs', { model_id: modelId, test_suite_id: testSuiteId }),
  getResults: (runId: string) => api.get(`/evaluation/runs/${runId}/results`),
};

// AI Providers
export const providerService = {
  getAll: () => api.get<AIProvider[]>('/providers'),
  getById: (id: string) => api.get<AIProvider>(`/providers/${id}`),
  create: (data: Partial<AIProvider>) => api.post<AIProvider>('/providers', data),
  update: (id: string, data: Partial<AIProvider>) => api.put<AIProvider>(`/providers/${id}`, data),
  delete: (id: string) => api.delete(`/providers/${id}`),
  enable: (id: string) => api.post(`/providers/${id}/enable`),
  disable: (id: string) => api.post(`/providers/${id}/disable`),
  test: (id: string) => api.post(`/providers/${id}/test`),
};

// AI Router
export const routerService = {
  getConfig: () => api.get('/router/config'),
  updateConfig: (config: Record<string, unknown>) => api.put('/router/config', config),
  getRules: () => api.get('/router/rules'),
  createRule: (rule: Record<string, unknown>) => api.post('/router/rules', rule),
  updateRule: (id: string, rule: Record<string, unknown>) => api.put(`/router/rules/${id}`, rule),
  deleteRule: (id: string) => api.delete(`/router/rules/${id}`),
};

// Conversations
export const conversationService = {
  getAll: () => api.get('/conversations'),
  getById: (id: string) => api.get(`/conversations/${id}`),
  create: (title?: string) => api.post('/conversations', { title }),
  sendMessage: (conversationId: string, message: string, attachments?: File[]) => {
    const formData = new FormData();
    formData.append('message', message);
    attachments?.forEach((file, i) => formData.append(`attachments`, file));
    return api.post(`/conversations/${conversationId}/messages`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  delete: (id: string) => api.delete(`/conversations/${id}`),
};

// Knowledge Gaps
export const knowledgeGapService = {
  getAll: () => api.get<KnowledgeGap[]>('/knowledge-gaps'),
  getById: (id: string) => api.get<KnowledgeGap>(`/knowledge-gaps/${id}`),
  resolve: (id: string) => api.post(`/knowledge-gaps/${id}/resolve`),
  ignore: (id: string) => api.post(`/knowledge-gaps/${id}/ignore`),
  research: (id: string) => api.post(`/knowledge-gaps/${id}/research`),
};

// Agents
export const agentService = {
  getTasks: () => api.get<AgentTask[]>('/agents/tasks'),
  createTask: (agentType: string, task: string) => 
    api.post<AgentTask>('/agents/tasks', { agent_type: agentType, task }),
  cancelTask: (id: string) => api.post(`/agents/tasks/${id}/cancel`),
};

// Settings
export const settingsService = {
  get: () => api.get<SystemSettings>('/settings'),
  update: (settings: Partial<SystemSettings>) => api.put<SystemSettings>('/settings', settings),
};

// Logs
export const logService = {
  getAll: (level?: string) => api.get<LogEntry[]>(`/logs${level ? `?level=${level}` : ''}`),
  getRecent: () => api.get<LogEntry[]>('/logs/recent'),
};

// System Control
export const systemControlService = {
  getStatus: () => api.get('/system/status'),
  stopAll: () => api.post('/system/stop-all'),
  pauseTraining: () => api.post('/system/pause-training'),
  freezeProduction: () => api.post('/system/freeze-production'),
  disableApiAccess: () => api.post('/system/disable-api-access'),
};

export default api;
