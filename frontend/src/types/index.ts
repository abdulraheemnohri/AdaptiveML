export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'viewer';
  created_at: string;
}

export interface DashboardStats {
  current_mode: 'training' | 'serving';
  production_model: string;
  model_version: string;
  training_status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  serving_status: 'active' | 'inactive';
  knowledge_growth: number;
  forgetting_score: number;
  model_quality: number;
  dataset_count: number;
  active_jobs: number;
  gpu_usage: number;
  ram_usage: number;
  storage_used: number;
  storage_total: number;
}

export interface TrainingJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  epoch: number;
  total_epochs: number;
  loss: number;
  val_loss: number;
  learning_rate: number;
  gpu_usage: number;
  vram_usage: number;
  eta_seconds: number;
  started_at: string;
  completed_at: string | null;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  version: string;
  sample_count: number;
  token_count: number;
  quality_score: number;
  trust_score: number;
  duplicate_count: number;
  languages: string[];
  sources: DataSource[];
  created_at: string;
  updated_at: string;
  status: 'draft' | 'processing' | 'ready' | 'archived';
}

export interface DataSource {
  id: string;
  name: string;
  type: 'website' | 'rss' | 'youtube' | 'file' | 'database' | 'github' | 'api';
  url: string;
  enabled: boolean;
  last_sync: string | null;
  next_sync: string | null;
  status: 'active' | 'error' | 'disabled';
}

export interface Model {
  id: string;
  name: string;
  version: string;
  parent_model_id: string | null;
  status: 'draft' | 'training' | 'candidate' | 'testing' | 'approved' | 'production' | 'archived';
  dataset_id: string;
  dataset_version: string;
  training_config: Record<string, unknown>;
  hardware: string;
  training_date: string;
  benchmark_results: BenchmarkResults;
  forgetting_score: number;
  safety_score: number;
  created_at: string;
}

export interface BenchmarkResults {
  accuracy: number;
  quality: number;
  reasoning: number;
  math: number;
  coding: number;
  vision: number;
  audio: number;
  speech: number;
  multilingual: number;
  safety: number;
}

export interface EvaluationRun {
  id: string;
  model_id: string;
  test_suite_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  results: TestResult[];
  overall_score: number;
  started_at: string;
  completed_at: string | null;
}

export interface TestResult {
  test_name: string;
  category: string;
  score: number;
  passed: boolean;
  details: Record<string, unknown>;
}

export interface AIProvider {
  id: string;
  name: string;
  type: 'openai' | 'anthropic' | 'gemini' | 'qwen' | 'deepseek' | 'mistral' | 'custom';
  api_key: string;
  endpoint: string;
  enabled: boolean;
  priority: number;
  models: string[];
}

export interface RoutingRule {
  id: string;
  name: string;
  condition: string;
  target: 'local' | string;
  priority: number;
  enabled: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  model_id: string;
  provider_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  attachments?: Attachment[];
  created_at: string;
}

export interface Attachment {
  type: 'image' | 'audio' | 'video' | 'file';
  url: string;
  name: string;
  size: number;
}

export interface KnowledgeGap {
  id: string;
  topic: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  source: 'user_request' | 'failed_response' | 'low_confidence' | 'benchmark';
  status: 'open' | 'researching' | 'resolved' | 'ignored';
  created_at: string;
}

export interface SystemSettings {
  theme: 'dark' | 'light' | 'auto';
  language: string;
  startup_mode: 'training' | 'serving';
  default_mode: 'training' | 'serving';
  notifications_enabled: boolean;
  training: TrainingSettings;
  continual_learning: ContinualLearningSettings;
  evaluation: EvaluationSettings;
  serving: ServingSettings;
  privacy: PrivacySettings;
}

export interface TrainingSettings {
  base_model: string;
  training_directory: string;
  dataset_directory: string;
  checkpoint_frequency: number;
  auto_training: boolean;
}

export interface ContinualLearningSettings {
  replay_ratio: number;
  replay_buffer_size: number;
  distillation_weight: number;
  ewc_strength: number;
  forgetting_threshold: number;
}

export interface EvaluationSettings {
  automatic_testing: boolean;
  test_frequency: number;
  required_benchmark_score: number;
  regression_threshold: number;
}

export interface ServingSettings {
  default_option: 'local' | 'api' | 'local_first' | 'api_first' | 'automatic';
  local_model_version: string;
  gpu_limit: number;
  context_length: number;
}

export interface PrivacySettings {
  local_only: boolean;
  send_files: boolean;
  send_images: boolean;
  send_audio: boolean;
  send_video: boolean;
  send_history: boolean;
}

export interface LogEntry {
  id: string;
  level: 'debug' | 'info' | 'warning' | 'error' | 'critical';
  message: string;
  source: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface AgentTask {
  id: string;
  agent_type: 'research' | 'verification' | 'fact_check' | 'synthesis';
  task: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result: string | null;
  created_at: string;
  completed_at: string | null;
}
