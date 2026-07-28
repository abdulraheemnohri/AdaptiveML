"""
Adaptive Omni ML Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "Adaptive Omni ML"
    DEBUG: bool = True
    VERSION: str = "1.0.0"
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/adaptive_omni_ml"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Base Model
    BASE_MODEL_NAME: str = "Qwen/Qwen2.5-Omni-3B"
    BASE_MODEL_PATH: str = "./models/base"
    
    # Directories
    MODELS_DIR: str = "./models"
    DATASETS_DIR: str = "./datasets"
    CHECKPOINTS_DIR: str = "./checkpoints"
    ADAPTERS_DIR: str = "./adapters"
    LOGS_DIR: str = "./logs"
    BACKUPS_DIR: str = "./backups"
    CONFIGS_DIR: str = "./configs"
    
    # Training
    TRAINING_BATCH_SIZE: int = 4
    TRAINING_LEARNING_RATE: float = 2e-4
    TRAINING_EPOCHS: int = 3
    TRAINING_GRADIENT_ACCUMULATION_STEPS: int = 4
    CHECKPOINT_FREQUENCY: int = 100
    
    # Continual Learning
    REPLAY_BUFFER_SIZE: int = 10000
    REPLAY_RATIO: float = 0.2
    DISTILLATION_WEIGHT: float = 0.5
    EWC_STRENGTH: float = 1000.0
    PROTECTED_CAPABILITY_WEIGHT: float = 0.3
    
    # Anti-Forgetting
    FORGETTING_THRESHOLD: float = 0.02  # 2%
    REGRESSION_THRESHOLD: float = 0.01  # 1%
    QUALITY_GATE_THRESHOLD: float = 0.90  # 90%
    SAFETY_GATE_THRESHOLD: float = 0.95  # 95%
    
    # Serving
    DEFAULT_SERVING_MODE: str = "local_first"  # local_only, api_only, local_first, api_first, automatic, manual
    INFERENCE_DEVICE: str = "cuda"  # cuda, cpu, mps
    MAX_CONTEXT_LENGTH: int = 8192
    INFERENCE_BATCH_SIZE: int = 1
    QUANTIZATION: str = "int4"  # int4, int8, fp16, fp32
    
    # AI Providers
    ENABLED_PROVIDERS: List[str] = ["openai", "anthropic", "gemini", "qwen", "deepseek"]
    DEFAULT_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Privacy
    LOCAL_ONLY_MODE: bool = False
    ALLOW_API_FILE_UPLOAD: bool = True
    ALLOW_API_IMAGE_UPLOAD: bool = True
    ALLOW_API_HISTORY_SEND: bool = False
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
