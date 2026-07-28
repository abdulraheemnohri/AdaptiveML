"""
Application Settings

Configuration management using Pydantic Settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    APP_NAME: str = "Adaptive Omni ML"
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"
    
    # Backend
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "sqlite:///./adaptive_omni_ml.db"
    
    # Model Settings
    BASE_MODEL: str = "Qwen/Qwen2.5-Omni-3B"
    MODEL_DIR: Path = Path("./models")
    CHECKPOINT_DIR: Path = Path("./checkpoints")
    ADAPTER_DIR: Path = Path("./adapters")
    
    # Training Settings
    TRAINING_BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 4
    MAX_SEQ_LENGTH: int = 2048
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    
    # Continual Learning
    REPLAY_RATIO: float = 0.3
    REPLAY_BUFFER_SIZE: int = 10000
    EWC_LAMBDA: float = 1000.0
    DISTILLATION_WEIGHT: float = 0.5
    FORGETTING_THRESHOLD: float = 0.02
    
    # Evaluation
    AUTO_EVALUATION: bool = True
    REGRESSION_THRESHOLD: float = 0.01
    SAFETY_THRESHOLD: float = 0.95
    QUALITY_THRESHOLD: float = 0.90
    
    # Serving
    DEFAULT_SERVING_MODE: str = "local"
    LOCAL_MODEL_VERSION: str = "latest"
    INFERENCE_DEVICE: str = "cuda"
    INFERENCE_BATCH_SIZE: int = 1
    MAX_CONTEXT_LENGTH: int = 4096
    
    # AI Providers
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"
    
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_BASE_URL: str = "https://generativelanguage.googleapis.com"
    
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    
    MISTRAL_API_KEY: Optional[str] = None
    MISTRAL_BASE_URL: str = "https://api.mistral.ai/v1"
    
    QWEN_API_KEY: Optional[str] = None
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    REQUEST_TIMEOUT: int = 120
    MAX_RETRIES: int = 3
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("./logs/app.log")
    
    # Security
    API_KEY_ENCRYPTION_KEY: str = "change-me-in-production"
    ENABLE_AUTH: bool = False
    JWT_SECRET_KEY: str = "change-me-in-production"
    
    # Storage
    DATASET_DIR: Path = Path("./datasets")
    EXPERIMENT_DIR: Path = Path("./experiments")
    EVALUATION_REPORT_DIR: Path = Path("./evaluation_reports")
    BACKUP_DIR: Path = Path("./backups")
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    @property
    def model_dir_abs(self) -> Path:
        """Get absolute path for model directory."""
        return self.MODEL_DIR.absolute()
    
    @property
    def checkpoint_dir_abs(self) -> Path:
        """Get absolute path for checkpoint directory."""
        return self.CHECKPOINT_DIR.absolute()
    
    @property
    def adapter_dir_abs(self) -> Path:
        """Get absolute path for adapter directory."""
        return self.ADAPTER_DIR.absolute()
    
    @property
    def dataset_dir_abs(self) -> Path:
        """Get absolute path for dataset directory."""
        return self.DATASET_DIR.absolute()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
