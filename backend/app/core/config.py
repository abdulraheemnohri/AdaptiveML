"""
Application Configuration using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Adaptive Omni ML Platform"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "A Self-Evolving Multimodal AI Learning Platform"
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Server
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./adaptive_omni.db"
    
    # Model Storage
    MODEL_STORAGE_PATH: str = "./models"
    BASE_MODEL_NAME: str = "Qwen2.5-Omni-3B"
    BASE_MODEL_PATH: Optional[str] = None
    
    # Data Storage
    DATA_STORAGE_PATH: str = "./data"
    RAW_DATA_PATH: str = "./data/raw"
    PROCESSED_DATA_PATH: str = "./data/processed"
    VALIDATED_DATA_PATH: str = "./data/validated"
    QUARANTINE_DATA_PATH: str = "./data/quarantine"
    REPLAY_BUFFER_PATH: str = "./data/replay"
    
    # Training
    MAX_CONTEXT_LENGTH: int = 4096
    TRAINING_BATCH_SIZE: int = 4
    TRAINING_EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5
    REPLAY_RATIO: float = 0.2
    EWC_STRENGTH: float = 0.1
    DISTILLATION_STRENGTH: float = 0.5
    
    # Anti-Forgetting
    FORGETTING_THRESHOLD: float = 0.05
    MIN_RETENTION_SCORE: float = 0.95
    
    # Safety
    SAFETY_THRESHOLD: float = 0.9
    POISONING_DETECTION_ENABLED: bool = True
    TRUST_THRESHOLD: float = 0.7
    
    # Automation
    AUTO_COLLECTION_ENABLED: bool = False
    AUTO_TRAINING_ENABLED: bool = False
    AUTO_EVALUATION_ENABLED: bool = False
    AUTO_PROMOTION_ENABLED: bool = False
    AUTO_ROLLBACK_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True)
os.makedirs(settings.DATA_STORAGE_PATH, exist_ok=True)
os.makedirs(settings.RAW_DATA_PATH, exist_ok=True)
os.makedirs(settings.PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(settings.VALIDATED_DATA_PATH, exist_ok=True)
os.makedirs(settings.QUARANTINE_DATA_PATH, exist_ok=True)
os.makedirs(settings.REPLAY_BUFFER_PATH, exist_ok=True)