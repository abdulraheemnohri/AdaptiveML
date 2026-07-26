"""
Application Configuration for Adaptive Omni ML Platform
"""

from pydantic import BaseSettings, Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings using Pydantic"""
    
    # Application
    APP_NAME: str = "Adaptive Omni ML Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./adaptive_omni.db",
        env="DATABASE_URL"
    )
    
    # PostgreSQL (Production)
    POSTGRES_USER: Optional[str] = Field(default=None, env="POSTGRES_USER")
    POSTGRES_PASSWORD: Optional[str] = Field(default=None, env="POSTGRES_PASSWORD")
    POSTGRES_HOST: Optional[str] = Field(default=None, env="POSTGRES_HOST")
    POSTGRES_PORT: Optional[int] = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_DB: Optional[str] = Field(default=None, env="POSTGRES_DB")
    
    # Qdrant Vector Database
    QDRANT_HOST: str = Field(default="localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(default=6333, env="QDRANT_PORT")
    QDRANT_COLLECTION_PREFIX: str = "adaptive_omni"
    
    # Neo4j Knowledge Graph
    NEO4J_URI: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    NEO4J_USER: str = Field(default="neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field(default="password", env="NEO4J_PASSWORD")
    
    # Model Configuration
    BASE_MODEL_NAME: str = "Qwen2.5-Omni-3B"
    MODEL_CACHE_DIR: str = "./models"
    MAX_MODEL_CACHE_SIZE: int = 100
    
    # Training Configuration
    TRAINING_OUTPUT_DIR: str = "./training_output"
    MAX_TRAINING_SESSIONS: int = 10
    
    # Data Configuration
    DATA_STORAGE_DIR: str = "./data"
    MAX_DATA_STORAGE_SIZE: int = 500
    
    # API Keys
    GITHUB_TOKEN: Optional[str] = Field(default=None, env="GITHUB_TOKEN")
    YOUTUBE_API_KEY: Optional[str] = Field(default=None, env="YOUTUBE_API_KEY")
    
    # Security
    SECRET_KEY: str = Field(
        default="change-me-in-production",
        env="SECRET_KEY"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

def get_settings() -> Settings:
    return settings