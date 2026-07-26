"""
Data Source Model - Defines where data comes from
"""

from sqlalchemy import Column, String, Text, Boolean, JSON, DateTime, Enum
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
import enum


class DataSourceType(str, enum.Enum):
    WEB = "web"
    RSS = "rss"
    YOUTUBE = "youtube"
    PDF = "pdf"
    DOCUMENT = "document"
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    SQLite = "sqlite"
    POSTGRESQL = "postgresql"
    GITHUB = "github"
    GIT = "git"
    LOCAL_FOLDER = "local_folder"
    CLOUD_STORAGE = "cloud_storage"
    API = "api"
    CUSTOM = "custom"


class DataSource(Base):
    __tablename__ = "data_sources"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    source_type = Column(Enum(DataSourceType), nullable=False, index=True)
    url = Column(String(1024))
    path = Column(String(1024))
    config = Column(JSON, default={})
    enabled = Column(Boolean, default=True)
    schedule = Column(String(100))
    last_collected = Column(DateTime(timezone=True))
    next_collection = Column(DateTime(timezone=True))
    total_collected = Column(Integer, default=0)
    total_processed = Column(Integer, default=0)
    total_failed = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    trust_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<DataSource(id={self.id}, name={self.name}, type={self.source_type.value})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type.value,
            "url": self.url,
            "path": self.path,
            "config": self.config or {},
            "enabled": self.enabled,
            "schedule": self.schedule,
            "last_collected": self.last_collected.isoformat() if self.last_collected else None,
            "next_collection": self.next_collection.isoformat() if self.next_collection else None,
            "total_collected": self.total_collected,
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "quality_score": self.quality_score,
            "trust_score": self.trust_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }