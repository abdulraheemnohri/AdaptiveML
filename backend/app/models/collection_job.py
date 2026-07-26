"""
Collection Job Model - Records of data collection jobs
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
import enum


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CollectionJob(Base):
    __tablename__ = "collection_jobs"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    source_id = Column(String(36), ForeignKey("data_sources.id"), nullable=False, index=True)
    name = Column(String(255))
    description = Column(Text)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, index=True)
    config = Column(JSON, default={})
    items_collected = Column(Integer, default=0)
    items_processed = Column(Integer, default=0)
    items_failed = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    trust_score = Column(Float, default=0.0)
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    duration = Column(Integer)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    source = relationship("DataSource", backref="collection_jobs")
    
    def __repr__(self):
        return f"<CollectionJob(id={self.id}, source_id={self.source_id}, status={self.status.value})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "source_id": self.source_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "config": self.config or {},
            "items_collected": self.items_collected,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
            "quality_score": self.quality_score,
            "trust_score": self.trust_score,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def get_progress(self) -> dict:
        total = self.items_collected + self.items_failed
        success_rate = (self.items_collected / total) * 100 if total > 0 else 0.0
        return {
            "items_collected": self.items_collected,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
            "success_rate": success_rate,
            "quality_score": self.quality_score,
            "trust_score": self.trust_score,
        }