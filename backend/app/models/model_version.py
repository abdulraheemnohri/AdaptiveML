"""
Model Version Model - Version history for models
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
import enum


class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    parent_version_id = Column(String(36), ForeignKey("model_versions.id"))
    changelog = Column(Text)
    commit_hash = Column(String(64))
    performance_scores = Column(JSON, default={})
    file_path = Column(String(1024))
    file_size = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    model = relationship("Model", backref="versions")
    parent_version = relationship("ModelVersion", remote_side=[id], backref="child_versions")
    
    def __repr__(self):
        return f"<ModelVersion(id={self.id}, model_id={self.model_id}, version={self.version})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "version": self.version,
            "parent_version_id": self.parent_version_id,
            "changelog": self.changelog,
            "commit_hash": self.commit_hash,
            "performance_scores": self.performance_scores or {},
            "file_path": self.file_path,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
        }