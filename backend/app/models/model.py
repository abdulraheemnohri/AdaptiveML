"""
Model and Model Version Models - AI models and their versions
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
import enum


class ModelStatus(str, enum.Enum):
    DRAFT = "draft"
    TRAINING = "training"
    TESTING = "testing"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class ModelType(str, enum.Enum):
    BASE = "base"
    FINE_TUNED = "fine_tuned"
    ADAPTER = "adapter"
    FUSED = "fused"
    DISTILLED = "distilled"


class Model(Base):
    __tablename__ = "models"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    model_type = Column(Enum(ModelType), default=ModelType.BASE, index=True)
    base_model_id = Column(String(36), ForeignKey("models.id"))
    status = Column(Enum(ModelStatus), default=ModelStatus.DRAFT, index=True)
    config = Column(JSON, default={})
    hyperparameters = Column(JSON, default={})
    training_dataset_id = Column(String(36), ForeignKey("datasets.id"))
    training_started_at = Column(DateTime(timezone=True))
    training_completed_at = Column(DateTime(timezone=True))
    training_duration = Column(Integer)
    overall_score = Column(Float)
    reasoning_score = Column(Float)
    coding_score = Column(Float)
    multimodal_score = Column(Float)
    language_score = Column(Float)
    safety_score = Column(Float)
    factuality_score = Column(Float)
    retention_score = Column(Float)
    forgetting_score = Column(Float)
    parameter_count = Column(Integer)
    quantization = Column(String(50))
    device = Column(String(50))
    file_path = Column(String(1024))
    file_size = Column(Integer)
    version = Column(String(50), default="1.0.0")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    base_model = relationship("Model", remote_side=[id], backref="derived_models")
    training_dataset = relationship("Dataset", foreign_keys=[training_dataset_id])
    versions = relationship("ModelVersion", backref="model", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", backref="model", cascade="all, delete-orphan")
    training_sessions = relationship("TrainingSession", backref="model", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Model(id={self.id}, name={self.name}, version={self.version}, status={self.status.value})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type.value,
            "base_model_id": self.base_model_id,
            "status": self.status.value,
            "config": self.config or {},
            "hyperparameters": self.hyperparameters or {},
            "training_dataset_id": self.training_dataset_id,
            "training_started_at": self.training_started_at.isoformat() if self.training_started_at else None,
            "training_completed_at": self.training_completed_at.isoformat() if self.training_completed_at else None,
            "training_duration": self.training_duration,
            "overall_score": self.overall_score,
            "reasoning_score": self.reasoning_score,
            "coding_score": self.coding_score,
            "multimodal_score": self.multimodal_score,
            "language_score": self.language_score,
            "safety_score": self.safety_score,
            "factuality_score": self.factuality_score,
            "retention_score": self.retention_score,
            "forgetting_score": self.forgetting_score,
            "parameter_count": self.parameter_count,
            "quantization": self.quantization,
            "device": self.device,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def get_performance_summary(self) -> dict:
        return {
            "overall": self.overall_score,
            "reasoning": self.reasoning_score,
            "coding": self.coding_score,
            "multimodal": self.multimodal_score,
            "language": self.language_score,
            "safety": self.safety_score,
            "factuality": self.factuality_score,
            "retention": self.retention_score,
            "forgetting": self.forgetting_score,
        }