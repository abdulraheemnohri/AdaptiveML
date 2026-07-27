"""
Training Session Model - Records of model training sessions
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
import enum


class TrainingStatus(str, enum.Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingSession(Base):
    __tablename__ = "training_sessions"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    training_dataset_id = Column(String(36), ForeignKey("datasets.id"))
    validation_dataset_id = Column(String(36), ForeignKey("datasets.id"))
    status = Column(Enum(TrainingStatus), default=TrainingStatus.PENDING, index=True)
    config = Column(JSON, default={})
    hyperparameters = Column(JSON, default={})
    epochs = Column(Integer, default=3)
    batch_size = Column(Integer, default=4)
    learning_rate = Column(Float, default=2e-5)
    replay_ratio = Column(Float, default=0.2)
    ewc_strength = Column(Float, default=0.1)
    distillation_strength = Column(Float, default=0.5)
    start_epoch = Column(Integer, default=0)
    current_epoch = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    current_step = Column(Integer, default=0)
    best_score = Column(Float)
    best_epoch = Column(Integer)
    loss = Column(Float)
    val_loss = Column(Float)
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    duration = Column(Integer)
    gpu_utilization = Column(Float)
    memory_used = Column(Integer)
    error_message = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    model = relationship("Model", backref="training_sessions")
    training_dataset = relationship("Dataset", foreign_keys=[training_dataset_id])
    validation_dataset = relationship("Dataset", foreign_keys=[validation_dataset_id])
    
    def __repr__(self):
        return f"<TrainingSession(id={self.id}, model_id={self.model_id}, status={self.status.value})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "training_dataset_id": self.training_dataset_id,
            "validation_dataset_id": self.validation_dataset_id,
            "status": self.status.value,
            "config": self.config or {},
            "hyperparameters": self.hyperparameters or {},
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "replay_ratio": self.replay_ratio,
            "ewc_strength": self.ewc_strength,
            "distillation_strength": self.distillation_strength,
            "start_epoch": self.start_epoch,
            "current_epoch": self.current_epoch,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "loss": self.loss,
            "val_loss": self.val_loss,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "gpu_utilization": self.gpu_utilization,
            "memory_used": self.memory_used,
            "error_message": self.error_message,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def get_progress(self) -> dict:
        progress = 0.0
        if self.epochs > 0:
            progress = (self.current_epoch / self.epochs) * 100
        return {
            "epoch_progress": progress,
            "step_progress": (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0,
            "current_epoch": self.current_epoch,
            "total_epochs": self.epochs,
        }