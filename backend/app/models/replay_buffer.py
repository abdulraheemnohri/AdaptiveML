"""
Replay Buffer Model - For continual learning with experience replay
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime


class ReplayBuffer(Base):
    __tablename__ = "replay_buffers"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    model_id = Column(String(36), ForeignKey("models.id"))
    capacity = Column(Integer, default=1000)
    size = Column(Integer, default=0)
    sampling_strategy = Column(String(50), default="random")
    priority_weights = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    model = relationship("Model", backref="replay_buffers")
    samples = relationship("ReplaySample", backref="replay_buffer", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ReplayBuffer(id={self.id}, name={self.name}, size={self.size}/{self.capacity})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model_id": self.model_id,
            "capacity": self.capacity,
            "size": self.size,
            "sampling_strategy": self.sampling_strategy,
            "priority_weights": self.priority_weights or {},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def is_full(self) -> bool:
        return self.size >= self.capacity
    
    def get_utilization(self) -> float:
        if self.capacity > 0:
            return (self.size / self.capacity) * 100
        return 0.0


class ReplaySample(Base):
    __tablename__ = "replay_samples"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    replay_buffer_id = Column(String(36), ForeignKey("replay_buffers.id"), nullable=False, index=True)
    data_sample_id = Column(String(36), ForeignKey("data_samples.id"))
    data = Column(JSON, default={})
    priority = Column(Float, default=1.0)
    last_used = Column(DateTime(timezone=True))
    use_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    replay_buffer = relationship("ReplayBuffer", backref="samples")
    data_sample = relationship("DataSample", backref="replay_samples")
    
    def __repr__(self):
        return f"<ReplaySample(id={self.id}, buffer_id={self.replay_buffer_id}, priority={self.priority})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "replay_buffer_id": self.replay_buffer_id,
            "data_sample_id": self.data_sample_id,
            "data": self.data or {},
            "priority": self.priority,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "use_count": self.use_count,
            "created_at": self.created_at.isoformat(),
        }