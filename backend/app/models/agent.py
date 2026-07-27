"""
Agent Model - AI agents for various tasks
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
import enum


class AgentType(str, enum.Enum):
    RESEARCH = "research"
    DATA_COLLECTOR = "data_collector"
    DATA_PROCESSOR = "data_processor"
    VERIFICATION = "verification"
    TEACHER = "teacher"
    CRITIC = "critic"
    EVALUATOR = "evaluator"
    FORGETTING_CHECKER = "forgetting_checker"
    SAFETY_CHECKER = "safety_checker"
    TRAINING_PLANNER = "training_planner"
    DEPLOYMENT_MANAGER = "deployment_manager"
    SUPERVISOR = "supervisor"
    CUSTOM = "custom"


class AgentStatus(str, enum.Enum):
    IDLE = "idle"
    WORKING = "working"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    name = Column(String(255), nullable=False, unique=True)
    agent_type = Column(Enum(AgentType), nullable=False, index=True)
    description = Column(Text)
    model_id = Column(String(36), ForeignKey("models.id"))
    config = Column(JSON, default={})
    status = Column(Enum(AgentStatus), default=AgentStatus.IDLE, index=True)
    current_task = Column(String(255))
    task_description = Column(Text)
    task_start_time = Column(DateTime(timezone=True))
    task_end_time = Column(DateTime(timezone=True))
    tasks_completed = Column(Integer, default=0)
    tasks_failed = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    last_error = Column(Text)
    capabilities = Column(JSON, default=[])
    permissions = Column(JSON, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    model = relationship("Model", backref="agents")
    
    def __repr__(self):
        return f"<Agent(id={self.id}, name={self.name}, type={self.agent_type.value}, status={self.status.value})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "agent_type": self.agent_type.value,
            "description": self.description,
            "model_id": self.model_id,
            "config": self.config or {},
            "status": self.status.value,
            "current_task": self.current_task,
            "task_description": self.task_description,
            "task_start_time": self.task_start_time.isoformat() if self.task_start_time else None,
            "task_end_time": self.task_end_time.isoformat() if self.task_end_time else None,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "success_rate": self.success_rate,
            "last_error": self.last_error,
            "capabilities": self.capabilities or [],
            "permissions": self.permissions or [],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }