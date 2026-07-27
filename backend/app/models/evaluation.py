"""
Evaluation Model - Records of model evaluations
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
import enum


class EvaluationType(str, enum.Enum):
    UNIT_TEST = "unit_test"
    CAPABILITY_TEST = "capability"
    MULTIMODAL_TEST = "multimodal"
    REGRESSION_TEST = "regression"
    ANTI_FORGETTING_TEST = "anti_forgetting"
    SAFETY_TEST = "safety"
    PERFORMANCE_TEST = "performance"
    HUMAN_EVALUATION = "human"
    CUSTOM_TEST = "custom"


class Evaluation(Base):
    __tablename__ = "evaluations"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False, index=True)
    evaluation_type = Column(Enum(EvaluationType), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    test_suite = Column(String(255))
    test_config = Column(JSON, default={})
    metrics = Column(JSON, default={})
    overall_score = Column(Float)
    passed = Column(Boolean, default=False)
    pass_threshold = Column(Float, default=70.0)
    results = Column(JSON, default={})
    error_message = Column(Text)
    evaluated_by = Column(String(255))
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    duration = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    model = relationship("Model", backref="evaluations")
    
    def __repr__(self):
        return f"<Evaluation(id={self.id}, model_id={self.model_id}, type={self.evaluation_type.value}, score={self.overall_score})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "evaluation_type": self.evaluation_type.value,
            "name": self.name,
            "description": self.description,
            "test_suite": self.test_suite,
            "test_config": self.test_config or {},
            "metrics": self.metrics or {},
            "overall_score": self.overall_score,
            "passed": self.passed,
            "pass_threshold": self.pass_threshold,
            "results": self.results or {},
            "error_message": self.error_message,
            "evaluated_by": self.evaluated_by,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }