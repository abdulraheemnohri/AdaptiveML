"""
Data Sample Model - Individual data items within a dataset
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
import enum


class DataQualityScore(str, enum.Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    BAD = "bad"


class DataSample(Base):
    __tablename__ = "data_samples"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    content = Column(Text)
    raw_content = Column(Text)
    metadata = Column(JSON, default={})
    modality = Column(JSON, default=[])
    language = Column(String(10))
    quality_score = Column(Float, default=0.0)
    quality_category = Column(Enum(DataQualityScore), default=DataQualityScore.POOR)
    relevance_score = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    freshness_score = Column(Float, default=0.0)
    duplication_score = Column(Float, default=100.0)
    safety_score = Column(Float, default=0.0)
    trust_score = Column(Float, default=0.0)
    is_poisoned = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    is_quarantined = Column(Boolean, default=False)
    verification_notes = Column(Text)
    processing_errors = Column(JSON, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    dataset = relationship("Dataset", backref="samples")
    
    def __repr__(self):
        return f"<DataSample(id={self.id}, dataset_id={self.dataset_id}, quality={self.quality_score})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "content": self.content,
            "raw_content": self.raw_content,
            "metadata": self.metadata or {},
            "modality": self.modality or [],
            "language": self.language,
            "quality_score": self.quality_score,
            "quality_category": self.quality_category.value,
            "relevance_score": self.relevance_score,
            "confidence_score": self.confidence_score,
            "freshness_score": self.freshness_score,
            "duplication_score": self.duplication_score,
            "safety_score": self.safety_score,
            "trust_score": self.trust_score,
            "is_poisoned": self.is_poisoned,
            "is_verified": self.is_verified,
            "is_quarantined": self.is_quarantined,
            "verification_notes": self.verification_notes,
            "processing_errors": self.processing_errors or [],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def get_overall_score(self) -> float:
        scores = [
            self.quality_score,
            self.relevance_score,
            self.confidence_score,
            self.freshness_score,
            self.safety_score,
            self.trust_score,
        ]
        return sum(scores) / len(scores) if scores else 0.0