"""
Dataset Model - Represents a collection of data samples
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
import enum


class DatasetStatus(str, enum.Enum):
    PENDING = "pending"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    VALIDATING = "validating"
    READY = "ready"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    source_id = Column(String(36), ForeignKey("data_sources.id"))
    status = Column(Enum(DatasetStatus), default=DatasetStatus.PENDING, index=True)
    file_path = Column(String(1024))
    file_size = Column(Integer)
    num_samples = Column(Integer, default=0)
    num_processed = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    trust_score = Column(Float, default=0.0)
    deduplication_score = Column(Float, default=100.0)
    language = Column(String(10))
    modality = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    tags = Column(JSON, default=[])
    version = Column(String(50), default="1.0.0")
    parent_dataset_id = Column(String(36), ForeignKey("datasets.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    source = relationship("DataSource", backref="datasets")
    parent_dataset = relationship("Dataset", remote_side=[id], backref="child_datasets")
    samples = relationship("DataSample", backref="dataset", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, name={self.name}, status={self.status.value})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source_id": self.source_id,
            "status": self.status.value,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "num_samples": self.num_samples,
            "num_processed": self.num_processed,
            "quality_score": self.quality_score,
            "trust_score": self.trust_score,
            "deduplication_score": self.deduplication_score,
            "language": self.language,
            "modality": self.modality,
            "metadata": self.metadata or {},
            "tags": self.tags or [],
            "version": self.version,
            "parent_dataset_id": self.parent_dataset_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }