"""
Dataset, DatasetVersion, and DatasetSample models.
"""

from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, Integer, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database.session import Base


class Dataset(Base):
    """Main dataset entity."""
    
    __tablename__ = "datasets"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft, active, archived
    is_locked: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Statistics
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    languages: Mapped[list] = mapped_column(JSON, default=list)
    
    # Quality metrics
    avg_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_trust_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    duplicate_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Source tracking
    source_ids: Mapped[list] = mapped_column(JSON, default=list)  # IDs of source documents
    
    # Metadata
    tags: Mapped[list] = mapped_column(JSON, default=list)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    samples: Mapped[list["DatasetSample"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name={self.name})>"


class DatasetVersion(Base):
    """Version snapshot of a dataset."""
    
    __tablename__ = "dataset_versions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Snapshot data
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)  # SHA256 of dataset
    
    # Change tracking
    changes: Mapped[str | None] = mapped_column(Text, nullable=True)  # Description of changes
    parent_version_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("dataset_versions.id"), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)  # User or system
    
    # Relationships
    dataset: Mapped["Dataset"] = relationship(back_populates="versions")
    parent_version: Mapped["DatasetVersion | None"] = relationship(
        remote_side="DatasetVersion.id",
        backref="child_versions",
    )
    
    def __repr__(self) -> str:
        return f"<DatasetVersion(id={self.id}, dataset_id={self.dataset_id}, v{self.version_number})>"


class DatasetSample(Base):
    """Individual sample within a dataset."""
    
    __tablename__ = "dataset_samples"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"), nullable=False)
    
    # Sample content
    text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # For instruction tuning / paired data
    instruction: Mapped[str | None] = mapped_column(Text, nullable=True)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Multimodal support
    image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    video_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    
    # Metadata
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    source_document_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    # Quality
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    trust_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Processing flags
    is_used_in_training: Mapped[bool] = mapped_column(Boolean, default=False)
    is_replay_sample: Mapped[bool] = mapped_column(Boolean, default=False)  # For experience replay
    is_protected: Mapped[bool] = mapped_column(Boolean, default=False)  # Protected capability sample
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dataset: Mapped["Dataset"] = relationship(back_populates="samples")
    
    def __repr__(self) -> str:
        return f"<DatasetSample(id={self.id}, dataset_id={self.dataset_id})>"
