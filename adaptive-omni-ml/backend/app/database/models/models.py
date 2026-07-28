"""
Model, ModelVersion, ModelDeployment, and Adapter models.
"""

from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, Integer, ForeignKey, Float, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship, Mapped, mapped_column
import enum
from backend.app.database.session import Base


class ModelStatus(str, enum.Enum):
    """Model lifecycle status."""
    DRAFT = "draft"
    TRAINING = "training"
    CANDIDATE = "candidate"
    TESTING = "testing"
    APPROVED = "approved"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class Model(Base):
    """Main model entity representing a trained ML model."""
    
    __tablename__ = "models"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Type and base
    model_type: Mapped[str] = mapped_column(String(50), default="custom")  # custom, base, adapter
    base_model_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("models.id"), nullable=True)
    base_model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="draft")  # ModelStatus enum
    
    # Architecture info
    architecture: Mapped[str | None] = mapped_column(String(100), nullable=True)
    num_parameters: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_context_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Capabilities
    capabilities: Mapped[list] = mapped_column(JSON, default=list)  # ['text', 'image', 'audio', 'video']
    languages: Mapped[list] = mapped_column(JSON, default=list)
    
    # Performance metrics
    avg_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    vram_usage_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Quality scores
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    safety_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    forgetting_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Storage
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_format: Mapped[str | None] = mapped_column(String(20), nullable=True)  # pytorch, safetensors, onnx
    quantization: Mapped[str | None] = mapped_column(String(20), nullable=True)  # fp16, int8, int4
    
    # Training info
    training_job_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("training_jobs.id"), nullable=True)
    dataset_ids: Mapped[list] = mapped_column(JSON, default=list)
    training_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Metadata
    tags: Mapped[list] = mapped_column(JSON, default=list)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
    )
    deployments: Mapped[list["ModelDeployment"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
    )
    adapters: Mapped[list["Adapter"]] = relationship(
        back_populates="base_model",
        cascade="all, delete-orphan",
    )
    parent_model: Mapped["Model | None"] = relationship(
        remote_side="Model.id",
        backref="child_models",
    )
    evaluations: Mapped[list["EvaluationRun"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
    )
    training_job: Mapped["TrainingJob | None"] = relationship(back_populates="output_model")
    
    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name={self.name})>"


class ModelVersion(Base):
    """A specific version of a model."""
    
    __tablename__ = "model_versions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("models.id"), nullable=False)
    version_number: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "1.0.0", "2.1.3"
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="draft")
    
    # Change tracking
    changes: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_version_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("model_versions.id"), nullable=True)
    
    # Checksums
    model_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    
    # Metrics at this version
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Storage
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    # Relationships
    model: Mapped["Model"] = relationship(back_populates="versions")
    parent_version: Mapped["ModelVersion | None"] = relationship(
        remote_side="ModelVersion.id",
        backref="child_versions",
    )
    
    def __repr__(self) -> str:
        return f"<ModelVersion(id={self.id}, model_id={self.model_id}, v{self.version_number})>"


class ModelDeployment(Base):
    """A deployed instance of a model."""
    
    __tablename__ = "model_deployments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("models.id"), nullable=False)
    model_version_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("model_versions.id"), nullable=True)
    
    # Deployment info
    deployment_type: Mapped[str] = mapped_column(String(20), default="local")  # local, api, edge
    environment: Mapped[str] = mapped_column(String(20), default="production")  # production, staging, development
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, deploying, active, failed, stopped
    
    # Configuration
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Runtime info
    endpoint_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    health_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    
    # Usage stats
    total_requests: Mapped[int | None] = mapped_column(Integer, default=0)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model: Mapped["Model"] = relationship(back_populates="deployments")
    
    def __repr__(self) -> str:
        return f"<ModelDeployment(id={self.id}, model_id={self.model_id})>"


class Adapter(Base):
    """A LoRA/QLoRA adapter for a base model."""
    
    __tablename__ = "adapters"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Base model
    base_model_id: Mapped[int] = mapped_column(Integer, ForeignKey("models.id"), nullable=False)
    
    # Adapter type
    adapter_type: Mapped[str] = mapped_column(String(20), default="lora")  # lora, qlora, prefix_tuning
    
    # Configuration
    r: Mapped[int | None] = mapped_column(Integer, nullable=True)
    alpha: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dropout: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_modules: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    # Training info
    dataset_ids: Mapped[list] = mapped_column(JSON, default=list)
    training_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Storage
    adapter_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Metrics
    evaluation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    base_model: Mapped["Model"] = relationship(back_populates="adapters")
    
    def __repr__(self) -> str:
        return f"<Adapter(id={self.id}, name={self.name})>"


# Import delayed to avoid circular imports
from backend.app.database.models.evaluation import EvaluationRun
from backend.app.database.models.training import TrainingJob
