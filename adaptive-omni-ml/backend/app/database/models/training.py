"""
Training job, checkpoint, and experiment models.
"""

from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, Integer, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database.session import Base


class TrainingJob(Base):
    """A model training job."""
    
    __tablename__ = "training_jobs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, running, paused, completed, failed, cancelled
    
    # Configuration
    base_model: Mapped[str] = mapped_column(String(200), nullable=False)
    training_type: Mapped[str] = mapped_column(String(50), default="lora")  # lora, qlora, full_finetune, continual
    datasets: Mapped[list] = mapped_column(JSON, default=list)  # Dataset IDs
    dataset_versions: Mapped[list] = mapped_column(JSON, default=list)
    
    # Hyperparameters
    batch_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gradient_accumulation_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    learning_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_seq_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lora_r: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lora_alpha: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lora_dropout: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Continual learning config
    replay_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    ewc_lambda: Mapped[float | None] = mapped_column(Float, nullable=True)
    distillation_weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Progress tracking
    current_epoch: Mapped[int | None] = mapped_column(Integer, default=0)
    current_step: Mapped[int | None] = mapped_column(Integer, default=0)
    total_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_percent: Mapped[float | None] = mapped_column(Float, default=0.0)
    
    # Metrics
    train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    learning_rate_current: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Resource usage
    gpu_usage: Mapped[float | None] = mapped_column(Float, nullable=True)
    vram_usage: Mapped[float | None] = mapped_column(Float, nullable=True)
    ram_usage: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    eta: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Error handling
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Output
    output_model_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("models.id"), nullable=True)
    checkpoint_dir: Mapped[str | None] = mapped_column(String(500), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    checkpoints: Mapped[list["TrainingCheckpoint"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )
    experiment: Mapped["Experiment | None"] = relationship(
        back_populates="training_job",
        uselist=False,
    )
    
    def __repr__(self) -> str:
        return f"<TrainingJob(id={self.id}, name={self.name})>"


class TrainingCheckpoint(Base):
    """A training checkpoint."""
    
    __tablename__ = "training_checkpoints"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(Integer, ForeignKey("training_jobs.id"), nullable=False)
    
    # Checkpoint info
    checkpoint_type: Mapped[str] = mapped_column(String(20), default="regular")  # regular, best, final
    epoch: Mapped[int] = mapped_column(Integer, default=0)
    step: Mapped[int] = mapped_column(Integer, default=0)
    
    # Metrics at checkpoint
    train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Storage
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job: Mapped["TrainingJob"] = relationship(back_populates="checkpoints")
    
    def __repr__(self) -> str:
        return f"<TrainingCheckpoint(id={self.id}, job_id={self.job_id}, epoch={self.epoch})>"


class Experiment(Base):
    """An ML experiment tracking configuration and results."""
    
    __tablename__ = "experiments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Link to training job
    training_job_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("training_jobs.id"), nullable=True)
    
    # Experiment parameters
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Results
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    evaluation_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    forgetting_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    safety_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft, running, completed, archived
    
    tags: Mapped[list] = mapped_column(JSON, default=list)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    training_job: Mapped["TrainingJob | None"] = relationship(back_populates="experiment")
    
    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name={self.name})>"
