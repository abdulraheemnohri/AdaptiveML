"""
Agent and AgentTask models.
"""

from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, Integer, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database.session import Base


class Agent(Base):
    """An autonomous AI agent."""
    
    __tablename__ = "agents"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Type
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)  # research, verification, synthesis, custom
    
    # Configuration
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    capabilities: Mapped[list] = mapped_column(JSON, default=list)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Tools available
    tools: Mapped[list] = mapped_column(JSON, default=list)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_autonomous: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Performance
    tasks_completed: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_completion_time: Mapped[float | None] = mapped_column(Float, nullable=True)  # seconds
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tasks: Mapped[list["AgentTask"]] = relationship(
        back_populates="agent",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Agent(id={self.id}, name={self.name})>"


class AgentTask(Base):
    """A task assigned to an agent."""
    
    __tablename__ = "agent_tasks"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[int] = mapped_column(Integer, ForeignKey("agents.id"), nullable=False)
    
    # Task details
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Input
    input_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, running, completed, failed, cancelled
    priority: Mapped[str] = mapped_column(String(20), default="medium")  # low, medium, high, critical
    
    # Progress
    progress_percent: Mapped[float | None] = mapped_column(Float, default=0.0)
    current_step: Mapped[str | None] = mapped_column(String(200), nullable=True)
    
    # Output
    output_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Error handling
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    
    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Approval workflow
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=True)
    approved_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    agent: Mapped["Agent"] = relationship(back_populates="tasks")
    
    def __repr__(self) -> str:
        return f"<AgentTask(id={self.id}, title={self.title})>"
