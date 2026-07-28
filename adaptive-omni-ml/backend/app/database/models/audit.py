"""
Audit log, system event, and alert models.
"""

from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database.session import Base


class AuditLog(Base):
    """Audit log entry for security and compliance."""
    
    __tablename__ = "audit_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Actor
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    user_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    # Action
    action: Mapped[str] = mapped_column(String(100), nullable=False)  # create, read, update, delete, login, logout
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)  # model, dataset, training_job, etc.
    resource_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Details
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    changes: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # Before/after values
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Context
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)
    
    # Result
    status: Mapped[str] = mapped_column(String(20), default="success")  # success, failure, denied
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user: Mapped["User | None"] = relationship(back_populates="audit_logs")
    
    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action={self.action})>"


class SystemEvent(Base):
    """System-level event for monitoring."""
    
    __tablename__ = "system_events"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Event info
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="info")  # debug, info, warning, error, critical
    
    # Source
    source: Mapped[str] = mapped_column(String(100), nullable=False)  # Component name
    component: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    # Content
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Context
    context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Metrics snapshot
    gpu_usage: Mapped[float | None] = mapped_column(Float, nullable=True)
    memory_usage: Mapped[float | None] = mapped_column(Float, nullable=True)
    disk_usage: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Acknowledgment
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self) -> str:
        return f"<SystemEvent(id={self.id}, type={self.event_type})>"


class Alert(Base):
    """Alert for critical conditions requiring attention."""
    
    __tablename__ = "alerts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Alert info
    alert_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # low, medium, high, critical
    
    # Content
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Source
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    related_resource_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    related_resource_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="active")  # active, acknowledged, resolved, suppressed
    
    # Resolution
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Notification
    notification_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    notification_channels: Mapped[list] = mapped_column(JSON, default=list)  # email, slack, webhook
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<Alert(id={self.id}, type={self.alert_type})>"


# Import delayed to avoid circular imports
from backend.app.database.models.users import User
