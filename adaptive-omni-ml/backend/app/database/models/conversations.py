"""
Conversation and Message models.
"""

from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database.session import Base


class Conversation(Base):
    """A conversation session."""
    
    __tablename__ = "conversations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    
    # Mode used
    serving_mode: Mapped[str] = mapped_column(String(20), default="local")  # local, api, compare, auto
    model_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("models.id"), nullable=True)
    provider_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("providers.id"), nullable=True)
    
    # Settings
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, default=0.7)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Statistics
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int | None] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title={self.title})>"


class Message(Base):
    """A single message in a conversation."""
    
    __tablename__ = "messages"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(Integer, ForeignKey("conversations.id"), nullable=False)
    
    # Content
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Multimodal attachments
    attachments: Mapped[list | None] = mapped_column(JSON, nullable=True)  # [{type: 'image', path: '...'}]
    
    # Model info (for assistant messages)
    model_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    # Token usage
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Quality
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # User feedback
    feedback_score: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-5
    feedback_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # For comparison mode
    comparison_group_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_winner: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role})>"
