"""
Memory, KnowledgeItem, and KnowledgeGap models.
"""

from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, Integer, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database.session import Base


class Memory(Base):
    """Long-term memory storage for the AI system."""
    
    __tablename__ = "memories"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(20), default="text")  # text, embedding, structured
    
    # Category
    category: Mapped[str] = mapped_column(String(100), nullable=False)  # facts, procedures, preferences, context
    tags: Mapped[list] = mapped_column(JSON, default=list)
    
    # Embedding (for semantic search)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    # Importance
    importance_score: Mapped[float | None] = mapped_column(Float, default=0.5)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Source
    source: Mapped[str | None] = mapped_column(String(200), nullable=True)  # conversation, document, user_input
    source_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Expiration
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_permanent: Mapped[bool] = mapped_column(Boolean, default=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<Memory(id={self.id}, category={self.category})>"


class KnowledgeItem(Base):
    """A structured knowledge item in the knowledge base."""
    
    __tablename__ = "knowledge_items"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Content
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Type
    knowledge_type: Mapped[str] = mapped_column(String(50), default="fact")  # fact, concept, procedure, reference
    
    # Category and tags
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    subcategory: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    
    # Relationships
    related_items: Mapped[list] = mapped_column(JSON, default=list)  # IDs of related knowledge items
    
    # Quality
    confidence_score: Mapped[float | None] = mapped_column(Float, default=1.0)
    trust_score: Mapped[float | None] = mapped_column(Float, default=1.0)
    
    # Source tracking
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_document_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Usage
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Status
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<KnowledgeItem(id={self.id}, title={self.title})>"


class KnowledgeGap(Base):
    """Identified gap in the model's knowledge."""
    
    __tablename__ = "knowledge_gaps"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Gap description
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Category
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    topic: Mapped[str | None] = mapped_column(String(200), nullable=True)
    
    # Detection info
    detection_method: Mapped[str] = mapped_column(String(50), default="user_feedback")  # user_feedback, failed_response, benchmark
    detected_from: Mapped[str | None] = mapped_column(String(200), nullable=True)  # Conversation ID, benchmark name
    
    # Priority
    priority: Mapped[str] = mapped_column(String(20), default="medium")  # low, medium, high, critical
    impact_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="identified")  # identified, researching, collecting, resolved, ignored
    
    # Resolution
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    # Action items
    action_items: Mapped[list] = mapped_column(JSON, default=list)
    datasets_created: Mapped[list] = mapped_column(JSON, default=list)  # Dataset IDs
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<KnowledgeGap(id={self.id}, title={self.title})>"
