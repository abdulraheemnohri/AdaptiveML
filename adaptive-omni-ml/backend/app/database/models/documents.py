"""
Document model for processed data.
"""

from datetime import datetime
from sqlalchemy import String, DateTime, Text, Integer, ForeignKey, Float, Boolean
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database.session import Base


class Document(Base):
    """A processed document from data collection."""
    
    __tablename__ = "documents"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection_id: Mapped[int] = mapped_column(Integer, ForeignKey("data_collections.id"), nullable=False)
    
    # Content
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # SHA256 for deduplication
    
    # Metadata
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    
    # Quality metrics
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    trust_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Processing status
    is_deduplicated: Mapped[bool] = mapped_column(Boolean, default=False)
    is_cleaned: Mapped[bool] = mapped_column(Boolean, default=False)
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_errors: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Timestamps
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    collection: Mapped["DataCollection"] = relationship(back_populates="documents")
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title={self.title})>"


# Import delayed to avoid circular imports
from backend.app.database.models.data_sources import DataCollection
