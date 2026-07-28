"""
Data source and collection models.
"""

from datetime import datetime
from enum import Enum
from sqlalchemy import String, Boolean, DateTime, Text, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database.session import Base


class SourceType(str, Enum):
    """Types of data sources."""
    WEBSITE = "website"
    RSS = "rss"
    SITEMAP = "sitemap"
    YOUTUBE = "youtube"
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    GITHUB = "github"
    GIT = "git"
    LOCAL_FOLDER = "local_folder"
    DATABASE = "database"
    CUSTOM = "custom"


class DataSource(Base):
    """Data source configuration."""
    
    __tablename__ = "data_sources"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # SourceType enum as string
    url: Mapped[str | None] = mapped_column(Text, nullable=True)  # URL or path
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # Source-specific configuration
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_sync: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    sync_schedule: Mapped[str | None] = mapped_column(String(50), nullable=True)  # Cron expression
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    collections: Mapped[list["DataCollection"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<DataSource(id={self.id}, name={self.name})>"


class DataCollection(Base):
    """A collection run from a data source."""
    
    __tablename__ = "data_collections"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(Integer, ForeignKey("data_sources.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, running, completed, failed
    items_collected: Mapped[int] = mapped_column(Integer, default=0)
    items_processed: Mapped[int] = mapped_column(Integer, default=0)
    items_failed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    source: Mapped["DataSource"] = relationship(back_populates="collections")
    documents: Mapped[list["Document"]] = relationship(
        back_populates="collection",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<DataCollection(id={self.id}, source_id={self.source_id})>"


# Import delayed to avoid circular imports
from backend.app.database.models.documents import Document
