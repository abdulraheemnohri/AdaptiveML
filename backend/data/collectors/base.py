"""
Base Collector Interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class SourceType(str, Enum):
    WEBSITE = "website"
    RSS = "rss"
    YOUTUBE = "youtube"
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    GITHUB = "github"
    DATABASE = "database"
    API = "api"
    LOCAL_FOLDER = "local_folder"
    CUSTOM = "custom"


@dataclass
class CollectorConfig:
    """Configuration for a data collector"""
    source_type: SourceType
    name: str
    url: Optional[str] = None
    path: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None
    schedule: Optional[str] = None  # Cron expression
    enabled: bool = True
    max_items: int = 1000
    timeout: int = 300
    retry_count: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectedData:
    """Represents collected data item"""
    content: str
    source_url: str
    source_type: SourceType
    title: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    collected_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[bytes] = None
    mime_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source_url": self.source_url,
            "source_type": self.source_type.value,
            "title": self.title,
            "author": self.author,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "collected_at": self.collected_at.isoformat(),
            "metadata": self.metadata,
            "mime_type": self.mime_type,
        }


class BaseCollector(ABC):
    """Abstract base class for all data collectors"""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self._session = None
    
    @abstractmethod
    async def collect(self) -> List[CollectedData]:
        """Collect data from the source"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the collector can connect to the source"""
        pass
    
    async def validate(self) -> List[str]:
        """Validate the collector configuration"""
        errors = []
        if not self.config.enabled:
            errors.append("Collector is disabled")
        if self.config.max_items <= 0:
            errors.append("max_items must be positive")
        return errors
    
    async def close(self):
        """Clean up resources"""
        if self._session:
            await self._session.close()
