"""
Data Collectors Module
Handles data acquisition from multiple sources
"""

from .base import BaseCollector, CollectorConfig, CollectedData
from .web import WebCollector
from .youtube import YouTubeCollector
from .file import FileCollector
from .database import DatabaseCollector
from .github import GitHubCollector
from .rss import RSSCollector
from .api import APICollector

__all__ = [
    "BaseCollector",
    "CollectorConfig",
    "CollectedData",
    "WebCollector",
    "YouTubeCollector",
    "FileCollector",
    "DatabaseCollector",
    "GitHubCollector",
    "RSSCollector",
    "APICollector",
]
