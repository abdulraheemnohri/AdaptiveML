"""
Data Processors Module
Handles parsing, extraction, normalization, and transformation of collected data
"""

from .base import BaseProcessor, ProcessorConfig, ProcessingResult
from .text import TextProcessor
from .html import HTMLProcessor
from .code import CodeProcessor
from .multimodal import MultimodalProcessor

__all__ = [
    "BaseProcessor",
    "ProcessorConfig",
    "ProcessingResult",
    "TextProcessor",
    "HTMLProcessor",
    "CodeProcessor",
    "MultimodalProcessor",
]
