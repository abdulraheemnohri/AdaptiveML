"""
Data Cleaning for Adaptive Qwen Omni.
Provides text cleaning, normalization, and preprocessing.
"""

import re
import html
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Union
import logging

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    MultimodalData,
    MultimodalEntry,
)

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning."""
    # Text cleaning
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_special_chars: bool = False
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    lowercase: bool = False
    
    # Language-specific
    language: str = "en"
    remove_non_ascii: bool = False
    
    # Content filtering
    min_length: int = 10
    max_length: int = 4096
    remove_empty: bool = True
    
    # Custom patterns
    custom_patterns: List[str] = field(default_factory=list)
    
    # Modality-specific
    clean_text: bool = True
    clean_image: bool = False
    clean_audio: bool = False
    clean_video: bool = False
    clean_speech: bool = False


@dataclass
class CleaningResult:
    """Result of a cleaning operation."""
    original: str
    cleaned: str
    changes_made: List[str] = field(default_factory=list)
    length_before: int = 0
    length_after: int = 0
    is_valid: bool = True
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "cleaned": self.cleaned,
            "changes_made": self.changes_made,
            "length_before": self.length_before,
            "length_after": self.length_after,
            "is_valid": self.is_valid,
            "reason": self.reason,
        }


class TextCleaner:
    """
    Cleans and normalizes text data.
    """
    
    # Common patterns
    URL_PATTERN: Pattern[str] = re.compile(
        r'https?://\S+|www\.\S+|http://\S+|\.com/\S+'
    )
    EMAIL_PATTERN: Pattern[str] = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    HTML_PATTERN: Pattern[str] = re.compile(r'<[^>]+>')
    SPECIAL_CHAR_PATTERN: Pattern[str] = re.compile(r'[^\w\s.,!?;:\-\'"()\[\]{}]')
    EXTRA_WHITESPACE_PATTERN: Pattern[str] = re.compile(r'\s+')
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize the text cleaner.
        
        Args:
            config: Cleaning configuration
        """
        self.config = config or CleaningConfig()
        
        # Compile custom patterns
        self.custom_patterns: List[Pattern[str]] = []
        for pattern in self.config.custom_patterns:
            self.custom_patterns.append(re.compile(pattern))
    
    def clean(self, text: str) -> CleaningResult:
        """
        Clean text according to configuration.
        
        Args:
            text: Input text
            
        Returns:
            CleaningResult with cleaned text and metadata
        """
        if not text or not isinstance(text, str):
            return CleaningResult(
                original=text or "",
                cleaned="",
                is_valid=False,
                reason="Empty or invalid text",
            )
        
        changes = []
        cleaned = text
        original = text
        
        # Track length
        length_before = len(text)
        
        # Normalize unicode
        if self.config.normalize_unicode:
            cleaned = unicodedata.normalize('NFKC', cleaned)
            changes.append("normalized_unicode")
        
        # Remove HTML
        if self.config.remove_html:
            cleaned = self.HTML_PATTERN.sub('', cleaned)
            changes.append("removed_html")
        
        # Remove URLs
        if self.config.remove_urls:
            cleaned = self.URL_PATTERN.sub('', cleaned)
            changes.append("removed_urls")
        
        # Remove emails
        if self.config.remove_emails:
            cleaned = self.EMAIL_PATTERN.sub('', cleaned)
            changes.append("removed_emails")
        
        # Remove special characters
        if self.config.remove_special_chars:
            cleaned = self.SPECIAL_CHAR_PATTERN.sub('', cleaned)
            changes.append("removed_special_chars")
        
        # Remove non-ASCII
        if self.config.remove_non_ascii:
            cleaned = cleaned.encode('ascii', 'ignore').decode('ascii')
            changes.append("removed_non_ascii")
        
        # Apply custom patterns
        for pattern in self.custom_patterns:
            if pattern.sub('', cleaned) != cleaned:
                cleaned = pattern.sub('', cleaned)
                changes.append(f"applied_custom_pattern:{pattern.pattern}")
        
        # Remove extra whitespace
        if self.config.remove_extra_whitespace:
            cleaned = self.EXTRA_WHITESPACE_PATTERN.sub(' ', cleaned).strip()
            changes.append("removed_extra_whitespace")
        
        # Lowercase
        if self.config.lowercase:
            cleaned = cleaned.lower()
            changes.append("lowercased")
        
        # Decode HTML entities
        cleaned = html.unescape(cleaned)
        
        # Check length
        length_after = len(cleaned)
        
        # Validate
        is_valid = True
        reason = None
        
        if self.config.remove_empty and not cleaned.strip():
            is_valid = False
            reason = "Empty after cleaning"
        
        if self.config.min_length and len(cleaned) < self.config.min_length:
            is_valid = False
            reason = f"Too short (min: {self.config.min_length})"
        
        if self.config.max_length and len(cleaned) > self.config.max_length:
            is_valid = False
            reason = f"Too long (max: {self.config.max_length})"
        
        return CleaningResult(
            original=original,
            cleaned=cleaned,
            changes_made=changes,
            length_before=length_before,
            length_after=length_after,
            is_valid=is_valid,
            reason=reason,
        )
    
    def batch_clean(self, texts: List[str]) -> List[CleaningResult]:
        """Clean a batch of texts."""
        return [self.clean(text) for text in texts]


class DataCleaning:
    """
    Main data cleaning class for multimodal data.
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize data cleaning.
        
        Args:
            config: Cleaning configuration
        """
        self.config = config or CleaningConfig()
        self.text_cleaner = TextCleaner(self.config)
    
    def clean_entry(self, entry: MultimodalEntry) -> MultimodalEntry:
        """
        Clean a multimodal entry.
        
        Args:
            entry: Input entry
            
        Returns:
            Cleaned entry
        """
        cleaned_data = self.clean_data(entry.data)
        
        # Clean instruction and output
        cleaned_instruction = None
        if entry.instruction:
            result = self.text_cleaner.clean(entry.instruction)
            if result.is_valid:
                cleaned_instruction = result.cleaned
        
        cleaned_output = None
        if entry.expected_output:
            result = self.text_cleaner.clean(entry.expected_output)
            if result.is_valid:
                cleaned_output = result.cleaned
        
        return MultimodalEntry(
            id=entry.id,
            data=cleaned_data,
            instruction=cleaned_instruction,
            expected_output=cleaned_output,
            domain=entry.domain,
            language=entry.language,
            importance=entry.importance,
            novelty=entry.novelty,
            difficulty=entry.difficulty,
            source=entry.source,
            version=entry.version,
            timestamp=entry.timestamp,
            priority=entry.priority,
            forgetting_risk=entry.forgetting_risk,
            error_rate=entry.error_rate,
            last_accessed=entry.last_accessed,
            access_count=entry.access_count,
        )
    
    def clean_data(self, data: MultimodalData) -> MultimodalData:
        """
        Clean multimodal data.
        
        Args:
            data: Input data
            
        Returns:
            Cleaned data
        """
        cleaned_text = None
        if data.text and self.config.clean_text:
            result = self.text_cleaner.clean(data.text)
            if result.is_valid:
                cleaned_text = result.cleaned
        
        return MultimodalData(
            text=cleaned_text or data.text,
            image=data.image,
            audio=data.audio,
            video=data.video,
            speech=data.speech,
        )
    
    def batch_clean_entries(self, entries: List[MultimodalEntry]) -> List[MultimodalEntry]:
        """Clean a batch of entries."""
        return [self.clean_entry(entry) for entry in entries]
    
    def filter_valid(self, entries: List[MultimodalEntry]) -> List[MultimodalEntry]:
        """Filter entries that are valid after cleaning."""
        valid_entries = []
        for entry in entries:
            cleaned = self.clean_entry(entry)
            # Check if text is valid
            if cleaned.data.text and len(cleaned.data.text) >= self.config.min_length:
                valid_entries.append(cleaned)
        return valid_entries
    
    def get_cleaning_stats(self, entries: List[MultimodalEntry]) -> Dict[str, Any]:
        """Get statistics about cleaning operations."""
        total = len(entries)
        valid = 0
        removed = 0
        avg_length_before = 0
        avg_length_after = 0
        
        for entry in entries:
            result = self.text_cleaner.clean(entry.data.text or "")
            if result.is_valid:
                valid += 1
                avg_length_before += result.length_before
                avg_length_after += result.length_after
            else:
                removed += 1
        
        return {
            "total": total,
            "valid": valid,
            "removed": removed,
            "avg_length_before": avg_length_before / valid if valid > 0 else 0,
            "avg_length_after": avg_length_after / valid if valid > 0 else 0,
            "removal_rate": removed / total if total > 0 else 0,
        }
