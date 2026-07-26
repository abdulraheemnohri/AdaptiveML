"""
Quality Filter for Adaptive Qwen Omni.
Filters entries based on quality criteria.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import re

from adaptive_ml.qwen_omni.core import (
    DomainType,
    MultimodalData,
    MultimodalEntry,
)
from adaptive_ml.qwen_omni.datasets.cleaning import TextCleaner, CleaningConfig

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """Configuration for quality filtering."""
    # Text quality
    min_text_length: int = 50
    max_text_length: int = 4096
    min_word_count: int = 10
    max_word_count: int = 1000
    
    # Content quality
    min_sentence_count: int = 2
    max_sentence_count: int = 100
    
    # Language quality
    check_language: bool = True
    allowed_languages: List[str] = field(default_factory=lambda: ["en", "ur", "ar"])
    
    # Content filters
    remove_profanity: bool = True
    remove_hate_speech: bool = True
    remove_personal_info: bool = True
    
    # Domain filters
    allowed_domains: List[DomainType] = field(default_factory=list)
    blocked_domains: List[DomainType] = field(default_factory=list)
    
    # Safety
    check_safety: bool = True
    safety_threshold: float = 0.5
    
    # Custom filters
    custom_filters: List[Callable[[MultimodalEntry], bool]] = field(default_factory=list)


@dataclass
class QualityResult:
    """Result of quality filtering."""
    entry: MultimodalEntry
    is_valid: bool = True
    reasons: List[str] = field(default_factory=list)
    score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "reasons": self.reasons,
            "score": self.score,
        }


class ProfanityFilter:
    """Filters profane content."""
    
    # Common profanity lists (can be extended)
    PROFANITY_WORDS: List[str] = [
        # English profanity
        "fuck", "shit", "bitch", "asshole", "dick", "pussy", "cunt",
        "whore", "slut", "bastard", "damn", "hell", "crap",
        # Urdu profanity (transliterated)
        "gandu", "chut", "bhosda", "bhosdi", "madarchod", "behenchod",
        "lauda", "loda", "kutta", "kutti", "randi", "harami",
    ]
    
    def __init__(self):
        """Initialize profanity filter."""
        self.patterns = [re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE) 
                        for word in self.PROFANITY_WORDS]
    
    def contains_profanity(self, text: str) -> bool:
        """Check if text contains profanity."""
        if not text:
            return False
        
        text_lower = text.lower()
        for pattern in self.patterns:
            if pattern.search(text_lower):
                return True
        return False
    
    def filter(self, text: str) -> Tuple[bool, List[str]]:
        """Filter profanity from text."""
        if not text:
            return True, []
        
        found = []
        for pattern in self.patterns:
            if pattern.search(text):
                found.append(pattern.pattern)
        
        return len(found) == 0, found


class HateSpeechFilter:
    """Filters hate speech content."""
    
    HATE_SPEECH_TERMS: List[str] = [
        "racist", "racism", "sexist", "sexism", "homophobic", "homophobia",
        "xenophobic", "xenophobia", "antisemitic", "antisemitism",
        "islamophobic", "islamophobia", "hate", "discrimination",
        "supremacy", "superior", "inferior", "nazi", "kkk",
    ]
    
    def __init__(self):
        """Initialize hate speech filter."""
        self.patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                        for term in self.HATE_SPEECH_TERMS]
    
    def contains_hate_speech(self, text: str) -> bool:
        """Check if text contains hate speech."""
        if not text:
            return False
        
        text_lower = text.lower()
        for pattern in self.patterns:
            if pattern.search(text_lower):
                return True
        return False


class PersonalInfoFilter:
    """Filters personal information."""
    
    # Patterns for personal information
    EMAIL_PATTERN: re.Pattern[str] = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    PHONE_PATTERN: re.Pattern[str] = re.compile(
        r'(\+?\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}'
    )
    SSN_PATTERN: re.Pattern[str] = re.compile(
        r'\d{3}-\d{2}-\d{4}'
    )
    IP_ADDRESS_PATTERN: re.Pattern[str] = re.compile(
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    )
    
    def contains_personal_info(self, text: str) -> bool:
        """Check if text contains personal information."""
        if not text:
            return False
        
        patterns = [
            self.EMAIL_PATTERN,
            self.PHONE_PATTERN,
            self.SSN_PATTERN,
            self.IP_ADDRESS_PATTERN,
        ]
        
        for pattern in patterns:
            if pattern.search(text):
                return True
        return False


class LanguageDetector:
    """Detects language of text."""
    
    def __init__(self):
        """Initialize language detector."""
        try:
            import langdetect
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("langdetect not available, language detection disabled")
    
    def detect(self, text: str) -> Optional[str]:
        """Detect language of text."""
        if not self._available or not text:
            return None
        
        try:
            import langdetect
            return langdetect.detect(text)
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return None
    
    def is_allowed(self, text: str, allowed_languages: List[str]) -> bool:
        """Check if text is in allowed languages."""
        lang = self.detect(text)
        return lang in allowed_languages


class QualityFilter:
    """
    Main quality filter for multimodal entries.
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize quality filter.
        
        Args:
            config: Quality configuration
        """
        self.config = config or QualityConfig()
        self.profanity_filter = ProfanityFilter()
        self.hate_speech_filter = HateSpeechFilter()
        self.personal_info_filter = PersonalInfoFilter()
        self.language_detector = LanguageDetector()
        self.text_cleaner = TextCleaner(CleaningConfig())
    
    def check_entry(self, entry: MultimodalEntry) -> QualityResult:
        """
        Check if an entry passes quality filters.
        
        Args:
            entry: Entry to check
            
        Returns:
            QualityResult with validation information
        """
        reasons = []
        score = 1.0
        
        # Check text length
        if entry.data.text:
            text_length = len(entry.data.text)
            if text_length < self.config.min_text_length:
                reasons.append(f"Text too short (min: {self.config.min_text_length})")
                score -= 0.3
            if text_length > self.config.max_text_length:
                reasons.append(f"Text too long (max: {self.config.max_text_length})")
                score -= 0.3
            
            # Check word count
            word_count = len(entry.data.text.split())
            if word_count < self.config.min_word_count:
                reasons.append(f"Too few words (min: {self.config.min_word_count})")
                score -= 0.2
            if word_count > self.config.max_word_count:
                reasons.append(f"Too many words (max: {self.config.max_word_count})")
                score -= 0.2
            
            # Check sentence count
            sentence_count = len(re.split(r'[.!?]', entry.data.text))
            if sentence_count < self.config.min_sentence_count:
                reasons.append(f"Too few sentences (min: {self.config.min_sentence_count})")
                score -= 0.1
            if sentence_count > self.config.max_sentence_count:
                reasons.append(f"Too many sentences (max: {self.config.max_sentence_count})")
                score -= 0.1
        
        # Check language
        if self.config.check_language and entry.data.text:
            if not self.language_detector.is_allowed(
                entry.data.text, 
                self.config.allowed_languages
            ):
                reasons.append(f"Language not allowed (allowed: {self.config.allowed_languages})")
                score -= 0.2
        
        # Check profanity
        if self.config.remove_profanity and entry.data.text:
            if self.profanity_filter.contains_profanity(entry.data.text):
                reasons.append("Contains profanity")
                score -= 0.4
        
        # Check hate speech
        if self.config.remove_hate_speech and entry.data.text:
            if self.hate_speech_filter.contains_hate_speech(entry.data.text):
                reasons.append("Contains hate speech")
                score -= 0.5
        
        # Check personal info
        if self.config.remove_personal_info and entry.data.text:
            if self.personal_info_filter.contains_personal_info(entry.data.text):
                reasons.append("Contains personal information")
                score -= 0.4
        
        # Check domain
        if self.config.allowed_domains:
            if entry.domain not in self.config.allowed_domains:
                reasons.append(f"Domain not allowed (allowed: {[d.value for d in self.config.allowed_domains]})")
                score -= 0.2
        
        if self.config.blocked_domains:
            if entry.domain in self.config.blocked_domains:
                reasons.append(f"Domain blocked")
                score -= 0.5
        
        # Apply custom filters
        for custom_filter in self.config.custom_filters:
            if not custom_filter(entry):
                reasons.append("Failed custom filter")
                score -= 0.1
        
        # Determine validity
        is_valid = len(reasons) == 0
        
        return QualityResult(
            entry=entry,
            is_valid=is_valid,
            reasons=reasons,
            score=max(0.0, min(1.0, score)),
        )
    
    def filter_entries(self, entries: List[MultimodalEntry]) -> Tuple[List[MultimodalEntry], List[QualityResult]]:
        """
        Filter a list of entries by quality.
        
        Args:
            entries: List of entries to filter
            
        Returns:
            Tuple of (valid_entries, invalid_results)
        """
        valid_entries = []
        invalid_results = []
        
        for entry in entries:
            result = self.check_entry(entry)
            if result.is_valid:
                valid_entries.append(entry)
            else:
                invalid_results.append(result)
        
        return valid_entries, invalid_results
    
    def get_quality_stats(self, entries: List[MultimodalEntry]) -> Dict[str, Any]:
        """Get statistics about quality filtering."""
        total = len(entries)
        valid = 0
        invalid = 0
        avg_score = 0.0
        
        for entry in entries:
            result = self.check_entry(entry)
            if result.is_valid:
                valid += 1
            else:
                invalid += 1
            avg_score += result.score
        
        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "avg_score": avg_score / total if total > 0 else 0,
            "valid_rate": valid / total if total > 0 else 0,
        }
    
    def rank_by_quality(self, entries: List[MultimodalEntry]) -> List[MultimodalEntry]:
        """Rank entries by quality score."""
        scored_entries = []
        for entry in entries:
            result = self.check_entry(entry)
            scored_entries.append((entry, result.score))
        
        # Sort by score (descending)
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        
        return [entry for entry, score in scored_entries]
    
    def get_top_quality(self, entries: List[MultimodalEntry], n: int = 10) -> List[MultimodalEntry]:
        """Get top N entries by quality."""
        ranked = self.rank_by_quality(entries)
        return ranked[:n]
    
    def add_custom_filter(self, filter_func: Callable[[MultimodalEntry], bool]) -> None:
        """Add a custom filter function."""
        self.config.custom_filters.append(filter_func)
    
    def remove_custom_filters(self) -> None:
        """Remove all custom filters."""
        self.config.custom_filters.clear()
