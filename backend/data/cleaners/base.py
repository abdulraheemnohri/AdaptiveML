"""
Base Cleaner Interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CleanerConfig:
    """Configuration for a data cleaner"""
    name: str
    enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)
    
    # General options
    min_length: int = 10
    max_length: int = 100000
    
    # Deduplication options
    similarity_threshold: float = 0.9
    
    # Quality options
    min_quality_score: float = 0.5


@dataclass
class CleaningResult:
    """Result of data cleaning"""
    content: str
    original_content: str
    cleaner_name: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    removed_count: int = 0
    modified_count: int = 0
    quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "original_content": self.original_content,
            "cleaner_name": self.cleaner_name,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "removed_count": self.removed_count,
            "modified_count": self.modified_count,
            "quality_score": self.quality_score,
        }


class BaseCleaner(ABC):
    """Abstract base class for all data cleaners"""
    
    def __init__(self, config: CleanerConfig):
        self.config = config
    
    @abstractmethod
    async def clean(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> CleaningResult:
        """Clean content and return result"""
        pass
    
    @abstractmethod
    async def clean_batch(self, items: List[Dict[str, Any]]) -> List[CleaningResult]:
        """Clean multiple items"""
        pass
    
    async def validate(self, content: str) -> List[str]:
        """Validate content before cleaning"""
        errors = []
        
        if not content:
            errors.append("Content is empty")
        
        if len(content) < self.config.min_length:
            errors.append(f"Content too short (min: {self.config.min_length})")
        
        if len(content) > self.config.max_length:
            errors.append(f"Content too long (max: {self.config.max_length})")
        
        return errors
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity"""
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
