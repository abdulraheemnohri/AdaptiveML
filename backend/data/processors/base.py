"""
Base Processor Interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ProcessorType(str, Enum):
    TEXT = "text"
    HTML = "html"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


@dataclass
class ProcessorConfig:
    """Configuration for a data processor"""
    processor_type: ProcessorType
    name: str
    enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)
    
    # Text processing options
    max_length: int = 100000
    min_length: int = 10
    normalize_whitespace: bool = True
    remove_special_chars: bool = False
    
    # Language options
    detect_language: bool = True
    target_language: Optional[str] = None
    
    # Chunking options
    chunk_enabled: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Quality options
    min_quality_score: float = 0.5


@dataclass
class ProcessingResult:
    """Result of data processing"""
    content: str
    original_content: str
    processor_type: ProcessorType
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[str] = field(default_factory=list)
    language: Optional[str] = None
    quality_score: float = 1.0
    tokens_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "original_content": self.original_content,
            "processor_type": self.processor_type.value,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "chunks": self.chunks,
            "language": self.language,
            "quality_score": self.quality_score,
            "tokens_count": self.tokens_count,
        }


class BaseProcessor(ABC):
    """Abstract base class for all data processors"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
    
    @abstractmethod
    async def process(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process content and return result"""
        pass
    
    async def validate(self, content: str) -> List[str]:
        """Validate content before processing"""
        errors = []
        
        if not content:
            errors.append("Content is empty")
        
        if len(content) < self.config.min_length:
            errors.append(f"Content too short (min: {self.config.min_length})")
        
        if len(content) > self.config.max_length:
            errors.append(f"Content too long (max: {self.config.max_length})")
        
        return errors
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Simple approximation: 1 token ≈ 4 characters for English
        # For more accurate counting, use tiktoken or similar
        return len(text) // 4
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        import re
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Strip leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters"""
        import re
        # Keep alphanumeric, spaces, and basic punctuation
        return re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        if not self.config.chunk_enabled:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1  # +1 for space
            
            if current_length >= self.config.chunk_size:
                chunks.append(' '.join(current_chunk))
                # Overlap: keep some words from end of current chunk
                overlap_words = current_chunk[-self.config.chunk_overlap:] if self.config.chunk_overlap > 0 else []
                current_chunk = overlap_words
                current_length = sum(len(w) + 1 for w in overlap_words)
            
            current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of text"""
        try:
            import langdetect
            return langdetect.detect(text[:10000])  # Limit for performance
        except ImportError:
            return None
        except Exception:
            return None
    
    def _calculate_quality_score(self, text: str, metadata: Optional[Dict] = None) -> float:
        """Calculate quality score for content"""
        score = 1.0
        
        # Penalize very short content
        if len(text) < 100:
            score -= 0.2
        
        # Penalize content with too many special characters
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_ratio > 0.3:
            score -= 0.2
        
        # Penalize content with many repeated characters
        if len(set(text)) / max(len(text), 1) < 0.1:
            score -= 0.3
        
        # Bonus for having proper sentence structure
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        if sentence_count > 0 and len(text) / sentence_count < 300:
            score += 0.1
        
        return max(0.0, min(1.0, score))
