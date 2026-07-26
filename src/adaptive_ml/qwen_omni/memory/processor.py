"""
Multimodal Data Processor for Adaptive Qwen Omni.
Handles data ingestion, cleaning, deduplication, and quality filtering.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from adaptive_ml.qwen_omni.core import (
    DomainType,
    MemoryCandidate,
    ModalityType,
    MultimodalData,
    MultimodalEntry,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Result of a quality check."""
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "messages": self.messages,
            "score": self.score,
        }


class MultimodalDataProcessor:
    """
    Processes multimodal data for the replay memory.
    Handles cleaning, deduplication, and quality filtering.
    """
    
    def __init__(
        self,
        use_deduplication: bool = True,
        use_quality_filter: bool = True,
        use_safety_filter: bool = True,
        min_text_length: int = 10,
        min_word_count: int = 3,
    ):
        self.use_deduplication = use_deduplication
        self.use_quality_filter = use_quality_filter
        self.use_safety_filter = use_safety_filter
        self.min_text_length = min_text_length
        self.min_word_count = min_word_count
        
        # Memory for deduplication
        self._seen_hashes: set = set()
        
    def process(
        self,
        data: MultimodalData,
        instruction: Optional[str] = None,
        expected_output: Optional[str] = None,
        domain: DomainType = DomainType.GENERAL,
        source: str = "unknown",
    ) -> Optional[MemoryCandidate]:
        """
        Process multimodal data into a memory candidate.
        
        Args:
            data: Raw multimodal data
            instruction: Optional instruction/prompt
            expected_output: Optional expected output
            domain: Domain classification
            source: Source of the data
            
        Returns:
            MemoryCandidate if data passes all checks, None otherwise
        """
        # Create candidate
        candidate = MemoryCandidate(
            data=data,
            instruction=instruction,
            expected_output=expected_output,
        )
        
        # Run checks
        if self.use_deduplication:
            if self._is_duplicate(data, instruction):
                candidate.is_duplicate = True
                return candidate
        
        if self.use_quality_filter:
            quality_result = self._check_quality(data, instruction)
            candidate.quality_score = quality_result.score
            candidate.is_low_quality = not quality_result.passed
            
            if not quality_result.passed:
                return candidate
        
        if self.use_safety_filter:
            if self._is_unsafe(data, instruction):
                candidate.is_unsafe = True
                return candidate
        
        # Compute novelty score (placeholder - would use embedding model in practice)
        candidate.novelty_score = self._compute_novelty(data, instruction)
        
        # Compute importance score
        candidate.importance_score = self._compute_importance(data, instruction, domain)
        
        return candidate
    
    def _is_duplicate(self, data: MultimodalData, instruction: Optional[str] = None) -> bool:
        """Check if data is a duplicate."""
        text = instruction or data.text or ""
        
        # Create hash of text
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        if text_hash in self._seen_hashes:
            return True
        
        self._seen_hashes.add(text_hash)
        return False
    
    def _check_quality(self, data: MultimodalData, instruction: Optional[str] = None) -> QualityCheckResult:
        """Check data quality."""
        text = instruction or data.text or ""
        
        checks = {}
        messages = []
        
        # Check text length
        if len(text.strip()) < self.min_text_length:
            checks["text_length"] = False
            messages.append(f"Text too short: {len(text)} chars")
        else:
            checks["text_length"] = True
        
        # Check word count
        word_count = len(text.split())
        if word_count < self.min_word_count:
            checks["word_count"] = False
            messages.append(f"Too few words: {word_count}")
        else:
            checks["word_count"] = True
        
        # Check for excessive repetition
        words = text.lower().split()
        if words:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio < 0.3:
                checks["repetition"] = False
                messages.append(f"Excessive repetition: {repetition_ratio:.2f}")
            else:
                checks["repetition"] = True
        
        # Check for empty modalities
        if not data.modalities:
            checks["modalities"] = False
            messages.append("No modalities detected")
        else:
            checks["modalities"] = True
        
        # Compute score
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        return QualityCheckResult(
            passed=score >= 0.8,
            checks=checks,
            messages=messages,
            score=score,
        )
    
    def _is_unsafe(self, data: MultimodalData, instruction: Optional[str] = None) -> bool:
        """Check if data contains unsafe content."""
        text = instruction or data.text or ""
        text_lower = text.lower()
        
        # List of unsafe keywords (in practice, use a proper safety classifier)
        unsafe_keywords = [
            "hate", "violence", "illegal", "harm", "abuse",
            "racist", "sexist", "toxic", "dangerous", "unethical"
        ]
        
        for keyword in unsafe_keywords:
            if keyword in text_lower:
                return True
        
        return False
    
    def _compute_novelty(self, data: MultimodalData, instruction: Optional[str] = None) -> float:
        """Compute novelty score (placeholder)."""
        # In a full implementation, this would:
        # 1. Embed the text using a sentence transformer
        # 2. Compare against existing embeddings in memory
        # 3. Return similarity score (1 - max_similarity)
        
        # Placeholder: return medium novelty
        return 0.5
    
    def _compute_importance(self, data: MultimodalData, instruction: Optional[str] = None, domain: DomainType = DomainType.GENERAL) -> float:
        """Compute importance score."""
        # Base importance
        importance = 0.5
        
        # Boost for certain domains
        domain_weights = {
            DomainType.CODING: 0.9,
            DomainType.MATHEMATICS: 0.8,
            DomainType.URDU: 0.8,
            DomainType.VISION: 0.7,
            DomainType.AUDIO: 0.7,
            DomainType.VIDEO: 0.7,
        }
        
        if domain in domain_weights:
            importance = domain_weights[domain]
        
        # Boost for multimodal data
        if len(data.modalities) > 1:
            importance *= 1.2
        
        return min(1.0, importance)
    
    def create_entry(
        self,
        candidate: MemoryCandidate,
        domain: DomainType = DomainType.GENERAL,
        language: str = "en",
        version: str = "base-v1",
    ) -> Optional[MultimodalEntry]:
        """
        Create a memory entry from a candidate.
        
        Args:
            candidate: Memory candidate
            domain: Domain classification
            language: Language
            version: Model version
            
        Returns:
            MultimodalEntry if candidate should be stored, None otherwise
        """
        if not candidate.should_store():
            return None
        
        return MultimodalEntry(
            id=str(len(self._seen_hashes)),  # Simple ID generation
            data=candidate.data,
            instruction=candidate.instruction,
            expected_output=candidate.expected_output,
            domain=domain,
            language=language,
            importance=candidate.importance_score,
            novelty=candidate.novelty_score,
            difficulty=0.5,  # Placeholder
            source=candidate.source if hasattr(candidate, 'source') else "unknown",
            version=version,
            priority=self._importance_to_priority(candidate.importance_score),
        )
    
    def _importance_to_priority(self, importance: float) -> Any:
        """Convert importance score to priority level."""
        from adaptive_ml.qwen_omni.core import MemoryPriority
        
        if importance >= 0.8:
            return MemoryPriority.CRITICAL
        elif importance >= 0.6:
            return MemoryPriority.HIGH
        elif importance >= 0.4:
            return MemoryPriority.MEDIUM
        else:
            return MemoryPriority.LOW
    
    def clear_memory(self) -> None:
        """Clear deduplication memory."""
        self._seen_hashes.clear()


class DataQualityGate:
    """
    Quality gate that filters incoming data.
    """
    
    def __init__(
        self,
        min_quality_score: float = 0.8,
        block_duplicates: bool = True,
        block_unsafe: bool = True,
        block_low_quality: bool = True,
    ):
        self.min_quality_score = min_quality_score
        self.block_duplicates = block_duplicates
        self.block_unsafe = block_unsafe
        self.block_low_quality = block_low_quality
        
        # Statistics
        self._total_processed = 0
        self._total_accepted = 0
        self._total_rejected = 0
        self._rejection_reasons: Dict[str, int] = {}
        
    def check(self, candidate: MemoryCandidate) -> Tuple[bool, str]:
        """
        Check if a candidate should be accepted.
        
        Args:
            candidate: Memory candidate to check
            
        Returns:
            Tuple of (accepted, reason)
        """
        self._total_processed += 1
        
        if self.block_duplicates and candidate.is_duplicate:
            self._total_rejected += 1
            self._rejection_reasons["duplicate"] = self._rejection_reasons.get("duplicate", 0) + 1
            return False, "Duplicate content"
        
        if self.block_unsafe and candidate.is_unsafe:
            self._total_rejected += 1
            self._rejection_reasons["unsafe"] = self._rejection_reasons.get("unsafe", 0) + 1
            return False, "Unsafe content"
        
        if self.block_low_quality and candidate.is_low_quality:
            self._total_rejected += 1
            self._rejection_reasons["low_quality"] = self._rejection_reasons.get("low_quality", 0) + 1
            return False, "Low quality"
        
        if candidate.quality_score < self.min_quality_score:
            self._total_rejected += 1
            self._rejection_reasons["quality_threshold"] = self._rejection_reasons.get("quality_threshold", 0) + 1
            return False, f"Quality score below threshold: {candidate.quality_score:.2f}"
        
        self._total_accepted += 1
        return True, "Accepted"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quality gate statistics."""
        return {
            "total_processed": self._total_processed,
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_rejected,
            "acceptance_rate": self._total_accepted / max(self._total_processed, 1),
            "rejection_reasons": self._rejection_reasons,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_processed = 0
        self._total_accepted = 0
        self._total_rejected = 0
        self._rejection_reasons.clear()
