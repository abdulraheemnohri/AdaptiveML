"""
Deduplication for Adaptive Qwen Omni.
Identifies and removes duplicate entries from datasets.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import hashlib
import numpy as np

from adaptive_ml.qwen_omni.core import (
    MultimodalData,
    MultimodalEntry,
)

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    # Text deduplication
    text_similarity_threshold: float = 0.95
    text_min_length: int = 50
    
    # Semantic deduplication (requires embeddings)
    use_semantic: bool = False
    semantic_threshold: float = 0.98
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Fuzzy matching
    use_fuzzy: bool = True
    fuzzy_threshold: int = 90  # 0-100
    
    # Exact matching
    use_exact: bool = True
    exact_fields: List[str] = field(default_factory=lambda: ["text", "instruction"])
    
    # Memory
    use_memory: bool = True
    memory_size: int = 100000
    
    # Performance
    batch_size: int = 1000
    num_workers: int = 4


@dataclass
class DuplicateResult:
    """Result of duplicate detection."""
    entry: MultimodalEntry
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    similarity_score: float = 0.0
    hash_value: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "similarity_score": self.similarity_score,
            "hash_value": self.hash_value,
        }


class TextHasher:
    """Generates hashes for text content."""
    
    def __init__(self):
        """Initialize the text hasher."""
        pass
    
    def hash_text(self, text: str) -> str:
        """Generate a hash for text."""
        if not text:
            return ""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def hash_entry(self, entry: MultimodalEntry) -> str:
        """Generate a hash for an entry based on text fields."""
        text_parts = []
        
        if entry.data.text:
            text_parts.append(entry.data.text)
        if entry.instruction:
            text_parts.append(entry.instruction)
        if entry.expected_output:
            text_parts.append(entry.expected_output)
        
        combined = " | ".join(text_parts)
        return self.hash_text(combined)


class FuzzyMatcher:
    """Fuzzy string matching for duplicate detection."""
    
    def __init__(self, threshold: int = 90):
        """
        Initialize fuzzy matcher.
        
        Args:
            threshold: Similarity threshold (0-100)
        """
        self.threshold = threshold
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()
    
    def is_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar."""
        if not text1 or not text2:
            return False
        return self.similarity(text1, text2) >= (self.threshold / 100)


class SemanticMatcher:
    """Semantic similarity matching using embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize semantic matcher.
        
        Args:
            model_name: Name of the embedding model
        """
        self.model_name = model_name
        self.model = None
        self._initialized = False
    
    def _init_model(self) -> None:
        """Initialize the embedding model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, semantic matching disabled")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    
    def embed(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text."""
        if not self._initialized:
            self._init_model()
        
        if self.model is None:
            return None
        
        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


class Deduplication:
    """
    Main deduplication class for multimodal datasets.
    """
    
    def __init__(self, config: Optional[DeduplicationConfig] = None):
        """
        Initialize deduplication.
        
        Args:
            config: Deduplication configuration
        """
        self.config = config or DeduplicationConfig()
        self.hasher = TextHasher()
        self.fuzzy_matcher = FuzzyMatcher(self.config.fuzzy_threshold)
        self.semantic_matcher = SemanticMatcher(self.config.embedding_model)
        
        # Track seen hashes and texts
        self.seen_hashes: Set[str] = set()
        self.seen_texts: Dict[str, MultimodalEntry] = {}
        self.seen_embeddings: Dict[str, np.ndarray] = {}
    
    def check_duplicate(self, entry: MultimodalEntry) -> DuplicateResult:
        """
        Check if an entry is a duplicate.
        
        Args:
            entry: Entry to check
            
        Returns:
            DuplicateResult with detection information
        """
        # Check exact duplicates first
        if self.config.use_exact:
            entry_hash = self.hasher.hash_entry(entry)
            if entry_hash in self.seen_hashes:
                return DuplicateResult(
                    entry=entry,
                    is_duplicate=True,
                    duplicate_of=self.seen_hashes[entry_hash],
                    similarity_score=1.0,
                    hash_value=entry_hash,
                )
        
        # Check fuzzy duplicates
        if self.config.use_fuzzy and entry.data.text and len(entry.data.text) >= self.config.text_min_length:
            for seen_id, seen_entry in self.seen_texts.items():
                if seen_entry.data.text:
                    sim = self.fuzzy_matcher.similarity(
                        entry.data.text,
                        seen_entry.data.text
                    )
                    if sim >= (self.config.fuzzy_threshold / 100):
                        return DuplicateResult(
                            entry=entry,
                            is_duplicate=True,
                            duplicate_of=seen_id,
                            similarity_score=sim,
                            hash_value=self.hasher.hash_entry(entry),
                        )
        
        # Check semantic duplicates
        if self.config.use_semantic and entry.data.text and len(entry.data.text) >= self.config.text_min_length:
            entry_embedding = self.semantic_matcher.embed(entry.data.text)
            if entry_embedding is not None:
                for seen_id, seen_embedding in self.seen_embeddings.items():
                    sim = float(np.dot(entry_embedding, seen_embedding) / (
                        np.linalg.norm(entry_embedding) * np.linalg.norm(seen_embedding)
                    ))
                    if sim >= self.config.semantic_threshold:
                        return DuplicateResult(
                            entry=entry,
                            is_duplicate=True,
                            duplicate_of=seen_id,
                            similarity_score=sim,
                            hash_value=self.hasher.hash_entry(entry),
                        )
        
        # Not a duplicate
        entry_hash = self.hasher.hash_entry(entry)
        self.seen_hashes.add(entry_hash)
        
        if entry.data.text:
            self.seen_texts[entry.id] = entry
            if self.config.use_semantic:
                embedding = self.semantic_matcher.embed(entry.data.text)
                if embedding is not None:
                    self.seen_embeddings[entry.id] = embedding
        
        return DuplicateResult(
            entry=entry,
            is_duplicate=False,
            similarity_score=0.0,
            hash_value=entry_hash,
        )
    
    def deduplicate(self, entries: List[MultimodalEntry]) -> Tuple[List[MultimodalEntry], List[DuplicateResult]]:
        """
        Remove duplicates from a list of entries.
        
        Args:
            entries: List of entries to deduplicate
            
        Returns:
            Tuple of (unique_entries, duplicates)
        """
        unique_entries = []
        duplicates = []
        
        # Reset seen tracking
        self.seen_hashes.clear()
        self.seen_texts.clear()
        self.seen_embeddings.clear()
        
        for entry in entries:
            result = self.check_duplicate(entry)
            if result.is_duplicate:
                duplicates.append(result)
            else:
                unique_entries.append(entry)
        
        return unique_entries, duplicates
    
    def batch_deduplicate(
        self,
        entries: List[MultimodalEntry],
        batch_size: Optional[int] = None,
    ) -> Tuple[List[MultimodalEntry], List[DuplicateResult]]:
        """
        Deduplicate entries in batches.
        
        Args:
            entries: List of entries
            batch_size: Batch size (default: from config)
            
        Returns:
            Tuple of (unique_entries, duplicates)
        """
        batch_size = batch_size or self.config.batch_size
        
        all_unique = []
        all_duplicates = []
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            unique, duplicates = self.deduplicate(batch)
            all_unique.extend(unique)
            all_duplicates.extend(duplicates)
        
        return all_unique, all_duplicates
    
    def get_duplicate_stats(self, entries: List[MultimodalEntry]) -> Dict[str, Any]:
        """Get statistics about duplicates in a dataset."""
        # Reset tracking
        self.seen_hashes.clear()
        self.seen_texts.clear()
        self.seen_embeddings.clear()
        
        total = len(entries)
        duplicates = 0
        unique = 0
        
        for entry in entries:
            result = self.check_duplicate(entry)
            if result.is_duplicate:
                duplicates += 1
            else:
                unique += 1
        
        return {
            "total": total,
            "unique": unique,
            "duplicates": duplicates,
            "duplicate_rate": duplicates / total if total > 0 else 0,
        }
    
    def find_duplicates(self, entries: List[MultimodalEntry]) -> List[List[MultimodalEntry]]:
        """
        Find all duplicate groups in entries.
        
        Args:
            entries: List of entries
            
        Returns:
            List of duplicate groups (each group has 2+ entries)
        """
        # Build hash to entries mapping
        hash_to_entries: Dict[str, List[MultimodalEntry]] = {}
        
        for entry in entries:
            entry_hash = self.hasher.hash_entry(entry)
            if entry_hash not in hash_to_entries:
                hash_to_entries[entry_hash] = []
            hash_to_entries[entry_hash].append(entry)
        
        # Filter to groups with 2+ entries
        duplicate_groups = [
            group for group in hash_to_entries.values() 
            if len(group) >= 2
        ]
        
        return duplicate_groups
    
    def clear(self) -> None:
        """Clear all tracked hashes and texts."""
        self.seen_hashes.clear()
        self.seen_texts.clear()
        self.seen_embeddings.clear()
