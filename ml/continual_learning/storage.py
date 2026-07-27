"""
Storage module for Adaptive Qwen Omni.
Implements episodic and semantic memory storage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adaptive_ml.qwen_omni.core import (
    DomainType,
    ModalityType,
    MultimodalEntry,
)


@dataclass
class EpisodicMemory:
    """
    Episodic memory that stores specific experiences.
    """
    entries: List[MultimodalEntry] = field(default_factory=list)
    max_entries: int = 10000

    def add(self, entry: MultimodalEntry) -> bool:
        """Add an entry to episodic memory."""
        if len(self.entries) >= self.max_entries:
            # Remove oldest entry
            self.entries.pop(0)

        self.entries.append(entry)
        return True

    def get_by_modality(self, modality: ModalityType) -> List[MultimodalEntry]:
        """Get entries for a specific modality."""
        return [e for e in self.entries if modality in e.data.modalities]

    def get_by_domain(self, domain: DomainType) -> List[MultimodalEntry]:
        """Get entries for a specific domain."""
        return [e for e in self.entries if e.domain == domain]

    def search(self, query: str, top_k: int = 5) -> List[MultimodalEntry]:
        """Search for similar entries (placeholder)."""
        # In a full implementation, this would use semantic search
        return self.entries[:min(top_k, len(self.entries))]

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()


@dataclass
class SemanticMemory:
    """
    Semantic memory that stores generalized knowledge.
    """
    concepts: Dict[str, Any] = field(default_factory=dict)
    max_concepts: int = 5000

    def add_concept(self, concept_id: str, concept_data: Any) -> bool:
        """Add a concept to semantic memory."""
        if len(self.concepts) >= self.max_concepts:
            # Remove a random concept
            if self.concepts:
                self.concepts.pop(next(iter(self.concepts)))

        self.concepts[concept_id] = concept_data
        return True

    def get_concept(self, concept_id: str) -> Optional[Any]:
        """Get a concept by ID."""
        return self.concepts.get(concept_id)

    def search_concepts(self, query: str, top_k: int = 5) -> List[Any]:
        """Search for related concepts (placeholder)."""
        # In a full implementation, this would use semantic search
        return list(self.concepts.values())[:min(top_k, len(self.concepts))]

    def clear(self) -> None:
        """Clear all concepts."""
        self.concepts.clear()


@dataclass
class ReplayStorage:
    """
    Unified storage for replay memory.
    Combines episodic and semantic memory.
    """
    episodic_memory: EpisodicMemory = field(default_factory=EpisodicMemory)
    semantic_memory: SemanticMemory = field(default_factory=SemanticMemory)

    def add_episode(self, entry: MultimodalEntry) -> bool:
        """Add an episode to episodic memory."""
        return self.episodic_memory.add(entry)

    def add_concept(self, concept_id: str, concept_data: Any) -> bool:
        """Add a concept to semantic memory."""
        return self.semantic_memory.add_concept(concept_id, concept_data)

    def get_episodes(self, modality: Optional[ModalityType] = None) -> List[MultimodalEntry]:
        """Get episodes, optionally filtered by modality."""
        if modality:
            return self.episodic_memory.get_by_modality(modality)
        return self.episodic_memory.entries

    def get_concepts(self) -> Dict[str, Any]:
        """Get all concepts."""
        return self.semantic_memory.concepts

    def clear(self) -> None:
        """Clear all memory."""
        self.episodic_memory.clear()
        self.semantic_memory.clear()
