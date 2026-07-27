"""
Memory Selector for Adaptive Qwen Omni.
Selects which data to store in memory and which to use for replay.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adaptive_ml.qwen_omni.core import (
    DomainType,
    MemoryPriority,
    ModalityType,
    MultimodalEntry,
)


@dataclass
class SelectionCriteria:
    """Criteria for memory selection."""
    min_importance: float = 0.3
    min_novelty: float = 0.3
    max_forgetting_risk: float = 0.5
    max_error_rate: float = 0.5

    # Priority thresholds
    critical_threshold: float = 0.8
    high_threshold: float = 0.6
    medium_threshold: float = 0.4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_importance": self.min_importance,
            "min_novelty": self.min_novelty,
            "max_forgetting_risk": self.max_forgetting_risk,
            "max_error_rate": self.max_error_rate,
            "critical_threshold": self.critical_threshold,
            "high_threshold": self.high_threshold,
            "medium_threshold": self.medium_threshold,
        }


class MemorySelector:
    """
    Selects which entries to use for replay based on various criteria.
    """

    def __init__(
        self,
        criteria: Optional[SelectionCriteria] = None,
    ):
        self.criteria = criteria or SelectionCriteria()

    def select_for_replay(
        self,
        entries: List[MultimodalEntry],
        batch_size: int,
        modality: Optional[ModalityType] = None,
        domain: Optional[DomainType] = None,
        priority: Optional[MemoryPriority] = None,
    ) -> List[MultimodalEntry]:
        """
        Select entries for replay.

        Args:
            entries: List of entries to select from
            batch_size: Number of entries to select
            modality: Optional modality filter
            domain: Optional domain filter
            priority: Optional priority filter

        Returns:
            List of selected entries
        """
        # Apply filters
        filtered = []
        for entry in entries:
            if modality and modality not in entry.data.modalities:
                continue
            if domain and entry.domain != domain:
                continue
            if priority and entry.priority != priority:
                continue

            # Apply quality criteria
            if entry.importance < self.criteria.min_importance:
                continue
            if entry.novelty < self.criteria.min_novelty:
                continue
            if entry.forgetting_risk > self.criteria.max_forgetting_risk:
                continue
            if entry.error_rate > self.criteria.max_error_rate:
                continue

            filtered.append(entry)

        # Sort by priority score (descending)
        filtered.sort(key=lambda e: e.get_priority_score(), reverse=True)

        # Select top entries
        return filtered[:min(batch_size, len(filtered))]

    def select_for_removal(
        self,
        entries: List[MultimodalEntry],
        count: int,
    ) -> List[MultimodalEntry]:
        """
        Select entries for removal (least important/novel).

        Args:
            entries: List of entries to select from
            count: Number of entries to select for removal

        Returns:
            List of entries to remove
        """
        # Sort by priority score (ascending - lowest priority first)
        sorted_entries = sorted(entries, key=lambda e: e.get_priority_score())

        # Also consider access count and timestamp
        # Prefer to remove old, rarely accessed entries
        sorted_entries.sort(key=lambda e: (
            e.get_priority_score(),
            e.access_count,
            e.timestamp.timestamp()
        ))

        return sorted_entries[:min(count, len(sorted_entries))]

    def get_high_priority_entries(
        self,
        entries: List[MultimodalEntry],
    ) -> List[MultimodalEntry]:
        """Get entries with high priority."""
        return [
            e for e in entries
            if e.priority in [MemoryPriority.CRITICAL, MemoryPriority.HIGH]
        ]

    def get_entries_by_forgetting_risk(
        self,
        entries: List[MultimodalEntry],
        min_risk: float = 0.5,
    ) -> List[MultimodalEntry]:
        """Get entries with high forgetting risk."""
        return [e for e in entries if e.forgetting_risk >= min_risk]

    def get_entries_by_error_rate(
        self,
        entries: List[MultimodalEntry],
        min_error: float = 0.3,
    ) -> List[MultimodalEntry]:
        """Get entries with high error rate."""
        return [e for e in entries if e.error_rate >= min_error]


class MemoryPriorityCalculator:
    """
    Calculates priority scores for memory entries.
    """

    def __init__(
        self,
        importance_weight: float = 0.5,
        novelty_weight: float = 0.3,
        difficulty_weight: float = 0.2,
        forgetting_risk_weight: float = 2.0,
        error_rate_weight: float = 1.5,
    ):
        self.importance_weight = importance_weight
        self.novelty_weight = novelty_weight
        self.difficulty_weight = difficulty_weight
        self.forgetting_risk_weight = forgetting_risk_weight
        self.error_rate_weight = error_rate_weight

    def calculate_priority(self, entry: MultimodalEntry) -> float:
        """
        Calculate priority score for an entry.

        Args:
            entry: The entry to calculate priority for

        Returns:
            Priority score (higher = more important)
        """
        # Base score from priority level
        priority_weights = {
            MemoryPriority.CRITICAL: 1.0,
            MemoryPriority.HIGH: 0.8,
            MemoryPriority.MEDIUM: 0.5,
            MemoryPriority.LOW: 0.2,
        }
        base_score = priority_weights.get(entry.priority, 0.5)

        # Boost by various factors
        score = base_score * (1 + entry.importance * self.importance_weight)
        score *= (1 + entry.novelty * self.novelty_weight)
        score *= (1 + entry.difficulty * self.difficulty_weight)
        score *= (1 + entry.forgetting_risk * self.forgetting_risk_weight)
        score *= (1 + entry.error_rate * self.error_rate_weight)

        return score

    def update_entry_priority(self, entry: MultimodalEntry) -> None:
        """Update an entry's priority based on current metrics."""
        # Recalculate priority
        new_priority_score = self.calculate_priority(entry)

        # Update priority level
        if new_priority_score >= 0.8:
            entry.priority = MemoryPriority.CRITICAL
        elif new_priority_score >= 0.6:
            entry.priority = MemoryPriority.HIGH
        elif new_priority_score >= 0.4:
            entry.priority = MemoryPriority.MEDIUM
        else:
            entry.priority = MemoryPriority.LOW
