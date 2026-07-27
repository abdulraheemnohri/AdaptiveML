"""
Multimodal Replay Memory for Qwen2.5-Omni-3B.
Implements priority-based sampling, modality-aware storage, and adaptive replay ratios.
"""

import logging
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from adaptive_ml.qwen_omni.core import (
    DomainType,
    MemoryPriority,
    ModalityType,
    MultimodalData,
    MultimodalEntry,
    MemoryCandidate,
    ReplayStats,
)

logger = logging.getLogger(__name__)


class MultimodalReplayBuffer:
    """
    Multimodal replay buffer that stores and samples experiences.
    Supports priority-based sampling, modality filtering, and domain filtering.
    """

    def __init__(
        self,
        max_entries: int = 10000,
        max_entries_per_modality: Optional[Dict[ModalityType, int]] = None,
        max_entries_per_domain: int = 1000,
        sampling_strategy: str = "priority_based",
        temperature: float = 1.0,
        importance_weight: float = 0.5,
        novelty_weight: float = 0.3,
        difficulty_weight: float = 0.2,
        forgetting_risk_weight: float = 2.0,
        error_rate_weight: float = 1.5,
    ):
        self.max_entries = max_entries
        self.max_entries_per_modality = max_entries_per_modality or {
            ModalityType.TEXT: 3000,
            ModalityType.VISION: 2000,
            ModalityType.AUDIO: 2000,
            ModalityType.VIDEO: 1500,
            ModalityType.SPEECH: 1500,
            ModalityType.MULTI_MODAL: 1000,
        }
        self.max_entries_per_domain = max_entries_per_domain
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature

        # Priority weights
        self.importance_weight = importance_weight
        self.novelty_weight = novelty_weight
        self.difficulty_weight = difficulty_weight
        self.forgetting_risk_weight = forgetting_risk_weight
        self.error_rate_weight = error_rate_weight

        # Storage
        self._entries: Dict[str, MultimodalEntry] = {}
        self._entries_by_modality: Dict[ModalityType, Dict[str, MultimodalEntry]] = defaultdict(dict)
        self._entries_by_domain: Dict[DomainType, Dict[str, MultimodalEntry]] = defaultdict(dict)
        self._entries_by_priority: Dict[MemoryPriority, Dict[str, MultimodalEntry]] = defaultdict(dict)

        # Indexes for fast lookup
        self._id_to_entry: Dict[str, MultimodalEntry] = {}
        self._modality_counts: Dict[ModalityType, int] = defaultdict(int)
        self._domain_counts: Dict[DomainType, int] = defaultdict(int)
        self._priority_counts: Dict[MemoryPriority, int] = defaultdict(int)

        # Statistics
        self._stats = ReplayStats()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry_id: str) -> bool:
        return entry_id in self._entries

    def __getitem__(self, entry_id: str) -> MultimodalEntry:
        return self._entries[entry_id]

    def add(self, entry: MultimodalEntry) -> str:
        """
        Add a new entry to the replay buffer.

        Args:
            entry: MultimodalEntry to add

        Returns:
            The ID of the added entry
        """
        # Generate ID if not provided
        if not entry.id:
            entry.id = str(uuid.uuid4())

        # Check if we're at capacity
        if len(self._entries) >= self.max_entries:
            # Remove oldest entry
            self._evict_oldest()

        # Check modality capacity
        for modality in entry.data.modalities:
            if self._modality_counts[modality] >= self.max_entries_per_modality.get(modality, 1000):
                # Remove oldest entry for this modality
                self._evict_oldest_for_modality(modality)

        # Check domain capacity
        if self._domain_counts[entry.domain] >= self.max_entries_per_domain:
            self._evict_oldest_for_domain(entry.domain)

        # Add entry
        self._entries[entry.id] = entry
        self._id_to_entry[entry.id] = entry

        # Add to modality index
        for modality in entry.data.modalities:
            self._entries_by_modality[modality][entry.id] = entry
            self._modality_counts[modality] += 1

        # Add to domain index
        self._entries_by_domain[entry.domain][entry.id] = entry
        self._domain_counts[entry.domain] += 1

        # Add to priority index
        self._entries_by_priority[entry.priority][entry.id] = entry
        self._priority_counts[entry.priority] += 1

        # Update statistics
        self._update_stats()

        logger.debug(f"Added entry {entry.id} with modalities {entry.data.modalities}")

        return entry.id

    def _evict_oldest(self) -> None:
        """Remove the oldest entry from the buffer."""
        if not self._entries:
            return

        # Find oldest entry
        oldest_id = min(self._entries.keys(), key=lambda x: self._entries[x].timestamp)
        self._remove_entry(oldest_id)

    def _evict_oldest_for_modality(self, modality: ModalityType) -> None:
        """Remove the oldest entry for a specific modality."""
        if modality not in self._entries_by_modality or not self._entries_by_modality[modality]:
            return

        oldest_id = min(
            self._entries_by_modality[modality].keys(),
            key=lambda x: self._entries_by_modality[modality][x].timestamp
        )
        self._remove_entry(oldest_id)

    def _evict_oldest_for_domain(self, domain: DomainType) -> None:
        """Remove the oldest entry for a specific domain."""
        if domain not in self._entries_by_domain or not self._entries_by_domain[domain]:
            return

        oldest_id = min(
            self._entries_by_domain[domain].keys(),
            key=lambda x: self._entries_by_domain[domain][x].timestamp
        )
        self._remove_entry(oldest_id)

    def _remove_entry(self, entry_id: str) -> None:
        """Remove an entry from all indexes."""
        if entry_id not in self._entries:
            return

        entry = self._entries[entry_id]

        # Remove from main storage
        del self._entries[entry_id]
        del self._id_to_entry[entry_id]

        # Remove from modality index
        for modality in entry.data.modalities:
            if entry_id in self._entries_by_modality[modality]:
                del self._entries_by_modality[modality][entry_id]
                self._modality_counts[modality] -= 1

        # Remove from domain index
        if entry_id in self._entries_by_domain[entry.domain]:
            del self._entries_by_domain[entry.domain][entry_id]
            self._domain_counts[entry.domain] -= 1

        # Remove from priority index
        if entry_id in self._entries_by_priority[entry.priority]:
            del self._entries_by_priority[entry.priority][entry_id]
            self._priority_counts[entry.priority] -= 1

        # Update statistics
        self._update_stats()

        logger.debug(f"Removed entry {entry_id}")

    def _update_stats(self) -> None:
        """Update statistics."""
        self._stats.total_entries = len(self._entries)
        self._stats.entries_by_modality = dict(self._modality_counts)
        self._stats.entries_by_domain = dict(self._domain_counts)
        self._stats.entries_by_priority = dict(self._priority_counts)

        # Calculate averages
        if self._entries:
            all_importance = [e.importance for e in self._entries.values()]
            all_novelty = [e.novelty for e in self._entries.values()]
            all_forgetting_risk = [e.forgetting_risk for e in self._entries.values()]
            all_error_rate = [e.error_rate for e in self._entries.values()]

            self._stats.average_importance = np.mean(all_importance)
            self._stats.average_novelty = np.mean(all_novelty)
            self._stats.average_forgetting_risk = np.mean(all_forgetting_risk)
            self._stats.average_error_rate = np.mean(all_error_rate)
        else:
            self._stats.average_importance = 0.0
            self._stats.average_novelty = 0.0
            self._stats.average_forgetting_risk = 0.0
            self._stats.average_error_rate = 0.0

        # Calculate utilization rate
        total_capacity = sum(self.max_entries_per_modality.values())
        self._stats.utilization_rate = len(self._entries) / max(total_capacity, 1)

    def sample(
        self,
        batch_size: int = 1,
        modalities: Optional[List[ModalityType]] = None,
        domains: Optional[List[DomainType]] = None,
        priorities: Optional[List[MemoryPriority]] = None,
        min_importance: float = 0.0,
        min_novelty: float = 0.0,
    ) -> List[MultimodalEntry]:
        """
        Sample entries from the replay buffer.

        Args:
            batch_size: Number of entries to sample
            modalities: Filter by modalities (None = all)
            domains: Filter by domains (None = all)
            priorities: Filter by priorities (None = all)
            min_importance: Minimum importance threshold
            min_novelty: Minimum novelty threshold

        Returns:
            List of sampled entries
        """
        if not self._entries:
            return []

        # Get candidate entries
        candidates = list(self._entries.values())

        # Apply filters
        if modalities:
            candidates = [e for e in candidates if any(m in modalities for m in e.data.modalities)]

        if domains:
            candidates = [e for e in candidates if e.domain in domains]

        if priorities:
            candidates = [e for e in candidates if e.priority in priorities]

        candidates = [e for e in candidates if e.importance >= min_importance]
        candidates = [e for e in candidates if e.novelty >= min_novelty]

        if not candidates:
            return []

        # Sample based on strategy
        if self.sampling_strategy == "uniform":
            sampled = random.sample(candidates, min(batch_size, len(candidates)))

        elif self.sampling_strategy == "balanced":
            sampled = self._sample_balanced(candidates, batch_size)

        elif self.sampling_strategy == "priority_based":
            sampled = self._sample_priority_based(candidates, batch_size)

        else:  # adaptive
            sampled = self._sample_adaptive(candidates, batch_size)

        # Update access statistics
        for entry in sampled:
            entry.last_accessed = datetime.now()
            entry.access_count += 1

        return sampled

    def _sample_balanced(self, candidates: List[MultimodalEntry], batch_size: int) -> List[MultimodalEntry]:
        """Sample with balanced representation across modalities."""
        # Group by modality
        modality_groups = defaultdict(list)
        for entry in candidates:
            for modality in entry.data.modalities:
                modality_groups[modality].append(entry)

        # Sample proportionally from each modality
        sampled = []
        remaining = batch_size

        for modality, entries in modality_groups.items():
            if not entries:
                continue

            # Allocate proportionally
            proportion = len(entries) / len(candidates)
            count = max(1, int(proportion * batch_size))
            count = min(count, len(entries), remaining)

            sampled.extend(random.sample(entries, count))
            remaining -= count

            if remaining <= 0:
                break

        # If we still need more, sample from remaining
        if remaining > 0 and len(sampled) < batch_size:
            remaining_candidates = [e for e in candidates if e not in sampled]
            sampled.extend(random.sample(remaining_candidates, min(remaining, len(remaining_candidates))))

        return sampled[:batch_size]

    def _sample_priority_based(self, candidates: List[MultimodalEntry], batch_size: int) -> List[MultimodalEntry]:
        """Sample based on priority scores."""
        # Calculate priority scores
        scored_entries = []
        for entry in candidates:
            score = entry.get_priority_score()
            scored_entries.append((entry, score))

        # Sort by score (descending)
        scored_entries.sort(key=lambda x: x[1], reverse=True)

        # Sample with temperature
        if self.temperature > 0:
            # Apply softmax with temperature
            scores = np.array([s for _, s in scored_entries])
            scores = scores - np.max(scores)  # For numerical stability
            exp_scores = np.exp(scores / self.temperature)
            probs = exp_scores / np.sum(exp_scores)

            # Sample based on probabilities
            indices = np.random.choice(len(scored_entries), size=min(batch_size, len(scored_entries)), p=probs, replace=False)
            sampled = [scored_entries[i][0] for i in indices]
        else:
            # Deterministic sampling (highest priority first)
            sampled = [entry for entry, _ in scored_entries[:batch_size]]

        return sampled

    def _sample_adaptive(self, candidates: List[MultimodalEntry], batch_size: int) -> List[MultimodalEntry]:
        """Adaptive sampling that considers multiple factors."""
        # Calculate composite scores
        scored_entries = []
        for entry in candidates:
            # Base score from priority
            base_score = entry.get_priority_score()

            # Boost by forgetting risk and error rate
            forgetting_boost = 1.0 + entry.forgetting_risk * self.forgetting_risk_weight
            error_boost = 1.0 + entry.error_rate * self.error_rate_weight

            # Combine scores
            composite_score = base_score * forgetting_boost * error_boost

            scored_entries.append((entry, composite_score))

        # Sort by composite score
        scored_entries.sort(key=lambda x: x[1], reverse=True)

        # Sample with temperature
        if self.temperature > 0:
            scores = np.array([s for _, s in scored_entries])
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores / self.temperature)
            probs = exp_scores / np.sum(exp_scores)

            indices = np.random.choice(len(scored_entries), size=min(batch_size, len(scored_entries)), p=probs, replace=False)
            sampled = [scored_entries[i][0] for i in indices]
        else:
            sampled = [entry for entry, _ in scored_entries[:batch_size]]

        return sampled

    def get_stats(self) -> ReplayStats:
        """Get current statistics."""
        return self._stats

    def get_entries_by_modality(self, modality: ModalityType) -> List[MultimodalEntry]:
        """Get all entries for a specific modality."""
        return list(self._entries_by_modality.get(modality, {}).values())

    def get_entries_by_domain(self, domain: DomainType) -> List[MultimodalEntry]:
        """Get all entries for a specific domain."""
        return list(self._entries_by_domain.get(domain, {}).values())

    def get_entries_by_priority(self, priority: MemoryPriority) -> List[MultimodalEntry]:
        """Get all entries for a specific priority level."""
        return list(self._entries_by_priority.get(priority, {}).values())

    def update_entry(self, entry_id: str, **kwargs: Any) -> bool:
        """Update an existing entry."""
        if entry_id not in self._entries:
            return False

        entry = self._entries[entry_id]

        # Update fields
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        # Update in all indexes
        # Remove old entry
        self._remove_entry(entry_id)
        # Add updated entry
        self.add(entry)

        return True

    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID."""
        if entry_id not in self._entries:
            return False

        self._remove_entry(entry_id)
        return True

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._entries_by_modality.clear()
        self._entries_by_domain.clear()
        self._entries_by_priority.clear()
        self._id_to_entry.clear()
        self._modality_counts.clear()
        self._domain_counts.clear()
        self._priority_counts.clear()
        self._stats = ReplayStats()

        logger.info("Cleared replay buffer")

    def to_list(self) -> List[MultimodalEntry]:
        """Get all entries as a list."""
        return list(self._entries.values())

    def save(self, path: str) -> None:
        """Save replay buffer to file (placeholder)."""
        # In a full implementation, this would save to disk
        logger.info(f"Saving replay buffer to {path}")

    def load(self, path: str) -> None:
        """Load replay buffer from file (placeholder)."""
        # In a full implementation, this would load from disk
        logger.info(f"Loading replay buffer from {path}")


class ReplayMemory:
    """
    High-level replay memory manager.
    Combines multiple replay buffers and provides unified interface.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        general_buffer: Optional[MultimodalReplayBuffer] = None,
        domain_buffers: Optional[Dict[DomainType, MultimodalReplayBuffer]] = None,
    ):
        self.config = config

        # Create buffers
        self.general_buffer = general_buffer or MultimodalReplayBuffer(
            max_entries=5000,
            max_entries_per_modality={
                ModalityType.TEXT: 2000,
                ModalityType.VISION: 1000,
                ModalityType.AUDIO: 1000,
                ModalityType.VIDEO: 500,
                ModalityType.SPEECH: 500,
                ModalityType.MULTI_MODAL: 500,
            }
        )

        self.domain_buffers = domain_buffers or {}

        # Statistics
        self._total_stats = ReplayStats()

    def add(self, entry: MultimodalEntry) -> str:
        """
        Add an entry to the appropriate buffer.

        Args:
            entry: Entry to add

        Returns:
            Entry ID
        """
        # Add to general buffer
        entry_id = self.general_buffer.add(entry)

        # Add to domain-specific buffer if it exists
        if entry.domain not in self.domain_buffers:
            self.domain_buffers[entry.domain] = MultimodalReplayBuffer(
                max_entries=1000,
                max_entries_per_modality={
                    ModalityType.TEXT: 400,
                    ModalityType.VISION: 200,
                    ModalityType.AUDIO: 200,
                    ModalityType.VIDEO: 100,
                    ModalityType.SPEECH: 100,
                    ModalityType.MULTI_MODAL: 100,
                }
            )

        self.domain_buffers[entry.domain].add(entry)

        # Update total stats
        self._update_total_stats()

        return entry_id

    def sample(
        self,
        batch_size: int = 1,
        domain: Optional[DomainType] = None,
        **kwargs: Any
    ) -> List[MultimodalEntry]:
        """
        Sample entries from memory.

        Args:
            batch_size: Number of entries to sample
            domain: Optional domain to sample from
            **kwargs: Additional sampling parameters

        Returns:
            List of sampled entries
        """
        if domain and domain in self.domain_buffers:
            return self.domain_buffers[domain].sample(batch_size, **kwargs)
        else:
            return self.general_buffer.sample(batch_size, **kwargs)

    def _update_total_stats(self) -> None:
        """Update total statistics."""
        total_entries = len(self.general_buffer)

        entries_by_modality = defaultdict(int)
        entries_by_domain = defaultdict(int)
        entries_by_priority = defaultdict(int)

        all_importance = []
        all_novelty = []
        all_forgetting_risk = []
        all_error_rate = []

        # Aggregate from general buffer
        for entry in self.general_buffer.to_list():
            for modality in entry.data.modalities:
                entries_by_modality[modality] += 1
            entries_by_domain[entry.domain] += 1
            entries_by_priority[entry.priority] += 1
            all_importance.append(entry.importance)
            all_novelty.append(entry.novelty)
            all_forgetting_risk.append(entry.forgetting_risk)
            all_error_rate.append(entry.error_rate)

        # Aggregate from domain buffers
        for domain, buffer in self.domain_buffers.items():
            for entry in buffer.to_list():
                for modality in entry.data.modalities:
                    entries_by_modality[modality] += 1
                entries_by_domain[domain] += 1
                entries_by_priority[entry.priority] += 1
                all_importance.append(entry.importance)
                all_novelty.append(entry.novelty)
                all_forgetting_risk.append(entry.forgetting_risk)
                all_error_rate.append(entry.error_rate)

        self._total_stats.total_entries = total_entries
        self._total_stats.entries_by_modality = dict(entries_by_modality)
        self._total_stats.entries_by_domain = dict(entries_by_domain)
        self._total_stats.entries_by_priority = dict(entries_by_priority)

        if all_importance:
            self._total_stats.average_importance = np.mean(all_importance)
            self._total_stats.average_novelty = np.mean(all_novelty)
            self._total_stats.average_forgetting_risk = np.mean(all_forgetting_risk)
            self._total_stats.average_error_rate = np.mean(all_error_rate)

        total_capacity = sum(self.general_buffer.max_entries_per_modality.values())
        self._total_stats.utilization_rate = total_entries / max(total_capacity, 1)

    def get_stats(self) -> ReplayStats:
        """Get total statistics."""
        return self._total_stats

    def get_buffer(self, domain: Optional[DomainType] = None) -> MultimodalReplayBuffer:
        """Get a specific buffer."""
        if domain and domain in self.domain_buffers:
            return self.domain_buffers[domain]
        return self.general_buffer

    def clear(self) -> None:
        """Clear all buffers."""
        self.general_buffer.clear()
        for buffer in self.domain_buffers.values():
            buffer.clear()
        self.domain_buffers.clear()
        self._total_stats = ReplayStats()


class ExperienceReplay:
    """
    Experience replay manager that handles replay scheduling and mixing.
    """

    def __init__(
        self,
        replay_buffer: Optional[MultimodalReplayBuffer] = None,
        replay_ratio: float = 0.3,
        min_replay_ratio: float = 0.1,
        max_replay_ratio: float = 0.7,
    ):
        self.replay_buffer = replay_buffer or MultimodalReplayBuffer()
        self.replay_ratio = replay_ratio
        self.min_replay_ratio = min_replay_ratio
        self.max_replay_ratio = max_replay_ratio

        # Adaptive replay
        self._current_replay_ratio = replay_ratio
        self._forgetting_detected = False

    def get_replay_batch(
        self,
        new_data_batch: List[Any],
        batch_size: int,
        **kwargs: Any
    ) -> Tuple[List[Any], List[Any]]:
        """
        Get a batch with replay data mixed in.

        Args:
            new_data_batch: New data batch
            batch_size: Total batch size
            **kwargs: Sampling parameters

        Returns:
            Tuple of (new_data, replay_data)
        """
        # Calculate number of replay samples
        replay_count = int(self._current_replay_ratio * batch_size)
        new_count = batch_size - replay_count

        # Ensure we have enough new data
        if len(new_data_batch) < new_count:
            new_count = len(new_data_batch)
            replay_count = batch_size - new_count

        # Sample replay data
        replay_data = self.replay_buffer.sample(replay_count, **kwargs)

        # Select new data
        new_data = new_data_batch[:new_count]

        return new_data, replay_data

    def adjust_replay_ratio(self, forgetting_detected: bool, forgetting_level: float = 0.0) -> None:
        """
        Adjust replay ratio based on forgetting detection.

        Args:
            forgetting_detected: Whether forgetting was detected
            forgetting_level: Level of forgetting (0-1)
        """
        self._forgetting_detected = forgetting_detected

        if forgetting_detected:
            # Increase replay ratio based on forgetting level
            increase = forgetting_level * 0.4  # Max increase of 0.4
            self._current_replay_ratio = min(
                self.max_replay_ratio,
                self._current_replay_ratio + increase
            )
        else:
            # Gradually decrease replay ratio
            self._current_replay_ratio = max(
                self.min_replay_ratio,
                self._current_replay_ratio - 0.01
            )

        logger.info(f"Adjusted replay ratio to {self._current_replay_ratio:.2f}")

    def get_current_replay_ratio(self) -> float:
        """Get current replay ratio."""
        return self._current_replay_ratio

    def add_to_buffer(self, entry: MultimodalEntry) -> str:
        """Add an entry to the replay buffer."""
        return self.replay_buffer.add(entry)

    def get_buffer_stats(self) -> ReplayStats:
        """Get replay buffer statistics."""
        return self.replay_buffer.get_stats()
