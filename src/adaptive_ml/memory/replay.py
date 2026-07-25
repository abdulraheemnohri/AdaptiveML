"""
Replay Buffer for Adaptive ML Framework.
Implements experience replay with various sampling strategies for continual learning.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import MemoryEntry, SamplingStrategy


@dataclass
class ReplayStats:
    """Statistics for the replay buffer."""

    size: int = 0
    capacity: int = 0
    num_tasks: int = 0
    task_distribution: Dict[str, int] = field(default_factory=dict)
    utilization: float = field(default=0.0)  # size / capacity
    
    def __post_init__(self):
        if self.capacity > 0:
            self.utilization = self.size / self.capacity
        else:
            self.utilization = 0.0


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.
    
    Supports multiple sampling strategies:
    - Uniform: Random sampling
    - Balanced: Class/task-balanced sampling
    - Importance: Importance-weighted sampling
    - Diversity: Diversity-based sampling (using embeddings)
    - Hard Example: High-uncertainty examples
    
    Features:
    - Reservoir sampling for streaming data
    - Task-aware storage and sampling
    - Importance and uncertainty tracking
    - Embedding-based diversity sampling
    - Efficient batch sampling
    
    Usage:
        buffer = ReplayBuffer(config)
        
        # Add examples
        buffer.add(task_id="task_a", data=x, label=y)
        
        # Sample a batch
        batch = buffer.sample(batch_size=32, strategy=SamplingStrategy.BALANCED)
        
        # Get statistics
        stats = buffer.get_stats()
    """

    def __init__(
        self,
        config: Optional[AdaptiveMLConfig] = None,
        capacity: Optional[int] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
    ):
        """
        Initialize ReplayBuffer.
        
        Args:
            config: AdaptiveMLConfig instance
            capacity: Maximum number of entries (overrides config)
            sampling_strategy: Default sampling strategy (overrides config)
        """
        self.config = config or AdaptiveMLConfig()
        self.capacity = capacity or self.config.memory.buffer_size
        self.default_strategy = sampling_strategy or self.config.memory.sampling_strategy
        
        # Storage
        self.buffer: List[MemoryEntry] = []
        self.task_indices: Dict[str, List[int]] = defaultdict(list)  # task_id -> [indices]
        self.task_counts: Dict[str, int] = defaultdict(int)
        
        # Reservoir sampling state
        self.reservoir_size = self.capacity
        self.num_seen = 0
        
        # Diversity sampling (FAISS index placeholder)
        self.embeddings: List[np.ndarray] = []
        self.use_faiss = False
        self.faiss_index = None
        
        # Importance and uncertainty tracking
        self.importance_scores: List[float] = []
        self.uncertainty_scores: List[float] = []

    def add(
        self,
        task_id: str,
        data: Any,
        label: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
        importance: float = 1.0,
        uncertainty: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add an entry to the replay buffer.
        
        Args:
            task_id: Task identifier
            data: Input data
            label: Ground truth label (optional)
            embedding: Precomputed embedding for diversity sampling (optional)
            importance: Importance score for weighted sampling
            uncertainty: Model uncertainty for hard example replay
            metadata: Additional metadata
        
        Returns:
            Index of the added entry
        """
        # Create entry
        entry = MemoryEntry(
            task_id=task_id,
            data=data,
            label=label,
            embedding=embedding,
            importance=importance,
            uncertainty=uncertainty,
            metadata=metadata or {},
        )
        
        # Reservoir sampling
        if len(self.buffer) < self.capacity:
            # Buffer not full, add directly
            idx = len(self.buffer)
            self.buffer.append(entry)
            self.task_indices[task_id].append(idx)
            self.task_counts[task_id] += 1
            
            if embedding is not None:
                self.embeddings.append(embedding)
            self.importance_scores.append(importance)
            self.uncertainty_scores.append(uncertainty)
        else:
            # Buffer full, use reservoir sampling
            self.num_seen += 1
            replace_idx = random.randint(0, self.num_seen - 1)
            
            if replace_idx < self.capacity:
                # Replace existing entry
                old_entry = self.buffer[replace_idx]
                old_task_id = old_entry.task_id
                
                # Remove from task indices
                self.task_indices[old_task_id].remove(replace_idx)
                self.task_counts[old_task_id] -= 1
                if self.task_counts[old_task_id] == 0:
                    del self.task_counts[old_task_id]
                    del self.task_indices[old_task_id]
                
                # Replace entry
                self.buffer[replace_idx] = entry
                self.task_indices[task_id].append(replace_idx)
                self.task_counts[task_id] += 1
                
                if embedding is not None:
                    if len(self.embeddings) > replace_idx:
                        self.embeddings[replace_idx] = embedding
                    else:
                        self.embeddings.append(embedding)
                
                if len(self.importance_scores) > replace_idx:
                    self.importance_scores[replace_idx] = importance
                else:
                    self.importance_scores.append(importance)
                
                if len(self.uncertainty_scores) > replace_idx:
                    self.uncertainty_scores[replace_idx] = uncertainty
                else:
                    self.uncertainty_scores.append(uncertainty)
        
        return len(self.buffer) - 1

    def sample(
        self,
        batch_size: int,
        strategy: Optional[SamplingStrategy] = None,
        task_weights: Optional[Dict[str, float]] = None,
        return_entries: bool = False,
    ) -> Union[List[MemoryEntry], Tuple[List[Any], List[Optional[Any]], Dict[str, Any]]]:
        """
        Sample a batch from the replay buffer.
        
        Args:
            batch_size: Number of samples to return
            strategy: Sampling strategy to use (defaults to buffer's default)
            task_weights: Optional weights for task-balanced sampling
            return_entries: If True, return MemoryEntry objects; else return (data, labels, metadata)
        
        Returns:
            List of MemoryEntry objects or tuple of (data, labels, metadata)
        """
        strategy = strategy or self.default_strategy
        
        if len(self.buffer) == 0:
            if return_entries:
                return []
            return [], [], {}
        
        # Sample indices
        indices = self._sample_indices(batch_size, strategy, task_weights)
        
        # Get entries
        entries = [self.buffer[i] for i in indices]
        
        if return_entries:
            return entries
        
        # Extract data, labels, and metadata
        data = [e.data for e in entries]
        labels = [e.label for e in entries]
        metadata = {
            "task_ids": [e.task_id for e in entries],
            "importances": [e.importance for e in entries],
            "uncertainties": [e.uncertainty for e in entries],
            "is_replay": [True] * len(entries),
        }
        
        return data, labels, metadata

    def _sample_indices(
        self,
        batch_size: int,
        strategy: SamplingStrategy,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> List[int]:
        """Sample indices using the specified strategy."""
        if strategy == SamplingStrategy.UNIFORM:
            return self._sample_uniform(batch_size)
        elif strategy == SamplingStrategy.BALANCED:
            return self._sample_balanced(batch_size, task_weights)
        elif strategy == SamplingStrategy.IMPORTANCE:
            return self._sample_importance(batch_size)
        elif strategy == SamplingStrategy.DIVERSITY:
            return self._sample_diversity(batch_size)
        elif strategy == SamplingStrategy.HARD_EXAMPLE:
            return self._sample_hard_examples(batch_size)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def _sample_uniform(self, batch_size: int) -> List[int]:
        """Uniform random sampling."""
        return random.choices(
            range(len(self.buffer)),
            k=min(batch_size, len(self.buffer)),
        )

    def _sample_balanced(
        self,
        batch_size: int,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> List[int]:
        """Task-balanced sampling."""
        if not self.task_indices:
            return self._sample_uniform(batch_size)
        
        tasks = list(self.task_indices.keys())
        if not tasks:
            return self._sample_uniform(batch_size)
        
        # Use provided weights or uniform
        weights = task_weights or {t: 1.0 for t in tasks}
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            weights = {t: 1.0 for t in tasks}
            total_weight = len(tasks)
        
        # Calculate samples per task
        indices = []
        for task in tasks:
            weight = weights.get(task, 1.0)
            n = int(batch_size * weight / total_weight)
            
            task_idxs = self.task_indices[task]
            if n > 0 and task_idxs:
                sampled = random.choices(
                    task_idxs,
                    k=min(n, len(task_idxs)),
                )
                indices.extend(sampled)
        
        # Fill remaining with random samples
        if len(indices) < batch_size:
            remaining = batch_size - len(indices)
            extra = random.choices(
                range(len(self.buffer)),
                k=remaining,
            )
            indices.extend(extra)
        
        return indices[:batch_size]

    def _sample_importance(self, batch_size: int) -> List[int]:
        """Importance-weighted sampling."""
        if len(self.importance_scores) == 0 or sum(self.importance_scores) == 0:
            return self._sample_uniform(batch_size)
        
        # Normalize importance scores
        total = sum(self.importance_scores)
        probs = [i / total for i in self.importance_scores]
        
        return random.choices(
            range(len(self.buffer)),
            weights=probs,
            k=batch_size,
        )

    def _sample_diversity(self, batch_size: int) -> List[int]:
        """
        Diversity-based sampling using embeddings.
        
        Uses farthest-point sampling for diversity.
        """
        if len(self.embeddings) == 0:
            return self._sample_uniform(batch_size)
        
        # Convert to numpy array
        embeddings = np.array(self.embeddings)
        
        # Simple diversity: select points farthest from mean
        mean_embedding = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - mean_embedding, axis=1)
        
        # Get top-k most diverse
        top_k = min(batch_size, len(self.buffer))
        selected = np.argsort(-distances)[:top_k]
        
        return selected.tolist()

    def _sample_hard_examples(self, batch_size: int) -> List[int]:
        """Sample based on uncertainty (hard examples)."""
        if len(self.uncertainty_scores) == 0 or sum(self.uncertainty_scores) == 0:
            return self._sample_uniform(batch_size)
        
        # Normalize uncertainty scores
        total = sum(self.uncertainty_scores)
        probs = [u / total for u in self.uncertainty_scores]
        
        return random.choices(
            range(len(self.buffer)),
            weights=probs,
            k=batch_size,
        )

    def get_stats(self) -> ReplayStats:
        """Get statistics about the replay buffer."""
        return ReplayStats(
            size=len(self.buffer),
            capacity=self.capacity,
            num_tasks=len(self.task_counts),
            task_distribution=dict(self.task_counts),
            utilization=len(self.buffer) / self.capacity if self.capacity > 0 else 0.0,
        )

    def get_task_data(self, task_id: str) -> List[MemoryEntry]:
        """Get all entries for a specific task."""
        indices = self.task_indices.get(task_id, [])
        return [self.buffer[i] for i in indices]

    def clear(self) -> None:
        """Clear the replay buffer."""
        self.buffer = []
        self.task_indices = defaultdict(list)
        self.task_counts = defaultdict(int)
        self.embeddings = []
        self.importance_scores = []
        self.uncertainty_scores = []
        self.num_seen = 0

    def remove_task(self, task_id: str) -> int:
        """
        Remove all entries for a specific task.
        
        Returns:
            Number of entries removed
        """
        if task_id not in self.task_indices:
            return 0
        
        indices = self.task_indices[task_id]
        count = len(indices)
        
        # Remove from buffer (mark as None for now)
        for idx in indices:
            if idx < len(self.buffer):
                self.buffer[idx] = None
        
        # Clean up
        self.buffer = [e for e in self.buffer if e is not None]
        del self.task_indices[task_id]
        del self.task_counts[task_id]
        
        # Rebuild indices
        self._rebuild_indices()
        
        return count

    def _rebuild_indices(self) -> None:
        """Rebuild task indices after modifications."""
        self.task_indices = defaultdict(list)
        self.task_counts = defaultdict(int)
        
        for idx, entry in enumerate(self.buffer):
            if entry is not None:
                self.task_indices[entry.task_id].append(idx)
                self.task_counts[entry.task_id] += 1

    def to_list(self) -> List[MemoryEntry]:
        """Get all entries as a list."""
        return list(self.buffer)

    def __len__(self) -> int:
        """Number of entries in the buffer."""
        return len(self.buffer)

    def __getitem__(self, idx: int) -> MemoryEntry:
        """Get entry by index."""
        return self.buffer[idx]

    def __iter__(self):
        """Iterate over entries."""
        return iter(self.buffer)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ReplayBuffer(size={stats.size}, capacity={stats.capacity}, "
            f"tasks={stats.num_tasks}, utilization={stats.utilization:.2%})"
        )
