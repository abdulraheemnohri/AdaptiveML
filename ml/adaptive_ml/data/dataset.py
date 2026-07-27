"""
Continual Dataset for Adaptive ML Framework.
Combines new data with replay buffer samples for continual learning.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from adaptive_ml.core.types import DatasetEntry, MemoryEntry, SamplingStrategy, Task


class ContinualDataset(Dataset):
    """
    Dataset for continual learning that combines new data with replay samples.
    
    Features:
    - Dynamic mixing of new data and replay buffer samples
    - Task-aware sampling
    - Support for various data formats (text, tensors, etc.)
    - Optional preprocessing hooks
    """

    def __init__(
        self,
        new_data: List[DatasetEntry],
        replay_buffer: Optional[List[MemoryEntry]] = None,
        replay_ratio: float = 0.3,
        sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        preprocess_fn: Optional[Callable] = None,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize ContinualDataset.
        
        Args:
            new_data: List of new data entries
            replay_buffer: List of replay buffer entries (optional)
            replay_ratio: Fraction of batch to sample from replay buffer (0-1)
            sampling_strategy: Strategy for sampling from replay buffer
            preprocess_fn: Optional preprocessing function for each entry
            task_weights: Optional weights for task-balanced sampling
        """
        self.new_data = new_data
        self.replay_buffer = replay_buffer or []
        self.replay_ratio = replay_ratio
        self.sampling_strategy = sampling_strategy
        self.preprocess_fn = preprocess_fn
        self.task_weights = task_weights or {}
        
        # Validate inputs
        if not 0 <= replay_ratio <= 1:
            raise ValueError(f"replay_ratio must be between 0 and 1, got {replay_ratio}")
        
        # Build indices for efficient sampling
        self._build_indices()

    def _build_indices(self) -> None:
        """Build indices for efficient sampling."""
        # New data indices
        self.new_indices = list(range(len(self.new_data)))
        
        # Replay buffer indices by task
        self.replay_indices_by_task: Dict[str, List[int]] = {}
        for idx, entry in enumerate(self.replay_buffer):
            if entry.task_id not in self.replay_indices_by_task:
                self.replay_indices_by_task[entry.task_id] = []
            self.replay_indices_by_task[entry.task_id].append(idx)
        
        # All replay indices
        self.replay_indices = list(range(len(self.replay_buffer)))

    def __len__(self) -> int:
        """Total number of samples (new + replay)."""
        return len(self.new_data) + len(self.replay_buffer)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Get a sample by index.
        
        Returns a tuple of (data, label) for compatibility with DataLoader.
        
        Note: This is a simplified version. For actual training, use sample_batch()
        which properly mixes new and replay data.
        """
        if idx < len(self.new_data):
            entry = self.new_data[idx]
            return entry.data, entry.label
        else:
            replay_idx = idx - len(self.new_data)
            entry = self.replay_buffer[replay_idx]
            return entry.data, entry.label

    def sample_batch(
        self,
        batch_size: int,
        return_tensors: str = "pt",  # "pt" for PyTorch, "np" for NumPy
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Sample a batch with proper mixing of new and replay data.
        
        Args:
            batch_size: Number of samples in the batch
            return_tensors: Format to return tensors in ("pt" or "np")
        
        Returns:
            Tuple of (inputs, labels, metadata)
        """
        # Calculate number of samples from each source
        num_replay = int(batch_size * self.replay_ratio)
        num_new = batch_size - num_replay
        
        # Sample new data
        new_samples = self._sample_new(num_new)
        
        # Sample replay data
        replay_samples = self._sample_replay(num_replay)
        
        # Combine samples
        all_samples = new_samples + replay_samples
        
        # Stack inputs and labels
        inputs = [s.data for s in all_samples]
        labels = [s.label for s in all_samples]
        metadata = {
            "task_ids": [s.task_id for s in all_samples],
            "is_replay": [s.is_replay for s in all_samples],
        }
        
        # Convert to tensors if needed
        if return_tensors == "pt":
            inputs = self._to_tensor(inputs)
            labels = self._to_tensor(labels) if labels[0] is not None else None
        elif return_tensors == "np":
            inputs = np.array(inputs)
            labels = np.array(labels) if labels[0] is not None else None
        
        return inputs, labels, metadata

    def _sample_new(self, num_samples: int) -> List[DatasetEntry]:
        """Sample from new data."""
        if num_samples <= 0:
            return []
        
        indices = np.random.choice(
            self.new_indices,
            size=min(num_samples, len(self.new_indices)),
            replace=num_samples > len(self.new_indices),
        )
        
        return [self.new_data[i] for i in indices]

    def _sample_replay(self, num_samples: int) -> List[DatasetEntry]:
        """Sample from replay buffer using the configured strategy."""
        if num_samples <= 0 or len(self.replay_buffer) == 0:
            return []
        
        if self.sampling_strategy == SamplingStrategy.UNIFORM:
            indices = np.random.choice(
                self.replay_indices,
                size=min(num_samples, len(self.replay_indices)),
                replace=num_samples > len(self.replay_indices),
            )
        elif self.sampling_strategy == SamplingStrategy.BALANCED:
            indices = self._sample_balanced(num_samples)
        elif self.sampling_strategy == SamplingStrategy.IMPORTANCE:
            indices = self._sample_importance(num_samples)
        elif self.sampling_strategy == SamplingStrategy.DIVERSITY:
            indices = self._sample_diversity(num_samples)
        elif self.sampling_strategy == SamplingStrategy.HARD_EXAMPLE:
            indices = self._sample_hard_examples(num_samples)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        return [
            DatasetEntry(
                data=entry.data,
                label=entry.label,
                task_id=entry.task_id,
                is_replay=True,
                metadata=entry.metadata,
            )
            for entry in [self.replay_buffer[i] for i in indices]
        ]

    def _sample_balanced(self, num_samples: int) -> List[int]:
        """Sample with class/task balancing."""
        if not self.replay_indices_by_task:
            return []
        
        # Get all tasks and their weights
        tasks = list(self.replay_indices_by_task.keys())
        if not tasks:
            return []
        
        # Use task weights if provided, otherwise uniform
        weights = [self.task_weights.get(t, 1.0) for t in tasks]
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(tasks)
            total_weight = len(tasks)
        
        # Calculate samples per task
        samples_per_task = []
        for t, w in zip(tasks, weights):
            n = int(num_samples * w / total_weight)
            samples_per_task.append((t, n))
        
        # Sample from each task
        indices = []
        for task, n in samples_per_task:
            task_indices = self.replay_indices_by_task[task]
            if n > 0 and task_indices:
                sampled = np.random.choice(
                    task_indices,
                    size=min(n, len(task_indices)),
                    replace=n > len(task_indices),
                )
                indices.extend(sampled)
        
        # If we didn't get enough samples, fill with random
        if len(indices) < num_samples:
            remaining = num_samples - len(indices)
            extra = np.random.choice(
                self.replay_indices,
                size=remaining,
                replace=remaining > len(self.replay_indices),
            )
            indices.extend(extra)
        
        return indices[:num_samples]

    def _sample_importance(self, num_samples: int) -> List[int]:
        """Sample based on importance scores."""
        if len(self.replay_buffer) == 0:
            return []
        
        # Get importance scores
        importances = [entry.importance for entry in self.replay_buffer]
        total = sum(importances)
        
        if total == 0:
            # Fall back to uniform sampling
            return self._sample_replay_uniform(num_samples)
        
        # Normalize to probabilities
        probs = [i / total for i in importances]
        
        # Sample with replacement
        indices = np.random.choice(
            self.replay_indices,
            size=num_samples,
            p=probs,
            replace=True,
        )
        
        return indices.tolist()

    def _sample_diversity(self, num_samples: int) -> List[int]:
        """
        Sample based on diversity (using embeddings).
        
        Note: This is a simplified version. For production, use FAISS.
        """
        if len(self.replay_buffer) == 0:
            return []
        
        # Get embeddings
        embeddings = []
        valid_indices = []
        for idx, entry in enumerate(self.replay_buffer):
            if entry.embedding is not None:
                embeddings.append(entry.embedding)
                valid_indices.append(idx)
        
        if len(embeddings) == 0:
            return self._sample_replay_uniform(num_samples)
        
        # Convert to numpy
        embeddings = np.array(embeddings)
        
        # Simple diversity sampling: farthest point from mean
        mean_embedding = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - mean_embedding, axis=1)
        
        # Get top-k most diverse
        top_k = min(num_samples, len(valid_indices))
        selected = np.argsort(-distances)[:top_k]
        
        return [valid_indices[i] for i in selected]

    def _sample_hard_examples(self, num_samples: int) -> List[int]:
        """Sample based on uncertainty (hard examples)."""
        if len(self.replay_buffer) == 0:
            return []
        
        # Get uncertainties
        uncertainties = [entry.uncertainty for entry in self.replay_buffer]
        
        # Sample with probability proportional to uncertainty
        total = sum(uncertainties)
        if total == 0:
            return self._sample_replay_uniform(num_samples)
        
        probs = [u / total for u in uncertainties]
        indices = np.random.choice(
            self.replay_indices,
            size=num_samples,
            p=probs,
            replace=True,
        )
        
        return indices.tolist()

    def _sample_replay_uniform(self, num_samples: int) -> List[int]:
        """Uniform sampling from replay buffer."""
        return np.random.choice(
            self.replay_indices,
            size=min(num_samples, len(self.replay_indices)),
            replace=num_samples > len(self.replay_indices),
        ).tolist()

    def _to_tensor(self, data: List[Any]) -> torch.Tensor:
        """Convert data to PyTorch tensor."""
        if isinstance(data[0], torch.Tensor):
            return torch.stack(data)
        elif isinstance(data[0], np.ndarray):
            return torch.from_numpy(np.stack(data))
        elif isinstance(data[0], (int, float)):
            return torch.tensor(data)
        else:
            # For text or other types, return as-is
            return data

    def add_data(self, entries: List[DatasetEntry]) -> None:
        """Add new data entries."""
        self.new_data.extend(entries)
        self._build_indices()

    def add_replay_data(self, entries: List[MemoryEntry]) -> None:
        """Add replay buffer entries."""
        self.replay_buffer.extend(entries)
        self._build_indices()

    def clear_replay(self) -> None:
        """Clear the replay buffer."""
        self.replay_buffer = []
        self._build_indices()

    def get_task_distribution(self) -> Dict[str, int]:
        """Get the distribution of tasks in the dataset."""
        task_counts: Dict[str, int] = {}
        
        for entry in self.new_data:
            task_counts[entry.task_id] = task_counts.get(entry.task_id, 0) + 1
        
        for entry in self.replay_buffer:
            task_counts[entry.task_id] = task_counts.get(entry.task_id, 0) + 1
        
        return task_counts
