"""
Dynamic LoRA Rank Adaptation for Adaptive ML Framework.
Automatically adjusts LoRA rank based on task complexity and performance.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import AdapterType


@dataclass
class DynamicLoRAConfig:
    """Configuration for dynamic LoRA rank adaptation."""

    enabled: bool = True
    min_rank: int = 8
    max_rank: int = 64
    initial_rank: int = 16
    growth_rate: float = 1.1  # Multiplier for rank growth
    shrink_threshold: float = 0.8  # Accuracy threshold to shrink rank
    growth_threshold: float = 0.9  # Accuracy threshold to grow rank
    
    # Performance monitoring
    window_size: int = 5  # Number of recent tasks to consider
    min_samples_per_rank: int = 100  # Minimum samples before considering rank change


@dataclass
class RankAdaptationStats:
    """Statistics for rank adaptation."""

    current_rank: int = 0
    previous_ranks: List[int] = field(default_factory=list)
    rank_changes: int = 0
    growth_count: int = 0
    shrink_count: int = 0
    last_change_reason: str = ""
    last_change_time: Optional[str] = None


class DynamicLoRAManager:
    """
    Manages dynamic LoRA rank adaptation for continual learning.
    
    Automatically adjusts the LoRA rank based on:
    - Task complexity (gradient magnitude)
    - Performance on recent tasks
    - Memory constraints
    
    The rank adaptation follows these rules:
    1. Start with initial_rank
    2. If performance > growth_threshold and rank < max_rank: rank *= growth_rate
    3. If performance < shrink_threshold and rank > min_rank: rank /= growth_rate
    4. Rank is always clamped to [min_rank, max_rank]
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AdaptiveMLConfig] = None,
        dynamic_config: Optional[DynamicLoRAConfig] = None,
    ):
        """
        Initialize DynamicLoRAManager.
        
        Args:
            model: The base model
            config: AdaptiveMLConfig instance
            dynamic_config: DynamicLoRAConfig instance
        """
        self.model = model
        self.config = config or AdaptiveMLConfig()
        self.dynamic_config = dynamic_config or DynamicLoRAConfig(
            enabled=self.config.training.dynamic_lora,
            min_rank=self.config.adapters.min_rank,
            max_rank=self.config.adapters.max_rank,
            growth_rate=self.config.adapters.rank_growth_rate,
            shrink_threshold=self.config.adapters.rank_shrink_threshold,
        )
        
        # Current state
        self.current_rank = self.dynamic_config.initial_rank
        self.task_performance: Dict[str, float] = {}  # task_id -> accuracy
        self.rank_history: Dict[str, int] = {}  # task_id -> rank used
        self.gradient_stats: Dict[str, float] = {}  # task_id -> mean gradient magnitude
        
        # Statistics
        self.stats = RankAdaptationStats(
            current_rank=self.current_rank,
            previous_ranks=[self.current_rank],
        )
        
        # Task complexity estimates
        self.task_complexity: Dict[str, float] = {}

    def get_optimal_rank(self, task_id: str, performance: float) -> int:
        """
        Determine the optimal LoRA rank for a task based on performance.
        
        Args:
            task_id: Task identifier
            performance: Current performance on the task (0-1)
            
        Returns:
            Optimal rank for the task
        """
        if not self.dynamic_config.enabled:
            return self.current_rank
        
        # Store performance
        self.task_performance[task_id] = performance
        
        # Get task complexity estimate
        complexity = self._estimate_task_complexity(task_id)
        
        # Calculate target rank based on performance and complexity
        target_rank = self._calculate_target_rank(performance, complexity)
        
        # Clamp to valid range
        target_rank = max(self.dynamic_config.min_rank, 
                         min(self.dynamic_config.max_rank, target_rank))
        
        # Round to nearest valid integer
        target_rank = int(round(target_rank))
        
        return target_rank

    def _estimate_task_complexity(self, task_id: str) -> float:
        """
        Estimate task complexity based on gradient statistics.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Complexity score (higher = more complex)
        """
        if task_id in self.task_complexity:
            return self.task_complexity[task_id]
        
        # Default complexity based on gradient magnitude
        if task_id in self.gradient_stats:
            # Normalize gradient magnitude to 0-1 range
            grad_mag = self.gradient_stats[task_id]
            # Assume typical gradient magnitudes are in range [0.01, 1.0]
            complexity = min(1.0, max(0.0, (grad_mag - 0.01) / 0.99))
        else:
            # Default complexity
            complexity = 0.5
        
        self.task_complexity[task_id] = complexity
        return complexity

    def _calculate_target_rank(self, performance: float, complexity: float) -> float:
        """
        Calculate target rank based on performance and complexity.
        
        Args:
            performance: Current performance (0-1)
            complexity: Task complexity (0-1)
            
        Returns:
            Target rank (float)
        """
        # Base rank based on current rank
        base_rank = self.current_rank
        
        # Adjust based on performance
        if performance > self.dynamic_config.growth_threshold:
            # Good performance: try higher rank for more capacity
            rank_multiplier = self.dynamic_config.growth_rate
        elif performance < self.dynamic_config.shrink_threshold:
            # Poor performance: try lower rank to avoid overfitting
            rank_multiplier = 1.0 / self.dynamic_config.growth_rate
        else:
            # Acceptable performance: keep current rank
            rank_multiplier = 1.0
        
        # Adjust based on complexity
        complexity_factor = 1.0 + (complexity - 0.5) * 0.5  # Range [0.75, 1.25]
        
        # Calculate target rank
        target_rank = base_rank * rank_multiplier * complexity_factor
        
        return target_rank

    def update_rank(self, task_id: str, performance: float) -> int:
        """
        Update the current rank based on task performance.
        
        Args:
            task_id: Task identifier
            performance: Current performance on the task (0-1)
            
        Returns:
            New rank after update
        """
        old_rank = self.current_rank
        new_rank = self.get_optimal_rank(task_id, performance)
        
        if new_rank != old_rank:
            self.current_rank = new_rank
            self.stats.current_rank = new_rank
            self.stats.previous_ranks.append(new_rank)
            self.stats.rank_changes += 1
            
            if new_rank > old_rank:
                self.stats.growth_count += 1
                self.stats.last_change_reason = f"Growth: performance={performance:.3f}"
            else:
                self.stats.shrink_count += 1
                self.stats.last_change_reason = f"Shrink: performance={performance:.3f}"
            
            from datetime import datetime
            self.stats.last_change_time = datetime.now().isoformat()
        
        self.rank_history[task_id] = self.current_rank
        return self.current_rank

    def update_gradient_stats(self, task_id: str, gradients: Dict[str, torch.Tensor]) -> None:
        """
        Update gradient statistics for a task.
        
        Args:
            task_id: Task identifier
            gradients: Dictionary of parameter gradients
        """
        if not gradients:
            return
        
        # Calculate mean gradient magnitude
        grad_magnitudes = []
        for grad in gradients.values():
            if grad is not None:
                grad_magnitudes.append(torch.norm(grad).item())
        
        if grad_magnitudes:
            mean_grad_mag = float(np.mean(grad_magnitudes))
            self.gradient_stats[task_id] = mean_grad_mag
            # Update complexity estimate
            if task_id in self.task_complexity:
                # Smooth the complexity estimate
                self.task_complexity[task_id] = 0.7 * self.task_complexity[task_id] + 0.3 * min(1.0, mean_grad_mag)

    def create_lora_config(self, task_id: Optional[str] = None) -> LoraConfig:
        """
        Create a LoRA configuration with the current rank.
        
        Args:
            task_id: Optional task identifier for task-specific rank
            
        Returns:
            LoraConfig with current rank
        """
        # Get rank for this task
        if task_id and task_id in self.rank_history:
            rank = self.rank_history[task_id]
        else:
            rank = self.current_rank
        
        return LoraConfig(
            r=rank,
            lora_alpha=self.config.adapters.lora_alpha,
            lora_dropout=self.config.adapters.lora_dropout,
            target_modules=self.config.adapters.target_modules,
            task_type=self.config.adapters.task_type,
            inference_mode=False,
            bias="none",
        )

    def get_adapter_rank(self, task_id: Optional[str] = None) -> int:
        """
        Get the current adapter rank for a task.
        
        Args:
            task_id: Optional task identifier
            
        Returns:
            Current rank
        """
        if task_id and task_id in self.rank_history:
            return self.rank_history[task_id]
        return self.current_rank

    def get_stats(self) -> RankAdaptationStats:
        """Get rank adaptation statistics."""
        return self.stats

    def reset(self) -> None:
        """Reset rank adaptation state."""
        self.current_rank = self.dynamic_config.initial_rank
        self.task_performance = {}
        self.rank_history = {}
        self.gradient_stats = {}
        self.task_complexity = {}
        self.stats = RankAdaptationStats(
            current_rank=self.current_rank,
            previous_ranks=[self.current_rank],
        )

    def __repr__(self) -> str:
        return (
            f"DynamicLoRAManager(current_rank={self.current_rank}, "
            f"min={self.dynamic_config.min_rank}, "
            f"max={self.dynamic_config.max_rank}, "
            f"enabled={self.dynamic_config.enabled})"
        )


class LoRARankAdapter:
    """
    Adapter that dynamically adjusts LoRA rank during training.
    
    This wraps a PEFT model and automatically adjusts the LoRA rank
    based on task performance and complexity.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AdaptiveMLConfig] = None,
        dynamic_config: Optional[DynamicLoRAConfig] = None,
    ):
        """
        Initialize LoRARankAdapter.
        
        Args:
            model: The base model
            config: AdaptiveMLConfig instance
            dynamic_config: DynamicLoRAConfig instance
        """
        self.model = model
        self.config = config or AdaptiveMLConfig()
        self.dynamic_manager = DynamicLoRAManager(model, config, dynamic_config)
        
        # Track LoRA adapters for different tasks
        self.task_adapters: Dict[str, PeftModel] = {}
        self.current_adapter: Optional[PeftModel] = None

    def add_task_adapter(self, task_id: str) -> PeftModel:
        """
        Add a LoRA adapter for a new task with optimal rank.
        
        Args:
            task_id: Task identifier
            
        Returns:
            The created PeftModel
        """
        # Get optimal rank for this task
        rank = self.dynamic_manager.current_rank
        
        # Create LoRA config
        lora_config = self.dynamic_manager.create_lora_config(task_id)
        
        # Create adapter
        adapter = get_peft_model(self.model, lora_config)
        self.task_adapters[task_id] = adapter
        
        # Track rank history
        self.dynamic_manager.rank_history[task_id] = rank
        
        return adapter

    def get_task_adapter(self, task_id: str) -> Optional[PeftModel]:
        """Get the adapter for a specific task."""
        return self.task_adapters.get(task_id)

    def set_current_adapter(self, task_id: str) -> None:
        """Set the current active adapter."""
        if task_id in self.task_adapters:
            self.current_adapter = self.task_adapters[task_id]
        else:
            self.current_adapter = None

    def update_rank(self, task_id: str, performance: float) -> int:
        """
        Update the rank for a task based on performance.
        
        Args:
            task_id: Task identifier
            performance: Current performance on the task
            
        Returns:
            New rank
        """
        new_rank = self.dynamic_manager.update_rank(task_id, performance)
        
        # If rank changed, we need to recreate the adapter
        if task_id in self.task_adapters:
            old_rank = self.dynamic_manager.rank_history.get(task_id, new_rank)
            if new_rank != old_rank:
                # Remove old adapter
                del self.task_adapters[task_id]
                self.current_adapter = None
                
                # Create new adapter with new rank
                return self.add_task_adapter(task_id).peft_config.r
        
        return new_rank

    def get_current_rank(self) -> int:
        """Get the current rank."""
        return self.dynamic_manager.current_rank

    def get_stats(self) -> RankAdaptationStats:
        """Get rank adaptation statistics."""
        return self.dynamic_manager.get_stats()

    def __repr__(self) -> str:
        return (
            f"LoRARankAdapter(adapters={len(self.task_adapters)}, "
            f"current_rank={self.dynamic_manager.current_rank})"
        )
