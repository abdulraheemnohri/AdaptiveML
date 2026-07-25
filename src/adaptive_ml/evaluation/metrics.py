"""
Metrics for evaluating continual learning performance.
Includes retention metrics, forgetting metrics, and comprehensive evaluation suites.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import Task
from adaptive_ml.data.dataset import DatasetEntry


@dataclass
class EvaluationResult:
    """Result of evaluating a model on a dataset."""

    task_id: str
    loss: float
    accuracy: float
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    num_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetentionMetrics:
    """
    Metrics for measuring knowledge retention in continual learning.
    
    Retention measures how well the model remembers previous tasks after
    learning new tasks.
    """

    # Performance on old tasks
    old_task_accuracy: float = 0.0
    old_task_loss: float = 0.0
    
    # Performance on new tasks
    new_task_accuracy: float = 0.0
    new_task_loss: float = 0.0
    
    # Overall performance
    average_accuracy: float = 0.0
    average_loss: float = 0.0
    
    # Retention score (0-1, higher is better)
    retention_score: float = 0.0
    
    # Forgetting penalty (0-1, lower is better)
    forgetting_penalty: float = 0.0
    
    # Per-task metrics
    task_metrics: Dict[str, EvaluationResult] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"RetentionMetrics(retention={self.retention_score:.4f}, "
            f"forgetting={self.forgetting_penalty:.4f}, "
            f"old_acc={self.old_task_accuracy:.4f}, new_acc={self.new_task_accuracy:.4f})"
        )


@dataclass
class ForgettingMetrics:
    """
    Metrics for measuring catastrophic forgetting.
    
    Forgetting measures how much the model has forgotten previous tasks
    after learning new tasks.
    """

    # Forgetting on each old task
    task_forgetting: Dict[str, float] = field(default_factory=dict)
    
    # Average forgetting across all old tasks
    average_forgetting: float = 0.0
    
    # Maximum forgetting on any single task
    max_forgetting: float = 0.0
    
    # Number of tasks with significant forgetting (> 10%)
    num_forgotten_tasks: int = 0
    
    # Forgetting penalty (0-1, lower is better)
    forgetting_penalty: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"ForgettingMetrics(avg={self.average_forgetting:.4f}, "
            f"max={self.max_forgetting:.4f}, "
            f"forgotten_tasks={self.num_forgotten_tasks})"
        )


class ContinualEvaluator:
    """
    Evaluates model performance on continual learning tasks.
    
    Computes:
    - Task-specific metrics (accuracy, loss, F1, etc.)
    - Retention metrics (how well old tasks are remembered)
    - Forgetting metrics (how much has been forgotten)
    - Comprehensive evaluation reports
    
    Usage:
        evaluator = ContinualEvaluator(model, config)
        
        # Evaluate on a task
        result = evaluator.evaluate_task("task_a", test_data_a)
        
        # Compute retention metrics
        retention = evaluator.compute_retention(
            old_tasks=["task_a", "task_b"],
            new_task="task_c",
        )
        
        # Compute forgetting metrics
        forgetting = evaluator.compute_forgetting(
            baseline_metrics={"task_a": 0.95, "task_b": 0.90},
            current_metrics={"task_a": 0.85, "task_b": 0.88},
        )
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AdaptiveMLConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize ContinualEvaluator.
        
        Args:
            model: The model to evaluate
            config: AdaptiveMLConfig instance
            device: Device to run evaluation on
        """
        self.model = model
        self.config = config or AdaptiveMLConfig()
        # Resolve device
        device_str = device or self.config.training.device
        if device_str == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_str
        
        # Store baseline metrics for forgetting computation
        self.baseline_metrics: Dict[str, EvaluationResult] = {}
        
        # Store evaluation history
        self.history: Dict[str, List[EvaluationResult]] = {}

    def evaluate_task(
        self,
        task_id: str,
        data: List[DatasetEntry],
        batch_size: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
    ) -> EvaluationResult:
        """
        Evaluate the model on a specific task.
        
        Args:
            task_id: Task identifier
            data: List of dataset entries
            batch_size: Batch size (defaults to config)
            loss_fn: Custom loss function (defaults to cross-entropy)
        
        Returns:
            EvaluationResult with task metrics
        """
        batch_size = batch_size or self.config.training.batch_size
        loss_fn = loss_fn or F.cross_entropy
        
        # Create dataset and dataloader
        from adaptive_ml.data.dataset import ContinualDataset
        dataset = ContinualDataset(new_data=data, replay_buffer=[])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch.get("input", batch.get("input_ids"))
                    labels = batch.get("label", batch.get("labels"))
                
                inputs = inputs.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                
                # Compute loss
                if labels is not None:
                    loss = loss_fn(logits, labels)
                    total_loss += loss.item() * inputs.shape[0]
                    
                    # Compute accuracy
                    preds = torch.argmax(logits, dim=-1)
                    correct = (preds == labels).float().sum()
                    total_correct += correct.item()
                    
                    # Store for F1 computation
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                total_samples += inputs.shape[0]
        
        self.model.train()
        
        # Compute metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Compute F1, precision, recall
        f1, precision, recall = self._compute_f1(all_preds, all_labels)
        
        result = EvaluationResult(
            task_id=task_id,
            loss=loss,
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            num_samples=total_samples,
        )
        
        # Store in history
        if task_id not in self.history:
            self.history[task_id] = []
        self.history[task_id].append(result)
        
        return result

    def evaluate_tasks(
        self,
        task_ids: List[str],
        data_dict: Dict[str, List[DatasetEntry]],
        batch_size: Optional[int] = None,
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate the model on multiple tasks.
        
        Args:
            task_ids: List of task identifiers
            data_dict: Dictionary mapping task_id to data
            batch_size: Batch size (defaults to config)
        
        Returns:
            Dictionary mapping task_id to EvaluationResult
        """
        results = {}
        for task_id in task_ids:
            if task_id in data_dict:
                results[task_id] = self.evaluate_task(task_id, data_dict[task_id], batch_size)
        return results

    def compute_retention(
        self,
        old_tasks: List[str],
        new_task: str,
        old_data: Dict[str, List[DatasetEntry]],
        new_data: List[DatasetEntry],
        weights: Optional[Dict[str, float]] = None,
    ) -> RetentionMetrics:
        """
        Compute retention metrics for continual learning.
        
        Args:
            old_tasks: List of old task identifiers
            new_task: New task identifier
            old_data: Dictionary mapping old task_id to data
            new_data: Data for the new task
            weights: Optional weights for computing retention score
        
        Returns:
            RetentionMetrics with comprehensive retention analysis
        """
        # Evaluate on all tasks
        all_results = {}
        for task_id in old_tasks:
            if task_id in old_data:
                all_results[task_id] = self.evaluate_task(task_id, old_data[task_id])
        
        if new_task and new_data:
            all_results[new_task] = self.evaluate_task(new_task, new_data)
        
        # Separate old and new task metrics
        old_results = {k: v for k, v in all_results.items() if k in old_tasks}
        new_results = {k: v for k, v in all_results.items() if k == new_task}
        
        # Compute average metrics
        old_accuracy = np.mean([r.accuracy for r in old_results.values()]) if old_results else 0.0
        old_loss = np.mean([r.loss for r in old_results.values()]) if old_results else 0.0
        new_accuracy = np.mean([r.accuracy for r in new_results.values()]) if new_results else 0.0
        new_loss = np.mean([r.loss for r in new_results.values()]) if new_results else 0.0
        
        # Use config weights if not provided
        if weights is None:
            weights = {
                "new_score": self.config.evaluation.new_score_weight,
                "old_score": self.config.evaluation.old_score_weight,
                "forgetting": self.config.evaluation.forgetting_penalty_weight,
            }
        
        # Compute retention score
        # Retention = weighted sum of old and new performance
        retention_score = (
            weights.get("old_score", 0.4) * old_accuracy +
            weights.get("new_score", 0.4) * new_accuracy
        )
        
        # Compute forgetting penalty
        forgetting_penalty = self._compute_forgetting_penalty(old_results)
        
        # Adjust retention score by forgetting penalty
        retention_score = max(0.0, retention_score - weights.get("forgetting", 0.2) * forgetting_penalty)
        
        return RetentionMetrics(
            old_task_accuracy=old_accuracy,
            old_task_loss=old_loss,
            new_task_accuracy=new_accuracy,
            new_task_loss=new_loss,
            average_accuracy=(old_accuracy + new_accuracy) / 2 if (old_accuracy + new_accuracy) > 0 else 0.0,
            average_loss=(old_loss + new_loss) / 2 if (old_loss + new_loss) > 0 else 0.0,
            retention_score=retention_score,
            forgetting_penalty=forgetting_penalty,
            task_metrics=all_results,
        )

    def compute_forgetting(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
    ) -> ForgettingMetrics:
        """
        Compute forgetting metrics by comparing baseline and current performance.
        
        Args:
            baseline_metrics: Dictionary mapping task_id to baseline accuracy
            current_metrics: Dictionary mapping task_id to current accuracy
        
        Returns:
            ForgettingMetrics with comprehensive forgetting analysis
        """
        task_forgetting = {}
        total_forgetting = 0.0
        max_forgetting = 0.0
        num_forgotten = 0
        
        for task_id in baseline_metrics:
            if task_id in current_metrics:
                baseline = baseline_metrics[task_id]
                current = current_metrics[task_id]
                forgetting = baseline - current
                
                task_forgetting[task_id] = forgetting
                total_forgetting += forgetting
                max_forgetting = max(max_forgetting, forgetting)
                
                if forgetting > 0.1:  # Significant forgetting
                    num_forgotten += 1
        
        # Average forgetting
        avg_forgetting = total_forgetting / len(baseline_metrics) if baseline_metrics else 0.0
        
        # Forgetting penalty (normalized to 0-1)
        # Higher forgetting = higher penalty
        forgetting_penalty = min(1.0, avg_forgetting / 0.5)  # Normalize by 50% forgetting
        
        return ForgettingMetrics(
            task_forgetting=task_forgetting,
            average_forgetting=avg_forgetting,
            max_forgetting=max_forgetting,
            num_forgotten_tasks=num_forgotten,
            forgetting_penalty=forgetting_penalty,
        )

    def _compute_f1(
        self,
        preds: List[int],
        labels: List[int],
    ) -> Tuple[float, float, float]:
        """
        Compute F1 score, precision, and recall.
        
        Args:
            preds: List of predicted labels
            labels: List of true labels
        
        Returns:
            Tuple of (f1, precision, recall)
        """
        if len(preds) == 0 or len(labels) == 0:
            return 0.0, 0.0, 0.0
        
        # Convert to numpy
        preds = np.array(preds)
        labels = np.array(labels)
        
        # Compute true positives, false positives, false negatives
        tp = np.sum((preds == labels) & (preds == 1))
        fp = np.sum((preds != labels) & (preds == 1))
        fn = np.sum((preds != labels) & (labels == 1))
        
        # Compute precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Compute F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall

    def _compute_forgetting_penalty(
        self,
        old_results: Dict[str, EvaluationResult],
    ) -> float:
        """
        Compute forgetting penalty based on old task performance.
        
        Args:
            old_results: Dictionary mapping task_id to EvaluationResult
        
        Returns:
            Forgetting penalty (0-1, lower is better)
        """
        if not old_results:
            return 0.0
        
        # Get baseline metrics if available
        baseline_accuracies = {}
        for task_id, result in old_results.items():
            if task_id in self.baseline_metrics:
                baseline_accuracies[task_id] = self.baseline_metrics[task_id].accuracy
            else:
                # If no baseline, assume perfect performance
                baseline_accuracies[task_id] = 1.0
        
        # Compute forgetting for each task
        total_forgetting = 0.0
        for task_id, result in old_results.items():
            baseline = baseline_accuracies.get(task_id, 1.0)
            current = result.accuracy
            forgetting = baseline - current
            total_forgetting += max(0, forgetting)
        
        # Average forgetting
        avg_forgetting = total_forgetting / len(old_results)
        
        # Normalize to 0-1
        return min(1.0, avg_forgetting / 0.5)

    def set_baseline_metrics(
        self,
        task_id: str,
        result: EvaluationResult,
    ) -> None:
        """
        Set baseline metrics for a task (to be used for forgetting computation).
        
        Args:
            task_id: Task identifier
            result: EvaluationResult with baseline metrics
        """
        self.baseline_metrics[task_id] = result

    def get_history(self, task_id: str) -> List[EvaluationResult]:
        """Get evaluation history for a task."""
        return self.history.get(task_id, [])

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self.history = {}

    def __repr__(self) -> str:
        return f"ContinualEvaluator(model={type(self.model).__name__}, device={self.device})"
