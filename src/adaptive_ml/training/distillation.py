"""
Knowledge Distillation for Adaptive ML Framework.
Preserves old model behavior while learning new tasks.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adaptive_ml.core.config import AdaptiveMLConfig


@dataclass
class DistillationStats:
    """Statistics for knowledge distillation."""

    teacher_loss: float = 0.0
    student_loss: float = 0.0
    distillation_loss: float = 0.0
    temperature: float = 1.0
    alpha: float = 0.5


class KnowledgeDistillation:
    """
    Knowledge Distillation for continual learning.
    
    Distillation preserves knowledge from a previous model (teacher) by training
    the new model (student) to match the teacher's soft predictions.
    
    The distillation loss is:
        L_distill = alpha * L_task + (1 - alpha) * L_KD
    
    where:
    - L_task: Task-specific loss (e.g., cross-entropy)
    - L_KD: Knowledge distillation loss (KL divergence between teacher and student softmax)
    - alpha: Weight for task loss (0-1)
    - temperature: Temperature for softmax (higher = softer predictions)
    
    Usage:
        # Create teacher model (previous model)
        teacher = ...
        
        # Create distillation
        kd = KnowledgeDistillation(teacher, alpha=0.5, temperature=2.0)
        
        # During training
        loss = kd.get_loss(student_outputs, targets)
    """

    def __init__(
        self,
        teacher: nn.Module,
        alpha: float = 0.5,
        temperature: float = 2.0,
        reduction: str = "batchmean",
    ):
        """
        Initialize KnowledgeDistillation.
        
        Args:
            teacher: The teacher model (previous model)
            alpha: Weight for task loss (0-1). Higher = more focus on new task.
            temperature: Temperature for softmax. Higher = softer predictions.
            reduction: Reduction method for loss ("batchmean", "mean", "sum")
        """
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        
        # Set teacher to evaluation mode
        self.teacher.eval()
        
        # Track statistics
        self.stats = DistillationStats(
            temperature=temperature,
            alpha=alpha,
        )

    def get_loss(
        self,
        student_outputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_outputs: Optional[torch.Tensor] = None,
        task_loss_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Compute the total loss with knowledge distillation.
        
        Args:
            student_outputs: Output logits from the student model
            targets: Ground truth targets (optional, for task loss)
            teacher_outputs: Output logits from the teacher model (optional)
            task_loss_fn: Custom task loss function (defaults to cross_entropy)
        
        Returns:
            Total loss = alpha * task_loss + (1 - alpha) * distillation_loss
        """
        # Compute teacher outputs if not provided
        if teacher_outputs is None:
            # Note: This requires the teacher to be run on the same inputs
            # In practice, you should pre-compute teacher outputs
            raise ValueError("teacher_outputs must be provided")
        
        # Compute task loss
        if targets is not None:
            if task_loss_fn is None:
                task_loss = F.cross_entropy(student_outputs, targets, reduction=self.reduction)
            else:
                task_loss = task_loss_fn(student_outputs, targets)
        else:
            task_loss = torch.tensor(0.0, device=student_outputs.device)
        
        # Compute distillation loss
        distill_loss = self._compute_distillation_loss(student_outputs, teacher_outputs)
        
        # Total loss
        total_loss = self.alpha * task_loss + (1 - self.alpha) * distill_loss
        
        # Update statistics
        self.stats.teacher_loss = float(task_loss.item()) if targets is not None else 0.0
        self.stats.student_loss = float(task_loss.item()) if targets is not None else 0.0
        self.stats.distillation_loss = float(distill_loss.item())
        
        return total_loss

    def _compute_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the knowledge distillation loss (KL divergence).
        
        Args:
            student_outputs: Output logits from student model
            teacher_outputs: Output logits from teacher model
        
        Returns:
            KL divergence loss
        """
        # Apply temperature
        student_logits = student_outputs / self.temperature
        teacher_logits = teacher_outputs / self.temperature
        
        # Compute softmax
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Compute KL divergence: KL(student || teacher)
        # This is equivalent to: teacher_probs * log(teacher_probs / student_probs)
        kl_loss = F.kl_div(
            torch.log(student_probs + 1e-10),
            teacher_probs,
            reduction=self.reduction,
            log_target=False,
        )
        
        # Scale by temperature^2 (as per original distillation paper)
        return kl_loss * (self.temperature ** 2)

    def get_teacher_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Get outputs from the teacher model.
        
        Args:
            inputs: Input tensor
        
        Returns:
            Teacher model outputs (logits)
        """
        with torch.no_grad():
            outputs = self.teacher(inputs)
            if isinstance(outputs, torch.Tensor):
                return outputs
            else:
                return outputs.logits if hasattr(outputs, "logits") else outputs[0]

    def update_alpha(self, alpha: float) -> None:
        """Update the alpha parameter."""
        self.alpha = alpha
        self.stats.alpha = alpha

    def update_temperature(self, temperature: float) -> None:
        """Update the temperature parameter."""
        self.temperature = temperature
        self.stats.temperature = temperature

    def get_stats(self) -> DistillationStats:
        """Get current statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = DistillationStats(
            temperature=self.temperature,
            alpha=self.alpha,
        )

    def __repr__(self) -> str:
        return (
            f"KnowledgeDistillation(alpha={self.alpha}, temperature={self.temperature}, "
            f"reduction={self.reduction})"
        )


class MultiTeacherDistillation:
    """
    Knowledge Distillation with multiple teacher models.
    
    Useful when you want to preserve knowledge from multiple previous models
    (e.g., after learning several tasks).
    
    The distillation loss is the average of KL divergences to all teachers:
        L_distill = (1 / N) * sum(L_KD(student, teacher_i) for i in range(N))
    """

    def __init__(
        self,
        teachers: List[nn.Module],
        alpha: float = 0.5,
        temperature: float = 2.0,
        reduction: str = "batchmean",
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize MultiTeacherDistillation.
        
        Args:
            teachers: List of teacher models
            alpha: Weight for task loss (0-1)
            temperature: Temperature for softmax
            reduction: Reduction method for loss
            weights: Optional weights for each teacher (must sum to 1)
        """
        self.teachers = teachers
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        
        # Set teachers to evaluation mode
        for teacher in self.teachers:
            teacher.eval()
        
        # Normalize weights
        if weights is None:
            weights = [1.0 / len(teachers)] * len(teachers)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        self.weights = weights
        
        # Track statistics
        self.stats = [DistillationStats() for _ in teachers]

    def get_loss(
        self,
        student_outputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_outputs: Optional[List[torch.Tensor]] = None,
        task_loss_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Compute the total loss with multi-teacher distillation.
        
        Args:
            student_outputs: Output logits from student model
            targets: Ground truth targets (optional)
            teacher_outputs: List of output logits from teacher models (optional)
            task_loss_fn: Custom task loss function
        
        Returns:
            Total loss
        """
        # Compute task loss
        if targets is not None:
            if task_loss_fn is None:
                task_loss = F.cross_entropy(student_outputs, targets, reduction=self.reduction)
            else:
                task_loss = task_loss_fn(student_outputs, targets)
        else:
            task_loss = torch.tensor(0.0, device=student_outputs.device)
        
        # Compute distillation loss (average over all teachers)
        if teacher_outputs is None:
            raise ValueError("teacher_outputs must be provided")
        
        distill_loss = torch.tensor(0.0, device=student_outputs.device)
        for i, (teacher_out, weight) in enumerate(zip(teacher_outputs, self.weights)):
            loss = self._compute_distillation_loss(student_outputs, teacher_out)
            distill_loss += weight * loss
            
            # Update stats
            self.stats[i].distillation_loss = float(loss.item())
        
        # Total loss
        total_loss = self.alpha * task_loss + (1 - self.alpha) * distill_loss
        
        return total_loss

    def _compute_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss for a single teacher."""
        student_logits = student_outputs / self.temperature
        teacher_logits = teacher_outputs / self.temperature
        
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        kl_loss = F.kl_div(
            torch.log(student_probs + 1e-10),
            teacher_probs,
            reduction=self.reduction,
            log_target=False,
        )
        
        return kl_loss * (self.temperature ** 2)

    def get_teacher_outputs(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """Get outputs from all teacher models."""
        outputs = []
        for teacher in self.teachers:
            with torch.no_grad():
                out = teacher(inputs)
                if isinstance(out, torch.Tensor):
                    outputs.append(out)
                else:
                    outputs.append(out.logits if hasattr(out, "logits") else out[0])
        return outputs

    def __repr__(self) -> str:
        return (
            f"MultiTeacherDistillation(teachers={len(self.teachers)}, "
            f"alpha={self.alpha}, temperature={self.temperature})"
        )
