"""
Knowledge Distillation for Qwen2.5-Omni-3B.
Implements teacher-student learning to preserve old knowledge while acquiring new knowledge.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    TrainingStats,
)

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 2.0
    distillation_weight: float = 0.5
    use_teacher_outputs: bool = True
    use_teacher_hidden: bool = False
    hidden_weight: float = 0.1
    layer_weights: Optional[List[float]] = None
    
    # Modality-specific settings
    modality_temperatures: Dict[ModalityType, float] = field(default_factory=dict)
    modality_weights: Dict[ModalityType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize default modality settings
        if not self.modality_temperatures:
            self.modality_temperatures = {
                ModalityType.TEXT: 2.0,
                ModalityType.VISION: 2.5,
                ModalityType.AUDIO: 2.0,
                ModalityType.VIDEO: 3.0,
                ModalityType.SPEECH: 2.0,
            }
        
        if not self.modality_weights:
            self.modality_weights = {
                ModalityType.TEXT: 1.0,
                ModalityType.VISION: 1.0,
                ModalityType.AUDIO: 1.0,
                ModalityType.VIDEO: 1.0,
                ModalityType.SPEECH: 1.0,
            }


@dataclass
class DistillationStats:
    """Statistics for knowledge distillation."""
    distillation_loss: float = 0.0
    temperature: float = 2.0
    weight: float = 0.5
    teacher_accuracy: float = 0.0
    student_accuracy: float = 0.0
    knowledge_transfer: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "distillation_loss": self.distillation_loss,
            "temperature": self.temperature,
            "weight": self.weight,
            "teacher_accuracy": self.teacher_accuracy,
            "student_accuracy": self.student_accuracy,
            "knowledge_transfer": self.knowledge_transfer,
        }


class KnowledgeDistillation:
    """
    Knowledge Distillation for continual learning.
    Uses a teacher model to guide the student model's learning.
    """
    
    def __init__(
        self,
        teacher_model: Optional[PreTrainedModel] = None,
        config: Optional[DistillationConfig] = None,
    ):
        self.teacher_model = teacher_model
        self.config = config or DistillationConfig()
        
        # Statistics
        self._stats = DistillationStats()
        
        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_teacher(self, teacher_model: PreTrainedModel) -> None:
        """Set the teacher model."""
        self.teacher_model = teacher_model
        self.teacher_model.to(self._device)
        self.teacher_model.eval()
        logger.info(f"Set teacher model: {teacher_model.__class__.__name__}")
    
    def set_config(self, config: DistillationConfig) -> None:
        """Set the distillation configuration."""
        self.config = config
        logger.info(f"Updated distillation config: temperature={config.temperature}, weight={config.distillation_weight}")
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        modality: Optional[ModalityType] = None,
    ) -> Tuple[torch.Tensor, DistillationStats]:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: Optional ground truth labels
            modality: Optional modality for modality-specific settings
            
        Returns:
            Tuple of (loss, updated stats)
        """
        # Get modality-specific settings
        if modality and modality in self.config.modality_temperatures:
            temperature = self.config.modality_temperatures[modality]
        else:
            temperature = self.config.temperature
        
        if modality and modality in self.config.modality_weights:
            weight = self.config.modality_weights[modality]
        else:
            weight = self.config.distillation_weight
        
        # Temperature scaling
        student_logits_scaled = student_logits / temperature
        teacher_logits_scaled = teacher_logits / temperature
        
        # Soft labels from teacher
        teacher_probs = F.softmax(teacher_logits_scaled, dim=-1)
        
        # KL divergence loss
        log_student_probs = F.log_softmax(student_logits_scaled, dim=-1)
        kl_loss = F.kl_div(log_student_probs, teacher_probs, reduction='batchmean')
        
        # Scale by temperature^2 (as per original distillation paper)
        distillation_loss = kl_loss * (temperature ** 2)
        
        # Combine with task loss if labels provided
        if labels is not None:
            # Cross entropy loss
            ce_loss = F.cross_entropy(student_logits, labels)
            
            # Weighted sum
            total_loss = weight * distillation_loss + (1 - weight) * ce_loss
        else:
            total_loss = weight * distillation_loss
        
        # Update stats
        stats = DistillationStats(
            distillation_loss=distillation_loss.item(),
            temperature=temperature,
            weight=weight,
        )
        
        return total_loss, stats
    
    def compute_multi_teacher_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        weights: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, DistillationStats]:
        """
        Compute loss with multiple teacher models.
        
        Args:
            student_logits: Logits from student model
            teacher_logits_list: List of logits from multiple teachers
            labels: Optional ground truth labels
            weights: Optional weights for each teacher
            
        Returns:
            Tuple of (loss, updated stats)
        """
        if not teacher_logits_list:
            if labels is not None:
                return F.cross_entropy(student_logits, labels), DistillationStats()
            else:
                return torch.tensor(0.0, device=self._device), DistillationStats()
        
        # Default weights
        if weights is None:
            weights = [1.0 / len(teacher_logits_list)] * len(teacher_logits_list)
        
        # Compute loss for each teacher
        total_distillation_loss = 0.0
        for teacher_logits, teacher_weight in zip(teacher_logits_list, weights):
            dist_loss, _ = self.compute_distillation_loss(
                student_logits, teacher_logits, labels=None
            )
            total_distillation_loss += teacher_weight * dist_loss
        
        # Combine with task loss
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.config.distillation_weight * total_distillation_loss + (1 - self.config.distillation_weight) * ce_loss
        else:
            total_loss = self.config.distillation_weight * total_distillation_loss
        
        stats = DistillationStats(
            distillation_loss=total_distillation_loss.item(),
            temperature=self.config.temperature,
            weight=self.config.distillation_weight,
        )
        
        return total_loss, stats
    
    def get_teacher_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Get outputs from teacher model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            **kwargs: Additional arguments for teacher model
            
        Returns:
            Dictionary with teacher outputs
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model not set")
        
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        return outputs
    
    def update_stats(self, stats: DistillationStats) -> None:
        """Update distillation statistics."""
        self._stats = stats
    
    def get_stats(self) -> DistillationStats:
        """Get current distillation statistics."""
        return self._stats


class MultiTeacherDistillation:
    """
    Multi-Teacher Knowledge Distillation.
    Uses an ensemble of teacher models for more robust knowledge transfer.
    """
    
    def __init__(
        self,
        teacher_models: Optional[List[PreTrainedModel]] = None,
        config: Optional[DistillationConfig] = None,
    ):
        self.teacher_models = teacher_models or []
        self.config = config or DistillationConfig()
        
        # Weights for each teacher
        self.teacher_weights: List[float] = []
        
        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add_teacher(self, teacher_model: PreTrainedModel, weight: float = 1.0) -> None:
        """Add a teacher model."""
        teacher_model.to(self._device)
        teacher_model.eval()
        self.teacher_models.append(teacher_model)
        self.teacher_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(self.teacher_weights)
        if total_weight > 0:
            self.teacher_weights = [w / total_weight for w in self.teacher_weights]
        
        logger.info(f"Added teacher model: {teacher_model.__class__.__name__} with weight {weight}")
    
    def remove_teacher(self, index: int) -> None:
        """Remove a teacher model."""
        if 0 <= index < len(self.teacher_models):
            del self.teacher_models[index]
            del self.teacher_weights[index]
            
            # Renormalize weights
            total_weight = sum(self.teacher_weights)
            if total_weight > 0:
                self.teacher_weights = [w / total_weight for w in self.teacher_weights]
        
    def get_ensemble_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Get ensemble outputs from all teacher models.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with ensemble outputs
        """
        if not self.teacher_models:
            raise ValueError("No teacher models available")
        
        # Get outputs from all teachers
        teacher_logits = []
        for teacher in self.teacher_models:
            with torch.no_grad():
                outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                teacher_logits.append(outputs.logits)
        
        # Average logits (logit ensemble)
        stacked_logits = torch.stack(teacher_logits, dim=0)
        ensemble_logits = torch.mean(stacked_logits, dim=0)
        
        return {
            "logits": ensemble_logits,
            "individual_logits": teacher_logits,
        }
    
    def compute_ensemble_loss(
        self,
        student_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, DistillationStats]:
        """
        Compute loss using ensemble of teachers.
        
        Args:
            student_logits: Logits from student model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            labels: Optional ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (loss, stats)
        """
        # Get ensemble outputs
        ensemble_outputs = self.get_ensemble_outputs(
            input_ids, attention_mask, **kwargs
        )
        
        # Use weighted ensemble
        if self.teacher_weights:
            weighted_logits = torch.zeros_like(ensemble_outputs["individual_logits"][0])
            for logits, weight in zip(ensemble_outputs["individual_logits"], self.teacher_weights):
                weighted_logits += weight * logits
            ensemble_logits = weighted_logits
        else:
            ensemble_logits = ensemble_outputs["logits"]
        
        # Compute distillation loss
        distillation = KnowledgeDistillation(config=self.config)
        loss, stats = distillation.compute_distillation_loss(
            student_logits, ensemble_logits, labels, modality=None
        )
        
        return loss, stats
    
    def get_teacher_count(self) -> int:
        """Get number of teacher models."""
        return len(self.teacher_models)
    
    def get_teacher_names(self) -> List[str]:
        """Get names of all teacher models."""
        return [teacher.__class__.__name__ for teacher in self.teacher_models]
