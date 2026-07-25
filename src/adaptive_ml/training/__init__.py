"""Training module for Adaptive ML Framework."""

from adaptive_ml.training.trainer import ContinualTrainer
from adaptive_ml.training.ewc import EWC
from adaptive_ml.training.distillation import KnowledgeDistillation

__all__ = ["ContinualTrainer", "EWC", "KnowledgeDistillation"]
