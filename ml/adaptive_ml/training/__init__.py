"""Training module for Adaptive ML Framework."""

from adaptive_ml.training.trainer import ContinualTrainer
from adaptive_ml.training.ewc import EWC, EWCStats
from adaptive_ml.training.mas import MAS, MASStats
from adaptive_ml.training.si import SI, SIStats
from adaptive_ml.training.distillation import KnowledgeDistillation, MultiTeacherDistillation

__all__ = [
    "ContinualTrainer", 
    "EWC", 
    "EWCStats",
    "MAS", 
    "MASStats", 
    "SI", 
    "SIStats",
    "KnowledgeDistillation", 
    "MultiTeacherDistillation"
]
