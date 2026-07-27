"""
Continual Learning module for Adaptive Qwen Omni.
Provides replay memory, knowledge distillation, parameter protection, and anti-forgetting.
"""

from adaptive_ml.qwen_omni.continual_learning.replay import (
    ReplayMemory,
    MultimodalReplayBuffer,
    ExperienceReplay,
    ReplayStats,
)

from adaptive_ml.qwen_omni.continual_learning.distillation import (
    KnowledgeDistillation,
    MultiTeacherDistillation,
)

from adaptive_ml.qwen_omni.continual_learning.protection import (
    ParameterProtection,
    EWCTrainer,
    MASTrainer,
    SITrainer,
    ProtectionStats,
)

from adaptive_ml.qwen_omni.continual_learning.forgetting import (
    ForgettingDetector,
    AntiForgettingEngine,
    ForgettingMetrics,
)

__all__ = [
    "ReplayMemory",
    "MultimodalReplayBuffer",
    "ExperienceReplay",
    "ReplayStats",
    "KnowledgeDistillation",
    "MultiTeacherDistillation",
    "ParameterProtection",
    "EWCTrainer",
    "MASTrainer",
    "SITrainer",
    "ProtectionStats",
    "ForgettingDetector",
    "AntiForgettingEngine",
    "ForgettingMetrics",
]
