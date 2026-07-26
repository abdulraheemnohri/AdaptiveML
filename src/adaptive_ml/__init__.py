"""
Adaptive ML Framework for Continual Learning with Catastrophic Forgetting Prevention.

Core Components:
- Replay Memory: Experience replay with reservoir sampling, diversity, and importance-based selection
- Knowledge Distillation: Preserve old model behavior while learning new tasks
- Parameter Importance: EWC, MAS, SI for protecting critical parameters
- Dynamic Adapters: LoRA, QLoRA, and adapter routing for task-specific knowledge
- Drift Detection: Statistical, semantic, and concept drift detection
- Evaluation & Promotion: Retention scoring, A/B testing, and rollback mechanisms
- Model Registry: Versioning, atomic promotion, and rollback
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "AdaptiveMLConfig":
        from adaptive_ml.core.config import AdaptiveMLConfig
        return AdaptiveMLConfig
    elif name == "Task":
        from adaptive_ml.core.types import Task
        return Task
    elif name == "MemoryEntry":
        from adaptive_ml.core.types import MemoryEntry
        return MemoryEntry
    elif name == "DriftType":
        from adaptive_ml.core.types import DriftType
        return DriftType
    elif name == "SamplingStrategy":
        from adaptive_ml.core.types import SamplingStrategy
        return SamplingStrategy
    elif name == "AdapterType":
        from adaptive_ml.core.types import AdapterType
        return AdapterType
    elif name == "PromotionStrategy":
        from adaptive_ml.core.types import PromotionStrategy
        return PromotionStrategy
    elif name == "ParameterImportanceMethod":
        from adaptive_ml.core.types import ParameterImportanceMethod
        return ParameterImportanceMethod
    elif name == "ModalityType":
        from adaptive_ml.core.types import ModalityType
        return ModalityType
    elif name == "ReplayBuffer":
        from adaptive_ml.memory.replay import ReplayBuffer
        return ReplayBuffer
    elif name == "MemoryCompressor":
        from adaptive_ml.memory.compression import MemoryCompressor
        return MemoryCompressor
    elif name == "CompressedReplayBuffer":
        from adaptive_ml.memory.compression import CompressedReplayBuffer
        return CompressedReplayBuffer
    elif name == "AdapterManager":
        from adaptive_ml.models.adapters import AdapterManager
        return AdapterManager
    elif name == "AdapterRouter":
        from adaptive_ml.models.adapters import AdapterRouter
        return AdapterRouter
    elif name == "DynamicLoRAManager":
        from adaptive_ml.models.dynamic_lora import DynamicLoRAManager
        return DynamicLoRAManager
    elif name == "LoRARankAdapter":
        from adaptive_ml.models.dynamic_lora import LoRARankAdapter
        return LoRARankAdapter
    elif name == "ContinualTrainer":
        from adaptive_ml.training.trainer import ContinualTrainer
        return ContinualTrainer
    elif name == "EWC":
        from adaptive_ml.training.ewc import EWC
        return EWC
    elif name == "MAS":
        from adaptive_ml.training.mas import MAS
        return MAS
    elif name == "SI":
        from adaptive_ml.training.si import SI
        return SI
    elif name == "KnowledgeDistillation":
        from adaptive_ml.training.distillation import KnowledgeDistillation
        return KnowledgeDistillation
    elif name == "DriftDetector":
        from adaptive_ml.data.drift import DriftDetector
        return DriftDetector
    elif name == "CLIPDriftDetector":
        from adaptive_ml.data.clip_drift import CLIPDriftDetector
        return CLIPDriftDetector
    elif name == "PromotionController":
        from adaptive_ml.evaluation.promoter import PromotionController
        return PromotionController
    elif name == "AdvancedEvaluator":
        from adaptive_ml.evaluation.advanced_metrics import AdvancedEvaluator
        return AdvancedEvaluator
    elif name == "PerplexityCalculator":
        from adaptive_ml.evaluation.advanced_metrics import PerplexityCalculator
        return PerplexityCalculator
    elif name == "ModelRegistry":
        from adaptive_ml.serving.registry import ModelRegistry
        return ModelRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AdaptiveMLConfig",
    "Task",
    "MemoryEntry",
    "DriftType",
    "SamplingStrategy",
    "AdapterType",
    "PromotionStrategy",
    "ParameterImportanceMethod",
    "ModalityType",
    "ReplayBuffer",
    "MemoryCompressor",
    "CompressedReplayBuffer",
    "AdapterManager",
    "AdapterRouter",
    "DynamicLoRAManager",
    "LoRARankAdapter",
    "ContinualTrainer",
    "EWC",
    "MAS",
    "SI",
    "KnowledgeDistillation",
    "DriftDetector",
    "CLIPDriftDetector",
    "PromotionController",
    "AdvancedEvaluator",
    "PerplexityCalculator",
    "ModelRegistry",
]
