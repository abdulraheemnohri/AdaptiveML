"""
Adaptive Qwen Omni - Continual Learning Framework for Qwen2.5-Omni-3B

This module provides a complete continual learning system specifically designed
for Qwen2.5-Omni-3B with its unique Thinker-Talker architecture and multimodal capabilities.

Architecture:
    Qwen2.5-Omni-3B (Frozen Base)
        │
        ├── Thinker (Understanding)
        └── Talker (Speech Output)
            │
            ▼
    Adaptive Learning Layer
        │
        ├── New Knowledge → Continual SFT
        ├── New Skills → LoRA/QLoRA Adapters
        └── New Domains → New Adapters
            │
            ▼
    Anti-Forgetting Engine
        │
        ├── Replay Memory (Multimodal)
        ├── Knowledge Distillation
        └── Parameter Protection (EWC/MAS/SI)
            │
            ▼
    Multimodal Evaluation
        │
        ├── Text Evaluation
        ├── Vision Evaluation
        ├── Audio Evaluation
        └── Video Evaluation
            │
            ▼
    Promotion Gate → Model Registry
"""

from adaptive_ml.qwen_omni.core import (
    QwenOmniConfig,
    QwenOmniModelConfig,
    QwenOmniTrainingConfig,
    QwenOmniAdapterConfig,
    QwenOmniMemoryConfig,
    QwenOmniDriftConfig,
    QwenOmniEvaluationConfig,
    QwenOmniRegistryConfig,
    ModalityType,
    AdapterType,
    TaskType,
    DomainType,
    ForgettingDetectionStrategy,
    AdaptationLevel,
    LearningDecision,
    MultimodalData,
    MultimodalEntry,
    MemoryCandidate,
    AdapterInfo,
    ModelVersion,
)

from adaptive_ml.qwen_omni.adaptive import (
    AdaptiveLearningOS,
    TaskDetector,
    DomainDetector,
    NoveltyDetector,
    LearningController,
    AdaptiveRouter,
    TaskClassificationResult,
    DomainClassificationResult,
    NoveltyResult,
    LearningStrategy,
)

from adaptive_ml.qwen_omni.continual_learning import (
    ReplayMemory,
    MultimodalReplayBuffer,
    ExperienceReplay,
    KnowledgeDistillation,
    ParameterProtection,
    EWCTrainer,
    MASTrainer,
    SITrainer,
    ForgettingDetector,
    AntiForgettingEngine,
    ReplayStats,
    ForgettingMetrics,
    ProtectionStats,
)

from adaptive_ml.qwen_omni.training import (
    QwenOmniTrainer,
    LoRATrainer,
    QLoRATrainer,
    MultimodalTrainer,
    CheckpointManager,
    TrainingMonitor,
    TrainingStats,
)

from adaptive_ml.qwen_omni.evaluation import (
    MultimodalEvaluator,
    ModalityEvaluator,
    TextEvaluator,
    VisionEvaluator,
    AudioEvaluator,
    VideoEvaluator,
    RegressionTester,
    BenchmarkRunner,
    RetentionScorer,
    PromotionGate,
    EvaluationResult,
    ModalityMetrics,
    ForgettingScore,
    RetentionScore,
)

from adaptive_ml.qwen_omni.memory import (
    MultimodalDataProcessor,
    DataQualityGate,
    MemorySelector,
    EpisodicMemory,
    SemanticMemory,
    ReplayStorage,
    MemoryPriorityCalculator,
)

from adaptive_ml.qwen_omni.registry import (
    QwenOmniModelRegistry,
    AdapterRegistry,
    VersionManager,
    ModelCard,
    AdapterCard,
)

from adaptive_ml.qwen_omni.inference import (
    QwenOmniInferenceEngine,
    AdapterLoader,
    MultimodalRouter,
    InferenceResult,
)

from adaptive_ml.qwen_omni.datasets import (
    MultimodalDataset,
    DataIngestion,
    DataCleaning,
    Deduplication,
    QualityFilter,
)

__all__ = [
    # Core
    "QwenOmniConfig",
    "QwenOmniModelConfig",
    "QwenOmniTrainingConfig",
    "QwenOmniAdapterConfig",
    "QwenOmniMemoryConfig",
    "QwenOmniDriftConfig",
    "QwenOmniEvaluationConfig",
    "QwenOmniRegistryConfig",
    "ModalityType",
    "AdapterType",
    "TaskType",
    "DomainType",
    "ForgettingDetectionStrategy",
    "AdaptationLevel",
    "LearningDecision",
    "MultimodalData",
    "MultimodalEntry",
    "MemoryCandidate",
    "AdapterInfo",
    "ModelVersion",
    # Adaptive
    "AdaptiveLearningOS",
    "TaskDetector",
    "DomainDetector",
    "NoveltyDetector",
    "LearningController",
    "AdaptiveRouter",
    "TaskClassificationResult",
    "DomainClassificationResult",
    "NoveltyResult",
    "LearningStrategy",
    # Continual Learning
    "ReplayMemory",
    "MultimodalReplayBuffer",
    "ExperienceReplay",
    "KnowledgeDistillation",
    "ParameterProtection",
    "EWCTrainer",
    "MASTrainer",
    "SITrainer",
    "ForgettingDetector",
    "AntiForgettingEngine",
    "ReplayStats",
    "ForgettingMetrics",
    "ProtectionStats",
    # Training
    "QwenOmniTrainer",
    "LoRATrainer",
    "QLoRATrainer",
    "MultimodalTrainer",
    "CheckpointManager",
    "TrainingMonitor",
    "TrainingStats",
    # Evaluation
    "MultimodalEvaluator",
    "ModalityEvaluator",
    "TextEvaluator",
    "VisionEvaluator",
    "AudioEvaluator",
    "VideoEvaluator",
    "RegressionTester",
    "BenchmarkRunner",
    "RetentionScorer",
    "PromotionGate",
    "EvaluationResult",
    "ModalityMetrics",
    "ForgettingScore",
    "RetentionScore",
    # Memory
    "MultimodalDataProcessor",
    "DataQualityGate",
    "MemorySelector",
    "EpisodicMemory",
    "SemanticMemory",
    "ReplayStorage",
    "MemoryPriorityCalculator",
    # Registry
    "QwenOmniModelRegistry",
    "AdapterRegistry",
    "VersionManager",
    "ModelCard",
    "AdapterCard",
    # Inference
    "QwenOmniInferenceEngine",
    "AdapterLoader",
    "MultimodalRouter",
    "InferenceResult",
    # Datasets
    "MultimodalDataset",
    "DataIngestion",
    "DataCleaning",
    "Deduplication",
    "QualityFilter",
]

__version__ = "1.0.0"
