"""
Evaluation module for Adaptive Qwen Omni.
Provides comprehensive evaluation for all modalities with forgetting detection.
"""

from adaptive_ml.qwen_omni.evaluation.evaluator import (
    MultimodalEvaluator,
    ModalityEvaluator,
    TextEvaluator,
    VisionEvaluator,
    AudioEvaluator,
    VideoEvaluator,
    EvaluationResult,
    ModalityMetrics,
)

from adaptive_ml.qwen_omni.evaluation.scoring import (
    RetentionScorer,
    ForgettingScore,
    RetentionScore,
)

from adaptive_ml.qwen_omni.evaluation.gate import (
    PromotionGate,
)

from adaptive_ml.qwen_omni.evaluation.regression import (
    RegressionTester,
)

from adaptive_ml.qwen_omni.evaluation.benchmark import (
    BenchmarkRunner,
)

__all__ = [
    "MultimodalEvaluator",
    "ModalityEvaluator",
    "TextEvaluator",
    "VisionEvaluator",
    "AudioEvaluator",
    "VideoEvaluator",
    "EvaluationResult",
    "ModalityMetrics",
    "RetentionScorer",
    "ForgettingScore",
    "RetentionScore",
    "PromotionGate",
    "RegressionTester",
    "BenchmarkRunner",
]
