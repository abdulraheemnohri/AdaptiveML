"""
Adaptive Learning OS for Qwen2.5-Omni-3B.
Core controller that orchestrates the continual learning process.
"""

from adaptive_ml.qwen_omni.adaptive.controller import (
    AdaptiveLearningOS,
    LearningController,
    LearningStrategy,
)

from adaptive_ml.qwen_omni.adaptive.detectors import (
    TaskDetector,
    DomainDetector,
    NoveltyDetector,
    TaskClassificationResult,
    DomainClassificationResult,
    NoveltyResult,
)

from adaptive_ml.qwen_omni.adaptive.router import (
    AdaptiveRouter,
    RouterConfig,
    RoutingDecision,
)

__all__ = [
    "AdaptiveLearningOS",
    "LearningController",
    "LearningStrategy",
    "TaskDetector",
    "DomainDetector",
    "NoveltyDetector",
    "TaskClassificationResult",
    "DomainClassificationResult",
    "NoveltyResult",
    "AdaptiveRouter",
    "RouterConfig",
    "RoutingDecision",
]
