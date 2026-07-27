"""
Controller module for Adaptive Qwen Omni system.
This file is kept for compatibility and imports from router.py
"""

# Import everything from router.py to maintain the module structure
from adaptive_ml.qwen_omni.adaptive.router import (
    AdaptiveLearningOS,
    AdaptiveRouter,
    LearningController,
    LearningStrategy,
    RouterConfig,
    RoutingDecision,
)

__all__ = [
    "AdaptiveLearningOS",
    "AdaptiveRouter",
    "LearningController",
    "LearningStrategy",
    "RouterConfig",
    "RoutingDecision",
]
