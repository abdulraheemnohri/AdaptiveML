"""
Inference module for Adaptive Qwen Omni.
Provides routing, loading, and execution of multimodal inference.
"""

from adaptive_ml.qwen_omni.inference.router import MultimodalRouter
from adaptive_ml.qwen_omni.inference.loader import AdapterLoader
from adaptive_ml.qwen_omni.inference.engine import QwenOmniInferenceEngine
from adaptive_ml.qwen_omni.core import InferenceResult

__all__ = [
    "MultimodalRouter",
    "AdapterLoader", 
    "QwenOmniInferenceEngine",
    "InferenceResult",
]
