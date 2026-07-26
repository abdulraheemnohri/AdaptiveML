"""
Tests for Adaptive Qwen Omni adaptive learning components.
"""

import pytest

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    TaskType,
    DomainType,
    AdapterType,
    MultimodalData,
    LearningDecision,
)
from adaptive_ml.qwen_omni.adaptive import (
    TaskDetector,
    DomainDetector,
    NoveltyDetector,
    AdaptiveRouter,
    LearningController,
    AdaptiveLearningOS,
)


class TestTaskDetector:
    """Test TaskDetector class."""
    
    def test_detector_initialization(self):
        """Test task detector initialization."""
        detector = TaskDetector()
        assert detector is not None
    
    def test_detect_code_generation(self):
        """Test detecting code generation task."""
        detector = TaskDetector()
        data = MultimodalData(text="Write a Python function")
        
        result = detector.detect(data, "Write a Python function")
        
        assert result is not None
        assert result.task_type == TaskType.CODE_GENERATION
        assert result.confidence > 0


class TestDomainDetector:
    """Test DomainDetector class."""
    
    def test_detector_initialization(self):
        """Test domain detector initialization."""
        detector = DomainDetector()
        assert detector is not None
    
    def test_detect_coding_domain(self):
        """Test detecting coding domain."""
        detector = DomainDetector()
        data = MultimodalData(text="Python code")
        
        result = detector.detect(data, "Write a Python function")
        
        assert result is not None
        assert result.domain == DomainType.CODING


class TestNoveltyDetector:
    """Test NoveltyDetector class."""
    
    def test_detector_initialization(self):
        """Test novelty detector initialization."""
        detector = NoveltyDetector()
        assert detector is not None


class TestAdaptiveRouter:
    """Test AdaptiveRouter class."""
    
    def test_router_initialization(self):
        """Test router initialization."""
        router = AdaptiveRouter()
        assert router is not None
    
    def test_detect_modality_text(self):
        """Test detecting text modality."""
        router = AdaptiveRouter()
        data = MultimodalData(text="Text only")
        
        modality = router.detect_modality(data)
        
        assert modality == ModalityType.TEXT
    
    def test_route(self):
        """Test routing."""
        router = AdaptiveRouter()
        data = MultimodalData(text="Write a Python function")
        
        routing = router.route(data, "Write a Python function")
        
        assert routing is not None
        assert routing.modality == ModalityType.TEXT


class TestLearningController:
    """Test LearningController class."""
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        controller = LearningController()
        assert controller is not None


class TestAdaptiveLearningOS:
    """Test AdaptiveLearningOS class."""
    
    def test_os_initialization(self):
        """Test OS initialization."""
        os = AdaptiveLearningOS()
        assert os is not None
