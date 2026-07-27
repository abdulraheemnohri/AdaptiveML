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


class TestModelServerEndpoints:
    """Test ModelServer endpoints and backend functionalities."""

    def test_server_routes(self):
        """Test the custom endpoints of the inference ModelServer."""
        import torch.nn as nn
        from fastapi.testclient import TestClient
        from adaptive_ml.serving.inference import ModelServer

        # Create a mock model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            def forward(self, x, attention_mask=None):
                return x

        model = DummyModel()
        server = ModelServer(model=model)
        app = server.get_app()
        client = TestClient(app)

        # Test health endpoint
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        # Test collect endpoint
        collect_resp = client.post("/collect?source=web&query=https://example.com")
        assert collect_resp.status_code == 200
        assert collect_resp.json()["status"] == "success"

        # Test process endpoint
        process_resp = client.post("/process?content=SomeRawContent")
        assert process_resp.status_code == 200
        assert process_resp.json()["status"] == "success"

        # Test deduplicate endpoint
        dedup_resp = client.post("/deduplicate", json=["hello world", "hello world", "different text"])
        assert dedup_resp.status_code == 200
        assert len(dedup_resp.json()["unique"]) >= 1

        # Test validate endpoint
        validate_resp = client.post("/validate?text=SafeReasonableContentLongTextForValidation")
        assert validate_resp.status_code == 200
        assert "result" in validate_resp.json()

        # Test research endpoint
        research_resp = client.post("/research?query=NewOmniModels")
        assert research_resp.status_code == 200
        assert research_resp.json()["confidence_score"] == 0.914

        # Test status endpoint
        status_resp = client.get("/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["base_model"] == "Qwen2.5-Omni-3B"
