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


class TestNewServicesSuite:
    """Test Suite for Data Collection, Research Agents, Knowledge Graph, Decisions and Testing Lab."""

    def test_data_collection_service(self):
        from backend.app.services.data_collection import DataCollectionService
        collector = DataCollectionService()

        sample_web = collector.collect("web", "https://qwenlm.github.io")
        assert sample_web.source == "web"
        assert sample_web.language == "en"
        assert sample_web.content_hash is not None

        sample_github = collector.collect("github", "https://github.com/huggingface/peft")
        assert sample_github.source == "github"
        assert sample_github.content_type == "code"

    def test_research_agent_service(self):
        from backend.app.services.research_agent import ResearchAgentService
        research = ResearchAgentService()
        synthesis = research.research_gap("continual learning")

        assert synthesis.confidence_score > 0.8
        assert synthesis.is_contradictory is False
        assert len(synthesis.claims) == 3

    def test_knowledge_graph_service(self):
        from backend.app.services.knowledge_graph import KnowledgeGraphService
        kg = KnowledgeGraphService()

        kg.add_entity("ContinualLearning", "Concept")
        kg.add_relation("Qwen2.5-Omni-3B", "Uses", "ContinualLearning", confidence=0.98)

        relations = kg.query_relationships("Qwen2.5-Omni-3B")
        assert len(relations) >= 5

        stats = kg.get_stats()
        assert stats["entity_count"] > 5

        # Test contradiction detection
        kg.add_relation("Qwen2.5-Omni-3B", "Developed By", "SomeOtherDeveloper")
        contradictions = kg.detect_contradictions("Qwen2.5-Omni-3B", "Developed By", "Qwen")
        assert len(contradictions) == 1

    def test_learning_decision_service(self):
        from backend.app.services.learning_decision import DecisionEngine, LearningSpeed
        engine = DecisionEngine()

        # Test Fast Speed
        decision_fast = engine.evaluate_decision("Today is breaking news about Qwen Omni release", "web")
        assert decision_fast["learning_speed"] == LearningSpeed.FAST

        # Test Medium Speed
        decision_medium = engine.evaluate_decision("def train_lora_adapter(weights): pass", "coding")
        assert decision_medium["learning_speed"] == LearningSpeed.MEDIUM

        # Test Slow Speed
        decision_slow = engine.evaluate_decision("Evolved cognitive capability logic", "general")
        assert decision_slow["learning_speed"] == LearningSpeed.SLOW

    def test_model_testing_lab_and_firewall(self):
        from backend.app.services.model_testing import ModelTestingLab, BenchmarkScores
        lab = ModelTestingLab()

        baseline = BenchmarkScores()
        candidate_pass = BenchmarkScores(mmlu_text=0.83, mmmu_vision=0.76)
        candidate_fail = BenchmarkScores(mmlu_text=0.70) # Major forgetting

        result_pass = lab.evaluate_candidate("v2.5.0", candidate_pass, baseline)
        assert result_pass.passed is True
        assert result_pass.retention_rate >= 0.98

        result_fail = lab.evaluate_candidate("v2.5.0-failed", candidate_fail, baseline)
        assert result_fail.passed is False
        assert result_fail.forgetting_rate > 0.03
        assert len(result_fail.failed_reasons) > 0
