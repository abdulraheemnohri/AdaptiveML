"""
Tests for Adaptive Qwen Omni core components.
"""

import pytest
from datetime import datetime
from typing import Optional

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    AdapterType,
    TaskType,
    DomainType,
    ForgettingDetectionStrategy,
    AdaptationLevel,
    LearningDecision,
    MemoryPriority,
    MultimodalData,
    MultimodalEntry,
    MemoryCandidate,
    AdapterInfo,
    ModelVersion,
    ReplayStats,
    ForgettingMetrics,
    ProtectionStats,
    TrainingStats,
    InferenceResult,
    QwenOmniModelConfig,
    QwenOmniTrainingConfig,
    QwenOmniAdapterConfig,
    QwenOmniMemoryConfig,
    QwenOmniDriftConfig,
    QwenOmniEvaluationConfig,
    QwenOmniRegistryConfig,
    QwenOmniConfig,
)


class TestEnums:
    """Test enum types."""
    
    def test_modality_type_values(self):
        """Test ModalityType enum values."""
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.VISION.value == "vision"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.VIDEO.value == "video"
        assert ModalityType.SPEECH.value == "speech"
        assert ModalityType.MULTI_MODAL.value == "multi_modal"
    
    def test_adapter_type_values(self):
        """Test AdapterType enum values."""
        assert AdapterType.GENERAL.value == "general"
        assert AdapterType.CODING.value == "coding"
        assert AdapterType.VISION.value == "vision"
        assert AdapterType.AUDIO.value == "audio"
        assert AdapterType.VIDEO.value == "video"
        assert AdapterType.SPEECH.value == "speech"
    
    def test_task_type_values(self):
        """Test TaskType enum values."""
        assert TaskType.TEXT_GENERATION.value == "text_generation"
        assert TaskType.CODE_GENERATION.value == "code_generation"
        assert TaskType.IMAGE_UNDERSTANDING.value == "image_understanding"
    
    def test_domain_type_values(self):
        """Test DomainType enum values."""
        assert DomainType.GENERAL.value == "general"
        assert DomainType.CODING.value == "coding"
        assert DomainType.URDU.value == "urdu"
    
    def test_learning_decision_values(self):
        """Test LearningDecision enum values."""
        assert LearningDecision.IGNORE.value == "ignore"
        assert LearningDecision.REPLAY.value == "replay"
        assert LearningDecision.UPDATE_ADAPTER.value == "update_adapter"
        assert LearningDecision.CREATE_ADAPTER.value == "create_adapter"
        assert LearningDecision.FULL_TRAINING.value == "full_training"
    
    def test_adaptation_level_values(self):
        """Test AdaptationLevel enum values."""
        assert AdaptationLevel.SAFE.value == "safe"
        assert AdaptationLevel.CORE.value == "core"


class TestMultimodalData:
    """Test MultimodalData dataclass."""
    
    def test_text_only(self):
        """Test text-only data."""
        data = MultimodalData(text="Hello world")
        assert data.text == "Hello world"
        assert data.image is None
        assert data.audio is None
        assert data.video is None
        assert data.speech is None
        assert ModalityType.TEXT in data.modalities
        assert len(data.modalities) == 1
    
    def test_multimodal(self):
        """Test multimodal data."""
        data = MultimodalData(
            text="Describe this image",
            image="path/to/image.jpg",
        )
        assert ModalityType.TEXT in data.modalities
        assert ModalityType.VISION in data.modalities
        assert ModalityType.MULTI_MODAL in data.modalities
    
    def test_all_modalities(self):
        """Test data with all modalities."""
        data = MultimodalData(
            text="Multimodal content",
            image="image.jpg",
            audio="audio.wav",
            video="video.mp4",
            speech="speech.wav",
        )
        assert len(data.modalities) == 6  # 5 + multi_modal
        assert ModalityType.MULTI_MODAL in data.modalities


class TestMultimodalEntry:
    """Test MultimodalEntry dataclass."""
    
    def test_entry_creation(self):
        """Test entry creation."""
        data = MultimodalData(text="Test text")
        entry = MultimodalEntry(
            id="test-001",
            data=data,
            instruction="Test instruction",
            expected_output="Test output",
            domain=DomainType.GENERAL,
            language="en",
            importance=0.8,
            novelty=0.7,
        )
        
        assert entry.id == "test-001"
        assert entry.data == data
        assert entry.instruction == "Test instruction"
        assert entry.expected_output == "Test output"
        assert entry.domain == DomainType.GENERAL
        assert entry.language == "en"
        assert entry.importance == 0.8
        assert entry.novelty == 0.7
    
    def test_priority_score(self):
        """Test priority score calculation."""
        data = MultimodalData(text="Test")
        entry = MultimodalEntry(
            id="test-001",
            data=data,
            priority=MemoryPriority.CRITICAL,
            importance=1.0,
            novelty=1.0,
            difficulty=1.0,
            forgetting_risk=1.0,
            error_rate=1.0,
        )
        
        score = entry.get_priority_score()
        assert score > 0
        assert score > 1.0  # With all boosts, should be > 1
    
    def test_priority_score_medium(self):
        """Test medium priority score."""
        data = MultimodalData(text="Test")
        entry = MultimodalEntry(
            id="test-001",
            data=data,
            priority=MemoryPriority.MEDIUM,
        )
        
        score = entry.get_priority_score()
        assert 0 < score <= 1.0


class TestMemoryCandidate:
    """Test MemoryCandidate dataclass."""
    
    def test_should_store_high_novelty(self):
        """Test should_store with high novelty."""
        data = MultimodalData(text="Test")
        candidate = MemoryCandidate(
            data=data,
            novelty_score=0.8,
            importance_score=0.5,
        )
        
        assert candidate.should_store() is True
    
    def test_should_store_low_novelty(self):
        """Test should_store with low novelty."""
        data = MultimodalData(text="Test")
        candidate = MemoryCandidate(
            data=data,
            novelty_score=0.3,
            importance_score=0.5,
        )
        
        assert candidate.should_store() is False
    
    def test_should_store_duplicate(self):
        """Test should_store with duplicate."""
        data = MultimodalData(text="Test")
        candidate = MemoryCandidate(
            data=data,
            novelty_score=0.8,
            is_duplicate=True,
        )
        
        assert candidate.should_store() is False


class TestAdapterInfo:
    """Test AdapterInfo dataclass."""
    
    def test_adapter_info(self):
        """Test adapter info creation."""
        info = AdapterInfo(
            name="test-adapter",
            adapter_type=AdapterType.CODING,
            domain=DomainType.CODING,
            modality=ModalityType.TEXT,
            rank=8,
            alpha=16.0,
            target_modules=["q_proj", "k_proj"],
            performance={"accuracy": 0.95, "f1": 0.92},
        )
        
        assert info.name == "test-adapter"
        assert info.adapter_type == AdapterType.CODING
        assert info.get_performance_score() == 0.935


class TestModelVersion:
    """Test ModelVersion dataclass."""
    
    def test_model_version(self):
        """Test model version creation."""
        version = ModelVersion(
            version="1.0.0",
            base_model="Qwen/Qwen2.5-Omni-3B",
            adapters=["general", "coding"],
            performance={"text": 0.95, "vision": 0.90},
            forgetting_scores={"text": 0.02, "vision": 0.03},
            retention_score=0.98,
        )
        
        assert version.version == "1.0.0"
        assert version.base_model == "Qwen/Qwen2.5-Omni-3B"
        assert version.get_overall_performance() == 0.925
        assert version.get_forgetting_level() == 0.025


class TestStats:
    """Test statistics dataclasses."""
    
    def test_replay_stats(self):
        """Test ReplayStats."""
        stats = ReplayStats(
            total_entries=100,
            average_importance=0.8,
            average_novelty=0.7,
        )
        
        result = stats.to_dict()
        assert result["total_entries"] == 100
        assert result["average_importance"] == 0.8
    
    def test_forgetting_metrics(self):
        """Test ForgettingMetrics."""
        metrics = ForgettingMetrics(
            modality_forgetting={ModalityType.TEXT: 0.02},
            overall_forgetting=0.01,
            retention_score=0.99,
            forgetting_detected=False,
        )
        
        result = metrics.to_dict()
        assert result["overall_forgetting"] == 0.01
        assert result["retention_score"] == 0.99
    
    def test_protection_stats(self):
        """Test ProtectionStats."""
        stats = ProtectionStats(
            ewc_lambda=0.1,
            mas_lambda=0.1,
            si_lambda=0.1,
            protected_parameters=1000,
            total_parameters=10000,
        )
        
        result = stats.to_dict()
        assert result["ewc_lambda"] == 0.1
        assert result["protection_ratio"] == 0.1
    
    def test_training_stats(self):
        """Test TrainingStats."""
        stats = TrainingStats(
            epoch=10,
            step=1000,
            loss=0.5,
            learning_rate=2e-5,
        )
        
        result = stats.to_dict()
        assert result["epoch"] == 10
        assert result["loss"] == 0.5


class TestInferenceResult:
    """Test InferenceResult dataclass."""
    
    def test_inference_result(self):
        """Test inference result creation."""
        data = MultimodalData(text="Test input")
        result = InferenceResult(
            request_id="req-001",
            input_data=data,
            output="Test output",
            modality=ModalityType.TEXT,
            task_type=TaskType.TEXT_GENERATION,
            domain=DomainType.GENERAL,
            adapter_used="general",
            adapters_used=["general"],
            inference_time_ms=100.0,
            tokens_generated=50,
            confidence=0.95,
            is_success=True,
        )
        
        assert result.request_id == "req-001"
        assert result.output == "Test output"
        assert result.is_success is True
        assert result.confidence == 0.95
    
    def test_inference_result_to_dict(self):
        """Test inference result to_dict."""
        data = MultimodalData(text="Test")
        result = InferenceResult(
            request_id="req-001",
            input_data=data,
            output="Output",
            modality=ModalityType.TEXT,
        )
        
        result_dict = result.to_dict()
        assert result_dict["request_id"] == "req-001"
        assert result_dict["output"] == "Output"
        assert result_dict["modality"] == "text"


class TestConfigurations:
    """Test configuration classes."""
    
    def test_model_config_defaults(self):
        """Test QwenOmniModelConfig defaults."""
        config = QwenOmniModelConfig()
        
        assert config.base_model == "Qwen/Qwen2.5-Omni-3B"
        assert config.model_type == "qwen2.5-omni"
        assert config.use_thinker is True
        assert config.use_talker is True
        assert config.use_flash_attention is True
    
    def test_training_config_defaults(self):
        """Test QwenOmniTrainingConfig defaults."""
        config = QwenOmniTrainingConfig()
        
        assert config.learning_rate == 2e-5
        assert config.batch_size == 4
        assert config.use_lora is True
        assert config.lora_rank == 8
        assert config.adaptation_level == AdaptationLevel.SAFE
    
    def test_adapter_config_defaults(self):
        """Test QwenOmniAdapterConfig defaults."""
        config = QwenOmniAdapterConfig()
        
        assert AdapterType.GENERAL in config.default_adapters
        assert AdapterType.CODING in config.default_adapters
        assert config.adapter_rank == 8
    
    def test_memory_config_defaults(self):
        """Test QwenOmniMemoryConfig defaults."""
        config = QwenOmniMemoryConfig()
        
        assert config.max_entries == 10000
        assert config.sampling_strategy == "priority_based"
        assert config.novelty_threshold == 0.5
    
    def test_drift_config_defaults(self):
        """Test QwenOmniDriftConfig defaults."""
        config = QwenOmniDriftConfig()
        
        assert config.use_statistical is True
        assert config.use_clip is True
        assert config.forgetting_threshold == 0.03
    
    def test_evaluation_config_defaults(self):
        """Test QwenOmniEvaluationConfig defaults."""
        config = QwenOmniEvaluationConfig()
        
        assert config.eval_interval == 100
        assert config.min_improvement == 0.05
        assert config.max_forgetting == 0.03
        assert config.min_retention == 0.98
    
    def test_registry_config_defaults(self):
        """Test QwenOmniRegistryConfig defaults."""
        config = QwenOmniRegistryConfig()
        
        assert config.registry_path == "./registry"
        assert config.enable_rollback is True
        assert config.rollback_window == 5
    
    def test_main_config(self):
        """Test QwenOmniConfig main configuration."""
        config = QwenOmniConfig()
        
        assert isinstance(config.model, QwenOmniModelConfig)
        assert isinstance(config.training, QwenOmniTrainingConfig)
        assert isinstance(config.adapters, QwenOmniAdapterConfig)
        assert isinstance(config.memory, QwenOmniMemoryConfig)
        assert isinstance(config.drift, QwenOmniDriftConfig)
        assert isinstance(config.evaluation, QwenOmniEvaluationConfig)
        assert isinstance(config.registry, QwenOmniRegistryConfig)
        assert config.project_name == "adaptive_qwen_omni"


class TestConfigSerialization:
    """Test configuration serialization."""
    
    def test_to_yaml(self, tmp_path):
        """Test saving config to YAML."""
        config = QwenOmniConfig()
        config_path = tmp_path / "config.yaml"
        
        config.to_yaml(str(config_path))
        
        assert config_path.exists()
        
        # Load it back
        loaded = QwenOmniConfig.from_yaml(str(config_path))
        assert loaded.project_name == config.project_name
    
    def test_update(self):
        """Test config update."""
        config = QwenOmniConfig()
        updated = config.update(project_name="test-project")
        
        assert updated.project_name == "test-project"
    
    def test_get_dict(self):
        """Test config to dict."""
        config = QwenOmniConfig()
        config_dict = config.get_dict()
        
        assert "model" in config_dict
        assert "training" in config_dict
        # project_name is at the top level
        assert config.project_name == "adaptive_qwen_omni"
