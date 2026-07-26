"""
Core types and configuration for Adaptive Qwen Omni system.
Specifically designed for Qwen2.5-Omni-3B with Thinker-Talker architecture.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field
from typing_extensions import Annotated


# =============================================================================
# ENUMS
# =============================================================================

class ModalityType(str, Enum):
    """Supported modalities for Qwen2.5-Omni-3B."""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    SPEECH = "speech"
    MULTI_MODAL = "multi_modal"


class AdapterType(str, Enum):
    """Types of adapters for different capabilities."""
    GENERAL = "general"
    CODING = "coding"
    MATH = "math"
    URDU = "urdu"
    ENGLISH = "english"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    SPEECH = "speech"
    DOMAIN_SPECIFIC = "domain_specific"
    CUSTOM = "custom"


class TaskType(str, Enum):
    """Types of tasks the system can handle."""
    TEXT_GENERATION = "text_generation"
    TEXT_UNDERSTANDING = "text_understanding"
    CODE_GENERATION = "code_generation"
    CODE_UNDERSTANDING = "code_understanding"
    IMAGE_UNDERSTANDING = "image_understanding"
    IMAGE_GENERATION = "image_generation"
    AUDIO_UNDERSTANDING = "audio_understanding"
    AUDIO_GENERATION = "audio_generation"
    VIDEO_UNDERSTANDING = "video_understanding"
    SPEECH_RECOGNITION = "speech_recognition"
    SPEECH_GENERATION = "speech_generation"
    MULTIMODAL_UNDERSTANDING = "multimodal_understanding"
    MULTIMODAL_GENERATION = "multimodal_generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    REASONING = "reasoning"


class DomainType(str, Enum):
    """Domain classifications."""
    GENERAL = "general"
    CODING = "coding"
    MATHEMATICS = "mathematics"
    URDU = "urdu"
    ENGLISH = "english"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    EDUCATION = "education"
    NEWS = "news"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCE = "finance"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    CUSTOM = "custom"


class ForgettingDetectionStrategy(str, Enum):
    """Strategies for detecting catastrophic forgetting."""
    PERFORMANCE_DROP = "performance_drop"
    ACCURACY_THRESHOLD = "accuracy_threshold"
    MODALITY_SPECIFIC = "modality_specific"
    WEIGHTED_AVERAGE = "weighted_average"
    REGRESSION_TESTING = "regression_testing"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"


class AdaptationLevel(str, Enum):
    """Levels of model adaptation."""
    SAFE = "safe"  # LoRA/QLoRA adapters only
    CORE = "core"  # Merge adapters into base model


class LearningDecision(str, Enum):
    """Decisions for handling new data."""
    IGNORE = "ignore"  # Already known
    REPLAY = "replay"  # Add to replay memory
    UPDATE_ADAPTER = "update_adapter"  # Update existing adapter
    CREATE_ADAPTER = "create_adapter"  # Create new adapter
    FULL_TRAINING = "full_training"  # Full model training (rare)


class MemoryPriority(str, Enum):
    """Priority levels for memory entries."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MultimodalData:
    """Represents multimodal input data."""
    text: Optional[str] = None
    image: Optional[Union[str, np.ndarray, torch.Tensor]] = None
    audio: Optional[Union[str, np.ndarray, torch.Tensor]] = None
    video: Optional[Union[str, np.ndarray, torch.Tensor]] = None
    speech: Optional[Union[str, np.ndarray, torch.Tensor]] = None
    modalities: List[ModalityType] = field(default_factory=list)
    
    def __post_init__(self):
        # Auto-detect modalities
        if self.text is not None:
            if ModalityType.TEXT not in self.modalities:
                self.modalities.append(ModalityType.TEXT)
        if self.image is not None:
            if ModalityType.VISION not in self.modalities:
                self.modalities.append(ModalityType.VISION)
        if self.audio is not None:
            if ModalityType.AUDIO not in self.modalities:
                self.modalities.append(ModalityType.AUDIO)
        if self.video is not None:
            if ModalityType.VIDEO not in self.modalities:
                self.modalities.append(ModalityType.VIDEO)
        if self.speech is not None:
            if ModalityType.SPEECH not in self.modalities:
                self.modalities.append(ModalityType.SPEECH)
        
        if len(self.modalities) > 1:
            if ModalityType.MULTI_MODAL not in self.modalities:
                self.modalities.append(ModalityType.MULTI_MODAL)


@dataclass
class MultimodalEntry:
    """A single entry in the multimodal replay memory."""
    id: str
    data: MultimodalData
    instruction: Optional[str] = None
    expected_output: Optional[str] = None
    domain: DomainType = DomainType.GENERAL
    language: str = "en"
    importance: float = 1.0
    novelty: float = 0.5
    difficulty: float = 0.5
    source: str = "unknown"
    version: str = "base-v1"
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MemoryPriority = MemoryPriority.MEDIUM
    forgetting_risk: float = 0.0
    error_rate: float = 0.0
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def get_priority_score(self) -> float:
        """Calculate priority score for sampling."""
        priority_weights = {
            MemoryPriority.CRITICAL: 1.0,
            MemoryPriority.HIGH: 0.8,
            MemoryPriority.MEDIUM: 0.5,
            MemoryPriority.LOW: 0.2
        }
        base_score = priority_weights.get(self.priority, 0.5)
        
        # Boost by importance, novelty, difficulty, forgetting risk, error rate
        score = base_score * (1 + self.importance * 0.5)
        score *= (1 + self.novelty * 0.3)
        score *= (1 + self.difficulty * 0.2)
        score *= (1 + self.forgetting_risk * 2.0)  # High forgetting risk = high priority
        score *= (1 + self.error_rate * 1.5)  # High error rate = high priority
        
        return score


@dataclass
class MemoryCandidate:
    """Candidate for adding to replay memory."""
    data: MultimodalData
    instruction: Optional[str] = None
    expected_output: Optional[str] = None
    novelty_score: float = 0.0
    importance_score: float = 0.0
    quality_score: float = 0.0
    is_duplicate: bool = False
    is_low_quality: bool = False
    is_unsafe: bool = False
    
    def should_store(self, threshold: float = 0.5) -> bool:
        """Determine if this candidate should be stored."""
        if self.is_duplicate or self.is_low_quality or self.is_unsafe:
            return False
        return self.novelty_score >= threshold or self.importance_score >= 0.8


@dataclass
class AdapterInfo:
    """Information about an adapter."""
    name: str
    adapter_type: AdapterType
    domain: DomainType = DomainType.GENERAL
    modality: ModalityType = ModalityType.TEXT
    rank: int = 8
    alpha: float = 16.0
    target_modules: List[str] = field(default_factory=list)
    is_quantized: bool = False
    quantization_bits: int = 4
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    performance: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score."""
        if not self.performance:
            return 0.0
        return sum(self.performance.values()) / len(self.performance)


@dataclass
class ModelVersion:
    """Information about a model version."""
    version: str
    base_model: str = "Qwen/Qwen2.5-Omni-3B"
    adapters: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    performance: Dict[str, float] = field(default_factory=dict)
    forgetting_scores: Dict[str, float] = field(default_factory=dict)
    retention_score: float = 1.0
    is_production: bool = False
    size_gb: float = 0.0
    
    def get_overall_performance(self) -> float:
        """Calculate overall performance across all modalities."""
        if not self.performance:
            return 0.0
        return sum(self.performance.values()) / len(self.performance)
    
    def get_forgetting_level(self) -> float:
        """Calculate overall forgetting level."""
        if not self.forgetting_scores:
            return 0.0
        return sum(self.forgetting_scores.values()) / len(self.forgetting_scores)


# =============================================================================
# STATS AND METRICS
# =============================================================================

@dataclass
class ReplayStats:
    """Statistics for replay memory."""
    total_entries: int = 0
    entries_by_modality: Dict[ModalityType, int] = field(default_factory=dict)
    entries_by_domain: Dict[DomainType, int] = field(default_factory=dict)
    entries_by_priority: Dict[MemoryPriority, int] = field(default_factory=dict)
    average_importance: float = 0.0
    average_novelty: float = 0.0
    average_forgetting_risk: float = 0.0
    average_error_rate: float = 0.0
    utilization_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "entries_by_modality": {k.value: v for k, v in self.entries_by_modality.items()},
            "entries_by_domain": {k.value: v for k, v in self.entries_by_domain.items()},
            "entries_by_priority": {k.value: v for k, v in self.entries_by_priority.items()},
            "average_importance": self.average_importance,
            "average_novelty": self.average_novelty,
            "average_forgetting_risk": self.average_forgetting_risk,
            "average_error_rate": self.average_error_rate,
            "utilization_rate": self.utilization_rate,
        }


@dataclass
class ForgettingMetrics:
    """Metrics for forgetting detection."""
    modality_forgetting: Dict[ModalityType, float] = field(default_factory=dict)
    domain_forgetting: Dict[DomainType, float] = field(default_factory=dict)
    overall_forgetting: float = 0.0
    retention_score: float = 1.0
    forgetting_detected: bool = False
    critical_modalities: List[ModalityType] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality_forgetting": {k.value: v for k, v in self.modality_forgetting.items()},
            "domain_forgetting": {k.value: v for k, v in self.domain_forgetting.items()},
            "overall_forgetting": self.overall_forgetting,
            "retention_score": self.retention_score,
            "forgetting_detected": self.forgetting_detected,
            "critical_modalities": [m.value for m in self.critical_modalities],
        }


@dataclass
class ProtectionStats:
    """Statistics for parameter protection."""
    ewc_lambda: float = 0.0
    mas_lambda: float = 0.0
    si_lambda: float = 0.0
    protected_parameters: int = 0
    total_parameters: int = 0
    protection_ratio: float = 0.0
    
    def __post_init__(self):
        """Calculate protection ratio."""
        if self.total_parameters > 0:
            self.protection_ratio = self.protected_parameters / self.total_parameters
        else:
            self.protection_ratio = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ewc_lambda": self.ewc_lambda,
            "mas_lambda": self.mas_lambda,
            "si_lambda": self.si_lambda,
            "protected_parameters": self.protected_parameters,
            "total_parameters": self.total_parameters,
            "protection_ratio": self.protection_ratio,
        }


@dataclass
class TrainingStats:
    """Statistics for training."""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    new_data_loss: float = 0.0
    replay_data_loss: float = 0.0
    distillation_loss: float = 0.0
    ewc_loss: float = 0.0
    gradient_norm: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "new_data_loss": self.new_data_loss,
            "replay_data_loss": self.replay_data_loss,
            "distillation_loss": self.distillation_loss,
            "ewc_loss": self.ewc_loss,
            "gradient_norm": self.gradient_norm,
        }


@dataclass
class InferenceResult:
    """Result of a multimodal inference operation."""
    request_id: str
    input_data: MultimodalData
    output: Optional[str] = None
    output_tokens: Optional[List[str]] = None
    output_embedding: Optional[torch.Tensor] = None
    modality: ModalityType = ModalityType.TEXT
    task_type: Optional[TaskType] = None
    domain: DomainType = DomainType.GENERAL
    adapter_used: Optional[str] = None
    adapters_used: List[str] = field(default_factory=list)
    inference_time_ms: float = 0.0
    tokens_generated: int = 0
    confidence: float = 0.0
    is_success: bool = True
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "request_id": self.request_id,
            "input_modalities": [m.value for m in self.input_data.modalities],
            "output": self.output,
            "modality": self.modality.value,
            "task_type": self.task_type.value if self.task_type else None,
            "domain": self.domain.value,
            "adapter_used": self.adapter_used,
            "adapters_used": self.adapters_used,
            "inference_time_ms": self.inference_time_ms,
            "tokens_generated": self.tokens_generated,
            "confidence": self.confidence,
            "is_success": self.is_success,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }
        return result


# =============================================================================
# CONFIGURATION
# =============================================================================

class QwenOmniModelConfig(BaseModel):
    """Configuration for Qwen2.5-Omni-3B model."""
    base_model: str = "Qwen/Qwen2.5-Omni-3B"
    model_type: str = "qwen2.5-omni"
    
    # Thinker-Talker architecture
    use_thinker: bool = True
    use_talker: bool = True
    
    # Quantization
    quantize: bool = False
    quantization_bits: int = 4
    quantization_method: str = "bitsandbytes"
    
    # Device
    device: str = "cuda"
    device_map: Optional[Dict[str, str]] = None
    
    # Memory optimization
    use_flash_attention: bool = True
    use_bfloat16: bool = True
    
    # Modality support
    supported_modalities: List[ModalityType] = Field(default_factory=lambda: [
        ModalityType.TEXT, ModalityType.VISION, ModalityType.AUDIO, 
        ModalityType.VIDEO, ModalityType.SPEECH, ModalityType.MULTI_MODAL
    ])
    
    # Memory requirements (for validation)
    min_memory_gb: float = 18.0  # Minimum for 15s video
    recommended_memory_gb: float = 24.0
    
    class Config:
        use_enum_values = True


class QwenOmniTrainingConfig(BaseModel):
    """Training configuration for Qwen Omni."""
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_steps: int = 1000
    warmup_steps: int = 100
    
    # LoRA configuration
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # QLoRA configuration
    use_qlora: bool = False
    quantization_bits: int = 4
    
    # Dynamic LoRA
    use_dynamic_lora: bool = True
    min_rank: int = 4
    max_rank: int = 32
    rank_growth_rate: float = 1.5
    rank_shrink_threshold: float = 0.8
    rank_growth_threshold: float = 0.9
    
    # Parameter importance
    use_ewc: bool = True
    ewc_lambda: float = 0.1
    use_mas: bool = True
    mas_lambda: float = 0.1
    use_si: bool = True
    si_lambda: float = 0.1
    use_fisher: bool = False
    
    # Knowledge distillation
    use_distillation: bool = True
    distillation_temperature: float = 2.0
    distillation_weight: float = 0.5
    
    # Replay configuration
    use_replay: bool = True
    replay_ratio: float = 0.3
    replay_batch_size: int = 4
    
    # Adaptation level
    adaptation_level: AdaptationLevel = AdaptationLevel.SAFE
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_interval: int = 500
    max_checkpoints: int = 5
    
    class Config:
        use_enum_values = True


class QwenOmniAdapterConfig(BaseModel):
    """Configuration for adapters."""
    # Default adapters
    default_adapters: List[AdapterType] = Field(default_factory=lambda: [
        AdapterType.GENERAL, AdapterType.CODING, AdapterType.URDU,
        AdapterType.VISION, AdapterType.AUDIO, AdapterType.VIDEO
    ])
    
    # Adapter settings
    adapter_rank: int = 8
    adapter_alpha: float = 16.0
    adapter_dropout: float = 0.05
    
    # Dynamic adapter creation
    create_new_adapters: bool = True
    adapter_creation_threshold: float = 0.7  # Novelty threshold
    
    # Adapter routing
    use_router: bool = True
    router_type: str = "task_based"  # or "modality_based", "hybrid"
    
    # Adapter merging (Level 2 adaptation)
    enable_adapter_merging: bool = True
    merge_threshold: float = 0.95  # Performance threshold for merging
    merge_interval: int = 1000  # Steps between merge checks
    
    # Domain-specific adapters
    domain_adapters: Dict[DomainType, List[AdapterType]] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class QwenOmniMemoryConfig(BaseModel):
    """Configuration for multimodal replay memory."""
    # Memory size
    max_entries: int = 10000
    max_entries_per_modality: Dict[ModalityType, int] = Field(default_factory=lambda: {
        ModalityType.TEXT: 3000,
        ModalityType.VISION: 2000,
        ModalityType.AUDIO: 2000,
        ModalityType.VIDEO: 1500,
        ModalityType.SPEECH: 1500,
        ModalityType.MULTI_MODAL: 1000,
    })
    max_entries_per_domain: int = 1000
    
    # Sampling
    sampling_strategy: str = "priority_based"  # uniform, balanced, priority, adaptive
    temperature: float = 1.0
    
    # Priority weights
    importance_weight: float = 0.5
    novelty_weight: float = 0.3
    difficulty_weight: float = 0.2
    forgetting_risk_weight: float = 2.0
    error_rate_weight: float = 1.5
    
    # Memory compression
    use_compression: bool = True
    compression_method: str = "faiss_ivf_pq"
    nlist: int = 100
    nprobe: int = 10
    m: int = 8
    nbits: int = 8
    
    # Novelty detection
    novelty_threshold: float = 0.5
    
    class Config:
        use_enum_values = True


class QwenOmniDriftConfig(BaseModel):
    """Configuration for drift detection."""
    # Drift detection methods
    use_statistical: bool = True
    use_semantic: bool = True
    use_clip: bool = True
    
    # CLIP configuration
    clip_model: str = "ViT-B/32"
    clip_threshold: float = 0.85
    
    # Statistical methods
    statistical_methods: List[str] = Field(default_factory=lambda: [
        "ks", "psi", "wasserstein", "pca"
    ])
    statistical_threshold: float = 0.05
    
    # Modality-specific drift
    detect_by_modality: bool = True
    modality_weights: Dict[ModalityType, float] = Field(default_factory=lambda: {
        ModalityType.TEXT: 1.0,
        ModalityType.VISION: 1.0,
        ModalityType.AUDIO: 1.0,
        ModalityType.VIDEO: 1.0,
        ModalityType.SPEECH: 1.0,
    })
    
    # Forgetting detection
    forgetting_strategy: ForgettingDetectionStrategy = ForgettingDetectionStrategy.MODALITY_SPECIFIC
    forgetting_threshold: float = 0.03  # 3% performance drop
    critical_modality_threshold: float = 0.05  # 5% for critical modalities
    
    class Config:
        use_enum_values = True


class QwenOmniEvaluationConfig(BaseModel):
    """Configuration for evaluation."""
    # Evaluation frequency
    eval_interval: int = 100
    eval_batch_size: int = 8
    
    # Metrics
    use_perplexity: bool = True
    use_bleu: bool = True
    use_rouge: bool = True
    use_f1: bool = True
    
    # Modality-specific evaluation
    evaluate_text: bool = True
    evaluate_vision: bool = True
    evaluate_audio: bool = True
    evaluate_video: bool = True
    evaluate_speech: bool = True
    
    # Benchmark datasets
    text_benchmark: str = "mmlu"
    vision_benchmark: str = "mmmu"
    audio_benchmark: str = "librispeech"
    video_benchmark: str = "msrvtt"
    
    # Retention scoring
    retention_weights: Dict[ModalityType, float] = Field(default_factory=lambda: {
        ModalityType.TEXT: 0.3,
        ModalityType.VISION: 0.2,
        ModalityType.AUDIO: 0.2,
        ModalityType.VIDEO: 0.2,
        ModalityType.SPEECH: 0.1,
    })
    
    # Promotion criteria
    min_improvement: float = 0.05  # 5% improvement required
    max_forgetting: float = 0.03  # 3% max forgetting allowed
    min_retention: float = 0.98  # 98% retention required
    
    class Config:
        use_enum_values = True


class QwenOmniRegistryConfig(BaseModel):
    """Configuration for model registry."""
    registry_path: str = "./registry"
    model_storage_path: str = "./models"
    adapter_storage_path: str = "./adapters"
    
    # Versioning
    version_format: str = "{base}-{timestamp}-{hash}"
    max_versions: int = 10
    
    # Rollback
    enable_rollback: bool = True
    rollback_window: int = 5  # Number of versions to keep for rollback
    
    # ONNX export
    export_onnx: bool = True
    onnx_opset: int = 14
    
    # MLflow integration
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "./mlruns"
    
    class Config:
        use_enum_values = True


class QwenOmniConfig(BaseModel):
    """
    Main configuration for Adaptive Qwen Omni system.
    Combines all sub-configurations with Qwen2.5-Omni-3B specific defaults.
    """
    # Model configuration
    model: QwenOmniModelConfig = Field(default_factory=QwenOmniModelConfig)
    
    # Training configuration
    training: QwenOmniTrainingConfig = Field(default_factory=QwenOmniTrainingConfig)
    
    # Adapter configuration
    adapters: QwenOmniAdapterConfig = Field(default_factory=QwenOmniAdapterConfig)
    
    # Memory configuration
    memory: QwenOmniMemoryConfig = Field(default_factory=QwenOmniMemoryConfig)
    
    # Drift detection configuration
    drift: QwenOmniDriftConfig = Field(default_factory=QwenOmniDriftConfig)
    
    # Evaluation configuration
    evaluation: QwenOmniEvaluationConfig = Field(default_factory=QwenOmniEvaluationConfig)
    
    # Registry configuration
    registry: QwenOmniRegistryConfig = Field(default_factory=QwenOmniRegistryConfig)
    
    # Project metadata
    project_name: str = "adaptive_qwen_omni"
    experiment_name: str = "default"
    run_id: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    
    class Config:
        use_enum_values = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "QwenOmniConfig":
        """Load configuration from YAML file."""
        from pathlib import Path
        import yaml
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        from pathlib import Path
        import yaml
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.model_dump(exclude_unset=True)
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def update(self, **kwargs: Any) -> "QwenOmniConfig":
        """Update configuration with new values."""
        config_dict = self.model_dump()
        config_dict.update(kwargs)
        return QwenOmniConfig(**config_dict)
    
    def get_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        result = self.model_dump(exclude_unset=True)
        # Ensure nested configs are included
        if 'model' not in result:
            result['model'] = self.model.model_dump(exclude_unset=True)
        if 'training' not in result:
            result['training'] = self.training.model_dump(exclude_unset=True)
        if 'adapters' not in result:
            result['adapters'] = self.adapters.model_dump(exclude_unset=True)
        if 'memory' not in result:
            result['memory'] = self.memory.model_dump(exclude_unset=True)
        if 'drift' not in result:
            result['drift'] = self.drift.model_dump(exclude_unset=True)
        if 'evaluation' not in result:
            result['evaluation'] = self.evaluation.model_dump(exclude_unset=True)
        if 'registry' not in result:
            result['registry'] = self.registry.model_dump(exclude_unset=True)
        return result


# Default configuration
DEFAULT_QWEN_OMNI_CONFIG = QwenOmniConfig()


def get_qwen_omni_config(config_path: Optional[str] = None) -> QwenOmniConfig:
    """
    Get Qwen Omni configuration from file or use defaults.
    
    Args:
        config_path: Path to YAML configuration file. If None, uses defaults.
    
    Returns:
        QwenOmniConfig instance
    """
    if config_path is None:
        return DEFAULT_QWEN_OMNI_CONFIG
    
    try:
        return QwenOmniConfig.from_yaml(config_path)
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        return DEFAULT_QWEN_OMNI_CONFIG
