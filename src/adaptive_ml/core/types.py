"""
Data structures and enums for Adaptive ML Framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class TaskStatus(Enum):
    """Status of a task in the continual learning pipeline."""

    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class DriftType(Enum):
    """Types of data drift."""

    NONE = "none"
    STATISTICAL = "statistical"  # KS-test, PSI, Wasserstein
    SEMANTIC = "semantic"  # Embedding-based drift
    CONCEPT = "concept"  # Both statistical and semantic


class SamplingStrategy(Enum):
    """Sampling strategies for replay buffer."""

    UNIFORM = "uniform"  # Random sampling
    BALANCED = "balanced"  # Class/task-balanced sampling
    IMPORTANCE = "importance"  # Importance-weighted sampling
    DIVERSITY = "diversity"  # Diversity-based sampling (FAISS)
    HARD_EXAMPLE = "hard_example"  # High-uncertainty examples


class ParameterImportanceMethod(Enum):
    """Methods for parameter importance estimation."""

    EWC = "ewc"  # Elastic Weight Consolidation
    MAS = "mas"  # Memory Aware Synapses
    SI = "si"  # Synaptic Intelligence
    FISHER = "fisher"  # Full Fisher Information
    GRADIENT = "gradient"  # Gradient-based importance


class AdapterType(Enum):
    """Types of adapters for parameter-efficient fine-tuning."""

    LORA = "lora"
    QLORA = "qlora"
    ADAPTER = "adapter"  # Standard adapter layers
    PREFIX = "prefix"  # Prefix tuning
    IA3 = "ia3"  # IA³ (Infused Adapter)


class PromotionStrategy(Enum):
    """Strategies for model promotion."""

    STRICT = "strict"  # Old knowledge >=95% preserved + new improvement
    BALANCED = "balanced"  # Allows A/B test gate for borderline cases
    AGGRESSIVE = "aggressive"  # Prioritizes new capability acquisition


@dataclass
class Task:
    """Represents a task in continual learning."""

    id: str
    name: str
    description: str = ""
    domain: str = "general"
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    adapter_id: Optional[str] = None
    data_path: Optional[str] = None
    num_examples: int = 0
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if self.completed_at and isinstance(self.completed_at, str):
            self.completed_at = datetime.fromisoformat(self.completed_at)


@dataclass
class DatasetEntry:
    """Represents an entry in a dataset for continual learning."""

    data: Any  # Input data (text, image, tensor, etc.)
    label: Optional[Any] = None  # Ground truth label
    task_id: str = "default"  # Task identifier
    is_replay: bool = False  # Whether this is from replay buffer
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    """Represents an entry in the replay buffer."""

    task_id: str
    data: Any  # Input data (text, image, etc.)
    label: Optional[Any] = None  # Ground truth label
    embedding: Optional[List[float]] = None  # Precomputed embedding for diversity sampling
    importance: float = 1.0  # Importance score for weighted sampling
    uncertainty: float = 0.0  # Model uncertainty (for hard example replay)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_embedding_array(self) -> Optional[np.ndarray]:
        """Convert embedding list to numpy array."""
        if self.embedding is None:
            return None
        return np.array(self.embedding)


@dataclass
class DriftResult:
    """Result of drift detection."""

    drift_type: DriftType
    score: float  # Drift magnitude (0-1)
    threshold: float  # Threshold for drift detection
    is_drift: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 3
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Continual learning specific
    ewc_lambda: float = 1000.0  # EWC regularization strength
    mas_lambda: float = 100.0  # MAS regularization strength
    si_lambda: float = 100.0  # SI regularization strength
    distill_alpha: float = 0.5  # Knowledge distillation weight
    replay_ratio: float = 0.3  # Fraction of batch from replay buffer
    
    # Parameter importance method
    importance_method: ParameterImportanceMethod = ParameterImportanceMethod.EWC
    
    # Dynamic architecture
    dynamic_lora: bool = True  # Enable dynamic LoRA rank adaptation
    lora_rank_range: Tuple[int, int] = (8, 64)  # Min and max LoRA rank
    lora_growth_rate: float = 1.1  # Rank growth multiplier
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    
    # Device
    device: str = "auto"  # Will be resolved at runtime to "cuda" or "cpu"


@dataclass
class AdapterConfig:
    """Configuration for adapters."""

    adapter_type: AdapterType = AdapterType.LORA
    r: int = 16  # Rank for LoRA
    lora_alpha: int = 32  # Scaling factor for LoRA
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    task_type: str = "CAUSAL_LM"  # For PEFT
    
    # QLoRA specific
    bits: int = 4  # Quantization bits
    double_quant: bool = True
    
    # Adapter routing
    use_router: bool = True
    router_type: str = "domain"  # "domain", "task", "hybrid"
    
    # Dynamic LoRA
    dynamic_rank: bool = True  # Enable dynamic rank adaptation
    min_rank: int = 8  # Minimum LoRA rank
    max_rank: int = 64  # Maximum LoRA rank
    rank_growth_rate: float = 1.1  # Multiplier for rank growth
    rank_shrink_threshold: float = 0.8  # Accuracy threshold to shrink rank
    
    # Multi-modal adapters
    use_vision_adapter: bool = True  # Use adapter for vision modality
    use_audio_adapter: bool = True  # Use adapter for audio modality
    vision_adapter_type: str = "lora"  # Adapter type for vision
    audio_adapter_type: str = "lora"  # Adapter type for audio


@dataclass
class MemoryConfig:
    """Configuration for replay buffer."""

    buffer_size: int = 10000
    sampling_strategy: SamplingStrategy = SamplingStrategy.BALANCED
    
    # Diversity sampling
    diversity_metric: str = "cosine"  # "cosine", "euclidean", "dot"
    diversity_threshold: float = 0.5
    
    # Importance sampling
    importance_metric: str = "uncertainty"  # "uncertainty", "loss", "custom"
    
    # Reservoir sampling
    use_reservoir: bool = True
    
    # Memory compression (for large buffers)
    use_compression: bool = True
    compression_method: str = "ivf"  # "ivf", "pq", "ivf_pq", "hnsw"
    nlist: int = 100  # Number of IVF clusters
    nprobe: int = 10  # Number of probes for IVF
    m: int = 8  # PQ codebook size
    nbits: int = 8  # PQ quantization bits


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    window_size: int = 1000
    reference_size: int = 1000
    
    # Statistical drift
    statistical_test: str = "ks"  # "ks", "psi", "wasserstein"
    statistical_threshold: float = 0.05
    
    # Semantic drift
    semantic_threshold: float = 0.15
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Multi-modal semantic drift
    use_clip: bool = True  # Use CLIP for image-text drift detection
    clip_model: str = "openai/clip-vit-base-patch32"  # CLIP model for multi-modal
    clip_threshold: float = 0.2  # Threshold for CLIP-based drift
    
    # Concept drift
    concept_threshold: float = 0.1
    
    # Advanced drift detection
    use_ensemble: bool = True  # Combine multiple drift detection methods
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "statistical": 0.4,
        "semantic": 0.3,
        "clip": 0.3
    })


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and promotion."""

    promotion_strategy: PromotionStrategy = PromotionStrategy.STRICT
    retention_threshold: float = 0.95  # Minimum retention score for promotion
    new_task_threshold: float = 0.1  # Minimum improvement on new tasks
    
    # Retention score weights
    new_score_weight: float = 0.4
    old_score_weight: float = 0.4
    forgetting_penalty_weight: float = 0.2
    
    # A/B testing
    ab_test_ratio: float = 0.1  # Fraction of traffic for A/B test
    ab_test_duration: str = "1h"  # Duration for A/B test
    
    # Advanced evaluation metrics
    use_perplexity: bool = True  # Calculate perplexity for language tasks
    use_bleu: bool = True  # Calculate BLEU score for text generation
    use_rouge: bool = True  # Calculate ROUGE score for text generation
    use_f1: bool = True  # Calculate F1 score for classification
    
    # Multi-modal evaluation
    use_vision_metrics: bool = True  # Calculate vision-specific metrics
    use_audio_metrics: bool = True  # Calculate audio-specific metrics


@dataclass
class RegistryConfig:
    """Configuration for model registry."""

    registry_path: str = "./model_registry"
    max_versions: int = 10  # Maximum number of versions to keep
    auto_archive: bool = True
    
    # ONNX export
    export_onnx: bool = True
    onnx_opset: int = 14


class ModalityType(Enum):
    """Types of data modalities."""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTI_MODAL = "multi_modal"


@dataclass
class ModelConfig:
    """Configuration for base model."""

    base_model: str = "Qwen/Qwen2.5-Omni-3B"  # Multi-modal model
    tokenizer: Optional[str] = None  # If None, uses base_model's tokenizer
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    device_map: Optional[Dict[str, str]] = None  # For multi-GPU
    
    # Multi-modal settings
    modality: ModalityType = ModalityType.MULTI_MODAL
    vision_encoder: Optional[str] = None  # Custom vision encoder
    audio_encoder: Optional[str] = None  # Custom audio encoder
    
    # Quantization for large models
    quantize: bool = True
    quantization_bits: int = 4  # 4, 8, or None
    
    def get_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        return dtype_map.get(self.dtype, torch.float32)
