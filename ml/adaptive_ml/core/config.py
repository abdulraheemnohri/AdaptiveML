"""
Configuration management for Adaptive ML Framework.
Uses Pydantic for validation and YAML for configuration files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, validator

from adaptive_ml.core.types import (
    AdapterConfig,
    DriftConfig,
    EvaluationConfig,
    MemoryConfig,
    ModelConfig,
    RegistryConfig,
    TrainingConfig,
)


class AdaptiveMLConfig(BaseModel):
    """
    Main configuration for Adaptive ML Framework.
    Combines all sub-configurations with sensible defaults.
    """

    # Model configuration
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Training configuration
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # Adapter configuration
    adapters: AdapterConfig = Field(default_factory=AdapterConfig)

    # Memory configuration
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # Drift detection configuration
    drift: DriftConfig = Field(default_factory=DriftConfig)

    # Evaluation configuration
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # Registry configuration
    registry: RegistryConfig = Field(default_factory=RegistryConfig)

    # Additional metadata
    project_name: str = "adaptive_ml"
    experiment_name: str = "default"
    run_id: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"

    # MLflow tracking
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: Optional[str] = None

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @classmethod
    def from_yaml(cls, path: str) -> "AdaptiveMLConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.model_dump(exclude_unset=True)
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def update(self, **kwargs: Any) -> "AdaptiveMLConfig":
        """Update configuration with new values."""
        config_dict = self.model_dump()
        config_dict.update(kwargs)
        return cls(**config_dict)

    def get_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.model_dump(exclude_unset=True)


# Default configuration
DEFAULT_CONFIG = AdaptiveMLConfig()


def get_config(config_path: Optional[str] = None) -> AdaptiveMLConfig:
    """
    Get configuration from file or use defaults.

    Args:
        config_path: Path to YAML configuration file. If None, uses defaults.

    Returns:
        AdaptiveMLConfig instance
    """
    if config_path is None:
        return DEFAULT_CONFIG

    try:
        return AdaptiveMLConfig.from_yaml(config_path)
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        return DEFAULT_CONFIG
