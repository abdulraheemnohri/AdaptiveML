"""
Model Registry for Adaptive ML Framework.
Manages model versioning, storage, and rollback for continual learning.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from adaptive_ml.core.config import AdaptiveMLConfig


@dataclass
class ModelVersion:
    """Information about a model version."""

    version: str
    model_path: str
    config_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    parameters: int = 0
    status: str = "active"  # "active", "archived", "deleted"


@dataclass
class RegistryStats:
    """Statistics for the model registry."""

    num_versions: int = 0
    current_version: Optional[str] = None
    total_size_bytes: int = 0
    max_versions: int = 0
    auto_archive: bool = False


class ModelRegistry:
    """
    Manages model versions for continual learning.

    Features:
    - Versioned model storage
    - Atomic promotion and rollback
    - Automatic archiving of old versions
    - ONNX export support
    - Metadata tracking
    - Model loading and unloading

    Usage:
        registry = ModelRegistry(config)

        # Save a model version
        registry.save_version("v1.0.0", model, metadata={"accuracy": 0.95})

        # Load a model version
        model = registry.load_version("v1.0.0")

        # Promote a version
        registry.promote("v1.1.0")

        # Rollback to previous version
        registry.rollback()
    """

    def __init__(
        self,
        config: Optional[AdaptiveMLConfig] = None,
        registry_path: Optional[str] = None,
    ):
        """
        Initialize ModelRegistry.

        Args:
            config: AdaptiveMLConfig instance
            registry_path: Path to the registry directory (defaults to config)
        """
        self.config = config or AdaptiveMLConfig()
        self.registry_path = Path(
            registry_path or self.config.registry.registry_path
        )
        self.max_versions = self.config.registry.max_versions
        self.auto_archive = self.config.registry.auto_archive
        self.export_onnx = self.config.registry.export_onnx
        self.onnx_opset = self.config.registry.onnx_opset

        # Initialize registry
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Version tracking
        self.versions: Dict[str, ModelVersion] = {}
        self.current_version: Optional[str] = None
        self.version_order: List[str] = []

        # Load existing versions
        self._load_versions()

    def _load_versions(self) -> None:
        """Load existing versions from the registry."""
        # Check for versions.json
        versions_file = self.registry_path / "versions.json"
        if versions_file.exists():
            with open(versions_file, "r") as f:
                versions_data = json.load(f)

            for version, data in versions_data.items():
                model_version = ModelVersion(
                    version=version,
                    model_path=data["model_path"],
                    config_path=data["config_path"],
                    metadata=data.get("metadata", {}),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    size_bytes=data.get("size_bytes", 0),
                    parameters=data.get("parameters", 0),
                    status=data.get("status", "active"),
                )
                self.versions[version] = model_version
                self.version_order.append(version)

            # Set current version
            if "current_version" in versions_data:
                self.current_version = versions_data["current_version"]

    def _save_versions(self) -> None:
        """Save versions to versions.json."""
        versions_data = {}
        for version, model_version in self.versions.items():
            versions_data[version] = {
                "model_path": model_version.model_path,
                "config_path": model_version.config_path,
                "metadata": model_version.metadata,
                "created_at": model_version.created_at.isoformat(),
                "size_bytes": model_version.size_bytes,
                "parameters": model_version.parameters,
                "status": model_version.status,
            }

        versions_data["current_version"] = self.current_version

        with open(self.registry_path / "versions.json", "w") as f:
            json.dump(versions_data, f, indent=2)

    def save_version(
        self,
        version: str,
        model: nn.Module,
        metadata: Optional[Dict[str, Any]] = None,
        export_onnx: Optional[bool] = None,
    ) -> ModelVersion:
        """
        Save a model as a new version.

        Args:
            version: Version identifier (e.g., "v1.0.0")
            model: The model to save
            metadata: Optional metadata to store with the version
            export_onnx: Whether to export to ONNX (defaults to config)

        Returns:
            ModelVersion with information about the saved version
        """
        # Create version directory
        version_path = self.registry_path / version
        version_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = version_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save config
        config_path = version_path / "config.json"
        model_config = model.config if hasattr(model, "config") else {}
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        # Export to ONNX if enabled
        if export_onnx is None:
            export_onnx = self.export_onnx

        onnx_path = None
        if export_onnx:
            try:
                onnx_path = version_path / "model.onnx"
                dummy_input = self._create_dummy_input(model)
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    opset_version=self.onnx_opset,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                )
            except Exception as e:
                print(f"Warning: Failed to export to ONNX: {e}")
                onnx_path = None

        # Get model size
        size_bytes = os.path.getsize(model_path)
        if onnx_path and onnx_path.exists():
            size_bytes += os.path.getsize(onnx_path)

        # Count parameters
        parameters = sum(p.numel() for p in model.parameters())

        # Create ModelVersion
        model_version = ModelVersion(
            version=version,
            model_path=str(model_path),
            config_path=str(config_path),
            metadata=metadata or {},
            created_at=datetime.now(),
            size_bytes=size_bytes,
            parameters=parameters,
            status="active",
        )

        # Add to versions
        self.versions[version] = model_version
        if version not in self.version_order:
            self.version_order.append(version)

        # Set as current version if this is the first version
        if self.current_version is None:
            self.current_version = version

        # Save versions
        self._save_versions()

        # Auto-archive if needed
        if self.auto_archive and len(self.versions) > self.max_versions:
            self._archive_old_versions()

        return model_version

    def load_version(
        self,
        version: str,
        model_class: Optional[Callable] = None,
        device: Optional[str] = None,
    ) -> nn.Module:
        """
        Load a model version.

        Args:
            version: Version identifier
            model_class: Optional model class to instantiate
            device: Device to load the model on

        Returns:
            The loaded model
        """
        if version not in self.versions:
            raise ValueError(f"Version {version} not found in registry")

        model_version = self.versions[version]

        # Load model
        if model_class is not None:
            # Create model instance
            model = model_class.from_pretrained(
                str(self.registry_path / version),
                state_dict=torch.load(model_version.model_path),
            )
        else:
            # Try to load state dict directly
            state_dict = torch.load(model_version.model_path)

            # Try to reconstruct model from config
            config_path = Path(model_version.config_path)
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)

                # This is a simplified approach; in practice, you'd need to
                # know the model class to properly reconstruct it
                model = nn.Module()
                model.load_state_dict(state_dict)
            else:
                raise ValueError(f"Cannot load model without model_class or config")

        # Move to device
        if device is not None:
            model.to(device)

        return model

    def promote(
        self,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Promote a version to be the current version.

        Args:
            version: Version identifier to promote
            metadata: Optional metadata to update

        Returns:
            True if promotion was successful
        """
        if version not in self.versions:
            return False

        # Update current version
        if self.current_version is not None:
            self.versions[self.current_version].status = "archived"

        self.current_version = version
        self.versions[version].status = "active"

        # Update metadata
        if metadata:
            self.versions[version].metadata.update(metadata)

        # Save versions
        self._save_versions()

        return True

    def rollback(self, version: Optional[str] = None) -> bool:
        """
        Rollback to a previous version.

        Args:
            version: Version to rollback to (defaults to previous version)

        Returns:
            True if rollback was successful
        """
        if version is None:
            # Find the most recent version before current
            if self.current_version in self.version_order:
                current_idx = self.version_order.index(self.current_version)
                if current_idx > 0:
                    version = self.version_order[current_idx - 1]
                else:
                    return False
            else:
                return False

        if version not in self.versions:
            return False

        # Promote the target version
        return self.promote(version)

    def delete_version(self, version: str) -> bool:
        """
        Delete a version from the registry.

        Args:
            version: Version identifier to delete

        Returns:
            True if deletion was successful
        """
        if version not in self.versions:
            return False

        # Remove from versions
        del self.versions[version]
        self.version_order.remove(version)

        # Delete files
        version_path = self.registry_path / version
        if version_path.exists():
            shutil.rmtree(version_path)

        # Update current version if needed
        if self.current_version == version:
            self.current_version = None
            if self.version_order:
                self.current_version = self.version_order[-1]

        # Save versions
        self._save_versions()

        return True

    def archive_version(self, version: str) -> bool:
        """
        Archive a version (mark as archived but keep files).

        Args:
            version: Version identifier to archive

        Returns:
            True if archiving was successful
        """
        if version not in self.versions:
            return False

        self.versions[version].status = "archived"
        self._save_versions()
        return True

    def _archive_old_versions(self) -> None:
        """Archive old versions if we exceed max_versions."""
        while len(self.versions) > self.max_versions:
            # Find the oldest version that's not current
            for version in self.version_order:
                if version != self.current_version:
                    self.archive_version(version)
                    break

    def _create_dummy_input(self, model: nn.Module) -> torch.Tensor:
        """Create a dummy input for ONNX export."""
        # Try to get input shape from model config
        if hasattr(model, "config") and hasattr(model.config, "input_shape"):
            shape = model.config.input_shape
        else:
            # Default to a reasonable shape
            shape = (1, 512)  # Batch size 1, sequence length 512

        return torch.randn(shape)

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get information about a version."""
        return self.versions.get(version)

    def get_versions(self) -> Dict[str, ModelVersion]:
        """Get all versions."""
        return self.versions

    def get_current_version(self) -> Optional[str]:
        """Get the current version."""
        return self.current_version

    def list_versions(self) -> List[str]:
        """List all version identifiers."""
        return list(self.version_order)

    def get_stats(self) -> RegistryStats:
        """Get registry statistics."""
        total_size = sum(v.size_bytes for v in self.versions.values())

        return RegistryStats(
            num_versions=len(self.versions),
            current_version=self.current_version,
            total_size_bytes=total_size,
            max_versions=self.max_versions,
            auto_archive=self.auto_archive,
        )

    def clear(self) -> None:
        """Clear all versions from the registry."""
        for version in list(self.versions.keys()):
            self.delete_version(version)

        self.versions = {}
        self.version_order = []
        self.current_version = None

        # Remove versions.json
        versions_file = self.registry_path / "versions.json"
        if versions_file.exists():
            versions_file.unlink()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ModelRegistry(path={self.registry_path}, "
            f"versions={stats.num_versions}, current={self.current_version})"
        )
