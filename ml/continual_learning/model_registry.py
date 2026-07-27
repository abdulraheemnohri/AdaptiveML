"""
Model Registry for Adaptive Qwen Omni.
Manages model versions, storage, and retrieval.
"""

import os
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from adaptive_ml.qwen_omni.core import (
    AdapterType,
    DomainType,
    ModalityType,
    ModelVersion,
)


@dataclass
class ModelCard:
    """Metadata card for a model version."""
    version: str
    base_model: str = "Qwen/Qwen2.5-Omni-3B"
    adapters: List[AdapterType] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    performance: Dict[str, float] = field(default_factory=dict)
    forgetting_scores: Dict[str, float] = field(default_factory=dict)
    retention_score: float = 1.0
    is_production: bool = False
    size_gb: float = 0.0
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "base_model": self.base_model,
            "adapters": [a.value for a in self.adapters],
            "created_at": self.created_at.isoformat(),
            "performance": self.performance,
            "forgetting_scores": self.forgetting_scores,
            "retention_score": self.retention_score,
            "is_production": self.is_production,
            "size_gb": self.size_gb,
            "description": self.description,
            "tags": self.tags,
        }


class QwenOmniModelRegistry:
    """
    Registry for Qwen2.5-Omni-3B models.
    Manages versioning, storage, and retrieval of models and adapters.
    """

    def __init__(
        self,
        registry_path: str = "./registry",
        model_storage_path: str = "./models",
        max_versions: int = 10,
    ):
        self.registry_path = registry_path
        self.model_storage_path = model_storage_path
        self.max_versions = max_versions

        # Create directories
        os.makedirs(registry_path, exist_ok=True)
        os.makedirs(model_storage_path, exist_ok=True)

        # Model versions
        self._versions: Dict[str, ModelCard] = {}
        self._model_paths: Dict[str, str] = {}  # version -> path

        # Current production version
        self._production_version: Optional[str] = None

        # Load existing registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load existing registry from disk."""
        # In a full implementation, this would load from a JSON file
        pass

    def _save_registry(self) -> None:
        """Save registry to disk."""
        # In a full implementation, this would save to a JSON file
        pass

    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_hash = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        return f"{timestamp}_{random_hash}"

    def register_model(
        self,
        model: Any,
        adapters: List[AdapterType] = None,
        performance: Dict[str, float] = None,
        forgetting_scores: Dict[str, float] = None,
        retention_score: float = 1.0,
        is_production: bool = False,
        description: str = "",
        tags: List[str] = None,
    ) -> str:
        """
        Register a new model version.

        Args:
            model: The model to register
            adapters: List of adapters used
            performance: Performance metrics
            forgetting_scores: Forgetting scores by modality
            retention_score: Overall retention score
            is_production: Whether this is a production version
            description: Description of the model
            tags: Tags for categorization

        Returns:
            Version ID of the registered model
        """
        version_id = self._generate_version_id()

        # Save model
        model_path = os.path.join(self.model_storage_path, version_id)
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)

        # Create model card
        model_card = ModelCard(
            version=version_id,
            base_model=getattr(model, 'base_model', 'Qwen/Qwen2.5-Omni-3B'),
            adapters=adapters or [],
            performance=performance or {},
            forgetting_scores=forgetting_scores or {},
            retention_score=retention_score,
            is_production=is_production,
            description=description,
            tags=tags or [],
        )

        # Store version
        self._versions[version_id] = model_card
        self._model_paths[version_id] = model_path

        # Update production version
        if is_production:
            self._production_version = version_id

        # Enforce max versions
        if len(self._versions) > self.max_versions:
            # Remove oldest non-production version
            oldest_version = min(
                v for v in self._versions
                if v != self._production_version
            )
            self._remove_version(oldest_version)

        # Save registry
        self._save_registry()

        return version_id

    def _remove_version(self, version_id: str) -> None:
        """Remove a version from the registry."""
        if version_id in self._versions:
            del self._versions[version_id]
        if version_id in self._model_paths:
            model_path = self._model_paths[version_id]
            if os.path.exists(model_path):
                import shutil
                shutil.rmtree(model_path)
            del self._model_paths[version_id]

    def get_model(self, version_id: str) -> Optional[Any]:
        """
        Get a model by version ID.

        Args:
            version_id: The version ID

        Returns:
            The model, or None if not found
        """
        if version_id not in self._model_paths:
            return None

        model_path = self._model_paths[version_id]

        # Load model
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path)

        return model

    def get_model_card(self, version_id: str) -> Optional[ModelCard]:
        """Get model card by version ID."""
        return self._versions.get(version_id)

    def get_production_model(self) -> Optional[Any]:
        """Get the current production model."""
        if self._production_version is None:
            return None
        return self.get_model(self._production_version)

    def get_production_version(self) -> Optional[str]:
        """Get the current production version ID."""
        return self._production_version

    def promote_to_production(self, version_id: str) -> bool:
        """
        Promote a version to production.

        Args:
            version_id: The version to promote

        Returns:
            True if successful, False otherwise
        """
        if version_id not in self._versions:
            return False

        # Demote current production version
        if self._production_version:
            old_card = self._versions[self._production_version]
            old_card.is_production = False

        # Promote new version
        self._production_version = version_id
        self._versions[version_id].is_production = True

        self._save_registry()
        return True

    def rollback(self, version_id: str) -> bool:
        """
        Rollback to a previous version.

        Args:
            version_id: The version to rollback to

        Returns:
            True if successful, False otherwise
        """
        return self.promote_to_production(version_id)

    def list_versions(self) -> List[str]:
        """List all available versions."""
        return list(self._versions.keys())

    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a version."""
        card = self._versions.get(version_id)
        if card is None:
            return None
        return card.to_dict()

    def delete_version(self, version_id: str) -> bool:
        """
        Delete a version.

        Args:
            version_id: The version to delete

        Returns:
            True if successful, False otherwise
        """
        if version_id == self._production_version:
            return False  # Cannot delete production version

        self._remove_version(version_id)
        self._save_registry()
        return True

    def export_to_onnx(self, version_id: str, output_path: str) -> bool:
        """
        Export a model to ONNX format.

        Args:
            version_id: The version to export
            output_path: Path to save the ONNX model

        Returns:
            True if successful, False otherwise
        """
        model = self.get_model(version_id)
        if model is None:
            return False

        # In a full implementation, this would export to ONNX
        # For now, just create a placeholder file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f"ONNX export of {version_id} (placeholder)")

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_versions": len(self._versions),
            "production_version": self._production_version,
            "max_versions": self.max_versions,
            "registry_path": self.registry_path,
            "model_storage_path": self.model_storage_path,
        }
