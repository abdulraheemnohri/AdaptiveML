"""
Adapter Registry for Adaptive Qwen Omni.
Manages adapter versions and storage.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from adaptive_ml.qwen_omni.core import (
    AdapterInfo,
    AdapterType,
    DomainType,
    ModalityType,
)


@dataclass
class AdapterCard:
    """Metadata card for an adapter."""
    name: str
    adapter_type: AdapterType
    version: str
    domain: DomainType = DomainType.GENERAL
    modality: ModalityType = ModalityType.TEXT
    rank: int = 8
    alpha: float = 16.0
    created_at: datetime = field(default_factory=datetime.now)
    performance: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "adapter_type": self.adapter_type.value,
            "version": self.version,
            "domain": self.domain.value,
            "modality": self.modality.value,
            "rank": self.rank,
            "alpha": self.alpha,
            "created_at": self.created_at.isoformat(),
            "performance": self.performance,
            "is_active": self.is_active,
            "description": self.description,
            "tags": self.tags,
        }


class AdapterRegistry:
    """
    Registry for adapters.
    Manages adapter versions, storage, and retrieval.
    """

    def __init__(
        self,
        registry_path: str = "./registry",
        adapter_storage_path: str = "./adapters",
        max_adapters: int = 50,
    ):
        self.registry_path = registry_path
        self.adapter_storage_path = adapter_storage_path
        self.max_adapters = max_adapters

        # Create directories
        os.makedirs(registry_path, exist_ok=True)
        os.makedirs(adapter_storage_path, exist_ok=True)

        # Adapters
        self._adapters: Dict[str, AdapterCard] = {}
        self._adapter_paths: Dict[str, str] = {}

    def register_adapter(
        self,
        adapter: Any,
        adapter_type: AdapterType,
        name: str = None,
        domain: DomainType = DomainType.GENERAL,
        modality: ModalityType = ModalityType.TEXT,
        rank: int = 8,
        alpha: float = 16.0,
        performance: Dict[str, float] = None,
        description: str = "",
        tags: List[str] = None,
    ) -> str:
        """
        Register a new adapter.

        Args:
            adapter: The adapter to register
            adapter_type: Type of adapter
            name: Name of the adapter
            domain: Domain
            modality: Modality
            rank: LoRA rank
            alpha: LoRA alpha
            performance: Performance metrics
            description: Description
            tags: Tags

        Returns:
            Adapter version ID
        """
        version_id = f"{adapter_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save adapter
        adapter_path = os.path.join(self.adapter_storage_path, version_id)
        os.makedirs(adapter_path, exist_ok=True)
        adapter.save_pretrained(adapter_path)

        # Create adapter card
        adapter_card = AdapterCard(
            name=name or version_id,
            adapter_type=adapter_type,
            version=version_id,
            domain=domain,
            modality=modality,
            rank=rank,
            alpha=alpha,
            performance=performance or {},
            description=description,
            tags=tags or [],
        )

        # Store adapter
        self._adapters[version_id] = adapter_card
        self._adapter_paths[version_id] = adapter_path

        # Enforce max adapters
        if len(self._adapters) > self.max_adapters:
            # Remove oldest adapter
            oldest = min(self._adapters.keys())
            self._remove_adapter(oldest)

        return version_id

    def _remove_adapter(self, version_id: str) -> None:
        """Remove an adapter from the registry."""
        if version_id in self._adapters:
            del self._adapters[version_id]
        if version_id in self._adapter_paths:
            adapter_path = self._adapter_paths[version_id]
            if os.path.exists(adapter_path):
                import shutil
                shutil.rmtree(adapter_path)
            del self._adapter_paths[version_id]

    def get_adapter(self, version_id: str) -> Optional[Any]:
        """Get an adapter by version ID."""
        if version_id not in self._adapter_paths:
            return None

        adapter_path = self._adapter_paths[version_id]

        # Load adapter
        from peft import PeftModel
        from transformers import AutoModel

        # In practice, you would need the base model to load the adapter
        # This is a simplified version
        try:
            adapter = PeftModel.from_pretrained(AutoModel.from_pretrained("Qwen/Qwen2.5-Omni-3B"), adapter_path)
            return adapter
        except:
            return None

    def get_adapter_card(self, version_id: str) -> Optional[AdapterCard]:
        """Get adapter card by version ID."""
        return self._adapters.get(version_id)

    def list_adapters(self) -> List[str]:
        """List all available adapters."""
        return list(self._adapters.keys())

    def get_adapters_by_type(self, adapter_type: AdapterType) -> List[str]:
        """Get adapters of a specific type."""
        return [
            v for v, card in self._adapters.items()
            if card.adapter_type == adapter_type
        ]

    def get_adapters_by_domain(self, domain: DomainType) -> List[str]:
        """Get adapters for a specific domain."""
        return [
            v for v, card in self._adapters.items()
            if card.domain == domain
        ]

    def get_adapters_by_modality(self, modality: ModalityType) -> List[str]:
        """Get adapters for a specific modality."""
        return [
            v for v, card in self._adapters.items()
            if card.modality == modality
        ]

    def activate_adapter(self, version_id: str) -> bool:
        """Activate an adapter."""
        if version_id in self._adapters:
            self._adapters[version_id].is_active = True
            return True
        return False

    def deactivate_adapter(self, version_id: str) -> bool:
        """Deactivate an adapter."""
        if version_id in self._adapters:
            self._adapters[version_id].is_active = False
            return True
        return False

    def delete_adapter(self, version_id: str) -> bool:
        """Delete an adapter."""
        if version_id not in self._adapters:
            return False

        self._remove_adapter(version_id)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        active_count = sum(1 for c in self._adapters.values() if c.is_active)

        return {
            "total_adapters": len(self._adapters),
            "active_adapters": active_count,
            "max_adapters": self.max_adapters,
            "registry_path": self.registry_path,
            "adapter_storage_path": self.adapter_storage_path,
        }
