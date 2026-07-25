"""
Adapter Manager for Adaptive ML Framework.
Manages LoRA, QLoRA, and other parameter-efficient adapters for continual learning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.tuners.lora import LoraLayer
from transformers import PreTrainedModel

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import AdapterType


@dataclass
class AdapterInfo:
    """Information about a loaded adapter."""

    adapter_id: str
    task_id: str
    adapter_type: AdapterType
    r: int
    lora_alpha: int
    target_modules: List[str]
    created_at: str
    num_parameters: int
    is_active: bool = False


class AdapterManager:
    """
    Manages multiple adapters for a base model.
    
    Features:
    - Create and manage LoRA/QLoRA adapters
    - Switch between adapters without reloading base model
    - Save and load adapters from disk
    - Merge adapters into base model
    - Track adapter metadata
    
    Usage:
        manager = AdapterManager(base_model, config)
        
        # Create adapter for a new task
        manager.create_adapter("task_a")
        
        # Activate adapter
        manager.activate_adapter("task_a")
        
        # Deactivate all adapters
        manager.deactivate_all()
        
        # Save adapter
        manager.save_adapter("task_a", "./adapters/task_a")
        
        # Load adapter
        manager.load_adapter("task_a", "./adapters/task_a")
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        config: Optional[AdaptiveMLConfig] = None,
    ):
        """
        Initialize AdapterManager.
        
        Args:
            base_model: The base pre-trained model
            config: AdaptiveMLConfig instance
        """
        self.base_model = base_model
        self.config = config or AdaptiveMLConfig()
        
        # Adapters storage
        self.adapters: Dict[str, nn.Module] = {}
        self.adapter_configs: Dict[str, LoraConfig] = {}
        self.adapter_info: Dict[str, AdapterInfo] = {}
        self.active_adapter: Optional[str] = None
        
        # Track which adapter is currently applied
        self.current_model: Optional[nn.Module] = None
        
        # Device
        self.device = self.config.training.device

    def create_adapter(
        self,
        adapter_id: str,
        task_id: Optional[str] = None,
        adapter_type: Optional[AdapterType] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Create a new adapter for the base model.
        
        Args:
            adapter_id: Unique identifier for the adapter
            task_id: Optional task identifier
            adapter_type: Type of adapter (defaults to config)
            **kwargs: Additional arguments for the adapter config
        
        Returns:
            The created adapter module
        """
        adapter_type = adapter_type or self.config.adapters.adapter_type
        task_id = task_id or adapter_id
        
        # Create LoRA config
        lora_config = self._create_lora_config(adapter_type, **kwargs)
        
        # Store config
        self.adapter_configs[adapter_id] = lora_config
        
        # Create adapter
        adapter = self._create_peft_adapter(adapter_id, lora_config)
        self.adapters[adapter_id] = adapter
        
        # Store info
        num_params = self._count_adapter_parameters(adapter)
        self.adapter_info[adapter_id] = AdapterInfo(
            adapter_id=adapter_id,
            task_id=task_id,
            adapter_type=adapter_type,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            created_at=torch.__version__,  # Simple timestamp
            num_parameters=num_params,
            is_active=False,
        )
        
        return adapter

    def _create_lora_config(
        self,
        adapter_type: AdapterType,
        **kwargs,
    ) -> LoraConfig:
        """Create a LoRA configuration."""
        config = self.config.adapters
        
        # Common parameters
        params = {
            "r": kwargs.get("r", config.r),
            "lora_alpha": kwargs.get("lora_alpha", config.lora_alpha),
            "lora_dropout": kwargs.get("lora_dropout", config.lora_dropout),
            "target_modules": kwargs.get("target_modules", config.target_modules),
            "task_type": TaskType(config.task_type),
            "bias": "none",
            "inference_mode": False,
        }
        
        # QLoRA specific
        if adapter_type == AdapterType.QLORA:
            params["use_rslora"] = False
            params["use_dora"] = False
        
        return LoraConfig(**params)

    def _create_peft_adapter(
        self,
        adapter_id: str,
        lora_config: LoraConfig,
    ) -> nn.Module:
        """Create a PEFT adapter using the base model."""
        # Create a copy of the base model with the adapter
        model = get_peft_model(self.base_model, lora_config)
        
        # Extract just the adapter layers
        # Note: This is a simplified approach; in practice, we'd need to
        # properly separate the adapter from the base model
        adapter = nn.ModuleDict({
            f"{adapter_id}_adapter": nn.ModuleList([
                layer for name, layer in model.named_modules()
                if isinstance(layer, LoraLayer)
            ])
        })
        
        return adapter

    def activate_adapter(self, adapter_id: str) -> nn.Module:
        """
        Activate an adapter by applying it to the base model.
        
        Args:
            adapter_id: ID of the adapter to activate
        
        Returns:
            The model with the adapter applied
        """
        if adapter_id not in self.adapters:
            raise ValueError(f"Adapter {adapter_id} not found")
        
        # Deactivate current adapter first
        if self.active_adapter:
            self.deactivate_adapter()
        
        # Apply the adapter
        lora_config = self.adapter_configs[adapter_id]
        self.current_model = get_peft_model(self.base_model, lora_config)
        
        # Load the adapter weights
        adapter_state = get_peft_model_state_dict(self.adapters[adapter_id])
        set_peft_model_state_dict(self.current_model, adapter_state)
        
        # Update active adapter
        self.active_adapter = adapter_id
        self.adapter_info[adapter_id].is_active = True
        
        return self.current_model

    def deactivate_adapter(self) -> None:
        """Deactivate the current adapter."""
        if self.active_adapter:
            self.adapter_info[self.active_adapter].is_active = False
            self.active_adapter = None
        
        self.current_model = None

    def deactivate_all(self) -> None:
        """Deactivate all adapters."""
        for adapter_id in self.adapter_info:
            self.adapter_info[adapter_id].is_active = False
        
        self.active_adapter = None
        self.current_model = None

    def get_model(self) -> nn.Module:
        """
        Get the current model (base model + active adapter if any).
        
        Returns:
            The model with active adapter or base model
        """
        if self.current_model is not None:
            return self.current_model
        return self.base_model

    def get_adapter(self, adapter_id: str) -> nn.Module:
        """Get a specific adapter by ID."""
        if adapter_id not in self.adapters:
            raise ValueError(f"Adapter {adapter_id} not found")
        return self.adapters[adapter_id]

    def save_adapter(
        self,
        adapter_id: str,
        path: Union[str, Path],
        save_config: bool = True,
    ) -> None:
        """
        Save an adapter to disk.
        
        Args:
            adapter_id: ID of the adapter to save
            path: Directory to save the adapter
            save_config: Whether to save the adapter config
        """
        if adapter_id not in self.adapters:
            raise ValueError(f"Adapter {adapter_id} not found")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save adapter weights
        adapter = self.adapters[adapter_id]
        torch.save(adapter.state_dict(), path / "adapter.pt")
        
        # Save config
        if save_config:
            lora_config = self.adapter_configs[adapter_id]
            torch.save(lora_config, path / "config.pt")
        
        # Save info
        info = self.adapter_info[adapter_id]
        with open(path / "info.json", "w") as f:
            import json
            json.dump({
                "adapter_id": info.adapter_id,
                "task_id": info.task_id,
                "adapter_type": info.adapter_type.value,
                "r": info.r,
                "lora_alpha": info.lora_alpha,
                "target_modules": info.target_modules,
                "created_at": info.created_at,
                "num_parameters": info.num_parameters,
            }, f, indent=2)

    def load_adapter(
        self,
        adapter_id: str,
        path: Union[str, Path],
        task_id: Optional[str] = None,
    ) -> nn.Module:
        """
        Load an adapter from disk.
        
        Args:
            adapter_id: ID to assign to the loaded adapter
            path: Directory containing the adapter files
            task_id: Optional task identifier
        
        Returns:
            The loaded adapter module
        """
        path = Path(path)
        
        # Load config
        lora_config = torch.load(path / "config.pt")
        self.adapter_configs[adapter_id] = lora_config
        
        # Load adapter weights
        adapter_state = torch.load(path / "adapter.pt")
        
        # Create adapter structure
        adapter = self._create_peft_adapter(adapter_id, lora_config)
        adapter.load_state_dict(adapter_state)
        
        self.adapters[adapter_id] = adapter
        
        # Load info
        task_id = task_id or adapter_id
        num_params = self._count_adapter_parameters(adapter)
        self.adapter_info[adapter_id] = AdapterInfo(
            adapter_id=adapter_id,
            task_id=task_id,
            adapter_type=self.config.adapters.adapter_type,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            created_at="",
            num_parameters=num_params,
            is_active=False,
        )
        
        return adapter

    def delete_adapter(self, adapter_id: str) -> bool:
        """
        Delete an adapter.
        
        Args:
            adapter_id: ID of the adapter to delete
        
        Returns:
            True if adapter was deleted, False if not found
        """
        if adapter_id not in self.adapters:
            return False
        
        # Deactivate if active
        if self.active_adapter == adapter_id:
            self.deactivate_adapter()
        
        # Remove from storage
        del self.adapters[adapter_id]
        del self.adapter_configs[adapter_id]
        del self.adapter_info[adapter_id]
        
        return True

    def list_adapters(self) -> List[AdapterInfo]:
        """List all adapters."""
        return list(self.adapter_info.values())

    def get_adapter_info(self, adapter_id: str) -> AdapterInfo:
        """Get information about a specific adapter."""
        if adapter_id not in self.adapter_info:
            raise ValueError(f"Adapter {adapter_id} not found")
        return self.adapter_info[adapter_id]

    def merge_adapter_into_base(self, adapter_id: str) -> nn.Module:
        """
        Merge an adapter into the base model.
        
        Note: This is a destructive operation that modifies the base model.
        
        Args:
            adapter_id: ID of the adapter to merge
        
        Returns:
            The merged model
        """
        if adapter_id not in self.adapters:
            raise ValueError(f"Adapter {adapter_id} not found")
        
        # Get the adapter config
        lora_config = self.adapter_configs[adapter_id]
        
        # Create a new model with the adapter
        merged_model = get_peft_model(self.base_model, lora_config)
        
        # Load the adapter weights
        adapter_state = get_peft_model_state_dict(self.adapters[adapter_id])
        set_peft_model_state_dict(merged_model, adapter_state)
        
        # Merge the adapter into the base model
        merged_model = merged_model.merge_and_unload()
        
        # Update base model
        self.base_model = merged_model
        
        # Remove the adapter
        self.delete_adapter(adapter_id)
        
        return merged_model

    def _count_adapter_parameters(self, adapter: nn.Module) -> int:
        """Count the number of parameters in an adapter."""
        return sum(p.numel() for p in adapter.parameters())

    def get_total_parameters(self) -> Dict[str, int]:
        """
        Get total parameter counts.
        
        Returns:
            Dictionary with parameter counts for base model and adapters
        """
        base_params = sum(p.numel() for p in self.base_model.parameters())
        adapter_params = sum(
            self._count_adapter_parameters(adapter)
            for adapter in self.adapters.values()
        )
        
        return {
            "base": base_params,
            "adapters": adapter_params,
            "total": base_params + adapter_params,
        }

    def __repr__(self) -> str:
        num_adapters = len(self.adapters)
        active = self.active_adapter or "none"
        return f"AdapterManager(base={type(self.base_model).__name__}, adapters={num_adapters}, active={active})"


class AdapterRouter:
    """
    Routes input to the appropriate adapter based on task/domain.
    
    Features:
    - Domain-based routing (e.g., coding, medical, legal)
    - Task-based routing
    - Hybrid routing (domain + task)
    - Fallback to default adapter
    
    Usage:
        router = AdapterRouter(adapter_manager, config)
        
        # Route input to appropriate adapter
        adapter_id = router.route(input_text)
        
        # Activate the routed adapter
        model = adapter_manager.activate_adapter(adapter_id)
    """

    def __init__(
        self,
        adapter_manager: AdapterManager,
        config: Optional[AdaptiveMLConfig] = None,
    ):
        """
        Initialize AdapterRouter.
        
        Args:
            adapter_manager: The AdapterManager instance
            config: AdaptiveMLConfig instance
        """
        self.adapter_manager = adapter_manager
        self.config = config or AdaptiveMLConfig()
        
        # Routing configuration
        self.router_type = self.config.adapters.router_type
        
        # Domain/task mappings
        self.domain_keywords: Dict[str, List[str]] = {
            "coding": ["code", "python", "java", "javascript", "function", "def", "class"],
            "medical": ["patient", "doctor", "hospital", "disease", "symptom", "treatment"],
            "legal": ["law", "contract", "court", "judge", "legal", "attorney"],
            "general": [],  # Default
        }
        
        # Task mappings
        self.task_keywords: Dict[str, List[str]] = {}
        
        # Default adapter
        self.default_adapter = "general"

    def route(self, input_data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Route input to the appropriate adapter.
        
        Args:
            input_data: Input data (text, tensor, etc.)
            metadata: Optional metadata containing task/domain info
        
        Returns:
            Adapter ID to use
        """
        # Check metadata first
        if metadata:
            if "adapter_id" in metadata:
                return metadata["adapter_id"]
            if "task_id" in metadata:
                return metadata["task_id"]
            if "domain" in metadata:
                return self._get_adapter_for_domain(metadata["domain"])
        
        # Extract text for keyword matching
        text = self._extract_text(input_data)
        
        if self.router_type == "domain":
            return self._route_by_domain(text)
        elif self.router_type == "task":
            return self._route_by_task(text)
        elif self.router_type == "hybrid":
            domain = self._detect_domain(text)
            task = self._detect_task(text)
            return self._route_hybrid(domain, task)
        else:
            return self.default_adapter

    def _extract_text(self, input_data: Any) -> str:
        """Extract text from input data for keyword matching."""
        if isinstance(input_data, str):
            return input_data.lower()
        elif isinstance(input_data, dict):
            # Try to find text in common keys
            for key in ["text", "prompt", "input", "query"]:
                if key in input_data and isinstance(input_data[key], str):
                    return input_data[key].lower()
        elif isinstance(input_data, list) and len(input_data) > 0:
            if isinstance(input_data[0], str):
                return " ".join(input_data).lower()
        
        return ""

    def _route_by_domain(self, text: str) -> str:
        """Route based on domain keywords."""
        domain = self._detect_domain(text)
        return self._get_adapter_for_domain(domain)

    def _route_by_task(self, text: str) -> str:
        """Route based on task keywords."""
        task = self._detect_task(text)
        return self._get_adapter_for_task(task)

    def _route_hybrid(self, domain: str, task: str) -> str:
        """Route using both domain and task."""
        # Try domain-specific task adapter first
        adapter_id = f"{domain}_{task}"
        if adapter_id in self.adapter_manager.adapters:
            return adapter_id
        
        # Fall back to domain
        if domain != "general":
            domain_adapter = self._get_adapter_for_domain(domain)
            if domain_adapter in self.adapter_manager.adapters:
                return domain_adapter
        
        # Fall back to task
        task_adapter = self._get_adapter_for_task(task)
        if task_adapter in self.adapter_manager.adapters:
            return task_adapter
        
        # Fall back to default
        return self.default_adapter

    def _detect_domain(self, text: str) -> str:
        """Detect domain from text using keywords."""
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return domain
        return "general"

    def _detect_task(self, text: str) -> str:
        """Detect task from text using keywords."""
        for task, keywords in self.task_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return task
        return "default"

    def _get_adapter_for_domain(self, domain: str) -> str:
        """Get adapter ID for a domain."""
        # Check if domain adapter exists
        if domain in self.adapter_manager.adapters:
            return domain
        
        # Check if there's a general adapter
        if "general" in self.adapter_manager.adapters:
            return "general"
        
        return self.default_adapter

    def _get_adapter_for_task(self, task: str) -> str:
        """Get adapter ID for a task."""
        # Check if task adapter exists
        if task in self.adapter_manager.adapters:
            return task
        
        # Check if there's a general adapter
        if "general" in self.adapter_manager.adapters:
            return "general"
        
        return self.default_adapter

    def add_domain(self, domain: str, keywords: List[str]) -> None:
        """Add a new domain with keywords."""
        self.domain_keywords[domain] = keywords

    def add_task(self, task: str, keywords: List[str]) -> None:
        """Add a new task with keywords."""
        self.task_keywords[task] = keywords

    def set_default_adapter(self, adapter_id: str) -> None:
        """Set the default adapter."""
        self.default_adapter = adapter_id

    def get_router_type(self) -> str:
        """Get the current router type."""
        return self.router_type

    def set_router_type(self, router_type: str) -> None:
        """Set the router type."""
        self.router_type = router_type

    def __repr__(self) -> str:
        return f"AdapterRouter(type={self.router_type}, default={self.default_adapter})"
