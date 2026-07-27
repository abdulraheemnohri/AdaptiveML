"""
Adapter Loader for Adaptive Qwen Omni inference.
Loads and manages adapters for the frozen base model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import torch

from transformers import PreTrainedModel, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from adaptive_ml.qwen_omni.core import (
    AdapterType,
    AdapterInfo,
    QwenOmniModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedAdapter:
    """Represents a loaded adapter."""
    name: str
    adapter_type: AdapterType
    peft_model: PeftModel
    config: PeftConfig
    info: AdapterInfo
    is_loaded: bool = True
    device: str = "cpu"

    def to(self, device: str) -> "LoadedAdapter":
        """Move adapter to device."""
        self.peft_model.to(device)
        self.device = device
        return self

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get adapter state dict."""
        return self.peft_model.state_dict()


@dataclass
class AdapterCache:
    """Cache for loaded adapters."""
    adapters: Dict[str, LoadedAdapter] = field(default_factory=dict)
    max_size: int = 10
    lru_order: List[str] = field(default_factory=list)

    def get(self, name: str) -> Optional[LoadedAdapter]:
        """Get adapter from cache."""
        if name in self.adapters:
            # Update LRU order
            if name in self.lru_order:
                self.lru_order.remove(name)
            self.lru_order.append(name)
            return self.adapters[name]
        return None

    def put(self, name: str, adapter: LoadedAdapter) -> None:
        """Add adapter to cache."""
        if name in self.adapters:
            # Update existing
            self.adapters[name] = adapter
            if name in self.lru_order:
                self.lru_order.remove(name)
            self.lru_order.append(name)
        else:
            # Add new
            if len(self.adapters) >= self.max_size:
                # Evict LRU
                lru_name = self.lru_order.pop(0)
                del self.adapters[lru_name]
                logger.info(f"Evicted adapter from cache: {lru_name}")

            self.adapters[name] = adapter
            self.lru_order.append(name)

    def remove(self, name: str) -> bool:
        """Remove adapter from cache."""
        if name in self.adapters:
            del self.adapters[name]
            if name in self.lru_order:
                self.lru_order.remove(name)
            return True
        return False

    def clear(self) -> None:
        """Clear all adapters from cache."""
        self.adapters.clear()
        self.lru_order.clear()

    def list_adapters(self) -> List[str]:
        """List all cached adapter names."""
        return list(self.adapters.keys())


class AdapterLoader:
    """
    Loads and manages LoRA/QLoRA adapters for Qwen2.5-Omni-3B.

    Features:
    - Lazy loading of adapters
    - LRU caching
    - Device management
    - Adapter merging (for Level 2 adaptation)
    """

    def __init__(
        self,
        base_model: Union[PreTrainedModel, str],
        model_config: Optional[QwenOmniModelConfig] = None,
        adapter_cache_size: int = 10,
        device: str = "cuda",
        use_cache: bool = True,
    ):
        """
        Initialize the adapter loader.

        Args:
            base_model: Base model or model name
            model_config: Model configuration
            adapter_cache_size: Maximum number of adapters to cache
            device: Default device for adapters
            use_cache: Whether to use adapter caching
        """
        self.model_config = model_config or QwenOmniModelConfig()
        self.device = device
        self.use_cache = use_cache

        # Initialize base model
        if isinstance(base_model, str):
            self.base_model = self._load_base_model(base_model)
        else:
            self.base_model = base_model

        # Adapter cache
        self.cache = AdapterCache(max_size=adapter_cache_size)

        # Track loaded adapters
        self.loaded_adapters: Dict[str, LoadedAdapter] = {}

        # Adapter storage paths
        self.adapter_paths: Dict[str, Path] = {}

    def _load_base_model(self, model_name: str) -> PreTrainedModel:
        """Load the base Qwen2.5-Omni-3B model."""
        logger.info(f"Loading base model: {model_name}")

        config = {
            "trust_remote_code": True,
            "device_map": self.device if self.device != "cpu" else None,
        }

        if self.model_config.use_flash_attention:
            config["use_flash_attention_2"] = True

        if self.model_config.use_bfloat16:
            config["torch_dtype"] = torch.bfloat16

        if self.model_config.quantize:
            config["quantization_config"] = {
                "bits": self.model_config.quantization_bits,
                "method": self.model_config.quantization_method,
            }

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **config
            )
            logger.info(f"Base model loaded successfully: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise

    def register_adapter_path(self, adapter_name: str, path: Union[str, Path]) -> None:
        """Register the path for an adapter."""
        self.adapter_paths[adapter_name] = Path(path)

    def load_adapter(
        self,
        adapter_name: str,
        adapter_type: Optional[AdapterType] = None,
        path: Optional[Union[str, Path]] = None,
        force_reload: bool = False,
    ) -> Optional[LoadedAdapter]:
        """
        Load an adapter by name.

        Args:
            adapter_name: Name of the adapter
            adapter_type: Type of adapter (optional)
            path: Path to adapter (optional, uses registered path if not provided)
            force_reload: Force reload even if cached

        Returns:
            LoadedAdapter or None if failed
        """
        # Check cache first
        if not force_reload and self.use_cache:
            cached = self.cache.get(adapter_name)
            if cached:
                logger.debug(f"Adapter loaded from cache: {adapter_name}")
                return cached

        # Get path
        if path is None:
            path = self.adapter_paths.get(adapter_name)
            if path is None:
                logger.warning(f"No path registered for adapter: {adapter_name}")
                return None

        path = Path(path)
        if not path.exists():
            logger.warning(f"Adapter path does not exist: {path}")
            return None

        try:
            logger.info(f"Loading adapter: {adapter_name} from {path}")

            # Load PEFT config
            config = PeftConfig.from_pretrained(path)

            # Load adapter
            peft_model = PeftModel.from_pretrained(
                self.base_model,
                str(path),
                is_trainable=False,
            )

            # Create adapter info
            info = AdapterInfo(
                name=adapter_name,
                adapter_type=adapter_type or AdapterType.CUSTOM,
                rank=config.r if hasattr(config, 'r') else 8,
                alpha=config.lora_alpha if hasattr(config, 'lora_alpha') else 16.0,
                target_modules=list(config.target_modules) if hasattr(config, 'target_modules') else [],
            )

            # Create loaded adapter
            loaded = LoadedAdapter(
                name=adapter_name,
                adapter_type=adapter_type or AdapterType.CUSTOM,
                peft_model=peft_model,
                config=config,
                info=info,
                device=self.device,
            )

            # Cache it
            if self.use_cache:
                self.cache.put(adapter_name, loaded)

            self.loaded_adapters[adapter_name] = loaded

            logger.info(f"Adapter loaded successfully: {adapter_name}")
            return loaded

        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_name}: {e}")
            return None

    def load_multiple_adapters(
        self,
        adapter_names: List[str],
    ) -> Dict[str, Optional[LoadedAdapter]]:
        """Load multiple adapters."""
        results = {}
        for name in adapter_names:
            results[name] = self.load_adapter(name)
        return results

    def unload_adapter(self, adapter_name: str) -> bool:
        """Unload an adapter."""
        if adapter_name in self.loaded_adapters:
            del self.loaded_adapters[adapter_name]
            self.cache.remove(adapter_name)
            logger.info(f"Unloaded adapter: {adapter_name}")
            return True
        return False

    def get_loaded_adapters(self) -> List[str]:
        """Get list of loaded adapter names."""
        return list(self.loaded_adapters.keys())

    def merge_adapters(
        self,
        adapter_names: List[str],
        merged_name: str,
        save_path: Union[str, Path],
    ) -> Optional[PeftModel]:
        """
        Merge multiple adapters into a single adapter (Level 2 adaptation).

        Args:
            adapter_names: Names of adapters to merge
            merged_name: Name for the merged adapter
            save_path: Path to save the merged adapter

        Returns:
            Merged PeftModel or None if failed
        """
        if len(adapter_names) < 2:
            logger.warning("Need at least 2 adapters to merge")
            return None

        try:
            # Load all adapters
            adapters = []
            for name in adapter_names:
                loaded = self.load_adapter(name)
                if loaded is None:
                    logger.warning(f"Failed to load adapter for merging: {name}")
                    return None
                adapters.append(loaded)

            # Merge state dicts
            merged_state = {}
            for adapter in adapters:
                state = adapter.get_state_dict()
                for key, value in state.items():
                    if key in merged_state:
                        # Average weights
                        merged_state[key] = (merged_state[key] + value) / 2.0
                    else:
                        merged_state[key] = value

            # Create new PeftModel with merged weights
            # Use config from first adapter
            first_adapter = adapters[0]
            merged_model = PeftModel.from_pretrained(
                self.base_model,
                str(first_adapter.config.id),
            )
            merged_model.load_state_dict(merged_state)

            # Save merged adapter
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(save_path))
            first_adapter.config.save_pretrained(str(save_path))

            # Register and cache
            self.register_adapter_path(merged_name, save_path)
            merged_loaded = LoadedAdapter(
                name=merged_name,
                adapter_type=AdapterType.CUSTOM,
                peft_model=merged_model,
                config=first_adapter.config,
                info=AdapterInfo(
                    name=merged_name,
                    adapter_type=AdapterType.CUSTOM,
                    rank=first_adapter.info.rank,
                    alpha=first_adapter.info.alpha,
                    target_modules=first_adapter.info.target_modules,
                ),
            )

            if self.use_cache:
                self.cache.put(merged_name, merged_loaded)

            self.loaded_adapters[merged_name] = merged_loaded

            logger.info(f"Merged {len(adapter_names)} adapters into: {merged_name}")
            return merged_model

        except Exception as e:
            logger.error(f"Failed to merge adapters: {e}")
            return None

    def get_adapter_info(self, adapter_name: str) -> Optional[AdapterInfo]:
        """Get information about an adapter."""
        loaded = self.loaded_adapters.get(adapter_name)
        if loaded:
            return loaded.info

        # Try to load from cache
        cached = self.cache.get(adapter_name)
        if cached:
            return cached.info

        return None

    def clear_cache(self) -> None:
        """Clear the adapter cache."""
        self.cache.clear()
        logger.info("Adapter cache cleared")

    def move_to_device(self, adapter_name: str, device: str) -> bool:
        """Move an adapter to a different device."""
        loaded = self.loaded_adapters.get(adapter_name)
        if loaded:
            loaded.to(device)
            logger.info(f"Moved adapter to device: {adapter_name} -> {device}")
            return True

        cached = self.cache.get(adapter_name)
        if cached:
            cached.to(device)
            return True

        return False
