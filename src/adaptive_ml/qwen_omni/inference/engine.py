"""
Inference Engine for Adaptive Qwen Omni.
Executes multimodal inference with adapter routing and result aggregation.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import numpy as np

from transformers import GenerationConfig, PreTrainedTokenizerFast, PreTrainedModel
from peft import PeftModel

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    TaskType,
    DomainType,
    AdapterType,
    MultimodalData,
    InferenceResult,
    QwenOmniModelConfig,
)
from adaptive_ml.qwen_omni.inference.router import MultimodalRouter, RoutingDecision
from adaptive_ml.qwen_omni.inference.loader import AdapterLoader, LoadedAdapter

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "early_stopping": self.early_stopping,
        }


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    use_router: bool = True
    use_fallback: bool = True
    fallback_threshold: float = 0.5
    max_retries: int = 3
    batch_size: int = 1
    
    # Modality-specific settings
    vision_config: Dict[str, Any] = field(default_factory=dict)
    audio_config: Dict[str, Any] = field(default_factory=dict)
    video_config: Dict[str, Any] = field(default_factory=dict)
    speech_config: Dict[str, Any] = field(default_factory=dict)


class QwenOmniInferenceEngine:
    """
    Main inference engine for Adaptive Qwen Omni.
    
    Features:
    - Multimodal inference (text, vision, audio, video, speech)
    - Adapter routing and selection
    - Fallback mechanisms
    - Result aggregation
    - Performance tracking
    """
    
    def __init__(
        self,
        base_model: Union[str, PreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        model_config: Optional[QwenOmniModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        router: Optional[MultimodalRouter] = None,
        adapter_loader: Optional[AdapterLoader] = None,
        device: str = "cuda",
    ):
        """
        Initialize the inference engine.
        
        Args:
            base_model: Base model name or instance
            tokenizer: Tokenizer for the model
            model_config: Model configuration
            inference_config: Inference configuration
            router: Multimodal router (created if None)
            adapter_loader: Adapter loader (created if None)
            device: Device to use for inference
        """
        self.model_config = model_config or QwenOmniModelConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.device = device
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = self._load_tokenizer()
        else:
            self.tokenizer = tokenizer
        
        # Initialize adapter loader
        if adapter_loader is None:
            self.adapter_loader = AdapterLoader(
                base_model=base_model,
                model_config=self.model_config,
                device=self.device,
            )
        else:
            self.adapter_loader = adapter_loader
        
        # Get base model from adapter loader
        self.base_model = self.adapter_loader.base_model
        
        # Initialize router
        if router is None:
            self.router = MultimodalRouter(
                available_adapters=self.model_config.adapters.default_adapters,
            )
        else:
            self.router = router
        
        # Track inference statistics
        self.stats: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_time_ms": 0.0,
            "requests_by_modality": {},
            "requests_by_adapter": {},
        }
        
        logger.info("Qwen Omni Inference Engine initialized")
    
    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        """Load the tokenizer for Qwen2.5-Omni-3B."""
        from transformers import AutoTokenizer
        
        logger.info(f"Loading tokenizer: {self.model_config.base_model}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.base_model,
                trust_remote_code=True,
            )
            logger.info("Tokenizer loaded successfully")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _prepare_input(
        self,
        data: MultimodalData,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare input for the model."""
        inputs = {}
        
        # Text input
        if data.text:
            inputs["text"] = data.text
        
        # Build prompt
        if instruction:
            prompt = f"{instruction}\n\n{data.text or ''}"
        else:
            prompt = data.text or ""
        
        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.inference_config.generation.max_new_tokens,
        )
        
        inputs.update(tokenized)
        
        # Handle other modalities (placeholders for actual implementation)
        if data.image is not None:
            inputs["images"] = self._process_image(data.image)
        if data.audio is not None:
            inputs["audio"] = self._process_audio(data.audio)
        if data.video is not None:
            inputs["video"] = self._process_video(data.video)
        if data.speech is not None:
            inputs["speech"] = self._process_speech(data.speech)
        
        return inputs
    
    def _process_image(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process image input (placeholder)."""
        # In actual implementation, this would use the model's image processor
        logger.warning("Image processing not fully implemented")
        return torch.zeros(1, 3, 224, 224)  # Placeholder
    
    def _process_audio(self, audio: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process audio input (placeholder)."""
        logger.warning("Audio processing not fully implemented")
        return torch.zeros(1, 16000)  # Placeholder
    
    def _process_video(self, video: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process video input (placeholder)."""
        logger.warning("Video processing not fully implemented")
        return torch.zeros(1, 16, 224, 224, 3)  # Placeholder
    
    def _process_speech(self, speech: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process speech input (placeholder)."""
        logger.warning("Speech processing not fully implemented")
        return torch.zeros(1, 16000)  # Placeholder
    
    def _generate_with_adapter(
        self,
        inputs: Dict[str, Any],
        adapter: LoadedAdapter,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate output with a specific adapter."""
        try:
            # Apply adapter
            model = self.base_model
            peft_model = adapter.peft_model
            
            # Move to device
            model.to(self.device)
            peft_model.to(self.device)
            
            # Prepare generation config
            gen_config = generation_config or self.inference_config.generation.to_dict()
            
            # Move inputs to device
            input_ids = inputs.get("input_ids").to(self.device)
            attention_mask = inputs.get("attention_mask").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = peft_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_config,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            
            return {
                "output": generated_text,
                "output_ids": outputs,
                "success": True,
                "error": None,
            }
            
        except Exception as e:
            logger.error(f"Generation failed with adapter {adapter.name}: {e}")
            return {
                "output": None,
                "output_ids": None,
                "success": False,
                "error": str(e),
            }
    
    def infer(
        self,
        data: MultimodalData,
        instruction: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        domain: Optional[DomainType] = None,
        adapter_name: Optional[str] = None,
        use_router: Optional[bool] = None,
    ) -> InferenceResult:
        """
        Perform inference on multimodal data.
        
        Args:
            data: Multimodal input data
            instruction: Optional instruction
            task_type: Optional task type
            domain: Optional domain
            adapter_name: Optional specific adapter to use
            use_router: Whether to use router (default: from config)
            
        Returns:
            InferenceResult with output and metadata
        """
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}"
        
        try:
            # Update stats
            self.stats["total_requests"] += 1
            
            # Detect modality
            modality = self.router.detect_modality(data)
            self.stats["requests_by_modality"][modality.value] = \
                self.stats["requests_by_modality"].get(modality.value, 0) + 1
            
            # Route if needed
            if adapter_name is None and (use_router or self.inference_config.use_router):
                routing = self.router.route(
                    data=data,
                    instruction=instruction,
                    task_type=task_type,
                    domain=domain,
                )
                adapter_name = routing.primary_adapter.value if routing.primary_adapter else None
                task_type = routing.task_type
                domain = routing.domain
            
            # Load adapter
            loaded_adapter = None
            if adapter_name:
                loaded_adapter = self.adapter_loader.load_adapter(adapter_name)
                if loaded_adapter:
                    self.stats["requests_by_adapter"][adapter_name] = \
                        self.stats["requests_by_adapter"].get(adapter_name, 0) + 1
            
            # Prepare input
            inputs = self._prepare_input(data, instruction)
            
            # Generate
            if loaded_adapter:
                result = self._generate_with_adapter(
                    inputs,
                    loaded_adapter,
                )
                adapter_used = adapter_name
                adapters_used = [adapter_name]
            else:
                # Use base model without adapter
                result = self._generate_with_base(inputs)
                adapter_used = None
                adapters_used = []
            
            # Process result
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000
            
            if result["success"]:
                self.stats["successful_requests"] += 1
                tokens_generated = len(result["output_ids"][0]) if result["output_ids"] else 0
                self.stats["total_tokens"] += tokens_generated
            else:
                self.stats["failed_requests"] += 1
            
            self.stats["total_time_ms"] += inference_time_ms
            
            # Create inference result
            inference_result = InferenceResult(
                request_id=request_id,
                input_data=data,
                output=result["output"],
                modality=modality,
                task_type=task_type,
                domain=domain or DomainType.GENERAL,
                adapter_used=adapter_used,
                adapters_used=adapters_used,
                inference_time_ms=inference_time_ms,
                tokens_generated=len(result["output_ids"][0]) if result.get("output_ids") else 0,
                confidence=1.0 if result["success"] else 0.0,
                is_success=result["success"],
                error_message=result["error"],
                metrics={
                    "inference_time_ms": inference_time_ms,
                    "tokens_per_sec": (self.stats["total_tokens"] / (self.stats["total_time_ms"] / 1000)) 
                        if self.stats["total_time_ms"] > 0 else 0,
                },
            )
            
            return inference_result
            
        except Exception as e:
            end_time = time.time()
            self.stats["failed_requests"] += 1
            self.stats["total_time_ms"] += (end_time - start_time) * 1000
            
            logger.error(f"Inference failed: {e}")
            
            return InferenceResult(
                request_id=request_id,
                input_data=data,
                output=None,
                modality=ModalityType.TEXT,
                task_type=task_type,
                domain=domain or DomainType.GENERAL,
                adapter_used=adapter_name,
                adapters_used=[adapter_name] if adapter_name else [],
                inference_time_ms=(end_time - start_time) * 1000,
                tokens_generated=0,
                confidence=0.0,
                is_success=False,
                error_message=str(e),
            )
    
    def _generate_with_base(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate with base model (no adapter)."""
        try:
            model = self.base_model.to(self.device)
            
            input_ids = inputs.get("input_ids").to(self.device)
            attention_mask = inputs.get("attention_mask").to(self.device)
            
            gen_config = self.inference_config.generation.to_dict()
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_config,
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            
            return {
                "output": generated_text,
                "output_ids": outputs,
                "success": True,
                "error": None,
            }
            
        except Exception as e:
            return {
                "output": None,
                "output_ids": None,
                "success": False,
                "error": str(e),
            }
    
    def batch_infer(
        self,
        batch: List[Tuple[MultimodalData, Optional[str]]],
        use_router: Optional[bool] = None,
    ) -> List[InferenceResult]:
        """Perform batch inference."""
        results = []
        for data, instruction in batch:
            result = self.infer(
                data=data,
                instruction=instruction,
                use_router=use_router,
            )
            results.append(result)
        return results
    
    def infer_with_fallback(
        self,
        data: MultimodalData,
        instruction: Optional[str] = None,
        max_adapters: int = 3,
    ) -> InferenceResult:
        """
        Perform inference with fallback to secondary adapters.
        
        Args:
            data: Multimodal input data
            instruction: Optional instruction
            max_adapters: Maximum number of adapters to try
            
        Returns:
            InferenceResult with best output
        """
        # Route to get primary and secondary adapters
        routing = self.router.route(
            data=data,
            instruction=instruction,
        )
        
        # Try primary adapter first
        if routing.primary_adapter:
            result = self.infer(
                data=data,
                instruction=instruction,
                adapter_name=routing.primary_adapter.value,
                use_router=False,
            )
            
            if result.is_success and result.confidence >= self.inference_config.fallback_threshold:
                return result
        
        # Try secondary adapters
        for adapter in routing.secondary_adapters[:max_adapters - 1]:
            result = self.infer(
                data=data,
                instruction=instruction,
                adapter_name=adapter.value,
                use_router=False,
            )
            
            if result.is_success and result.confidence >= self.inference_config.fallback_threshold:
                return result
        
        # Try fallback adapters
        for adapter in routing.fallback_adapters[:max_adapters - len(routing.secondary_adapters) - 1]:
            result = self.infer(
                data=data,
                instruction=instruction,
                adapter_name=adapter.value,
                use_router=False,
            )
            
            if result.is_success:
                return result
        
        # Return the best result we got (even if not perfect)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = self.stats.copy()
        
        # Calculate averages
        if stats["total_requests"] > 0:
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_requests"]
            stats["avg_tokens"] = stats["total_tokens"] / stats["total_requests"]
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
        else:
            stats["avg_time_ms"] = 0
            stats["avg_tokens"] = 0
            stats["success_rate"] = 0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset inference statistics."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_time_ms": 0.0,
            "requests_by_modality": {},
            "requests_by_adapter": {},
        }
    
    def warmup(
        self,
        adapter_names: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """
        Warmup the inference engine by loading adapters.
        
        Args:
            adapter_names: List of adapters to warmup (all available if None)
            
        Returns:
            Dictionary of adapter names to success status
        """
        if adapter_names is None:
            adapter_names = self.model_config.adapters.default_adapters
        
        results = {}
        for name in adapter_names:
            try:
                loaded = self.adapter_loader.load_adapter(name)
                results[name] = loaded is not None
            except Exception as e:
                logger.error(f"Failed to warmup adapter {name}: {e}")
                results[name] = False
        
        return results
