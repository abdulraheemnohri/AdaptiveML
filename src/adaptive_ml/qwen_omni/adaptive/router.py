"""
Adaptive Router for Qwen2.5-Omni-3B.
Intelligently routes requests to appropriate adapters based on task, domain, and modality.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from adaptive_ml.qwen_omni.core import (
    AdapterInfo,
    AdapterType,
    DomainType,
    LearningDecision,
    ModalityType,
    MultimodalData,
    TaskType,
)
from adaptive_ml.qwen_omni.adaptive.detectors import (
    DomainClassificationResult,
    TaskClassificationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """Configuration for the adaptive router."""
    router_type: str = "hybrid"  # task_based, modality_based, hybrid
    use_task_routing: bool = True
    use_domain_routing: bool = True
    use_modality_routing: bool = True
    
    # Task to adapter mapping
    task_adapter_map: Dict[TaskType, List[AdapterType]] = field(default_factory=dict)
    
    # Domain to adapter mapping
    domain_adapter_map: Dict[DomainType, List[AdapterType]] = field(default_factory=dict)
    
    # Modality to adapter mapping
    modality_adapter_map: Dict[ModalityType, List[AdapterType]] = field(default_factory=dict)
    
    # Default adapters
    default_adapters: List[AdapterType] = field(default_factory=lambda: [
        AdapterType.GENERAL
    ])
    
    # Multi-adapter composition
    allow_multi_adapter: bool = True
    max_adapters: int = 3
    
    def __post_init__(self):
        # Initialize default mappings
        if not self.task_adapter_map:
            self.task_adapter_map = {
                TaskType.TEXT_GENERATION: [AdapterType.GENERAL],
                TaskType.TEXT_UNDERSTANDING: [AdapterType.GENERAL],
                TaskType.CODE_GENERATION: [AdapterType.CODING, AdapterType.GENERAL],
                TaskType.CODE_UNDERSTANDING: [AdapterType.CODING, AdapterType.GENERAL],
                TaskType.IMAGE_UNDERSTANDING: [AdapterType.VISION, AdapterType.GENERAL],
                TaskType.IMAGE_GENERATION: [AdapterType.VISION, AdapterType.GENERAL],
                TaskType.AUDIO_UNDERSTANDING: [AdapterType.AUDIO, AdapterType.GENERAL],
                TaskType.AUDIO_GENERATION: [AdapterType.AUDIO, AdapterType.GENERAL],
                TaskType.VIDEO_UNDERSTANDING: [AdapterType.VIDEO, AdapterType.GENERAL],
                TaskType.SPEECH_RECOGNITION: [AdapterType.SPEECH, AdapterType.AUDIO, AdapterType.GENERAL],
                TaskType.SPEECH_GENERATION: [AdapterType.SPEECH, AdapterType.AUDIO, AdapterType.GENERAL],
                TaskType.MULTIMODAL_UNDERSTANDING: [AdapterType.GENERAL, AdapterType.VISION, AdapterType.AUDIO],
                TaskType.TRANSLATION: [AdapterType.GENERAL],
                TaskType.SUMMARIZATION: [AdapterType.GENERAL],
                TaskType.QUESTION_ANSWERING: [AdapterType.GENERAL],
                TaskType.REASONING: [AdapterType.GENERAL],
            }
        
        if not self.domain_adapter_map:
            self.domain_adapter_map = {
                DomainType.GENERAL: [AdapterType.GENERAL],
                DomainType.CODING: [AdapterType.CODING, AdapterType.GENERAL],
                DomainType.MATHEMATICS: [AdapterType.MATH, AdapterType.GENERAL],
                DomainType.URDU: [AdapterType.URDU, AdapterType.GENERAL],
                DomainType.ENGLISH: [AdapterType.ENGLISH, AdapterType.GENERAL],
                DomainType.VISION: [AdapterType.VISION, AdapterType.GENERAL],
                DomainType.AUDIO: [AdapterType.AUDIO, AdapterType.GENERAL],
                DomainType.VIDEO: [AdapterType.VIDEO, AdapterType.GENERAL],
                DomainType.EDUCATION: [AdapterType.GENERAL],
                DomainType.NEWS: [AdapterType.GENERAL],
                DomainType.MEDICAL: [AdapterType.GENERAL],
                DomainType.LEGAL: [AdapterType.GENERAL],
                DomainType.FINANCE: [AdapterType.GENERAL],
                DomainType.TECHNICAL: [AdapterType.GENERAL],
                DomainType.CREATIVE: [AdapterType.GENERAL],
            }
        
        if not self.modality_adapter_map:
            self.modality_adapter_map = {
                ModalityType.TEXT: [AdapterType.GENERAL],
                ModalityType.VISION: [AdapterType.VISION, AdapterType.GENERAL],
                ModalityType.AUDIO: [AdapterType.AUDIO, AdapterType.GENERAL],
                ModalityType.VIDEO: [AdapterType.VIDEO, AdapterType.GENERAL],
                ModalityType.SPEECH: [AdapterType.SPEECH, AdapterType.AUDIO, AdapterType.GENERAL],
                ModalityType.MULTI_MODAL: [AdapterType.GENERAL, AdapterType.VISION, AdapterType.AUDIO],
            }


@dataclass
class RoutingDecision:
    """Result of adapter routing."""
    adapters: List[AdapterType]  # Ordered list of adapters to use
    primary_adapter: AdapterType
    confidence: float
    explanation: str
    is_multi_adapter: bool = False
    modality: ModalityType = ModalityType.TEXT
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapters": [a.value for a in self.adapters],
            "primary_adapter": self.primary_adapter.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "is_multi_adapter": self.is_multi_adapter,
            "modality": self.modality.value,
        }


class AdaptiveRouter:
    """
    Intelligently routes requests to appropriate adapters.
    Supports task-based, domain-based, and modality-based routing.
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self._available_adapters: Set[AdapterType] = set()
        
    def detect_modality(self, data: MultimodalData) -> ModalityType:
        """Detect the overall/primary modality of the input data."""
        if not data.modalities:
            return ModalityType.TEXT
        if ModalityType.MULTI_MODAL in data.modalities:
            return ModalityType.MULTI_MODAL
        # Otherwise, return the first modality or MULTI_MODAL if more than one
        if len(data.modalities) > 1:
            return ModalityType.MULTI_MODAL
        return data.modalities[0]

    def register_adapter(self, adapter_type: AdapterType) -> None:
        """Register an adapter as available."""
        self._available_adapters.add(adapter_type)
        logger.info(f"Registered adapter: {adapter_type.value}")
    
    def unregister_adapter(self, adapter_type: AdapterType) -> None:
        """Unregister an adapter."""
        self._available_adapters.discard(adapter_type)
        logger.info(f"Unregistered adapter: {adapter_type.value}")
    
    def get_available_adapters(self) -> List[AdapterType]:
        """Get list of available adapters."""
        return list(self._available_adapters)
    
    def _get_adapters_for_task(self, task_result: TaskClassificationResult) -> List[AdapterType]:
        """Get adapters for a task type."""
        if self.config.use_task_routing and task_result.task_type in self.config.task_adapter_map:
            adapters = self.config.task_adapter_map[task_result.task_type]
            # Filter to available adapters
            return [a for a in adapters if a in self._available_adapters]
        return []
    
    def _get_adapters_for_domain(self, domain_result: DomainClassificationResult) -> List[AdapterType]:
        """Get adapters for a domain."""
        if self.config.use_domain_routing and domain_result.domain in self.config.domain_adapter_map:
            adapters = self.config.domain_adapter_map[domain_result.domain]
            # Filter to available adapters
            return [a for a in adapters if a in self._available_adapters]
        return []
    
    def _get_adapters_for_modality(self, modalities: List[ModalityType]) -> List[AdapterType]:
        """Get adapters for modalities."""
        if not self.config.use_modality_routing:
            return []
        
        adapters = []
        for modality in modalities:
            if modality in self.config.modality_adapter_map:
                modality_adapters = self.config.modality_adapter_map[modality]
                adapters.extend([a for a in modality_adapters if a in self._available_adapters])
        
        return list(set(adapters))  # Remove duplicates
    
    def _combine_adapters(
        self, 
        task_adapters: List[AdapterType],
        domain_adapters: List[AdapterType],
        modality_adapters: List[AdapterType]
    ) -> List[AdapterType]:
        """Combine adapters from different routing strategies."""
        all_adapters = []
        
        # Add adapters based on router type
        if self.config.router_type == "task_based":
            all_adapters = task_adapters
        elif self.config.router_type == "domain_based":
            all_adapters = domain_adapters
        elif self.config.router_type == "modality_based":
            all_adapters = modality_adapters
        else:  # hybrid
            # Combine all adapters, prioritizing task > domain > modality
            all_adapters = task_adapters + domain_adapters + modality_adapters
        
        # Remove duplicates while preserving order
        seen = set()
        unique_adapters = []
        for adapter in all_adapters:
            if adapter not in seen:
                seen.add(adapter)
                unique_adapters.append(adapter)
        
        # If no adapters found, use defaults
        if not unique_adapters:
            unique_adapters = [a for a in self.config.default_adapters if a in self._available_adapters]
        
        # Limit to max adapters
        if self.config.allow_multi_adapter:
            unique_adapters = unique_adapters[:self.config.max_adapters]
        else:
            unique_adapters = unique_adapters[:1]
        
        return unique_adapters
    
    def _calculate_confidence(
        self,
        task_result: TaskClassificationResult,
        domain_result: DomainClassificationResult,
        modalities: List[ModalityType]
    ) -> float:
        """Calculate routing confidence."""
        confidences = []
        
        if self.config.use_task_routing:
            confidences.append(task_result.confidence)
        if self.config.use_domain_routing:
            confidences.append(domain_result.confidence)
        if self.config.use_modality_routing and modalities:
            # Modality confidence based on number of modalities
            confidences.append(min(1.0, len(modalities) * 0.25))
        
        if not confidences:
            return 0.5
        
        return sum(confidences) / len(confidences)
    
    def route(
        self,
        data: MultimodalData,
        task_result: Optional[Union[TaskClassificationResult, str]] = None,
        domain_result: Optional[Union[DomainClassificationResult, str]] = None,
        instruction: Optional[str] = None
    ) -> RoutingDecision:
        """
        Route input data to appropriate adapters.
        
        Args:
            data: Multimodal input data
            task_result: Optional pre-computed task classification
            domain_result: Optional pre-computed domain classification
            instruction: Optional instruction/prompt
            
        Returns:
            RoutingDecision with selected adapters
        """
        # Handle positional string arguments gracefully (e.g. from tests)
        if isinstance(task_result, str):
            instruction = task_result
            task_result = None

        if isinstance(domain_result, str):
            instruction = domain_result
            domain_result = None

        # Classify if not provided
        if task_result is None:
            from adaptive_ml.qwen_omni.adaptive.detectors import TaskDetector
            task_detector = TaskDetector()
            task_result = task_detector.detect(data, instruction)
        
        if domain_result is None:
            from adaptive_ml.qwen_omni.adaptive.detectors import DomainDetector
            domain_detector = DomainDetector()
            domain_result = domain_detector.detect(data, instruction)
        
        # Get adapters from different strategies
        task_adapters = self._get_adapters_for_task(task_result)
        domain_adapters = self._get_adapters_for_domain(domain_result)
        modality_adapters = self._get_adapters_for_modality(data.modalities)
        
        # Combine adapters
        adapters = self._combine_adapters(task_adapters, domain_adapters, modality_adapters)
        
        # If still no adapters, use general
        if not adapters:
            general = AdapterType.GENERAL
            if general in self._available_adapters:
                adapters = [general]
        
        # Determine primary adapter (first in list)
        primary_adapter = adapters[0] if adapters else AdapterType.GENERAL
        
        # Calculate confidence
        confidence = self._calculate_confidence(task_result, domain_result, data.modalities)
        
        # Generate explanation
        explanation_parts = []
        if task_adapters:
            explanation_parts.append(f"task: {task_result.task_type.value}")
        if domain_adapters:
            explanation_parts.append(f"domain: {domain_result.domain.value}")
        if modality_adapters:
            explanation_parts.append(f"modalities: {[m.value for m in data.modalities]}")
        
        explanation = f"Routed based on {', '.join(explanation_parts)}" if explanation_parts else "Default routing"
        
        is_multi_adapter = len(adapters) > 1
        modality = self.detect_modality(data)
        
        return RoutingDecision(
            adapters=adapters,
            primary_adapter=primary_adapter,
            confidence=confidence,
            explanation=explanation,
            is_multi_adapter=is_multi_adapter,
            modality=modality
        )
    
    def route_by_task(self, task_type: TaskType) -> RoutingDecision:
        """Route by task type only."""
        adapters = self._get_adapters_for_task(
            TaskClassificationResult(task_type=task_type, confidence=1.0)
        )
        
        if not adapters:
            adapters = [a for a in self.config.default_adapters if a in self._available_adapters]
        
        primary_adapter = adapters[0] if adapters else AdapterType.GENERAL
        
        return RoutingDecision(
            adapters=adapters,
            primary_adapter=primary_adapter,
            confidence=1.0,
            explanation=f"Task-based routing: {task_type.value}",
            is_multi_adapter=len(adapters) > 1
        )
    
    def route_by_domain(self, domain: DomainType) -> RoutingDecision:
        """Route by domain only."""
        adapters = self._get_adapters_for_domain(
            DomainClassificationResult(domain=domain, confidence=1.0)
        )
        
        if not adapters:
            adapters = [a for a in self.config.default_adapters if a in self._available_adapters]
        
        primary_adapter = adapters[0] if adapters else AdapterType.GENERAL
        
        return RoutingDecision(
            adapters=adapters,
            primary_adapter=primary_adapter,
            confidence=1.0,
            explanation=f"Domain-based routing: {domain.value}",
            is_multi_adapter=len(adapters) > 1
        )
    
    def route_by_modality(self, modalities: List[ModalityType]) -> RoutingDecision:
        """Route by modalities only."""
        adapters = self._get_adapters_for_modality(modalities)
        
        if not adapters:
            adapters = [a for a in self.config.default_adapters if a in self._available_adapters]
        
        primary_adapter = adapters[0] if adapters else AdapterType.GENERAL
        
        return RoutingDecision(
            adapters=adapters,
            primary_adapter=primary_adapter,
            confidence=1.0,
            explanation=f"Modality-based routing: {[m.value for m in modalities]}",
            is_multi_adapter=len(adapters) > 1
        )


# =============================================================================
# LEARNING STRATEGY
# =============================================================================

@dataclass
class LearningStrategy:
    """Strategy for handling new data."""
    decision: LearningDecision
    adapters_to_use: List[AdapterType]
    adapters_to_create: List[AdapterType]
    use_replay: bool = True
    use_distillation: bool = True
    use_parameter_protection: bool = True
    replay_ratio: float = 0.3
    distillation_weight: float = 0.5
    protection_lambda: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "adapters_to_use": [a.value for a in self.adapters_to_use],
            "adapters_to_create": [a.value for a in self.adapters_to_create],
            "use_replay": self.use_replay,
            "use_distillation": self.use_distillation,
            "use_parameter_protection": self.use_parameter_protection,
            "replay_ratio": self.replay_ratio,
            "distillation_weight": self.distillation_weight,
            "protection_lambda": self.protection_lambda,
        }


class LearningController:
    """
    Core controller that determines the learning strategy.
    Combines task detection, domain detection, novelty detection, and routing.
    """
    
    def __init__(
        self,
        router: Optional[AdaptiveRouter] = None,
        task_detector: Optional[Any] = None,
        domain_detector: Optional[Any] = None,
        novelty_detector: Optional[Any] = None
    ):
        self.router = router or AdaptiveRouter()
        self.task_detector = task_detector
        self.domain_detector = domain_detector
        self.novelty_detector = novelty_detector
        
        # Initialize detectors if not provided
        if self.task_detector is None:
            from adaptive_ml.qwen_omni.adaptive.detectors import TaskDetector
            self.task_detector = TaskDetector()
        if self.domain_detector is None:
            from adaptive_ml.qwen_omni.adaptive.detectors import DomainDetector
            self.domain_detector = DomainDetector()
        if self.novelty_detector is None:
            from adaptive_ml.qwen_omni.adaptive.detectors import NoveltyDetector
            self.novelty_detector = NoveltyDetector()
    
    def determine_strategy(
        self,
        data: MultimodalData,
        instruction: Optional[str] = None,
        available_adapters: Optional[List[AdapterType]] = None
    ) -> LearningStrategy:
        """
        Determine the optimal learning strategy for new data.
        
        Args:
            data: Multimodal input data
            instruction: Optional instruction/prompt
            available_adapters: List of available adapters
            
        Returns:
            LearningStrategy with recommended approach
        """
        # Update router with available adapters
        if available_adapters:
            for adapter in available_adapters:
                self.router.register_adapter(adapter)
        
        # Update novelty detector with memory
        if hasattr(self.novelty_detector, 'update_memory'):
            # In practice, you would pass the current memory entries
            pass
        
        # Detect task, domain, and novelty
        task_result = self.task_detector.detect(data, instruction)
        domain_result = self.domain_detector.detect(data, instruction)
        novelty_result = self.novelty_detector.detect(data, instruction)
        
        # Route to adapters
        routing_decision = self.router.route(data, task_result, domain_result, instruction)
        
        # Determine strategy based on novelty
        decision = novelty_result.learning_decision
        
        # Map decision to strategy
        if decision == LearningDecision.IGNORE:
            # Ignore this data
            return LearningStrategy(
                decision=decision,
                adapters_to_use=[],
                adapters_to_create=[],
                use_replay=False,
                use_distillation=False,
                use_parameter_protection=False,
                explanation="Data ignored: " + novelty_result.explanation
            )
        
        elif decision == LearningDecision.REPLAY:
            # Add to replay memory only
            return LearningStrategy(
                decision=decision,
                adapters_to_use=routing_decision.adapters,
                adapters_to_create=[],
                use_replay=True,
                use_distillation=False,
                use_parameter_protection=False,
                replay_ratio=0.3,
                explanation="Add to replay memory: " + novelty_result.explanation
            )
        
        elif decision == LearningDecision.UPDATE_ADAPTER:
            # Update existing adapter
            return LearningStrategy(
                decision=decision,
                adapters_to_use=routing_decision.adapters,
                adapters_to_create=[],
                use_replay=True,
                use_distillation=True,
                use_parameter_protection=True,
                replay_ratio=0.4,
                distillation_weight=0.6,
                protection_lambda=0.15,
                explanation="Update existing adapter: " + novelty_result.explanation
            )
        
        elif decision == LearningDecision.CREATE_ADAPTER:
            # Create new adapter
            # Determine which adapter to create based on domain
            adapter_to_create = self._determine_adapter_to_create(domain_result, task_result)
            
            return LearningStrategy(
                decision=decision,
                adapters_to_use=routing_decision.adapters,
                adapters_to_create=[adapter_to_create],
                use_replay=True,
                use_distillation=True,
                use_parameter_protection=True,
                replay_ratio=0.5,
                distillation_weight=0.7,
                protection_lambda=0.2,
                explanation="Create new adapter: " + novelty_result.explanation
            )
        
        else:  # FULL_TRAINING
            return LearningStrategy(
                decision=decision,
                adapters_to_use=routing_decision.adapters,
                adapters_to_create=[],
                use_replay=True,
                use_distillation=True,
                use_parameter_protection=True,
                replay_ratio=0.7,
                distillation_weight=0.8,
                protection_lambda=0.3,
                explanation="Full training required: " + novelty_result.explanation
            )
    
    def _determine_adapter_to_create(
        self,
        domain_result: DomainClassificationResult,
        task_result: TaskClassificationResult
    ) -> AdapterType:
        """Determine which adapter to create based on domain and task."""
        # Priority: domain > task > general
        
        # Map domain to adapter
        domain_adapter_map = {
            DomainType.CODING: AdapterType.CODING,
            DomainType.MATHEMATICS: AdapterType.MATH,
            DomainType.URDU: AdapterType.URDU,
            DomainType.ENGLISH: AdapterType.ENGLISH,
            DomainType.VISION: AdapterType.VISION,
            DomainType.AUDIO: AdapterType.AUDIO,
            DomainType.VIDEO: AdapterType.VIDEO,
        }
        
        if domain_result.domain in domain_adapter_map:
            return domain_adapter_map[domain_result.domain]
        
        # Map task to adapter
        task_adapter_map = {
            TaskType.CODE_GENERATION: AdapterType.CODING,
            TaskType.CODE_UNDERSTANDING: AdapterType.CODING,
            TaskType.IMAGE_UNDERSTANDING: AdapterType.VISION,
            TaskType.IMAGE_GENERATION: AdapterType.VISION,
            TaskType.AUDIO_UNDERSTANDING: AdapterType.AUDIO,
            TaskType.AUDIO_GENERATION: AdapterType.AUDIO,
            TaskType.VIDEO_UNDERSTANDING: AdapterType.VIDEO,
            TaskType.SPEECH_RECOGNITION: AdapterType.SPEECH,
            TaskType.SPEECH_GENERATION: AdapterType.SPEECH,
        }
        
        if task_result.task_type in task_adapter_map:
            return task_adapter_map[task_result.task_type]
        
        # Default to domain-specific or custom
        if domain_result.domain == DomainType.CUSTOM:
            return AdapterType.CUSTOM
        
        return AdapterType.DOMAIN_SPECIFIC


class AdaptiveLearningOS:
    """
    Main Adaptive Learning OS for Qwen2.5-Omni-3B.
    Orchestrates the complete continual learning pipeline.
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        learning_controller: Optional[LearningController] = None
    ):
        self.config = config
        self.learning_controller = learning_controller or LearningController()
        self._is_initialized = False
        
    def initialize(self) -> None:
        """Initialize the Adaptive Learning OS."""
        logger.info("Initializing Adaptive Learning OS for Qwen2.5-Omni-3B")
        
        # Initialize all components
        # In a full implementation, this would load models, adapters, etc.
        
        self._is_initialized = True
        logger.info("Adaptive Learning OS initialized successfully")
    
    def process_new_data(
        self,
        data: MultimodalData,
        instruction: Optional[str] = None,
        available_adapters: Optional[List[AdapterType]] = None
    ) -> LearningStrategy:
        """
        Process new data and determine learning strategy.
        
        Args:
            data: New multimodal data
            instruction: Optional instruction/prompt
            available_adapters: List of currently available adapters
            
        Returns:
            LearningStrategy with recommended approach
        """
        if not self._is_initialized:
            self.initialize()
        
        return self.learning_controller.determine_strategy(
            data, instruction, available_adapters
        )
    
    def get_learning_decision(
        self,
        data: MultimodalData,
        instruction: Optional[str] = None
    ) -> LearningDecision:
        """
        Get the learning decision for new data.
        
        Args:
            data: New multimodal data
            instruction: Optional instruction/prompt
            
        Returns:
            LearningDecision enum value
        """
        strategy = self.process_new_data(data, instruction)
        return strategy.decision
    
    def get_router(self) -> AdaptiveRouter:
        """Get the adaptive router."""
        return self.learning_controller.router
    
    def get_task_detector(self) -> Any:
        """Get the task detector."""
        return self.learning_controller.task_detector
    
    def get_domain_detector(self) -> Any:
        """Get the domain detector."""
        return self.learning_controller.domain_detector
    
    def get_novelty_detector(self) -> Any:
        """Get the novelty detector."""
        return self.learning_controller.novelty_detector
