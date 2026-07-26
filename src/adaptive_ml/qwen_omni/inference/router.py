"""
Multimodal Router for Adaptive Qwen Omni inference.
Routes inference requests to appropriate adapters based on task, modality, and domain.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    TaskType,
    DomainType,
    AdapterType,
    MultimodalData,
    InferenceResult,
)

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Decision for routing an inference request."""
    primary_adapter: Optional[AdapterType] = None
    secondary_adapters: List[AdapterType] = field(default_factory=list)
    fallback_adapters: List[AdapterType] = field(default_factory=list)
    modality: ModalityType = ModalityType.TEXT
    task_type: Optional[TaskType] = None
    domain: DomainType = DomainType.GENERAL
    confidence: float = 0.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_adapter": self.primary_adapter.value if self.primary_adapter else None,
            "secondary_adapters": [a.value for a in self.secondary_adapters],
            "fallback_adapters": [a.value for a in self.fallback_adapters],
            "modality": self.modality.value,
            "task_type": self.task_type.value if self.task_type else None,
            "domain": self.domain.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class MultimodalRouter:
    """
    Routes multimodal inference requests to appropriate adapters.
    
    Uses a hierarchical approach:
    1. Modality-based routing (primary)
    2. Task-based routing (secondary)
    3. Domain-based routing (tertiary)
    4. Confidence-based fallback
    """
    
    # Mapping of modalities to default adapters
    MODALITY_ADAPTER_MAP: Dict[ModalityType, AdapterType] = {
        ModalityType.TEXT: AdapterType.GENERAL,
        ModalityType.VISION: AdapterType.VISION,
        ModalityType.AUDIO: AdapterType.AUDIO,
        ModalityType.VIDEO: AdapterType.VIDEO,
        ModalityType.SPEECH: AdapterType.SPEECH,
        ModalityType.MULTI_MODAL: AdapterType.GENERAL,
    }
    
    # Mapping of tasks to adapters
    TASK_ADAPTER_MAP: Dict[TaskType, List[AdapterType]] = {
        TaskType.TEXT_GENERATION: [AdapterType.GENERAL, AdapterType.ENGLISH],
        TaskType.TEXT_UNDERSTANDING: [AdapterType.GENERAL, AdapterType.ENGLISH],
        TaskType.CODE_GENERATION: [AdapterType.CODING, AdapterType.GENERAL],
        TaskType.CODE_UNDERSTANDING: [AdapterType.CODING, AdapterType.GENERAL],
        TaskType.IMAGE_UNDERSTANDING: [AdapterType.VISION, AdapterType.GENERAL],
        TaskType.IMAGE_GENERATION: [AdapterType.VISION, AdapterType.GENERAL],
        TaskType.AUDIO_UNDERSTANDING: [AdapterType.AUDIO, AdapterType.GENERAL],
        TaskType.AUDIO_GENERATION: [AdapterType.AUDIO, AdapterType.GENERAL],
        TaskType.VIDEO_UNDERSTANDING: [AdapterType.VIDEO, AdapterType.GENERAL],
        TaskType.SPEECH_RECOGNITION: [AdapterType.SPEECH, AdapterType.GENERAL],
        TaskType.SPEECH_GENERATION: [AdapterType.SPEECH, AdapterType.GENERAL],
        TaskType.MULTIMODAL_UNDERSTANDING: [AdapterType.GENERAL, AdapterType.VISION, AdapterType.AUDIO],
        TaskType.MULTIMODAL_GENERATION: [AdapterType.GENERAL, AdapterType.VISION, AdapterType.AUDIO],
        TaskType.TRANSLATION: [AdapterType.GENERAL, AdapterType.URDU, AdapterType.ENGLISH],
        TaskType.SUMMARIZATION: [AdapterType.GENERAL, AdapterType.ENGLISH],
        TaskType.QUESTION_ANSWERING: [AdapterType.GENERAL, AdapterType.ENGLISH],
        TaskType.REASONING: [AdapterType.GENERAL, AdapterType.MATH],
    }
    
    # Mapping of domains to adapters
    DOMAIN_ADAPTER_MAP: Dict[DomainType, List[AdapterType]] = {
        DomainType.GENERAL: [AdapterType.GENERAL],
        DomainType.CODING: [AdapterType.CODING, AdapterType.GENERAL],
        DomainType.MATHEMATICS: [AdapterType.MATH, AdapterType.GENERAL],
        DomainType.URDU: [AdapterType.URDU, AdapterType.GENERAL],
        DomainType.ENGLISH: [AdapterType.ENGLISH, AdapterType.GENERAL],
        DomainType.VISION: [AdapterType.VISION, AdapterType.GENERAL],
        DomainType.AUDIO: [AdapterType.AUDIO, AdapterType.GENERAL],
        DomainType.VIDEO: [AdapterType.VIDEO, AdapterType.GENERAL],
        DomainType.EDUCATION: [AdapterType.GENERAL],
        DomainType.NEWS: [AdapterType.GENERAL, AdapterType.ENGLISH],
        DomainType.MEDICAL: [AdapterType.GENERAL],
        DomainType.LEGAL: [AdapterType.GENERAL],
        DomainType.FINANCE: [AdapterType.GENERAL],
        DomainType.TECHNICAL: [AdapterType.GENERAL, AdapterType.CODING],
        DomainType.CREATIVE: [AdapterType.GENERAL],
        DomainType.CUSTOM: [AdapterType.CUSTOM, AdapterType.GENERAL],
    }
    
    def __init__(
        self,
        available_adapters: Optional[List[AdapterType]] = None,
        routing_strategy: str = "hierarchical",
        use_confidence: bool = True,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the multimodal router.
        
        Args:
            available_adapters: List of adapters available for routing
            routing_strategy: Strategy to use ('hierarchical', 'modality_only', 'task_only', 'domain_only')
            use_confidence: Whether to use confidence-based routing
            confidence_threshold: Minimum confidence for primary adapter
        """
        self.available_adapters = available_adapters or [
            AdapterType.GENERAL, AdapterType.CODING, AdapterType.URDU,
            AdapterType.ENGLISH, AdapterType.VISION, AdapterType.AUDIO,
            AdapterType.VIDEO, AdapterType.SPEECH, AdapterType.MATH
        ]
        self.routing_strategy = routing_strategy
        self.use_confidence = use_confidence
        self.confidence_threshold = confidence_threshold
        
        # Validate adapters
        for adapter in self.available_adapters:
            if adapter not in AdapterType:
                logger.warning(f"Unknown adapter type: {adapter}")
    
    def detect_modality(self, data: MultimodalData) -> ModalityType:
        """Detect the primary modality from input data."""
        if not data.modalities:
            return ModalityType.TEXT
        
        # Priority order for modality detection
        priority_order = [
            ModalityType.MULTI_MODAL,
            ModalityType.VIDEO,
            ModalityType.AUDIO,
            ModalityType.SPEECH,
            ModalityType.VISION,
            ModalityType.TEXT,
        ]
        
        for modality in priority_order:
            if modality in data.modalities:
                return modality
        
        return data.modalities[0]
    
    def detect_task_type(
        self,
        instruction: Optional[str] = None,
        input_text: Optional[str] = None,
    ) -> Optional[TaskType]:
        """Detect task type from instruction or input text."""
        if not instruction and not input_text:
            return None
        
        text = (instruction or "") + " " + (input_text or "")
        text_lower = text.lower()
        
        # Task detection keywords
        task_keywords = {
            TaskType.TEXT_GENERATION: ["generate", "write", "create", "compose", "produce"],
            TaskType.TEXT_UNDERSTANDING: ["understand", "analyze", "explain", "interpret"],
            TaskType.CODE_GENERATION: ["code", "program", "script", "function", "write code"],
            TaskType.CODE_UNDERSTANDING: ["explain code", "analyze code", "review code"],
            TaskType.IMAGE_UNDERSTANDING: ["describe image", "analyze image", "what is in the image"],
            TaskType.IMAGE_GENERATION: ["generate image", "create image", "draw"],
            TaskType.AUDIO_UNDERSTANDING: ["transcribe", "analyze audio", "what is in the audio"],
            TaskType.AUDIO_GENERATION: ["generate audio", "create audio", "synthesize"],
            TaskType.VIDEO_UNDERSTANDING: ["describe video", "analyze video", "what is in the video"],
            TaskType.SPEECH_RECOGNITION: ["transcribe speech", "convert speech to text"],
            TaskType.SPEECH_GENERATION: ["generate speech", "text to speech", "synthesize speech"],
            TaskType.MULTIMODAL_UNDERSTANDING: ["describe", "analyze", "explain"],
            TaskType.MULTIMODAL_GENERATION: ["generate", "create"],
            TaskType.TRANSLATION: ["translate", "convert language"],
            TaskType.SUMMARIZATION: ["summarize", "summary", "summarise"],
            TaskType.QUESTION_ANSWERING: ["answer", "question", "what is", "who is", "how to"],
            TaskType.REASONING: ["reason", "solve", "calculate", "explain step by step"],
        }
        
        for task, keywords in task_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return task
        
        return None
    
    def detect_domain(
        self,
        instruction: Optional[str] = None,
        input_text: Optional[str] = None,
    ) -> DomainType:
        """Detect domain from instruction or input text."""
        if not instruction and not input_text:
            return DomainType.GENERAL
        
        text = (instruction or "") + " " + (input_text or "")
        text_lower = text.lower()
        
        # Domain detection keywords
        domain_keywords = {
            DomainType.CODING: ["python", "javascript", "code", "program", "function", "algorithm"],
            DomainType.MATHEMATICS: ["math", "mathematics", "calculate", "equation", "formula"],
            DomainType.URDU: ["urdu", "اردو", "pakistan", "pakistani"],
            DomainType.ENGLISH: ["english", "grammar", "vocabulary"],
            DomainType.VISION: ["image", "photo", "picture", "visual", "see"],
            DomainType.AUDIO: ["audio", "sound", "music", "listen"],
            DomainType.VIDEO: ["video", "movie", "film", "watch"],
            DomainType.EDUCATION: ["teach", "learn", "study", "education", "student"],
            DomainType.NEWS: ["news", "article", "journalism", "report"],
            DomainType.MEDICAL: ["medical", "health", "doctor", "patient", "disease"],
            DomainType.LEGAL: ["legal", "law", "court", "attorney", "contract"],
            DomainType.FINANCE: ["finance", "money", "investment", "stock", "bank"],
            DomainType.TECHNICAL: ["technical", "engineering", "technology", "computer"],
            DomainType.CREATIVE: ["creative", "story", "poem", "art", "design"],
        }
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return domain
        
        return DomainType.GENERAL
    
    def route(
        self,
        data: MultimodalData,
        instruction: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        domain: Optional[DomainType] = None,
    ) -> RoutingDecision:
        """
        Route an inference request to appropriate adapters.
        
        Args:
            data: Multimodal input data
            instruction: Optional instruction for task detection
            task_type: Optional explicit task type
            domain: Optional explicit domain
            
        Returns:
            RoutingDecision with primary, secondary, and fallback adapters
        """
        # Detect modality
        modality = self.detect_modality(data)
        
        # Detect task type if not provided
        if task_type is None:
            task_type = self.detect_task_type(instruction, data.text)
        
        # Detect domain if not provided
        if domain is None:
            domain = self.detect_domain(instruction, data.text)
        
        # Get adapters based on strategy
        if self.routing_strategy == "modality_only":
            primary = self.MODALITY_ADAPTER_MAP.get(modality)
            adapters = [primary] if primary else []
            
        elif self.routing_strategy == "task_only":
            adapters = self.TASK_ADAPTER_MAP.get(task_type or TaskType.TEXT_GENERATION, [])
            primary = adapters[0] if adapters else None
            
        elif self.routing_strategy == "domain_only":
            adapters = self.DOMAIN_ADAPTER_MAP.get(domain, [])
            primary = adapters[0] if adapters else None
            
        else:  # hierarchical (default)
            # Start with modality-based adapter
            primary = self.MODALITY_ADAPTER_MAP.get(modality)
            
            # Add task-based adapters
            task_adapters = self.TASK_ADAPTER_MAP.get(task_type or TaskType.TEXT_GENERATION, [])
            
            # Add domain-based adapters
            domain_adapters = self.DOMAIN_ADAPTER_MAP.get(domain, [])
            
            # Combine and deduplicate
            all_adapters = list(set([primary] + task_adapters + domain_adapters))
            all_adapters = [a for a in all_adapters if a is not None]
            
            # Filter to available adapters
            adapters = [a for a in all_adapters if a in self.available_adapters]
            
            # If primary not available, use first available
            if primary and primary not in self.available_adapters:
                primary = adapters[0] if adapters else None
        
        # Separate into primary, secondary, and fallback
        primary_adapter = primary if primary and primary in self.available_adapters else None
        
        # Secondary adapters (excluding primary)
        secondary_adapters = [a for a in adapters if a != primary_adapter]
        
        # Fallback to GENERAL if no primary found
        if primary_adapter is None:
            if AdapterType.GENERAL in self.available_adapters:
                primary_adapter = AdapterType.GENERAL
            else:
                primary_adapter = self.available_adapters[0] if self.available_adapters else None
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            modality, task_type, domain, primary_adapter, adapters
        )
        
        # Build reasoning
        reasoning = self._build_reasoning(
            modality, task_type, domain, primary_adapter, secondary_adapters
        )
        
        return RoutingDecision(
            primary_adapter=primary_adapter,
            secondary_adapters=secondary_adapters,
            fallback_adapters=[AdapterType.GENERAL] if AdapterType.GENERAL in self.available_adapters else [],
            modality=modality,
            task_type=task_type,
            domain=domain,
            confidence=confidence,
            reasoning=reasoning,
        )
    
    def _calculate_confidence(
        self,
        modality: ModalityType,
        task_type: Optional[TaskType],
        domain: DomainType,
        primary_adapter: Optional[AdapterType],
        adapters: List[AdapterType],
    ) -> float:
        """Calculate routing confidence score."""
        if not self.use_confidence:
            return 1.0
        
        confidence = 0.5  # Base confidence
        
        # Modality match
        if primary_adapter and self.MODALITY_ADAPTER_MAP.get(modality) == primary_adapter:
            confidence += 0.2
        
        # Task match
        if task_type and primary_adapter in self.TASK_ADAPTER_MAP.get(task_type, []):
            confidence += 0.2
        
        # Domain match
        if primary_adapter in self.DOMAIN_ADAPTER_MAP.get(domain, []):
            confidence += 0.1
        
        # Multiple adapters available
        if len(adapters) > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _build_reasoning(
        self,
        modality: ModalityType,
        task_type: Optional[TaskType],
        domain: DomainType,
        primary_adapter: Optional[AdapterType],
        secondary_adapters: List[AdapterType],
    ) -> str:
        """Build human-readable reasoning for routing decision."""
        parts = []
        
        if modality:
            parts.append(f"modality={modality.value}")
        if task_type:
            parts.append(f"task={task_type.value}")
        if domain:
            parts.append(f"domain={domain.value}")
        
        if primary_adapter:
            parts.append(f"primary={primary_adapter.value}")
        if secondary_adapters:
            parts.append(f"secondary={[a.value for a in secondary_adapters]}")
        
        return "; ".join(parts)
    
    def batch_route(
        self,
        batch: List[Tuple[MultimodalData, Optional[str]]],
    ) -> List[RoutingDecision]:
        """Route a batch of inference requests."""
        return [
            self.route(data, instruction) 
            for data, instruction in batch
        ]
    
    def get_adapter_priority(
        self,
        adapter: AdapterType,
        modality: ModalityType,
        task_type: Optional[TaskType] = None,
        domain: Optional[DomainType] = None,
    ) -> float:
        """Get priority score for an adapter given context."""
        score = 0.0
        
        # Modality match
        if self.MODALITY_ADAPTER_MAP.get(modality) == adapter:
            score += 1.0
        
        # Task match
        if task_type and adapter in self.TASK_ADAPTER_MAP.get(task_type, []):
            score += 0.5
        
        # Domain match
        if domain and adapter in self.DOMAIN_ADAPTER_MAP.get(domain, []):
            score += 0.3
        
        return score
