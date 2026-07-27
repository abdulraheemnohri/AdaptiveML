"""
Detectors for Adaptive Qwen Omni system.
Task, domain, and novelty detection for intelligent routing and learning decisions.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from adaptive_ml.qwen_omni.core import (
    DomainType,
    LearningDecision,
    ModalityType,
    MultimodalData,
    TaskType,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskClassificationResult:
    """Result of task classification."""
    task_type: TaskType
    confidence: float
    sub_tasks: List[TaskType] = field(default_factory=list)
    sub_confidences: List[float] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "confidence": self.confidence,
            "sub_tasks": [t.value for t in self.sub_tasks],
            "sub_confidences": self.sub_confidences,
            "explanation": self.explanation,
        }


@dataclass
class DomainClassificationResult:
    """Result of domain classification."""
    domain: DomainType
    confidence: float
    sub_domains: List[DomainType] = field(default_factory=list)
    sub_confidences: List[float] = field(default_factory=list)
    language: str = "en"
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "confidence": self.confidence,
            "sub_domains": [d.value for d in self.sub_domains],
            "sub_confidences": self.sub_confidences,
            "language": self.language,
            "explanation": self.explanation,
        }


@dataclass
class NoveltyResult:
    """Result of novelty detection."""
    novelty_score: float
    is_novel: bool
    learning_decision: LearningDecision
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    closest_matches: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "novelty_score": self.novelty_score,
            "is_novel": self.is_novel,
            "learning_decision": self.learning_decision.value,
            "similarity_scores": self.similarity_scores,
            "closest_matches": self.closest_matches,
            "explanation": self.explanation,
        }


class TaskDetector:
    """
    Detects the task type from input data.
    Uses keyword matching, semantic analysis, and modality detection.
    """

    def __init__(self, use_semantic: bool = True, semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.use_semantic = use_semantic
        self.semantic_model = semantic_model
        self._tokenizer = None
        self._model = None
        self._task_keywords = self._build_task_keywords()
        self._task_weights = {
            TaskType.TEXT_GENERATION: 0.1,
            TaskType.TEXT_UNDERSTANDING: 0.2,
            TaskType.QUESTION_ANSWERING: 0.3,
            TaskType.SUMMARIZATION: 0.4,
            TaskType.TRANSLATION: 0.5,
            TaskType.REASONING: 0.6,
            TaskType.CODE_GENERATION: 0.8,
            TaskType.CODE_UNDERSTANDING: 0.8,
            TaskType.IMAGE_UNDERSTANDING: 0.8,
            TaskType.IMAGE_GENERATION: 0.8,
            TaskType.AUDIO_UNDERSTANDING: 0.8,
            TaskType.AUDIO_GENERATION: 0.8,
            TaskType.VIDEO_UNDERSTANDING: 0.8,
            TaskType.SPEECH_RECOGNITION: 0.8,
            TaskType.SPEECH_GENERATION: 0.8,
            TaskType.MULTIMODAL_UNDERSTANDING: 0.9,
            TaskType.MULTIMODAL_GENERATION: 0.9,
        }

    def _build_task_keywords(self) -> Dict[TaskType, List[str]]:
        """Build keyword mappings for task detection."""
        return {
            TaskType.TEXT_GENERATION: [
                "generate", "create", "write", "compose", "produce", "draft",
                "text generation", "content creation"
            ],
            TaskType.TEXT_UNDERSTANDING: [
                "understand", "comprehend", "analyze", "explain", "interpret",
                "text understanding", "reading comprehension"
            ],
            TaskType.CODE_GENERATION: [
                "code", "program", "script", "function", "algorithm", "implement",
                "write code", "generate code", "coding", "programming"
            ],
            TaskType.CODE_UNDERSTANDING: [
                "explain code", "understand code", "analyze code", "debug",
                "code review", "code analysis"
            ],
            TaskType.IMAGE_UNDERSTANDING: [
                "describe image", "analyze image", "understand image", "explain image",
                "image description", "visual analysis", "what is in this image"
            ],
            TaskType.IMAGE_GENERATION: [
                "generate image", "create image", "draw", "paint", "visualize",
                "image generation", "art generation"
            ],
            TaskType.AUDIO_UNDERSTANDING: [
                "transcribe", "audio analysis", "sound analysis", "music analysis",
                "speech to text", "audio description"
            ],
            TaskType.AUDIO_GENERATION: [
                "generate audio", "create audio", "synthesize", "music generation",
                "text to speech", "audio creation"
            ],
            TaskType.VIDEO_UNDERSTANDING: [
                "describe video", "analyze video", "understand video", "video summary",
                "video analysis", "what is in this video"
            ],
            TaskType.SPEECH_RECOGNITION: [
                "transcribe speech", "speech to text", "voice recognition",
                "audio transcription"
            ],
            TaskType.SPEECH_GENERATION: [
                "text to speech", "speech generation", "voice synthesis",
                "generate speech", "speak"
            ],
            TaskType.MULTIMODAL_UNDERSTANDING: [
                "analyze", "describe", "explain", "understand",
                "what is", "tell me about"
            ],
            TaskType.TRANSLATION: [
                "translate", "translation", "convert language",
                "language translation"
            ],
            TaskType.SUMMARIZATION: [
                "summarize", "summary", "summarization", "condense",
                "short version", "tl;dr"
            ],
            TaskType.QUESTION_ANSWERING: [
                "answer", "question", "what", "how", "why", "when", "where",
                "who", "?", "qa", "question answering"
            ],
            TaskType.REASONING: [
                "reason", "think", "solve", "calculate", "explain",
                "step by step", "logic", "reasoning"
            ],
        }

    def _detect_from_text(self, text: str) -> TaskClassificationResult:
        """Detect task from text using keyword matching."""
        text_lower = text.lower()

        best_task = TaskType.TEXT_GENERATION
        best_confidence = 0.0

        for task_type, keywords in self._task_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Simple confidence based on keyword position
                    pos = text_lower.find(keyword)
                    confidence = 1.0 - (pos / max(len(text_lower), 1))
                    # Apply task-specific weight to prefer more specific tasks over generic ones
                    weight = self._task_weights.get(task_type, 1.0)
                    weighted_confidence = confidence * weight

                    if weighted_confidence > best_confidence:
                        best_confidence = weighted_confidence
                        best_task = task_type

        return TaskClassificationResult(
            task_type=best_task,
            confidence=best_confidence,
            explanation=f"Detected from keywords: {best_task.value}"
        )

    def _detect_from_modalities(self, modalities: List[ModalityType]) -> TaskClassificationResult:
        """Detect task from modalities."""
        modality_task_map = {
            ModalityType.TEXT: TaskType.TEXT_UNDERSTANDING,
            ModalityType.VISION: TaskType.IMAGE_UNDERSTANDING,
            ModalityType.AUDIO: TaskType.AUDIO_UNDERSTANDING,
            ModalityType.VIDEO: TaskType.VIDEO_UNDERSTANDING,
            ModalityType.SPEECH: TaskType.SPEECH_RECOGNITION,
            ModalityType.MULTI_MODAL: TaskType.MULTIMODAL_UNDERSTANDING,
        }

        if len(modalities) == 1:
            return TaskClassificationResult(
                task_type=modality_task_map.get(modalities[0], TaskType.TEXT_GENERATION),
                confidence=0.9,
                explanation=f"Single modality: {modalities[0].value}"
            )
        else:
            return TaskClassificationResult(
                task_type=TaskType.MULTIMODAL_UNDERSTANDING,
                confidence=0.95,
                explanation=f"Multiple modalities: {[m.value for m in modalities]}"
            )

    def detect(self, data: MultimodalData, instruction: Optional[str] = None) -> TaskClassificationResult:
        """
        Detect task type from multimodal data.

        Args:
            data: Multimodal input data
            instruction: Optional instruction/prompt

        Returns:
            TaskClassificationResult with detected task and confidence
        """
        # Use instruction if provided
        text_to_analyze = instruction or data.text or ""

        # Detect from text
        if text_to_analyze:
            text_result = self._detect_from_text(text_to_analyze)
        else:
            text_result = TaskClassificationResult(
                task_type=TaskType.TEXT_GENERATION,
                confidence=0.1
            )

        # Detect from modalities
        if data.modalities:
            modality_result = self._detect_from_modalities(data.modalities)
        else:
            modality_result = TaskClassificationResult(
                task_type=TaskType.TEXT_GENERATION,
                confidence=0.1
            )

        # Combine results - prefer text result if keywords were matched
        if text_result.confidence > 0.0:
            return text_result
        elif text_result.confidence > modality_result.confidence:
            return text_result
        else:
            return modality_result


class DomainDetector:
    """
    Detects the domain from input data.
    Uses keyword matching, language detection, and semantic analysis.
    """

    def __init__(self):
        self._domain_keywords = self._build_domain_keywords()
        self._language_keywords = self._build_language_keywords()

    def _build_domain_keywords(self) -> Dict[DomainType, List[str]]:
        """Build keyword mappings for domain detection."""
        return {
            DomainType.GENERAL: ["general", "common", "basic"],
            DomainType.CODING: [
                "code", "program", "python", "java", "c++", "javascript",
                "function", "algorithm", "data structure", "api", "library",
                "framework", "debug", "compile", "execute"
            ],
            DomainType.MATHEMATICS: [
                "math", "mathematics", "calculate", "equation", "formula",
                "algebra", "geometry", "calculus", "statistics", "probability"
            ],
            DomainType.URDU: [
                "urdu", "اردو", "pakistan", "pakistani", "south asia",
                "urdu language", "urdu text"
            ],
            DomainType.ENGLISH: [
                "english", "england", "america", "british", "american",
                "grammar", "vocabulary", "english language"
            ],
            DomainType.VISION: [
                "image", "photo", "picture", "visual", "computer vision",
                "object detection", "image classification", "segmentation"
            ],
            DomainType.AUDIO: [
                "audio", "sound", "music", "speech", "voice",
                "audio processing", "speech recognition", "music generation"
            ],
            DomainType.VIDEO: [
                "video", "movie", "film", "motion", "frame",
                "video analysis", "video understanding", "action recognition"
            ],
            DomainType.EDUCATION: [
                "education", "learn", "teach", "student", "teacher",
                "school", "university", "course", "lesson", "tutorial"
            ],
            DomainType.NEWS: [
                "news", "current events", "journalism", "report",
                "headline", "article", "breaking news"
            ],
            DomainType.MEDICAL: [
                "medical", "health", "doctor", "patient", "hospital",
                "disease", "treatment", "diagnosis", "medicine"
            ],
            DomainType.LEGAL: [
                "legal", "law", "attorney", "court", "contract",
                "regulation", "compliance", "legal advice"
            ],
            DomainType.FINANCE: [
                "finance", "money", "bank", "investment", "stock",
                "market", "economy", "trading", "portfolio"
            ],
            DomainType.TECHNICAL: [
                "technical", "technology", "engineering", "science",
                "research", "development", "innovation", "gadget"
            ],
            DomainType.CREATIVE: [
                "creative", "art", "design", "writing", "poetry",
                "story", "fiction", "creativity", "imagination"
            ],
        }

    def _build_language_keywords(self) -> Dict[str, List[str]]:
        """Build language detection keywords."""
        return {
            "en": ["english", "eng", "uk", "us", "usa", "british", "american"],
            "ur": ["urdu", "اردو", "pakistan", "pk", "اردو زبان"],
            "hi": ["hindi", "हिन्दी", "india", "in", "भारत"],
            "es": ["spanish", "español", "spain", "mexico", "latino"],
            "fr": ["french", "français", "france", "paris"],
            "de": ["german", "deutsch", "germany", "berlin"],
            "zh": ["chinese", "中文", "china", "cn", "中国"],
            "ja": ["japanese", "日本語", "japan", "jp", "日本"],
            "ar": ["arabic", "العربية", "saudi", "egypt", "中东"],
        }

    def _detect_language(self, text: str) -> str:
        """Detect language from text."""
        text_lower = text.lower()

        for lang, keywords in self._language_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return lang

        return "en"  # Default to English

    def _detect_from_text(self, text: str) -> DomainClassificationResult:
        """Detect domain from text using keyword matching."""
        text_lower = text.lower()

        best_domain = DomainType.GENERAL
        best_confidence = 0.0

        for domain, keywords in self._domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    pos = text_lower.find(keyword)
                    confidence = 1.0 - (pos / max(len(text_lower), 1))

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_domain = domain

        language = self._detect_language(text)

        return DomainClassificationResult(
            domain=best_domain,
            confidence=best_confidence,
            language=language,
            explanation=f"Detected from keywords: {best_domain.value}"
        )

    def _detect_from_modality(self, modalities: List[ModalityType]) -> DomainClassificationResult:
        """Detect domain from modalities."""
        modality_domain_map = {
            ModalityType.TEXT: DomainType.GENERAL,
            ModalityType.VISION: DomainType.VISION,
            ModalityType.AUDIO: DomainType.AUDIO,
            ModalityType.VIDEO: DomainType.VIDEO,
            ModalityType.SPEECH: DomainType.AUDIO,
            ModalityType.MULTI_MODAL: DomainType.GENERAL,
        }

        if len(modalities) == 1:
            return DomainClassificationResult(
                domain=modality_domain_map.get(modalities[0], DomainType.GENERAL),
                confidence=0.8,
                language="en"
            )
        else:
            # For multimodal, use the most specific domain
            domains = [modality_domain_map.get(m, DomainType.GENERAL) for m in modalities]
            domain_counts = {d: domains.count(d) for d in set(domains)}
            best_domain = max(domain_counts, key=domain_counts.get)

            return DomainClassificationResult(
                domain=best_domain,
                confidence=0.7,
                language="en"
            )

    def detect(self, data: MultimodalData, instruction: Optional[str] = None) -> DomainClassificationResult:
        """
        Detect domain from multimodal data.

        Args:
            data: Multimodal input data
            instruction: Optional instruction/prompt

        Returns:
            DomainClassificationResult with detected domain and confidence
        """
        text_to_analyze = instruction or data.text or ""

        if text_to_analyze:
            text_result = self._detect_from_text(text_to_analyze)
        else:
            text_result = DomainClassificationResult(
                domain=DomainType.GENERAL,
                confidence=0.1,
                language="en"
            )

        if data.modalities:
            modality_result = self._detect_from_modality(data.modalities)
        else:
            modality_result = DomainClassificationResult(
                domain=DomainType.GENERAL,
                confidence=0.1,
                language="en"
            )

        # Combine results - prefer text detection
        if text_result.confidence > 0.5:
            return text_result
        elif modality_result.confidence > 0.5:
            return modality_result
        else:
            # Use text result as fallback
            return text_result


class NoveltyDetector:
    """
    Detects novelty of input data compared to existing knowledge.
    Uses semantic similarity, keyword overlap, and memory lookup.
    """

    def __init__(self, memory_entries: Optional[List[Any]] = None):
        self.memory_entries = memory_entries or []
        self._embedding_model = None
        self._tokenizer = None

    def _compute_hash(self, text: str) -> str:
        """Compute hash for text deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _is_duplicate(self, text: str, threshold: float = 0.95) -> Tuple[bool, List[str]]:
        """Check if text is a duplicate of existing entries."""
        if not self.memory_entries:
            return False, []

        text_hash = self._compute_hash(text)

        for entry in self.memory_entries:
            if hasattr(entry, 'data') and hasattr(entry.data, 'text'):
                entry_text = entry.data.text or ""
                entry_hash = self._compute_hash(entry_text)

                # Simple hash-based duplicate detection
                if text_hash == entry_hash:
                    return True, [entry.id]

        return False, []

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts (simple implementation)."""
        # Use Jaccard similarity for simplicity
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def _find_similar_entries(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar entries in memory."""
        if not self.memory_entries or not text:
            return []

        similarities = []

        for entry in self.memory_entries:
            if hasattr(entry, 'data') and hasattr(entry.data, 'text'):
                entry_text = entry.data.text or ""
                if entry_text:
                    similarity = self._compute_similarity(text, entry_text)
                    if similarity > 0.1:  # Only consider non-trivial similarities
                        similarities.append((entry.id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _check_quality(self, text: str) -> Tuple[bool, str]:
        """Check text quality."""
        if not text or len(text.strip()) < 10:
            return False, "Text too short"

        # Simple quality checks
        if len(text.split()) < 3:
            return False, "Too few words"

        # Check for excessive repetition
        words = text.lower().split()
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:
            return False, "Excessive repetition"

        return True, "Good quality"

    def _check_safety(self, text: str) -> bool:
        """Check text for unsafe content (placeholder)."""
        # In a production system, this would use a proper safety classifier
        unsafe_keywords = ["hate", "violence", "illegal", "harm"]
        text_lower = text.lower()

        for keyword in unsafe_keywords:
            if keyword in text_lower:
                return False

        return True

    def detect(self, data: MultimodalData, instruction: Optional[str] = None) -> NoveltyResult:
        """
        Detect novelty of input data.

        Args:
            data: Multimodal input data
            instruction: Optional instruction/prompt

        Returns:
            NoveltyResult with novelty score and learning decision
        """
        text_to_analyze = instruction or data.text or ""

        # Check for duplicates
        is_duplicate, duplicate_ids = self._is_duplicate(text_to_analyze)

        # Check quality
        is_good_quality, quality_reason = self._check_quality(text_to_analyze)
        is_low_quality = not is_good_quality

        # Check safety
        is_unsafe = not self._check_safety(text_to_analyze)

        # Find similar entries
        similar_entries = self._find_similar_entries(text_to_analyze)
        similarity_scores = {entry_id: score for entry_id, score in similar_entries}
        closest_matches = [entry_id for entry_id, _ in similar_entries]

        # Calculate novelty score (inverse of max similarity)
        if similar_entries:
            max_similarity = max(score for _, score in similar_entries)
            novelty_score = 1.0 - max_similarity
        else:
            novelty_score = 1.0  # Completely novel if no similar entries

        # Adjust novelty based on quality and safety
        if is_duplicate:
            novelty_score = 0.0
        elif is_low_quality:
            novelty_score *= 0.3
        elif is_unsafe:
            novelty_score = 0.0

        # Determine learning decision
        if is_duplicate or is_unsafe:
            learning_decision = LearningDecision.IGNORE
        elif is_low_quality:
            learning_decision = LearningDecision.IGNORE
        elif novelty_score >= 0.7:
            learning_decision = LearningDecision.CREATE_ADAPTER
        elif novelty_score >= 0.5:
            learning_decision = LearningDecision.UPDATE_ADAPTER
        elif novelty_score >= 0.3:
            learning_decision = LearningDecision.REPLAY
        else:
            learning_decision = LearningDecision.IGNORE

        # Generate explanation
        if is_duplicate:
            explanation = f"Duplicate of existing entries: {duplicate_ids}"
        elif is_unsafe:
            explanation = "Content flagged as unsafe"
        elif is_low_quality:
            explanation = f"Low quality: {quality_reason}"
        else:
            explanation = f"Novelty score: {novelty_score:.2f}, similar to: {closest_matches}"

        return NoveltyResult(
            novelty_score=novelty_score,
            is_novel=novelty_score >= 0.5,
            learning_decision=learning_decision,
            similarity_scores=similarity_scores,
            closest_matches=closest_matches,
            explanation=explanation
        )

    def update_memory(self, memory_entries: List[Any]) -> None:
        """Update the memory entries for novelty detection."""
        self.memory_entries = memory_entries
