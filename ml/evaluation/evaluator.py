"""
Evaluators for Adaptive Qwen Omni.
Implements comprehensive evaluation for all modalities.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from adaptive_ml.qwen_omni.core import (
    DomainType,
    ModalityType,
    MultimodalData,
)

logger = logging.getLogger(__name__)


@dataclass
class ModalityMetrics:
    """Metrics for a single modality."""
    modality: ModalityType
    accuracy: float = 0.0
    loss: float = 0.0
    perplexity: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    bleu: float = 0.0
    rouge: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "perplexity": self.perplexity,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "bleu": self.bleu,
            "rouge": self.rouge,
            **self.custom_metrics,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    overall_accuracy: float = 0.0
    overall_loss: float = 0.0
    overall_perplexity: float = 0.0
    modality_metrics: Dict[ModalityType, ModalityMetrics] = field(default_factory=dict)
    domain_metrics: Dict[DomainType, ModalityMetrics] = field(default_factory=dict)

    # Forgetting metrics
    forgetting_scores: Dict[ModalityType, float] = field(default_factory=dict)
    retention_score: float = 1.0

    # Timing
    evaluation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_accuracy": self.overall_accuracy,
            "overall_loss": self.overall_loss,
            "overall_perplexity": self.overall_perplexity,
            "modality_metrics": {k.value: v.to_dict() for k, v in self.modality_metrics.items()},
            "domain_metrics": {k.value: v.to_dict() for k, v in self.domain_metrics.items()},
            "forgetting_scores": {k.value: v for k, v in self.forgetting_scores.items()},
            "retention_score": self.retention_score,
            "evaluation_time": self.evaluation_time,
        }

    def get_modality_performance(self, modality: ModalityType) -> float:
        """Get performance score for a modality."""
        if modality in self.modality_metrics:
            return self.modality_metrics[modality].accuracy
        return 0.0

    def get_domain_performance(self, domain: DomainType) -> float:
        """Get performance score for a domain."""
        if domain in self.domain_metrics:
            return self.domain_metrics[domain].accuracy
        return 0.0


class ModalityEvaluator:
    """
    Base class for modality-specific evaluators.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        modality: ModalityType = ModalityType.TEXT,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.modality = modality

        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(
        self,
        dataset: Any,
        **kwargs: Any,
    ) -> ModalityMetrics:
        """
        Evaluate on a dataset.

        Args:
            dataset: Dataset to evaluate on
            **kwargs: Additional evaluation arguments

        Returns:
            ModalityMetrics with evaluation results
        """
        raise NotImplementedError("Subclasses must implement evaluate")

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute accuracy."""
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float().sum()
        total = labels.numel()
        return correct.item() / total

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute cross-entropy loss."""
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)).item()

    def compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from loss."""
        return torch.exp(torch.tensor(loss)).item()


class TextEvaluator(ModalityEvaluator):
    """
    Evaluator for text modality.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__(model, tokenizer, ModalityType.TEXT)

    def evaluate(
        self,
        dataset: Any,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> ModalityMetrics:
        """
        Evaluate on text dataset.

        Args:
            dataset: Text dataset
            batch_size: Batch size for evaluation
            **kwargs: Additional evaluation arguments

        Returns:
            ModalityMetrics with evaluation results
        """
        if self.model is None or self.tokenizer is None:
            return ModalityMetrics(modality=self.modality)

        self.model.eval()
        self.model.to(self._device)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Simple evaluation loop
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                [item.get("text", "") for item in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self._device)

            labels = torch.tensor(
                [item.get("label", 0) for item in batch],
                device=self._device,
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Compute metrics
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).float().sum()

            total_loss += loss.item() * len(batch)
            total_correct += correct.item()
            total_samples += len(batch)

        # Compute final metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()

        return ModalityMetrics(
            modality=self.modality,
            accuracy=accuracy,
            loss=avg_loss,
            perplexity=perplexity,
        )


class VisionEvaluator(ModalityEvaluator):
    """
    Evaluator for vision modality.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__(model, tokenizer, ModalityType.VISION)

    def evaluate(
        self,
        dataset: Any,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> ModalityMetrics:
        """
        Evaluate on vision dataset.

        Args:
            dataset: Vision dataset
            batch_size: Batch size for evaluation
            **kwargs: Additional evaluation arguments

        Returns:
            ModalityMetrics with evaluation results
        """
        if self.model is None:
            return ModalityMetrics(modality=self.modality)

        self.model.eval()
        self.model.to(self._device)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Simple evaluation loop
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]

            # Prepare inputs (simplified - in practice, handle images properly)
            # For Qwen2.5-Omni-3B, images are typically passed as pixel values
            inputs = {}

            # Check if batch has images
            if "images" in batch:
                # Convert images to tensor
                # This is a placeholder - actual implementation depends on data format
                pass

            # Check if batch has text
            if "text" in batch:
                text_inputs = self.tokenizer(
                    batch["text"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self._device)
                inputs.update(text_inputs)

            labels = torch.tensor(
                batch.get("label", [0] * len(batch)),
                device=self._device,
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Compute metrics
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).float().sum()

            total_loss += loss.item() * len(batch)
            total_correct += correct.item()
            total_samples += len(batch)

        # Compute final metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()

        return ModalityMetrics(
            modality=self.modality,
            accuracy=accuracy,
            loss=avg_loss,
            perplexity=perplexity,
        )


class AudioEvaluator(ModalityEvaluator):
    """
    Evaluator for audio modality.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__(model, tokenizer, ModalityType.AUDIO)

    def evaluate(
        self,
        dataset: Any,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> ModalityMetrics:
        """
        Evaluate on audio dataset.

        Args:
            dataset: Audio dataset
            batch_size: Batch size for evaluation
            **kwargs: Additional evaluation arguments

        Returns:
            ModalityMetrics with evaluation results
        """
        if self.model is None:
            return ModalityMetrics(modality=self.modality)

        self.model.eval()
        self.model.to(self._device)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Simple evaluation loop
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]

            # Prepare inputs (simplified - in practice, handle audio properly)
            inputs = {}

            # Check if batch has audio
            if "audio" in batch:
                # Convert audio to tensor
                # This is a placeholder - actual implementation depends on data format
                pass

            # Check if batch has text
            if "text" in batch:
                text_inputs = self.tokenizer(
                    batch["text"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self._device)
                inputs.update(text_inputs)

            labels = torch.tensor(
                batch.get("label", [0] * len(batch)),
                device=self._device,
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Compute metrics
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).float().sum()

            total_loss += loss.item() * len(batch)
            total_correct += correct.item()
            total_samples += len(batch)

        # Compute final metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()

        return ModalityMetrics(
            modality=self.modality,
            accuracy=accuracy,
            loss=avg_loss,
            perplexity=perplexity,
        )


class VideoEvaluator(ModalityEvaluator):
    """
    Evaluator for video modality.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__(model, tokenizer, ModalityType.VIDEO)

    def evaluate(
        self,
        dataset: Any,
        batch_size: int = 2,
        **kwargs: Any,
    ) -> ModalityMetrics:
        """
        Evaluate on video dataset.

        Args:
            dataset: Video dataset
            batch_size: Batch size for evaluation (smaller for video)
            **kwargs: Additional evaluation arguments

        Returns:
            ModalityMetrics with evaluation results
        """
        if self.model is None:
            return ModalityMetrics(modality=self.modality)

        self.model.eval()
        self.model.to(self._device)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Simple evaluation loop
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]

            # Prepare inputs (simplified - in practice, handle video properly)
            inputs = {}

            # Check if batch has video
            if "video" in batch:
                # Convert video to tensor
                # This is a placeholder - actual implementation depends on data format
                pass

            # Check if batch has text
            if "text" in batch:
                text_inputs = self.tokenizer(
                    batch["text"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self._device)
                inputs.update(text_inputs)

            labels = torch.tensor(
                batch.get("label", [0] * len(batch)),
                device=self._device,
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Compute metrics
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).float().sum()

            total_loss += loss.item() * len(batch)
            total_correct += correct.item()
            total_samples += len(batch)

        # Compute final metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()

        return ModalityMetrics(
            modality=self.modality,
            accuracy=accuracy,
            loss=avg_loss,
            perplexity=perplexity,
        )


class MultimodalEvaluator:
    """
    Multimodal Evaluator for Qwen2.5-Omni-3B.
    Coordinates evaluation across all modalities.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        modality_weights: Optional[Dict[ModalityType, float]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        # Modality weights
        self.modality_weights = modality_weights or {
            ModalityType.TEXT: 0.3,
            ModalityType.VISION: 0.2,
            ModalityType.AUDIO: 0.2,
            ModalityType.VIDEO: 0.2,
            ModalityType.SPEECH: 0.1,
        }

        # Initialize modality evaluators
        self._evaluators = {
            ModalityType.TEXT: TextEvaluator(model, tokenizer),
            ModalityType.VISION: VisionEvaluator(model, tokenizer),
            ModalityType.AUDIO: AudioEvaluator(model, tokenizer),
            ModalityType.VIDEO: VideoEvaluator(model, tokenizer),
            ModalityType.SPEECH: AudioEvaluator(model, tokenizer),  # Speech uses audio evaluator
        }

        # Previous performance for forgetting detection
        self._previous_performance: Dict[ModalityType, float] = {}

    def evaluate(
        self,
        datasets: Dict[ModalityType, Any],
        batch_size: int = 8,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate across all modalities.

        Args:
            datasets: Dictionary of modality to dataset
            batch_size: Batch size for evaluation
            **kwargs: Additional evaluation arguments

        Returns:
            EvaluationResult with all metrics
        """
        import time

        start_time = time.time()

        result = EvaluationResult()

        # Evaluate each modality
        for modality, dataset in datasets.items():
            if modality in self._evaluators and dataset is not None:
                metrics = self._evaluators[modality].evaluate(
                    dataset, batch_size=batch_size, **kwargs
                )
                result.modality_metrics[modality] = metrics

                # Store current performance for forgetting detection
                self._previous_performance[modality] = metrics.accuracy

        # Compute overall metrics
        if result.modality_metrics:
            # Weighted average
            total_weight = sum(self.modality_weights.values())
            weighted_accuracy = sum(
                m.accuracy * self.modality_weights.get(m.modality, 1.0)
                for m in result.modality_metrics.values()
            )
            weighted_loss = sum(
                m.loss * self.modality_weights.get(m.modality, 1.0)
                for m in result.modality_metrics.values()
            )

            result.overall_accuracy = weighted_accuracy / total_weight
            result.overall_loss = weighted_loss / total_weight
            result.overall_perplexity = torch.exp(torch.tensor(result.overall_loss)).item()

        # Compute forgetting scores
        for modality, metrics in result.modality_metrics.items():
            previous = self._previous_performance.get(modality, metrics.accuracy)
            forgetting = previous - metrics.accuracy
            result.forgetting_scores[modality] = max(0.0, forgetting)

        # Compute retention score
        if result.forgetting_scores:
            avg_forgetting = sum(result.forgetting_scores.values()) / len(result.forgetting_scores)
            result.retention_score = 1.0 - min(1.0, avg_forgetting)

        result.evaluation_time = time.time() - start_time

        return result

    def evaluate_single_modality(
        self,
        modality: ModalityType,
        dataset: Any,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> ModalityMetrics:
        """
        Evaluate a single modality.

        Args:
            modality: Modality to evaluate
            dataset: Dataset for the modality
            batch_size: Batch size for evaluation
            **kwargs: Additional evaluation arguments

        Returns:
            ModalityMetrics for the modality
        """
        if modality in self._evaluators:
            return self._evaluators[modality].evaluate(dataset, batch_size, **kwargs)
        else:
            return ModalityMetrics(modality=modality)

    def get_forgetting_scores(
        self,
        current_result: EvaluationResult,
    ) -> Dict[ModalityType, float]:
        """
        Compute forgetting scores based on current and previous performance.

        Args:
            current_result: Current evaluation result

        Returns:
            Dictionary of modality to forgetting score
        """
        forgetting_scores = {}

        for modality, metrics in current_result.modality_metrics.items():
            previous = self._previous_performance.get(modality, metrics.accuracy)
            forgetting = previous - metrics.accuracy
            forgetting_scores[modality] = max(0.0, forgetting)

        return forgetting_scores

    def update_previous_performance(self, result: EvaluationResult) -> None:
        """Update previous performance metrics."""
        for modality, metrics in result.modality_metrics.items():
            self._previous_performance[modality] = metrics.accuracy

    def set_model(self, model: PreTrainedModel, tokenizer: Optional[PreTrainedTokenizer] = None) -> None:
        """Set the model for all evaluators."""
        self.model = model
        if tokenizer is not None:
            self.tokenizer = tokenizer

        for evaluator in self._evaluators.values():
            evaluator.model = model
            evaluator.tokenizer = tokenizer or evaluator.tokenizer
