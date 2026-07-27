"""
Scoring module for Adaptive Qwen Omni.
Implements retention scoring and forgetting metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    DomainType,
)


@dataclass
class ForgettingScore:
    """Forgetting score for a modality or domain."""
    name: str
    score: float  # 0 = no forgetting, 1 = complete forgetting
    previous_performance: float
    current_performance: float
    threshold: float = 0.03  # 3% performance drop

    def is_critical(self) -> bool:
        """Check if forgetting is critical."""
        return self.score >= 0.05  # 5% or more

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "previous_performance": self.previous_performance,
            "current_performance": self.current_performance,
            "threshold": self.threshold,
            "is_critical": self.is_critical(),
        }


@dataclass
class RetentionScore:
    """Retention score across all modalities."""
    overall_score: float  # 0 = complete forgetting, 1 = perfect retention
    modality_scores: Dict[str, float] = field(default_factory=dict)
    domain_scores: Dict[str, float] = field(default_factory=dict)

    def is_acceptable(self, threshold: float = 0.95) -> bool:
        """Check if retention is acceptable."""
        return self.overall_score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "modality_scores": self.modality_scores,
            "domain_scores": self.domain_scores,
            "is_acceptable": self.is_acceptable(),
        }


class RetentionScorer:
    """
    Computes retention scores based on evaluation results.
    """

    def __init__(
        self,
        modality_weights: Optional[Dict[ModalityType, float]] = None,
        domain_weights: Optional[Dict[DomainType, float]] = None,
    ):
        self.modality_weights = modality_weights or {
            ModalityType.TEXT: 0.3,
            ModalityType.VISION: 0.2,
            ModalityType.AUDIO: 0.2,
            ModalityType.VIDEO: 0.2,
            ModalityType.SPEECH: 0.1,
        }

        self.domain_weights = domain_weights or {
            DomainType.GENERAL: 0.2,
            DomainType.CODING: 0.3,
            DomainType.MATHEMATICS: 0.2,
            DomainType.URDU: 0.2,
            DomainType.ENGLISH: 0.2,
            DomainType.VISION: 0.3,
            DomainType.AUDIO: 0.3,
            DomainType.VIDEO: 0.3,
        }

        # Previous performance storage
        self._previous_modality_performance: Dict[ModalityType, float] = {}
        self._previous_domain_performance: Dict[DomainType, float] = {}

    def compute_forgetting_scores(
        self,
        current_modality_performance: Dict[ModalityType, float],
        current_domain_performance: Dict[DomainType, float],
    ) -> Dict[str, ForgettingScore]:
        """
        Compute forgetting scores for all modalities and domains.

        Args:
            current_modality_performance: Current performance by modality
            current_domain_performance: Current performance by domain

        Returns:
            Dictionary of name to ForgettingScore
        """
        scores = {}

        # Compute modality forgetting scores
        for modality, current_perf in current_modality_performance.items():
            previous_perf = self._previous_modality_performance.get(modality, current_perf)
            forgetting = previous_perf - current_perf
            score = max(0.0, min(1.0, forgetting / previous_perf)) if previous_perf > 0 else 0.0

            scores[modality.value] = ForgettingScore(
                name=modality.value,
                score=score,
                previous_performance=previous_perf,
                current_performance=current_perf,
            )

        # Compute domain forgetting scores
        for domain, current_perf in current_domain_performance.items():
            previous_perf = self._previous_domain_performance.get(domain, current_perf)
            forgetting = previous_perf - current_perf
            score = max(0.0, min(1.0, forgetting / previous_perf)) if previous_perf > 0 else 0.0

            scores[domain.value] = ForgettingScore(
                name=domain.value,
                score=score,
                previous_performance=previous_perf,
                current_performance=current_perf,
            )

        return scores

    def compute_retention_score(
        self,
        forgetting_scores: Dict[str, ForgettingScore],
    ) -> RetentionScore:
        """
        Compute overall retention score from forgetting scores.

        Args:
            forgetting_scores: Dictionary of forgetting scores

        Returns:
            RetentionScore with overall and per-modality/domain scores
        """
        modality_scores = {}
        domain_scores = {}

        # Separate modality and domain scores
        for name, score in forgetting_scores.items():
            if name in [m.value for m in ModalityType]:
                modality_scores[name] = 1.0 - score.score
            elif name in [d.value for d in DomainType]:
                domain_scores[name] = 1.0 - score.score

        # Compute weighted average for modalities
        if modality_scores:
            total_modality_weight = sum(
                self.modality_weights.get(ModalityType(m), 0.0) for m in modality_scores
            )
            weighted_modality_score = sum(
                modality_scores[m] * self.modality_weights.get(ModalityType(m), 0.0)
                for m in modality_scores
            )
            modality_retention = weighted_modality_score / total_modality_weight if total_modality_weight > 0 else 0.0
        else:
            modality_retention = 1.0

        # Compute weighted average for domains
        if domain_scores:
            total_domain_weight = sum(
                self.domain_weights.get(DomainType(d), 0.0) for d in domain_scores
            )
            weighted_domain_score = sum(
                domain_scores[d] * self.domain_weights.get(DomainType(d), 0.0)
                for d in domain_scores
            )
            domain_retention = weighted_domain_score / total_domain_weight if total_domain_weight > 0 else 0.0
        else:
            domain_retention = 1.0

        # Overall retention score (weighted average of modality and domain)
        overall_score = 0.7 * modality_retention + 0.3 * domain_retention

        return RetentionScore(
            overall_score=overall_score,
            modality_scores=modality_scores,
            domain_scores=domain_scores,
        )

    def update_previous_performance(
        self,
        modality_performance: Dict[ModalityType, float],
        domain_performance: Dict[DomainType, float],
    ) -> None:
        """Update previous performance metrics."""
        self._previous_modality_performance.update(modality_performance)
        self._previous_domain_performance.update(domain_performance)

    def get_critical_modalities(
        self,
        forgetting_scores: Dict[str, ForgettingScore],
    ) -> List[str]:
        """Get list of modalities with critical forgetting."""
        return [
            name for name, score in forgetting_scores.items()
            if score.is_critical() and name in [m.value for m in ModalityType]
        ]

    def get_critical_domains(
        self,
        forgetting_scores: Dict[str, ForgettingScore],
    ) -> List[str]:
        """Get list of domains with critical forgetting."""
        return [
            name for name, score in forgetting_scores.items()
            if score.is_critical() and name in [d.value for d in DomainType]
        ]
