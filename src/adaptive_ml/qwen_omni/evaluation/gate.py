"""
Promotion Gate for Adaptive Qwen Omni.
Determines whether a model should be promoted based on evaluation results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    DomainType,
)
from adaptive_ml.qwen_omni.evaluation.evaluator import EvaluationResult
from adaptive_ml.qwen_omni.evaluation.scoring import RetentionScore, ForgettingScore


@dataclass
class PromotionDecision:
    """Decision for model promotion."""
    promote: bool
    rollback: bool = False
    reason: str = ""
    confidence: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "promote": self.promote,
            "rollback": self.rollback,
            "reason": self.reason,
            "confidence": self.confidence,
            "metrics": self.metrics,
        }


@dataclass
class PromotionCriteria:
    """Criteria for model promotion."""
    min_improvement: float = 0.05  # 5% improvement required
    max_forgetting: float = 0.03  # 3% max forgetting allowed
    min_retention: float = 0.98  # 98% retention required
    min_accuracy: float = 0.85  # Minimum accuracy
    
    # Modality-specific criteria
    modality_min_improvement: Dict[ModalityType, float] = field(default_factory=dict)
    modality_max_forgetting: Dict[ModalityType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize default modality criteria
        if not self.modality_min_improvement:
            self.modality_min_improvement = {
                ModalityType.TEXT: 0.05,
                ModalityType.VISION: 0.04,
                ModalityType.AUDIO: 0.04,
                ModalityType.VIDEO: 0.03,
                ModalityType.SPEECH: 0.03,
            }
        
        if not self.modality_max_forgetting:
            self.modality_max_forgetting = {
                ModalityType.TEXT: 0.03,
                ModalityType.VISION: 0.04,
                ModalityType.AUDIO: 0.04,
                ModalityType.VIDEO: 0.05,
                ModalityType.SPEECH: 0.05,
            }


class PromotionGate:
    """
    Promotion gate that determines whether a model should be promoted.
    """
    
    def __init__(
        self,
        criteria: Optional[PromotionCriteria] = None,
    ):
        self.criteria = criteria or PromotionCriteria()
        
        # Previous performance for comparison
        self._previous_result: Optional[EvaluationResult] = None
        
    def decide(
        self,
        current_result: EvaluationResult,
        previous_result: Optional[EvaluationResult] = None,
    ) -> PromotionDecision:
        """
        Make promotion decision based on evaluation results.
        
        Args:
            current_result: Current evaluation result
            previous_result: Previous evaluation result (optional)
            
        Returns:
            PromotionDecision with the decision
        """
        # Use provided previous result or stored one
        prev_result = previous_result or self._previous_result
        
        # Check if we have previous results for comparison
        if prev_result is not None:
            # Check for improvement
            improvement = current_result.overall_accuracy - prev_result.overall_accuracy
            
            # Check forgetting
            max_forgetting = max(current_result.forgetting_scores.values()) if current_result.forgetting_scores else 0.0
            avg_forgetting = sum(current_result.forgetting_scores.values()) / len(current_result.forgetting_scores) if current_result.forgetting_scores else 0.0
            
            # Check retention
            retention_ok = current_result.retention_score >= self.criteria.min_retention
            
            # Check accuracy
            accuracy_ok = current_result.overall_accuracy >= self.criteria.min_accuracy
            
            # Check modality-specific criteria
            modality_ok = True
            for modality, forgetting in current_result.forgetting_scores.items():
                modality_type = ModalityType(modality)
                max_forget = self.criteria.modality_max_forgetting.get(modality_type, self.criteria.max_forgetting)
                if forgetting > max_forget:
                    modality_ok = False
                    break
            
            # Make decision
            if improvement >= self.criteria.min_improvement and retention_ok and accuracy_ok and modality_ok:
                return PromotionDecision(
                    promote=True,
                    rollback=False,
                    reason=f"Improvement: {improvement:.2%}, Retention: {current_result.retention_score:.2%}",
                    confidence=min(1.0, improvement / self.criteria.min_improvement),
                    metrics={
                        "improvement": improvement,
                        "max_forgetting": max_forgetting,
                        "avg_forgetting": avg_forgetting,
                        "retention_score": current_result.retention_score,
                    }
                )
            elif max_forgetting > 0.1 or current_result.retention_score < 0.9:
                return PromotionDecision(
                    promote=False,
                    rollback=True,
                    reason=f"Critical forgetting detected: {max_forgetting:.2%}",
                    confidence=min(1.0, max_forgetting / 0.1),
                    metrics={
                        "improvement": improvement,
                        "max_forgetting": max_forgetting,
                        "retention_score": current_result.retention_score,
                    }
                )
            else:
                return PromotionDecision(
                    promote=False,
                    rollback=False,
                    reason=f"Insufficient improvement: {improvement:.2%}",
                    confidence=1.0 - min(1.0, improvement / self.criteria.min_improvement),
                    metrics={
                        "improvement": improvement,
                        "max_forgetting": max_forgetting,
                        "retention_score": current_result.retention_score,
                    }
                )
        else:
            # First evaluation - check absolute criteria
            if current_result.overall_accuracy >= self.criteria.min_accuracy:
                return PromotionDecision(
                    promote=True,
                    rollback=False,
                    reason="First evaluation - meets accuracy criteria",
                    confidence=0.8,
                    metrics={
                        "accuracy": current_result.overall_accuracy,
                        "retention_score": current_result.retention_score,
                    }
                )
            else:
                return PromotionDecision(
                    promote=False,
                    rollback=False,
                    reason="First evaluation - below accuracy threshold",
                    confidence=0.5,
                    metrics={
                        "accuracy": current_result.overall_accuracy,
                        "retention_score": current_result.retention_score,
                    }
                )
    
    def update_previous_result(self, result: EvaluationResult) -> None:
        """Update the previous evaluation result."""
        self._previous_result = result
    
    def get_criteria(self) -> PromotionCriteria:
        """Get the promotion criteria."""
        return self.criteria
    
    def set_criteria(self, criteria: PromotionCriteria) -> None:
        """Set the promotion criteria."""
        self.criteria = criteria
