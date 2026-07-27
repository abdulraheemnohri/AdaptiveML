"""
Promotion Controller for Adaptive ML Framework.
Manages model promotion, rollback, and A/B testing based on retention metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import PromotionStrategy
from adaptive_ml.evaluation.metrics import RetentionMetrics, EvaluationResult


class PromotionDecision(Enum):
    """Decision for model promotion."""

    PROMOTE = "promote"
    REJECT = "reject"
    AB_TEST = "ab_test"


@dataclass
class PromotionResult:
    """Result of a promotion decision."""

    decision: PromotionDecision
    retention_score: float
    old_task_score: float
    new_task_score: float
    forgetting_penalty: float
    passed: bool
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """Result of an A/B test."""

    candidate_version: str
    baseline_version: str
    candidate_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    winner: str  # "candidate", "baseline", or "tie"
    confidence: float  # 0-1
    duration: str
    completed_at: datetime = field(default_factory=datetime.now)


class PromotionController:
    """
    Controls model promotion based on retention metrics and evaluation results.
    
    Features:
    - Retention score computation
    - Promotion decision making (strict, balanced, aggressive)
    - A/B testing support
    - Rollback mechanisms
    - Version tracking
    
    The retention score formula:
        retention_score = w_new * new_score + w_old * old_score - w_forget * forgetting_penalty
    
    Promotion strategies:
    - STRICT: Require old knowledge >=95% preserved + new improvement
    - BALANCED: Allows A/B test for borderline cases
    - AGGRESSIVE: Prioritizes new capability acquisition
    
    Usage:
        promoter = PromotionController(config)
        
        # Evaluate candidate model
        result = promoter.evaluate_candidate(
            candidate_model=candidate,
            baseline_model=baseline,
            new_task_data=new_data,
            old_task_data=old_data,
        )
        
        # Make promotion decision
        decision = promoter.make_decision(result)
        
        if decision == PromotionDecision.PROMOTE:
            promoter.promote_candidate("v1.2.0")
        elif decision == PromotionDecision.REJECT:
            promoter.rollback()
    """

    def __init__(
        self,
        config: Optional[AdaptiveMLConfig] = None,
    ):
        """
        Initialize PromotionController.
        
        Args:
            config: AdaptiveMLConfig instance
        """
        self.config = config or AdaptiveMLConfig()
        self.strategy = self.config.evaluation.promotion_strategy
        
        # Thresholds
        self.retention_threshold = self.config.evaluation.retention_threshold
        self.new_task_threshold = self.config.evaluation.new_task_threshold
        
        # Weights for retention score
        self.weights = {
            "new_score": self.config.evaluation.new_score_weight,
            "old_score": self.config.evaluation.old_score_weight,
            "forgetting": self.config.evaluation.forgetting_penalty_weight,
        }
        
        # A/B testing configuration
        self.ab_test_ratio = self.config.evaluation.ab_test_ratio
        self.ab_test_duration = self.config.evaluation.ab_test_duration
        
        # Track versions and metrics
        self.versions: Dict[str, Dict[str, Any]] = {}
        self.current_version: Optional[str] = None
        self.candidate_version: Optional[str] = None
        
        # Track A/B tests
        self.ab_tests: Dict[str, ABTestResult] = {}
        self.active_ab_test: Optional[str] = None

    def evaluate_candidate(
        self,
        candidate_model: nn.Module,
        baseline_model: Optional[nn.Module] = None,
        new_task_data: Optional[List[Any]] = None,
        old_task_data: Optional[Dict[str, List[Any]]] = None,
        new_task_id: str = "new_task",
        old_task_ids: Optional[List[str]] = None,
        evaluator: Optional[Any] = None,
    ) -> PromotionResult:
        """
        Evaluate a candidate model for promotion.
        
        Args:
            candidate_model: The candidate model to evaluate
            baseline_model: Optional baseline model for comparison
            new_task_data: Data for the new task
            old_task_data: Dictionary mapping task_id to old task data
            new_task_id: Identifier for the new task
            old_task_ids: List of old task identifiers
            evaluator: Optional ContinualEvaluator instance
        
        Returns:
            PromotionResult with evaluation and decision
        """
        from adaptive_ml.evaluation.metrics import ContinualEvaluator
        
        evaluator = evaluator or ContinualEvaluator(candidate_model, self.config)
        
        # Evaluate on new task
        if new_task_data:
            new_result = evaluator.evaluate_task(new_task_id, new_task_data)
            new_score = new_result.accuracy
        else:
            new_score = 0.0
        
        # Evaluate on old tasks
        old_scores = {}
        if old_task_data and old_task_ids:
            for task_id in old_task_ids:
                if task_id in old_task_data:
                    result = evaluator.evaluate_task(task_id, old_task_data[task_id])
                    old_scores[task_id] = result.accuracy
        
        old_score = np.mean(list(old_scores.values())) if old_scores else 0.0
        
        # Compute forgetting penalty
        if baseline_model is not None and old_task_data and old_task_ids:
            # Evaluate baseline on old tasks
            baseline_evaluator = ContinualEvaluator(baseline_model, self.config)
            baseline_scores = {}
            for task_id in old_task_ids:
                if task_id in old_task_data:
                    result = baseline_evaluator.evaluate_task(task_id, old_task_data[task_id])
                    baseline_scores[task_id] = result.accuracy
            
            # Compute forgetting
            forgetting = 0.0
            for task_id in old_task_ids:
                if task_id in baseline_scores and task_id in old_scores:
                    forgetting += max(0, baseline_scores[task_id] - old_scores[task_id])
            
            forgetting_penalty = min(1.0, forgetting / len(old_task_ids)) if old_task_ids else 0.0
        else:
            forgetting_penalty = 0.0
        
        # Compute retention score
        retention_score = self._compute_retention_score(
            new_score=new_score,
            old_score=old_score,
            forgetting_penalty=forgetting_penalty,
        )
        
        # Make decision
        decision = self._make_decision(
            retention_score=retention_score,
            new_score=new_score,
            old_score=old_score,
            forgetting_penalty=forgetting_penalty,
        )
        
        # Check if passed
        passed = decision == PromotionDecision.PROMOTE
        
        # Generate message
        message = self._generate_message(
            decision=decision,
            retention_score=retention_score,
            new_score=new_score,
            old_score=old_score,
            forgetting_penalty=forgetting_penalty,
        )
        
        return PromotionResult(
            decision=decision,
            retention_score=retention_score,
            old_task_score=old_score,
            new_task_score=new_score,
            forgetting_penalty=forgetting_penalty,
            passed=passed,
            message=message,
            metadata={
                "old_scores": old_scores,
                "new_result": new_result.__dict__ if new_task_data else None,
            },
        )

    def _compute_retention_score(
        self,
        new_score: float,
        old_score: float,
        forgetting_penalty: float,
    ) -> float:
        """
        Compute the retention score.
        
        Formula:
            retention_score = w_new * new_score + w_old * old_score - w_forget * forgetting_penalty
        
        Args:
            new_score: Performance on new tasks (0-1)
            old_score: Performance on old tasks (0-1)
            forgetting_penalty: Forgetting penalty (0-1)
        
        Returns:
            Retention score (0-1, higher is better)
        """
        return (
            self.weights["new_score"] * new_score +
            self.weights["old_score"] * old_score -
            self.weights["forgetting"] * forgetting_penalty
        )

    def _make_decision(
        self,
        retention_score: float,
        new_score: float,
        old_score: float,
        forgetting_penalty: float,
    ) -> PromotionDecision:
        """
        Make a promotion decision based on the strategy.
        
        Args:
            retention_score: Computed retention score
            new_score: Performance on new tasks
            old_score: Performance on old tasks
            forgetting_penalty: Forgetting penalty
        
        Returns:
            PromotionDecision
        """
        if self.strategy == PromotionStrategy.STRICT:
            # Require: old knowledge >=95% preserved AND new improvement
            if old_score >= 0.95 and new_score > 0.0:
                return PromotionDecision.PROMOTE
            else:
                return PromotionDecision.REJECT
        
        elif self.strategy == PromotionStrategy.BALANCED:
            # Use retention score threshold
            if retention_score >= self.retention_threshold:
                return PromotionDecision.PROMOTE
            elif retention_score >= self.retention_threshold - 0.1:
                # Borderline case: A/B test
                return PromotionDecision.AB_TEST
            else:
                return PromotionDecision.REJECT
        
        elif self.strategy == PromotionStrategy.AGGRESSIVE:
            # Prioritize new capability
            if new_score >= self.new_task_threshold:
                return PromotionDecision.PROMOTE
            else:
                return PromotionDecision.REJECT
        
        return PromotionDecision.REJECT

    def _generate_message(
        self,
        decision: PromotionDecision,
        retention_score: float,
        new_score: float,
        old_score: float,
        forgetting_penalty: float,
    ) -> str:
        """Generate a human-readable message for the decision."""
        if decision == PromotionDecision.PROMOTE:
            return (
                f"PROMOTE: Retention score {retention_score:.4f} >= {self.retention_threshold:.2f}. "
                f"Old tasks: {old_score:.4f}, New tasks: {new_score:.4f}, "
                f"Forgetting: {forgetting_penalty:.4f}"
            )
        elif decision == PromotionDecision.REJECT:
            return (
                f"REJECT: Retention score {retention_score:.4f} < {self.retention_threshold:.2f}. "
                f"Old tasks: {old_score:.4f}, New tasks: {new_score:.4f}, "
                f"Forgetting: {forgetting_penalty:.4f}"
            )
        else:
            return (
                f"AB_TEST: Retention score {retention_score:.4f} is borderline. "
                f"Old tasks: {old_score:.4f}, New tasks: {new_score:.4f}"
            )

    def promote_candidate(
        self,
        version: str,
        model: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> bool:
        """
        Promote a candidate model to production.
        
        Args:
            version: Version identifier for the new model
            model: Optional model to promote (if None, uses candidate)
            metrics: Optional metrics to store with the version
            message: Optional message describing the promotion
        
        Returns:
            True if promotion was successful
        """
        # Store version info
        self.versions[version] = {
            "promoted_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "message": message,
            "status": "active",
        }
        
        # Update current version
        self.current_version = version
        self.candidate_version = None
        
        # Clear active A/B test
        self.active_ab_test = None
        
        return True

    def reject_candidate(self, version: str, message: str = "") -> bool:
        """
        Reject a candidate model.
        
        Args:
            version: Version identifier for the rejected model
            message: Optional message describing the rejection
        
        Returns:
            True if rejection was successful
        """
        if version in self.versions:
            self.versions[version]["status"] = "rejected"
            self.versions[version]["message"] = message
        
        self.candidate_version = None
        return True

    def rollback(self, version: Optional[str] = None) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            version: Version to rollback to (defaults to previous version)
        
        Returns:
            True if rollback was successful
        """
        if version is None:
            # Find the most recent version before current
            sorted_versions = sorted(
                [v for v in self.versions.keys() if v != self.current_version],
                reverse=True,
            )
            if sorted_versions:
                version = sorted_versions[0]
            else:
                return False
        
        if version not in self.versions:
            return False
        
        # Update current version
        self.versions[self.current_version]["status"] = "rolled_back"
        self.current_version = version
        self.versions[version]["status"] = "active"
        
        return True

    def start_ab_test(
        self,
        candidate_version: str,
        baseline_version: str,
        duration: Optional[str] = None,
    ) -> str:
        """
        Start an A/B test between candidate and baseline versions.
        
        Args:
            candidate_version: Version identifier for the candidate
            baseline_version: Version identifier for the baseline
            duration: Duration for the A/B test (defaults to config)
        
        Returns:
            A/B test identifier
        """
        import uuid
        
        ab_test_id = str(uuid.uuid4())
        
        self.ab_tests[ab_test_id] = ABTestResult(
            candidate_version=candidate_version,
            baseline_version=baseline_version,
            candidate_metrics={},
            baseline_metrics={},
            winner="tie",
            confidence=0.0,
            duration=duration or self.ab_test_duration,
        )
        
        self.active_ab_test = ab_test_id
        self.candidate_version = candidate_version
        
        return ab_test_id

    def update_ab_test(
        self,
        ab_test_id: str,
        candidate_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> Optional[ABTestResult]:
        """
        Update an A/B test with new metrics.
        
        Args:
            ab_test_id: A/B test identifier
            candidate_metrics: New metrics for the candidate
            baseline_metrics: New metrics for the baseline
        
        Returns:
            Updated ABTestResult or None if test not found
        """
        if ab_test_id not in self.ab_tests:
            return None
        
        ab_test = self.ab_tests[ab_test_id]
        
        # Update metrics
        for key, value in candidate_metrics.items():
            if key in ab_test.candidate_metrics:
                # Average with existing
                ab_test.candidate_metrics[key] = (
                    ab_test.candidate_metrics[key] + value
                ) / 2
            else:
                ab_test.candidate_metrics[key] = value
        
        for key, value in baseline_metrics.items():
            if key in ab_test.baseline_metrics:
                ab_test.baseline_metrics[key] = (
                    ab_test.baseline_metrics[key] + value
                ) / 2
            else:
                ab_test.baseline_metrics[key] = value
        
        # Check if we can determine a winner
        if len(ab_test.candidate_metrics) > 0:
            # Compare retention scores
            candidate_score = ab_test.candidate_metrics.get("retention_score", 0.0)
            baseline_score = ab_test.baseline_metrics.get("retention_score", 0.0)
            
            if candidate_score > baseline_score + 0.05:
                ab_test.winner = "candidate"
                ab_test.confidence = min(1.0, (candidate_score - baseline_score) / 0.2)
            elif baseline_score > candidate_score + 0.05:
                ab_test.winner = "baseline"
                ab_test.confidence = min(1.0, (baseline_score - candidate_score) / 0.2)
            else:
                ab_test.winner = "tie"
                ab_test.confidence = 0.5
        
        return ab_test

    def end_ab_test(
        self,
        ab_test_id: str,
    ) -> Optional[ABTestResult]:
        """
        End an A/B test and return the result.
        
        Args:
            ab_test_id: A/B test identifier
        
        Returns:
            ABTestResult or None if test not found
        """
        if ab_test_id not in self.ab_tests:
            return None
        
        ab_test = self.ab_tests[ab_test_id]
        ab_test.completed_at = datetime.now()
        
        if ab_test.winner == "candidate":
            # Promote candidate
            self.promote_candidate(ab_test.candidate_version)
        
        self.active_ab_test = None
        return ab_test

    def get_current_version(self) -> Optional[str]:
        """Get the current active version."""
        return self.current_version

    def get_candidate_version(self) -> Optional[str]:
        """Get the current candidate version."""
        return self.candidate_version

    def get_versions(self) -> Dict[str, Dict[str, Any]]:
        """Get all versions."""
        return self.versions

    def get_ab_tests(self) -> Dict[str, ABTestResult]:
        """Get all A/B tests."""
        return self.ab_tests

    def set_strategy(self, strategy: PromotionStrategy) -> None:
        """Set the promotion strategy."""
        self.strategy = strategy

    def set_retention_threshold(self, threshold: float) -> None:
        """Set the retention threshold."""
        self.retention_threshold = threshold

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set the weights for retention score computation."""
        self.weights.update(weights)

    def __repr__(self) -> str:
        return (
            f"PromotionController(strategy={self.strategy.value}, "
            f"current={self.current_version}, candidate={self.candidate_version})"
        )
