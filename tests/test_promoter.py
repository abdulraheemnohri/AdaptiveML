"""Tests for Promotion Controller."""

import pytest
import torch
import torch.nn as nn

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import PromotionStrategy
from adaptive_ml.evaluation.promoter import (
    PromotionController,
    PromotionDecision,
    PromotionResult,
    ABTestResult,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    config = AdaptiveMLConfig()
    config.evaluation.promotion_strategy = PromotionStrategy.BALANCED
    config.evaluation.retention_threshold = 0.8
    return config


@pytest.fixture
def promoter(config):
    """Create a test promotion controller."""
    return PromotionController(config)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )


class TestPromotionController:
    """Tests for PromotionController class."""

    def test_init(self, promoter):
        """Test initialization."""
        assert promoter.strategy == PromotionStrategy.BALANCED
        assert promoter.retention_threshold == 0.8
        assert promoter.current_version is None
        assert promoter.candidate_version is None

    def test_compute_retention_score(self, promoter):
        """Test retention score computation."""
        # Test with perfect scores
        score = promoter._compute_retention_score(
            new_score=1.0,
            old_score=1.0,
            forgetting_penalty=0.0,
        )
        # retention_score = 0.4 * 1.0 + 0.4 * 1.0 - 0.2 * 0.0 = 0.8
        assert abs(score - 0.8) < 0.01
        
        # Test with forgetting
        score = promoter._compute_retention_score(
            new_score=1.0,
            old_score=1.0,
            forgetting_penalty=0.5,
        )
        # retention_score = 0.4 * 1.0 + 0.4 * 1.0 - 0.2 * 0.5 = 0.7
        assert abs(score - 0.7) < 0.01

    def test_make_decision_strict(self, config):
        """Test strict promotion strategy."""
        config.evaluation.promotion_strategy = PromotionStrategy.STRICT
        promoter = PromotionController(config)
        
        # Should promote if old score >= 0.95 and new score > 0
        decision = promoter._make_decision(
            retention_score=0.9,
            new_score=0.1,
            old_score=0.96,
            forgetting_penalty=0.0,
        )
        assert decision == PromotionDecision.PROMOTE
        
        # Should reject if old score < 0.95
        decision = promoter._make_decision(
            retention_score=0.9,
            new_score=0.1,
            old_score=0.94,
            forgetting_penalty=0.0,
        )
        assert decision == PromotionDecision.REJECT

    def test_make_decision_balanced(self, promoter):
        """Test balanced promotion strategy."""
        # Should promote if retention score >= threshold
        decision = promoter._make_decision(
            retention_score=0.85,
            new_score=0.8,
            old_score=0.8,
            forgetting_penalty=0.0,
        )
        assert decision == PromotionDecision.PROMOTE
        
        # Should A/B test if borderline
        decision = promoter._make_decision(
            retention_score=0.75,  # threshold - 0.05
            new_score=0.7,
            old_score=0.7,
            forgetting_penalty=0.0,
        )
        assert decision == PromotionDecision.AB_TEST
        
        # Should reject if below threshold
        decision = promoter._make_decision(
            retention_score=0.7,
            new_score=0.6,
            old_score=0.6,
            forgetting_penalty=0.0,
        )
        assert decision == PromotionDecision.REJECT

    def test_make_decision_aggressive(self, config):
        """Test aggressive promotion strategy."""
        config.evaluation.promotion_strategy = PromotionStrategy.AGGRESSIVE
        config.evaluation.new_task_threshold = 0.1
        promoter = PromotionController(config)
        
        # Should promote if new score >= threshold
        decision = promoter._make_decision(
            retention_score=0.5,
            new_score=0.15,
            old_score=0.5,
            forgetting_penalty=0.0,
        )
        assert decision == PromotionDecision.PROMOTE
        
        # Should reject if new score < threshold
        decision = promoter._make_decision(
            retention_score=0.5,
            new_score=0.05,
            old_score=0.5,
            forgetting_penalty=0.0,
        )
        assert decision == PromotionDecision.REJECT

    def test_generate_message(self, promoter):
        """Test message generation."""
        message = promoter._generate_message(
            decision=PromotionDecision.PROMOTE,
            retention_score=0.85,
            new_score=0.8,
            old_score=0.8,
            forgetting_penalty=0.0,
        )
        
        assert "PROMOTE" in message
        assert "0.85" in message
        
        message = promoter._generate_message(
            decision=PromotionDecision.REJECT,
            retention_score=0.7,
            new_score=0.6,
            old_score=0.6,
            forgetting_penalty=0.0,
        )
        
        assert "REJECT" in message

    def test_promote_candidate(self, promoter):
        """Test promoting a candidate."""
        success = promoter.promote_candidate(
            version="v1.0.0",
            metrics={"accuracy": 0.95},
            message="Test promotion",
        )
        
        assert success is True
        assert promoter.current_version == "v1.0.0"
        assert "v1.0.0" in promoter.versions
        assert promoter.versions["v1.0.0"]["status"] == "active"

    def test_reject_candidate(self, promoter):
        """Test rejecting a candidate."""
        promoter.promote_candidate("v1.0.0")
        
        success = promoter.reject_candidate("v1.0.0", "Test rejection")
        
        assert success is True
        assert promoter.versions["v1.0.0"]["status"] == "rejected"

    def test_rollback(self, promoter):
        """Test rollback."""
        promoter.promote_candidate("v1.0.0")
        promoter.promote_candidate("v1.1.0")
        
        assert promoter.current_version == "v1.1.0"
        
        # Rollback to previous version
        success = promoter.rollback()
        
        assert success is True
        assert promoter.current_version == "v1.0.0"

    def test_start_ab_test(self, promoter):
        """Test starting an A/B test."""
        ab_test_id = promoter.start_ab_test(
            candidate_version="v1.1.0",
            baseline_version="v1.0.0",
        )
        
        assert ab_test_id in promoter.ab_tests
        assert promoter.active_ab_test == ab_test_id
        assert promoter.candidate_version == "v1.1.0"

    def test_update_ab_test(self, promoter):
        """Test updating an A/B test."""
        ab_test_id = promoter.start_ab_test("v1.1.0", "v1.0.0")
        
        # Update with metrics (candidate needs to be > 0.05 better than baseline)
        result = promoter.update_ab_test(
            ab_test_id=ab_test_id,
            candidate_metrics={"retention_score": 0.90},
            baseline_metrics={"retention_score": 0.80},
        )
        
        assert result is not None
        assert result.candidate_metrics["retention_score"] == 0.90
        assert result.baseline_metrics["retention_score"] == 0.80
        assert result.winner == "candidate"

    def test_end_ab_test(self, promoter):
        """Test ending an A/B test."""
        ab_test_id = promoter.start_ab_test("v1.1.0", "v1.0.0")
        promoter.update_ab_test(
            ab_test_id=ab_test_id,
            candidate_metrics={"retention_score": 0.90},
            baseline_metrics={"retention_score": 0.80},
        )
        
        result = promoter.end_ab_test(ab_test_id)
        
        assert result is not None
        assert result.winner == "candidate"
        assert promoter.current_version == "v1.1.0"
        assert promoter.active_ab_test is None

    def test_set_strategy(self, promoter):
        """Test setting strategy."""
        promoter.set_strategy(PromotionStrategy.STRICT)
        assert promoter.strategy == PromotionStrategy.STRICT

    def test_set_retention_threshold(self, promoter):
        """Test setting retention threshold."""
        promoter.set_retention_threshold(0.9)
        assert promoter.retention_threshold == 0.9

    def test_set_weights(self, promoter):
        """Test setting weights."""
        promoter.set_weights({"new_score": 0.5, "old_score": 0.5})
        assert promoter.weights["new_score"] == 0.5
        assert promoter.weights["old_score"] == 0.5

    def test_get_current_version(self, promoter):
        """Test getting current version."""
        assert promoter.get_current_version() is None
        
        promoter.promote_candidate("v1.0.0")
        assert promoter.get_current_version() == "v1.0.0"

    def test_get_versions(self, promoter):
        """Test getting all versions."""
        promoter.promote_candidate("v1.0.0")
        promoter.promote_candidate("v1.1.0")
        
        versions = promoter.get_versions()
        assert "v1.0.0" in versions
        assert "v1.1.0" in versions

    def test_repr(self, promoter):
        """Test __repr__ method."""
        repr_str = repr(promoter)
        
        assert "PromotionController" in repr_str
        assert "strategy=" in repr_str


class TestPromotionResult:
    """Tests for PromotionResult dataclass."""

    def test_init(self):
        """Test initialization."""
        result = PromotionResult(
            decision=PromotionDecision.PROMOTE,
            retention_score=0.85,
            old_task_score=0.8,
            new_task_score=0.8,
            forgetting_penalty=0.0,
            passed=True,
            message="Test message",
        )
        
        assert result.decision == PromotionDecision.PROMOTE
        assert result.retention_score == 0.85
        assert result.passed is True


class TestABTestResult:
    """Tests for ABTestResult dataclass."""

    def test_init(self):
        """Test initialization."""
        result = ABTestResult(
            candidate_version="v1.1.0",
            baseline_version="v1.0.0",
            candidate_metrics={},
            baseline_metrics={},
            winner="candidate",
            confidence=0.8,
            duration="1h",
        )
        
        assert result.candidate_version == "v1.1.0"
        assert result.baseline_version == "v1.0.0"
        assert result.winner == "candidate"
        assert result.confidence == 0.8
