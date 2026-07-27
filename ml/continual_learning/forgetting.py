"""
Forgetting Detection and Anti-Forgetting Engine for Qwen2.5-Omni-3B.
Implements modality-specific forgetting detection and adaptive anti-forgetting strategies.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive_ml.qwen_omni.core import (
    DomainType,
    ForgettingDetectionStrategy,
    ForgettingMetrics,
    ModalityType,
)

logger = logging.getLogger(__name__)


@dataclass
class ModalityPerformance:
    """Performance metrics for a single modality."""
    modality: ModalityType
    current_score: float
    previous_score: float
    forgetting: float  # positive = forgetting, negative = improvement
    weight: float = 1.0
    is_critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "current_score": self.current_score,
            "previous_score": self.previous_score,
            "forgetting": self.forgetting,
            "weight": self.weight,
            "is_critical": self.is_critical,
        }


@dataclass
class DomainPerformance:
    """Performance metrics for a single domain."""
    domain: DomainType
    current_score: float
    previous_score: float
    forgetting: float
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "current_score": self.current_score,
            "previous_score": self.previous_score,
            "forgetting": self.forgetting,
            "weight": self.weight,
        }


class ForgettingDetector:
    """
    Detects catastrophic forgetting across modalities and domains.
    Implements multiple detection strategies.
    """

    def __init__(
        self,
        strategy: ForgettingDetectionStrategy = ForgettingDetectionStrategy.MODALITY_SPECIFIC,
        forgetting_threshold: float = 0.03,
        critical_modality_threshold: float = 0.05,
        modality_weights: Optional[Dict[ModalityType, float]] = None,
        domain_weights: Optional[Dict[DomainType, float]] = None,
    ):
        self.strategy = strategy
        self.forgetting_threshold = forgetting_threshold
        self.critical_modality_threshold = critical_modality_threshold

        # Weights for different modalities and domains
        self.modality_weights = modality_weights or {
            ModalityType.TEXT: 0.3,
            ModalityType.VISION: 0.2,
            ModalityType.AUDIO: 0.2,
            ModalityType.VIDEO: 0.2,
            ModalityType.SPEECH: 0.1,
            ModalityType.MULTI_MODAL: 0.3,
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

        # Store previous performance for comparison
        self._previous_modality_performance: Dict[ModalityType, float] = {}
        self._previous_domain_performance: Dict[DomainType, float] = {}
        self._performance_history: Dict[str, List[float]] = {}  # key: modality/domain, value: history

    def update_previous_performance(
        self,
        modality_performance: Dict[ModalityType, float],
        domain_performance: Dict[DomainType, float]
    ) -> None:
        """Update previous performance metrics."""
        self._previous_modality_performance.update(modality_performance)
        self._previous_domain_performance.update(domain_performance)

        # Update history
        for modality, score in modality_performance.items():
            key = f"modality:{modality.value}"
            if key not in self._performance_history:
                self._performance_history[key] = []
            self._performance_history[key].append(score)
            # Keep only last 10 measurements
            if len(self._performance_history[key]) > 10:
                self._performance_history[key] = self._performance_history[key][-10:]

        for domain, score in domain_performance.items():
            key = f"domain:{domain.value}"
            if key not in self._performance_history:
                self._performance_history[key] = []
            self._performance_history[key].append(score)
            if len(self._performance_history[key]) > 10:
                self._performance_history[key] = self._performance_history[key][-10:]

    def detect(
        self,
        current_modality_performance: Dict[ModalityType, float],
        current_domain_performance: Dict[DomainType, float],
    ) -> ForgettingMetrics:
        """
        Detect forgetting based on current and previous performance.

        Args:
            current_modality_performance: Current performance scores by modality
            current_domain_performance: Current performance scores by domain

        Returns:
            ForgettingMetrics with detailed forgetting information
        """
        modality_forgetting: Dict[ModalityType, float] = {}
        domain_forgetting: Dict[DomainType, float] = {}

        critical_modalities: List[ModalityType] = []

        # Calculate modality forgetting
        for modality, current_score in current_modality_performance.items():
            previous_score = self._previous_modality_performance.get(modality, current_score)
            forgetting = previous_score - current_score
            modality_forgetting[modality] = forgetting

            # Check if critical
            if forgetting >= self.critical_modality_threshold:
                critical_modalities.append(modality)

        # Calculate domain forgetting
        for domain, current_score in current_domain_performance.items():
            previous_score = self._previous_domain_performance.get(domain, current_score)
            forgetting = previous_score - current_score
            domain_forgetting[domain] = forgetting

        # Calculate overall forgetting based on strategy
        if self.strategy == ForgettingDetectionStrategy.PERFORMANCE_DROP:
            # Simple average of all forgetting
            all_forgetting = list(modality_forgetting.values()) + list(domain_forgetting.values())
            overall_forgetting = np.mean(all_forgetting) if all_forgetting else 0.0

        elif self.strategy == ForgettingDetectionStrategy.ACCURACY_THRESHOLD:
            # Count modalities/domains above threshold
            above_threshold = [
                f for f in modality_forgetting.values() if f >= self.forgetting_threshold
            ] + [
                f for f in domain_forgetting.values() if f >= self.forgetting_threshold
            ]
            overall_forgetting = len(above_threshold) / max(len(modality_forgetting) + len(domain_forgetting), 1)

        elif self.strategy == ForgettingDetectionStrategy.MODALITY_SPECIFIC:
            # Weighted average by modality
            weighted_forgetting = []
            for modality, forgetting in modality_forgetting.items():
                weight = self.modality_weights.get(modality, 1.0)
                weighted_forgetting.append(forgetting * weight)
            overall_forgetting = np.mean(weighted_forgetting) if weighted_forgetting else 0.0

        elif self.strategy == ForgettingDetectionStrategy.WEIGHTED_AVERAGE:
            # Weighted average of modalities and domains
            modality_weights = [self.modality_weights.get(m, 1.0) for m in modality_forgetting.keys()]
            domain_weights = [self.domain_weights.get(d, 1.0) for d in domain_forgetting.keys()]
            all_weights = modality_weights + domain_weights
            all_forgetting = list(modality_forgetting.values()) + list(domain_forgetting.values())

            if all_forgetting and all_weights:
                total_weight = sum(all_weights)
                overall_forgetting = sum(f * w for f, w in zip(all_forgetting, all_weights)) / total_weight
            else:
                overall_forgetting = 0.0

        elif self.strategy == ForgettingDetectionStrategy.REGRESSION_TESTING:
            # Use performance history to detect trends
            overall_forgetting = self._detect_trend_forgetting()

        elif self.strategy == ForgettingDetectionStrategy.BENCHMARK_COMPARISON:
            # Compare against benchmark thresholds
            overall_forgetting = self._detect_benchmark_forgetting(current_modality_performance)

        else:  # ADAPTIVE_THRESHOLD
            # Adaptive threshold based on recent history
            overall_forgetting = self._detect_adaptive_forgetting(
                current_modality_performance, current_domain_performance
            )

        # Calculate retention score (1 - forgetting)
        retention_score = 1.0 - min(1.0, max(0.0, overall_forgetting))

        # Determine if forgetting is detected
        forgetting_detected = overall_forgetting >= self.forgetting_threshold

        return ForgettingMetrics(
            modality_forgetting=modality_forgetting,
            domain_forgetting=domain_forgetting,
            overall_forgetting=overall_forgetting,
            retention_score=retention_score,
            forgetting_detected=forgetting_detected,
            critical_modalities=critical_modalities,
        )

    def _detect_trend_forgetting(self) -> float:
        """Detect forgetting based on performance trends."""
        trends = []

        for key, history in self._performance_history.items():
            if len(history) >= 3:
                # Calculate linear trend
                x = np.arange(len(history))
                y = np.array(history)

                # Simple linear regression: y = mx + b
                # m = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
                x_mean = np.mean(x)
                y_mean = np.mean(y)

                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sum((x - x_mean) ** 2)

                if denominator > 0:
                    slope = numerator / denominator
                    trends.append(slope)

        if not trends:
            return 0.0

        # Negative slope = performance decreasing = forgetting
        average_trend = np.mean(trends)
        return max(0.0, -average_trend)  # Convert to positive forgetting score

    def _detect_benchmark_forgetting(
        self, current_performance: Dict[ModalityType, float]
    ) -> float:
        """Detect forgetting by comparing against benchmark thresholds."""
        # Define benchmark thresholds (minimum acceptable performance)
        benchmarks = {
            ModalityType.TEXT: 0.85,
            ModalityType.VISION: 0.80,
            ModalityType.AUDIO: 0.80,
            ModalityType.VIDEO: 0.75,
            ModalityType.SPEECH: 0.75,
        }

        below_benchmark = []
        for modality, score in current_performance.items():
            benchmark = benchmarks.get(modality, 0.7)
            if score < benchmark:
                below_benchmark.append(benchmark - score)

        if not below_benchmark:
            return 0.0

        return np.mean(below_benchmark)

    def _detect_adaptive_forgetting(
        self,
        current_modality_performance: Dict[ModalityType, float],
        current_domain_performance: Dict[DomainType, float]
    ) -> float:
        """Adaptive forgetting detection that adjusts thresholds based on history."""
        # Calculate forgetting for each modality
        modality_forgetting = []
        for modality, current_score in current_modality_performance.items():
            previous_score = self._previous_modality_performance.get(modality, current_score)
            forgetting = previous_score - current_score

            # Get historical volatility for this modality
            history_key = f"modality:{modality.value}"
            history = self._performance_history.get(history_key, [])

            if len(history) >= 3:
                # Calculate standard deviation of performance
                std_dev = np.std(history)
                # Adaptive threshold: higher volatility = higher threshold
                adaptive_threshold = self.forgetting_threshold * (1 + std_dev)

                # Normalize forgetting by volatility
                normalized_forgetting = forgetting / max(std_dev, 0.01)
                modality_forgetting.append(normalized_forgetting * self.modality_weights.get(modality, 1.0))
            else:
                modality_forgetting.append(forgetting * self.modality_weights.get(modality, 1.0))

        if not modality_forgetting:
            return 0.0

        return np.mean(modality_forgetting)

    def get_modality_performance_details(
        self, current_performance: Dict[ModalityType, float]
    ) -> List[ModalityPerformance]:
        """Get detailed performance information for each modality."""
        details = []

        for modality, current_score in current_performance.items():
            previous_score = self._previous_modality_performance.get(modality, current_score)
            forgetting = previous_score - current_score
            weight = self.modality_weights.get(modality, 1.0)

            # Determine if critical
            is_critical = forgetting >= self.critical_modality_threshold

            details.append(ModalityPerformance(
                modality=modality,
                current_score=current_score,
                previous_score=previous_score,
                forgetting=forgetting,
                weight=weight,
                is_critical=is_critical,
            ))

        return details

    def get_domain_performance_details(
        self, current_performance: Dict[DomainType, float]
    ) -> List[DomainPerformance]:
        """Get detailed performance information for each domain."""
        details = []

        for domain, current_score in current_performance.items():
            previous_score = self._previous_domain_performance.get(domain, current_score)
            forgetting = previous_score - current_score
            weight = self.domain_weights.get(domain, 1.0)

            details.append(DomainPerformance(
                domain=domain,
                current_score=current_score,
                previous_score=previous_score,
                forgetting=forgetting,
                weight=weight,
            ))

        return details


class AntiForgettingEngine:
    """
    Anti-Forgetting Engine that implements adaptive strategies to prevent forgetting.

    The engine implements a feedback loop:
    1. Detect forgetting (via ForgettingDetector)
    2. Analyze which modalities/domains are affected
    3. Apply appropriate anti-forgetting strategies
    4. Monitor effectiveness and adjust
    """

    def __init__(
        self,
        forgetting_detector: Optional[ForgettingDetector] = None,
        min_replay_ratio: float = 0.1,
        max_replay_ratio: float = 0.7,
        min_distillation_weight: float = 0.1,
        max_distillation_weight: float = 0.9,
        min_protection_lambda: float = 0.01,
        max_protection_lambda: float = 0.5,
    ):
        self.forgetting_detector = forgetting_detector or ForgettingDetector()

        # Strategy parameters
        self.min_replay_ratio = min_replay_ratio
        self.max_replay_ratio = max_replay_ratio
        self.min_distillation_weight = min_distillation_weight
        self.max_distillation_weight = max_distillation_weight
        self.min_protection_lambda = min_protection_lambda
        self.max_protection_lambda = max_protection_lambda

        # Current strategy parameters
        self._current_replay_ratios: Dict[ModalityType, float] = {}
        self._current_distillation_weights: Dict[ModalityType, float] = {}
        self._current_protection_lambdas: Dict[ModalityType, float] = {}

        # Default values
        self._default_replay_ratio = 0.3
        self._default_distillation_weight = 0.5
        self._default_protection_lambda = 0.1

        # Strategy state
        self._forgetting_history: List[ForgettingMetrics] = []
        self._max_history = 10

    def update_performance(
        self,
        modality_performance: Dict[ModalityType, float],
        domain_performance: Dict[DomainType, float]
    ) -> None:
        """Update performance metrics."""
        self.forgetting_detector.update_previous_performance(
            modality_performance, domain_performance
        )

    def detect_and_respond(
        self,
        current_modality_performance: Dict[ModalityType, float],
        current_domain_performance: Dict[DomainType, float],
    ) -> Tuple[ForgettingMetrics, Dict[str, Any]]:
        """
        Detect forgetting and generate response strategy.

        Args:
            current_modality_performance: Current performance by modality
            current_domain_performance: Current performance by domain

        Returns:
            Tuple of (ForgettingMetrics, response strategy)
        """
        # Detect forgetting
        forgetting_metrics = self.forgetting_detector.detect(
            current_modality_performance, current_domain_performance
        )

        # Store in history
        self._forgetting_history.append(forgetting_metrics)
        if len(self._forgetting_history) > self._max_history:
            self._forgetting_history = self._forgetting_history[-self._max_history:]

        # Generate response strategy
        strategy = self._generate_response_strategy(forgetting_metrics)

        return forgetting_metrics, strategy

    def _generate_response_strategy(
        self, forgetting_metrics: ForgettingMetrics
    ) -> Dict[str, Any]:
        """Generate response strategy based on forgetting metrics."""
        strategy = {
            "replay_ratios": {},
            "distillation_weights": {},
            "protection_lambdas": {},
            "actions": [],
            "explanation": "",
        }

        if not forgetting_metrics.forgetting_detected:
            # No forgetting detected - use defaults
            strategy["explanation"] = "No forgetting detected. Using default parameters."
            strategy["actions"].append("maintain_default_strategy")

            # Reset to defaults
            for modality in forgetting_metrics.modality_forgetting.keys():
                strategy["replay_ratios"][modality.value] = self._default_replay_ratio
                strategy["distillation_weights"][modality.value] = self._default_distillation_weight
                strategy["protection_lambdas"][modality.value] = self._default_protection_lambda

            return strategy

        # Forgetting detected - generate adaptive response
        actions = []
        explanations = []

        # Process each modality
        for modality, forgetting in forgetting_metrics.modality_forgetting.items():
            is_critical = modality in forgetting_metrics.critical_modalities

            # Calculate response intensity
            intensity = min(1.0, forgetting / self.forgetting_detector.critical_modality_threshold)

            # Adjust replay ratio
            replay_increase = intensity * 0.4  # Max increase of 0.4
            new_replay_ratio = min(
                self.max_replay_ratio,
                self._default_replay_ratio + replay_increase
            )
            strategy["replay_ratios"][modality.value] = new_replay_ratio

            # Adjust distillation weight
            distillation_increase = intensity * 0.4
            new_distillation_weight = min(
                self.max_distillation_weight,
                self._default_distillation_weight + distillation_increase
            )
            strategy["distillation_weights"][modality.value] = new_distillation_weight

            # Adjust protection lambda
            protection_increase = intensity * 0.4
            new_protection_lambda = min(
                self.max_protection_lambda,
                self._default_protection_lambda + protection_increase
            )
            strategy["protection_lambdas"][modality.value] = new_protection_lambda

            # Generate actions and explanations
            if is_critical:
                actions.append(f"critical_forgetting_{modality.value}")
                explanations.append(f"Critical forgetting detected in {modality.value} ({forgetting:.2%})")
            else:
                actions.append(f"forgetting_{modality.value}")
                explanations.append(f"Forgetting detected in {modality.value} ({forgetting:.2%})")

        # Process domains
        for domain, forgetting in forgetting_metrics.domain_forgetting.items():
            if forgetting >= self.forgetting_detector.forgetting_threshold:
                actions.append(f"domain_forgetting_{domain.value}")
                explanations.append(f"Forgetting detected in domain {domain.value} ({forgetting:.2%})")

        strategy["actions"] = actions
        strategy["explanation"] = "; ".join(explanations) if explanations else "Forgetting detected"

        return strategy

    def get_adaptive_parameters(
        self, modality: ModalityType
    ) -> Dict[str, float]:
        """
        Get adaptive parameters for a specific modality.

        Args:
            modality: The modality to get parameters for

        Returns:
            Dictionary with replay_ratio, distillation_weight, protection_lambda
        """
        # Check if we have modality-specific parameters
        if modality.value in self._current_replay_ratios:
            replay_ratio = self._current_replay_ratios[modality.value]
        else:
            replay_ratio = self._default_replay_ratio

        if modality.value in self._current_distillation_weights:
            distillation_weight = self._current_distillation_weights[modality.value]
        else:
            distillation_weight = self._default_distillation_weight

        if modality.value in self._current_protection_lambdas:
            protection_lambda = self._current_protection_lambdas[modality.value]
        else:
            protection_lambda = self._default_protection_lambda

        return {
            "replay_ratio": replay_ratio,
            "distillation_weight": distillation_weight,
            "protection_lambda": protection_lambda,
        }

    def apply_strategy(self, strategy: Dict[str, Any]) -> None:
        """Apply a response strategy."""
        if "replay_ratios" in strategy:
            self._current_replay_ratios.update(strategy["replay_ratios"])

        if "distillation_weights" in strategy:
            self._current_distillation_weights.update(strategy["distillation_weights"])

        if "protection_lambdas" in strategy:
            self._current_protection_lambdas.update(strategy["protection_lambdas"])

        logger.info(f"Applied anti-forgetting strategy: {strategy.get('explanation', '')}")

    def get_forgetting_trend(self) -> float:
        """
        Get the trend of forgetting over time.

        Returns:
            Positive value = increasing forgetting, negative = decreasing
        """
        if len(self._forgetting_history) < 3:
            return 0.0

        overall_forgetting = [m.overall_forgetting for m in self._forgetting_history]
        x = np.arange(len(overall_forgetting))
        y = np.array(overall_forgetting)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator > 0:
            slope = numerator / denominator
            return slope

        return 0.0

    def get_recommendations(self) -> List[str]:
        """
        Get recommendations for improving the system.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if len(self._forgetting_history) >= 3:
            trend = self.get_forgetting_trend()

            if trend > 0.01:  # Increasing forgetting
                recommendations.append(
                    "Forgetting is increasing over time. Consider increasing base replay ratio "
                    "or adding more diverse replay data."
                )
            elif trend < -0.01:  # Decreasing forgetting
                recommendations.append(
                    "Forgetting is decreasing. Current anti-forgetting strategies are effective."
                )

        if self._forgetting_history and self._forgetting_history[-1].forgetting_detected:
            last_metrics = self._forgetting_history[-1]

            if last_metrics.critical_modalities:
                modalities = [m.value for m in last_metrics.critical_modalities]
                recommendations.append(
                    f"Critical forgetting detected in: {', '.join(modalities)}. "
                    "Immediate action required: increase replay, distillation, and protection "
                    "for these modalities."
                )

        if not recommendations:
            recommendations.append(
                "No immediate concerns. Continue monitoring forgetting metrics."
            )

        return recommendations
