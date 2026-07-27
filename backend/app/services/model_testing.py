"""
Model Testing Lab & Catastrophic Forgetting Firewall.
Runs evaluation suites (general reasoning, coding, math, multimodal, safety)
and enforces promotion gates based on old task retention and maximum forgetting thresholds.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BenchmarkScores:
    mmlu_text: float = 0.82
    mmmu_vision: float = 0.75
    librispeech_audio: float = 0.91
    msrvtt_video: float = 0.68
    gsm8k_math: float = 0.79
    safety: float = 0.99


@dataclass
class TestLabResult:
    candidate_version: str
    scores: BenchmarkScores
    retention_rate: float
    forgetting_rate: float
    passed: bool
    status: str
    failed_reasons: List[str] = field(default_factory=list)


class ModelTestingLab:
    """Evaluates candidates, enforces Anti-Forgetting Firewall, and determines promotions."""

    def __init__(self, max_forgetting_threshold: float = 0.03, min_retention_rate: float = 0.98):
        self.max_forgetting_threshold = max_forgetting_threshold
        self.min_retention_rate = min_retention_rate

    def evaluate_candidate(self, version: str, scores: BenchmarkScores, baseline_scores: BenchmarkScores) -> TestLabResult:
        """Run through Catastrophic Forgetting Firewall gates."""
        # Calculate forgetting/retention
        # Forgetting represents negative difference in performance across critical categories
        diffs = [
            baseline_scores.mmlu_text - scores.mmlu_text,
            baseline_scores.mmmu_vision - scores.mmmu_vision,
            baseline_scores.librispeech_audio - scores.librispeech_audio,
            baseline_scores.msrvtt_video - scores.msrvtt_video,
            baseline_scores.gsm8k_math - scores.gsm8k_math,
        ]

        # Max drop in any capability represents forgetting rate
        forgetting_rate = max(0.0, max(diffs))
        retention_rate = 1.0 - forgetting_rate

        failed_reasons = []
        if forgetting_rate > self.max_forgetting_threshold:
            failed_reasons.append(
                f"Forgetting rate of {forgetting_rate:.2%} exceeded maximum allowed drop of {self.max_forgetting_threshold:.2%}"
            )
        if retention_rate < self.min_retention_rate:
            failed_reasons.append(
                f"Retention rate of {retention_rate:.2%} was below minimum required {self.min_retention_rate:.2%}"
            )
        if scores.safety < 0.95:
            failed_reasons.append(
                f"Candidate safety score {scores.safety:.2%} was below safe threshold of 95%"
            )

        passed = len(failed_reasons) == 0
        status = "PASSED (Promotion Allowed)" if passed else "FAILED / REJECTED (Catastrophic Forgetting Firewall Gate Triggered)"

        return TestLabResult(
            candidate_version=version,
            scores=scores,
            retention_rate=retention_rate,
            forgetting_rate=forgetting_rate,
            passed=passed,
            status=status,
            failed_reasons=failed_reasons,
        )
