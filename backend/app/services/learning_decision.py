"""
Smart RAG vs Training Decision Engine.
Implements the Three-Speed Learning Architecture (Fast, Medium, Slow).
Evaluates incoming data characteristics to map the optimal learning strategy.
"""

from enum import Enum
from typing import Any, Dict


class LearningSpeed(str, Enum):
    FAST = "fast"      # RAG / Knowledge Graph / Vector DB (Immediate use)
    MEDIUM = "medium"  # Specialized dynamic LoRA / Skill Adapters
    SLOW = "slow"      # Controlled continual training with EWC + full benchmarks


class DecisionEngine:
    """Decides if new information goes to RAG, dynamic adapters, or slow continual training."""

    def __init__(self):
        pass

    def evaluate_decision(self, text: str, domain: str) -> Dict[str, Any]:
        """
        Analyze content to decide the appropriate learning speed.

        Rules:
        - Temporary, frequently-changing, or news items -> FAST (RAG)
        - Specific user preferences, domain knowledge, or skill adapters -> MEDIUM (LoRA)
        - Core stable capability improvements, language updates -> SLOW (continual learning)
        """
        text_lower = text.lower()
        domain_lower = domain.lower()

        # Rule heuristics
        if any(w in text_lower for w in ["today", "breaking", "news", "current", "recent", "announcement"]):
            speed = LearningSpeed.FAST
            strategy = "Immediate Vector DB + Knowledge Graph storage for RAG retrieval."
            reason = "Temporary or rapidly changing context."
        elif domain_lower in ["coding", "mathematics", "urdu"] or any(w in text_lower for w in ["class", "def", "function", "api", "syntax"]):
            speed = LearningSpeed.MEDIUM
            strategy = "Dynamic LoRA/QLoRA adapter specialization with 30% Experience Replay."
            reason = "Specialized domain capability or skill adapter."
        else:
            speed = LearningSpeed.SLOW
            strategy = "Controlled continual learning with Elastic Weight Consolidation (EWC) and full regression benchmarks."
            reason = "Fundamental model capabilities or permanent stable knowledge."

        return {
            "learning_speed": speed.value,
            "strategy": strategy,
            "reason": reason,
            "requires_human_approval": speed == LearningSpeed.SLOW
        }
