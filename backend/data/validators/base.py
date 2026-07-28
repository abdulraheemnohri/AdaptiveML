"""
Base Validator Interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidatorConfig:
    """Configuration for a validator"""
    name: str
    enabled: bool = True
    strict_mode: bool = False
    options: Dict[str, Any] = field(default_factory=dict)
    
    # Thresholds
    confidence_threshold: float = 0.7
    severity_threshold: str = "medium"  # low, medium, high


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    code: str
    message: str
    severity: str  # low, medium, high, critical
    category: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "category": self.category,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of validation"""
    status: ValidationStatus
    validator_name: str
    content: Optional[str] = None
    score: float = 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED
    
    @property
    def has_warnings(self) -> bool:
        return self.status == ValidationStatus.WARNING or any(
            i.severity == "low" for i in self.issues
        )
    
    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "validator_name": self.validator_name,
            "content": self.content,
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "metadata": self.metadata,
            "error": self.error,
            "passed": self.passed,
            "has_warnings": self.has_warnings,
            "has_critical_issues": self.has_critical_issues,
        }


class BaseValidator(ABC):
    """Abstract base class for all validators"""
    
    def __init__(self, config: ValidatorConfig):
        self.config = config
    
    @abstractmethod
    async def validate(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate content and return result"""
        pass
    
    @abstractmethod
    async def validate_batch(self, items: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate multiple items"""
        pass
    
    def _create_issue(self, code: str, message: str, severity: str = "medium", 
                      category: Optional[str] = None, details: Optional[Dict] = None) -> ValidationIssue:
        """Create a validation issue"""
        return ValidationIssue(
            code=code,
            message=message,
            severity=severity,
            category=category or self.config.name,
            details=details,
        )
    
    def _calculate_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall score based on issues"""
        if not issues:
            return 1.0
        
        severity_weights = {
            "low": 0.05,
            "medium": 0.15,
            "high": 0.3,
            "critical": 0.5,
        }
        
        deduction = sum(
            severity_weights.get(issue.severity, 0.1) 
            for issue in issues
        )
        
        return max(0.0, 1.0 - deduction)
    
    def _determine_status(self, score: float, issues: List[ValidationIssue]) -> ValidationStatus:
        """Determine validation status based on score and issues"""
        # Check for critical issues
        if any(i.severity == "critical" for i in issues):
            return ValidationStatus.FAILED
        
        # Check strict mode
        if self.config.strict_mode:
            if issues:
                return ValidationStatus.FAILED
            return ValidationStatus.PASSED
        
        # Normal mode - use score thresholds
        if score >= self.config.confidence_threshold:
            return ValidationStatus.PASSED
        elif score >= self.config.confidence_threshold - 0.2:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.FAILED
