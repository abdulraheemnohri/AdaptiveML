"""
Safety Validator - Checks content for safety issues
"""

import re
from typing import Any, Dict, List, Optional, Set

from .base import BaseValidator, ValidatorConfig, ValidationResult, ValidationStatus


class SafetyValidator(BaseValidator):
    """Validates content for safety issues including harmful, toxic, or inappropriate content"""
    
    # Categories of unsafe content
    HATE_SPEECH_PATTERNS = [
        r'\b(nigger|nigga)\b',
        r'\b(fag|faggot)\b',
        r'\b(dyke)\b',
        r'\b(spaghetic)\b',
        r'\b(chink)\b',
        r'\b(gook)\b',
        r'\b(kike)\b',
        r'\b(paki)\b',
        r'\b(tranny)\b',
    ]
    
    VIOLENCE_PATTERNS = [
        r'\bkill\s+(yourself|myself|him|her|them|us)\b',
        r'\bmurder\b',
        r'\bmassacre\b',
        r'\bexecute\b',
        r'\bbeheading\b',
        r'\bslaughter\b',
        r'\bbombing\b',
        r'\bterrorist\s+attack\b',
    ]
    
    SELF_HARM_PATTERNS = [
        r'\bkill\s+(myself|yourself)\b',
        r'\bsuicide\b',
        r'\bharm\s+(myself|yourself)\b',
        r'\bcut\s+(myself|yourself)\b',
        r'\boverdose\b',
        r'\bend\s+(my|your)\s+life\b',
        r'\bdie\b',
    ]
    
    SEXUAL_CONTENT_PATTERNS = [
        r'\b(porn|pornography)\b',
        r'\b(xxx)\b',
        r'\b(onlyfans)\b',
        r'\b(sex\s+work)\b',
    ]
    
    DANGEROUS_ACTIVITIES_PATTERNS = [
        r'\bhow\s+to\s+(make\s+a\s+bomb|build\s+a\s+weapon)\b',
        r'\bhow\s+to\s+(hack\s+into|break\s+into)\b',
        r'\bhow\s+to\s+(steal|shoplift)\b',
        r'\bhow\s+to\s+(buy\s+drugs|make\s+drugs)\b',
        r'\bhow\s+to\s+(commit\s+fraud|scam)\b',
    ]
    
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone_us': r'\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?)[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }
    
    def __init__(self, config: ValidatorConfig):
        super().__init__(config)
        self._custom_blocked_terms: Set[str] = set()
        self._allowed_terms: Set[str] = set()
    
    async def validate(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate content for safety issues"""
        issues = []
        content_lower = content.lower()
        
        # Check hate speech
        hate_issues = self._check_patterns(content_lower, self.HATE_SPEECH_PATTERNS, "hate_speech")
        issues.extend(hate_issues)
        
        # Check violence
        violence_issues = self._check_patterns(content_lower, self.VIOLENCE_PATTERNS, "violence")
        issues.extend(violence_issues)
        
        # Check self-harm
        self_harm_issues = self._check_patterns(content_lower, self.SELF_HARM_PATTERNS, "self_harm")
        issues.extend(self_harm_issues)
        
        # Check sexual content
        sexual_issues = self._check_patterns(content_lower, self.SEXUAL_CONTENT_PATTERNS, "sexual_content")
        issues.extend(sexual_issues)
        
        # Check dangerous activities
        dangerous_issues = self._check_patterns(content_lower, self.DANGEROUS_ACTIVITIES_PATTERNS, "dangerous_activities")
        issues.extend(dangerous_issues)
        
        # Check for PII (Personally Identifiable Information)
        pii_issues = self._check_pii(content)
        issues.extend(pii_issues)
        
        # Check custom blocked terms
        custom_issues = self._check_custom_blocked(content_lower)
        issues.extend(custom_issues)
        
        # Calculate score and determine status
        score = self._calculate_score(issues)
        status = self._determine_status(score, issues)
        
        return ValidationResult(
            status=status,
            validator_name="safety",
            content=content if status != ValidationStatus.FAILED else None,
            score=score,
            issues=issues,
            metadata={
                'categories_checked': ['hate_speech', 'violence', 'self_harm', 
                                       'sexual_content', 'dangerous_activities', 'pii'],
                'has_critical_issues': any(i.severity == 'critical' for i in issues),
            },
        )
    
    async def validate_batch(self, items: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate multiple items"""
        results = []
        
        for item in items:
            content = item.get('content', '')
            metadata = item.get('metadata')
            
            result = await self.validate(content, metadata)
            results.append(result)
        
        return results
    
    def _check_patterns(self, content: str, patterns: List[str], category: str) -> List:
        """Check content against a list of patterns"""
        issues = []
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Determine severity based on category
                severity = self._get_severity(category)
                
                # Skip if term is in allowed list
                for match in matches:
                    if match.lower() in self._allowed_terms:
                        continue
                    
                    issues.append(self._create_issue(
                        code=f"safety_{category}",
                        message=f"Detected potentially unsafe content: {category}",
                        severity=severity,
                        category="safety",
                        details={
                            'pattern': pattern,
                            'matches_count': len(matches),
                            'sample': matches[0] if matches else None,
                        },
                    ))
                    break  # One issue per category
        
        return issues
    
    def _check_pii(self, content: str) -> List:
        """Check for personally identifiable information"""
        issues = []
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, content)
            if matches:
                issues.append(self._create_issue(
                    code=f"safety_pii_{pii_type}",
                    message=f"Detected potential {pii_type} in content",
                    severity="high",
                    category="safety",
                    details={
                        'pii_type': pii_type,
                        'matches_count': len(matches),
                    },
                ))
        
        return issues
    
    def _check_custom_blocked(self, content: str) -> List:
        """Check against custom blocked terms"""
        issues = []
        
        for term in self._custom_blocked_terms:
            if term.lower() in content:
                issues.append(self._create_issue(
                    code="safety_blocked_term",
                    message=f"Content contains blocked term",
                    severity="medium",
                    category="safety",
                    details={'term': term},
                ))
        
        return issues
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for a category"""
        severity_map = {
            'hate_speech': 'critical',
            'violence': 'high',
            'self_harm': 'critical',
            'sexual_content': 'high',
            'dangerous_activities': 'high',
        }
        return severity_map.get(category, 'medium')
    
    def add_blocked_term(self, term: str):
        """Add a term to the blocked list"""
        self._custom_blocked_terms.add(term.lower())
    
    def remove_blocked_term(self, term: str):
        """Remove a term from the blocked list"""
        self._custom_blocked_terms.discard(term.lower())
    
    def allow_term(self, term: str):
        """Add a term to the allowed list (override blocking)"""
        self._allowed_terms.add(term.lower())
    
    def clear_custom_lists(self):
        """Clear custom blocked and allowed terms"""
        self._custom_blocked_terms.clear()
        self._allowed_terms.clear()
    
    async def get_safety_report(self, content: str) -> Dict[str, Any]:
        """Get detailed safety report for content"""
        result = await self.validate(content)
        
        return {
            'overall_safe': result.passed,
            'score': result.score,
            'issues': [i.to_dict() for i in result.issues],
            'categories': {
                'hate_speech': any(i.code == 'safety_hate_speech' for i in result.issues),
                'violence': any(i.code == 'safety_violence' for i in result.issues),
                'self_harm': any(i.code == 'safety_self_harm' for i in result.issues),
                'sexual_content': any(i.code == 'safety_sexual_content' for i in result.issues),
                'dangerous_activities': any(i.code == 'safety_dangerous_activities' for i in result.issues),
                'pii': any('pii' in i.code for i in result.issues),
            },
            'recommendation': 'approve' if result.passed else 'reject',
        }
