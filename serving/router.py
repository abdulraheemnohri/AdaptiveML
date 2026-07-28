"""
AI Router for Adaptive Omni ML.

Routes requests between local model and multiple AI providers based on:
- Task type
- Capability requirements
- Privacy settings
- Latency requirements
- Cost considerations
- Context length
- Provider availability
- Model quality
"""
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import statistics


class RoutingMode(Enum):
    """Routing modes available."""
    LOCAL_ONLY = "local_only"
    API_ONLY = "api_only"
    LOCAL_FIRST = "local_first"
    API_FIRST = "api_first"
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class TaskType(Enum):
    """Types of tasks for routing decisions."""
    GENERAL_CHAT = "general_chat"
    CODING = "coding"
    MATH = "math"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    SPEECH = "speech"
    DATA_ANALYSIS = "data_analysis"
    PRIVATE_DATA = "private_data"
    LONG_CONTEXT = "long_context"
    REAL_TIME = "real_time"


@dataclass
class ProviderCapabilities:
    """Capabilities of a provider."""
    provider_id: str
    provider_name: str
    supported_task_types: List[TaskType] = field(default_factory=list)
    max_context_length: int = 4096
    supports_vision: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_speech: bool = False
    avg_latency_ms: float = 0.0
    cost_per_1k_tokens: float = 0.0
    reliability_score: float = 1.0
    is_available: bool = True
    rate_limit_remaining: int = 1000


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    selected_target: str  # "local" or provider_id
    target_type: str  # "local" or "api"
    reasoning: str
    fallback_target: Optional[str] = None
    estimated_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    confidence_score: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RoutingRule:
    """Custom routing rule."""
    id: str
    name: str
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    target: str = ""
    fallback_target: Optional[str] = None
    is_enabled: bool = True
    
    def matches(self, request: Dict[str, Any]) -> bool:
        """Check if request matches this rule's conditions."""
        if not self.is_enabled:
            return False
        
        for key, expected_value in self.conditions.items():
            request_value = request.get(key)
            
            if isinstance(expected_value, list):
                # Match any value in list
                if request_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict):
                # Complex condition (e.g., {"gt": 1000})
                for op, val in expected_value.items():
                    if op == "gt" and not (request_value > val):
                        return False
                    elif op == "lt" and not (request_value < val):
                        return False
                    elif op == "gte" and not (request_value >= val):
                        return False
                    elif op == "lte" and not (request_value <= val):
                        return False
                    elif op == "eq" and not (request_value == val):
                        return False
                    elif op == "ne" and not (request_value != val):
                        return False
                    elif op == "contains" and val not in str(request_value):
                        return False
            else:
                # Simple equality
                if request_value != expected_value:
                    return False
        
        return True


class AIRouter:
    """
    Main AI router that decides whether to use local model or external API.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.routing_mode = RoutingMode(
            self.config.get('routing_mode', 'automatic')
        )
        self.manual_selection: Optional[str] = None
        
        # Provider capabilities
        self.providers: Dict[str, ProviderCapabilities] = {}
        
        # Local model capabilities
        self.local_capabilities = ProviderCapabilities(
            provider_id="local",
            provider_name="Local Qwen Model",
            supported_task_types=list(TaskType),
            max_context_length=8192,
            supports_vision=True,
            supports_audio=True,
            supports_video=True,
            supports_speech=True,
            avg_latency_ms=100.0,
            cost_per_1k_tokens=0.0,
            reliability_score=1.0,
            is_available=False,  # Will be updated when model loads
            rate_limit_remaining=999999
        )
        
        # Custom routing rules
        self.rules: List[RoutingRule] = []
        
        # Statistics
        self.routing_history: List[Dict] = []
        self.stats = {
            'total_requests': 0,
            'local_requests': 0,
            'api_requests': 0,
            'fallbacks': 0
        }
    
    def register_provider(self, capabilities: ProviderCapabilities):
        """Register a provider with its capabilities."""
        self.providers[capabilities.provider_id] = capabilities
    
    def unregister_provider(self, provider_id: str):
        """Unregister a provider."""
        if provider_id in self.providers:
            del self.providers[provider_id]
    
    def update_provider_status(self, provider_id: str, is_available: bool):
        """Update provider availability status."""
        if provider_id in self.providers:
            self.providers[provider_id].is_available = is_available
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a custom routing rule."""
        self.rules.append(rule)
        # Sort by priority (higher first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def remove_routing_rule(self, rule_id: str):
        """Remove a routing rule."""
        self.rules = [r for r in self.rules if r.id != rule_id]
    
    def set_routing_mode(self, mode: RoutingMode):
        """Set the routing mode."""
        self.routing_mode = mode
    
    def set_manual_selection(self, target: Optional[str]):
        """Set manual target selection."""
        self.manual_selection = target
        if target:
            self.routing_mode = RoutingMode.MANUAL
    
    async def route_request(
        self,
        request: Dict[str, Any]
    ) -> RoutingDecision:
        """
        Route a request to the appropriate target.
        
        Args:
            request: Request details including task_type, privacy_level, etc.
        
        Returns:
            RoutingDecision with selected target and reasoning
        """
        self.stats['total_requests'] += 1
        
        # Check manual selection first
        if self.routing_mode == RoutingMode.MANUAL and self.manual_selection:
            return await self._route_manual(request)
        
        # Check custom rules
        rule_decision = await self._check_rules(request)
        if rule_decision:
            return rule_decision
        
        # Route based on mode
        if self.routing_mode == RoutingMode.LOCAL_ONLY:
            return await self._route_local_only(request)
        elif self.routing_mode == RoutingMode.API_ONLY:
            return await self._route_api_only(request)
        elif self.routing_mode == RoutingMode.LOCAL_FIRST:
            return await self._route_local_first(request)
        elif self.routing_mode == RoutingMode.API_FIRST:
            return await self._route_api_first(request)
        else:  # AUTOMATIC
            return await self._route_automatic(request)
    
    async def _route_manual(self, request: Dict[str, Any]) -> RoutingDecision:
        """Route based on manual selection."""
        if self.manual_selection == "local":
            return RoutingDecision(
                selected_target="local",
                target_type="local",
                reasoning="Manual selection: Local model",
                estimated_latency_ms=self.local_capabilities.avg_latency_ms,
                estimated_cost_usd=0.0
            )
        elif self.manual_selection in self.providers:
            provider = self.providers[self.manual_selection]
            return RoutingDecision(
                selected_target=self.manual_selection,
                target_type="api",
                reasoning=f"Manual selection: {provider.provider_name}",
                estimated_latency_ms=provider.avg_latency_ms,
                estimated_cost_usd=provider.cost_per_1k_tokens
            )
        else:
            # Fallback to automatic
            return await self._route_automatic(request)
    
    async def _check_rules(self, request: Dict[str, Any]) -> Optional[RoutingDecision]:
        """Check custom routing rules."""
        for rule in self.rules:
            if rule.matches(request):
                target_type = "local" if rule.target == "local" else "api"
                
                latency = 0.0
                cost = 0.0
                
                if rule.target == "local":
                    latency = self.local_capabilities.avg_latency_ms
                elif rule.target in self.providers:
                    provider = self.providers[rule.target]
                    latency = provider.avg_latency_ms
                    cost = provider.cost_per_1k_tokens
                
                return RoutingDecision(
                    selected_target=rule.target,
                    target_type=target_type,
                    reasoning=f"Rule matched: {rule.name}",
                    fallback_target=rule.fallback_target,
                    estimated_latency_ms=latency,
                    estimated_cost_usd=cost
                )
        
        return None
    
    async def _route_local_only(self, request: Dict[str, Any]) -> RoutingDecision:
        """Route to local model only."""
        if not self.local_capabilities.is_available:
            return RoutingDecision(
                selected_target="error",
                target_type="error",
                reasoning="Local model not available and local-only mode enabled",
                confidence_score=0.0
            )
        
        self.stats['local_requests'] += 1
        return RoutingDecision(
            selected_target="local",
            target_type="local",
            reasoning="Local-only mode",
            estimated_latency_ms=self.local_capabilities.avg_latency_ms,
            estimated_cost_usd=0.0
        )
    
    async def _route_api_only(self, request: Dict[str, Any]) -> RoutingDecision:
        """Route to API providers only."""
        available_providers = [
            (pid, p) for pid, p in self.providers.items()
            if p.is_available
        ]
        
        if not available_providers:
            return RoutingDecision(
                selected_target="error",
                target_type="error",
                reasoning="No API providers available",
                confidence_score=0.0
            )
        
        # Select best provider based on request
        best_provider = self._select_best_provider(available_providers, request)
        
        if best_provider:
            self.stats['api_requests'] += 1
            return RoutingDecision(
                selected_target=best_provider[0],
                target_type="api",
                reasoning="API-only mode",
                estimated_latency_ms=best_provider[1].avg_latency_ms,
                estimated_cost_usd=best_provider[1].cost_per_1k_tokens
            )
        
        return RoutingDecision(
            selected_target="error",
            target_type="error",
            reasoning="No suitable API provider found",
            confidence_score=0.0
        )
    
    async def _route_local_first(self, request: Dict[str, Any]) -> RoutingDecision:
        """Try local first, fallback to API."""
        if self.local_capabilities.is_available:
            # Check if local can handle this request
            if self._can_local_handle(request):
                self.stats['local_requests'] += 1
                return RoutingDecision(
                    selected_target="local",
                    target_type="local",
                    reasoning="Local-first mode: local capable",
                    estimated_latency_ms=self.local_capabilities.avg_latency_ms,
                    estimated_cost_usd=0.0,
                    fallback_target=self._get_best_fallback(request)
                )
        
        # Fallback to API
        return await self._route_api_only(request)
    
    async def _route_api_first(self, request: Dict[str, Any]) -> RoutingDecision:
        """Try API first, fallback to local."""
        available_providers = [
            (pid, p) for pid, p in self.providers.items()
            if p.is_available
        ]
        
        if available_providers:
            best_provider = self._select_best_provider(available_providers, request)
            if best_provider:
                self.stats['api_requests'] += 1
                return RoutingDecision(
                    selected_target=best_provider[0],
                    target_type="api",
                    reasoning="API-first mode",
                    estimated_latency_ms=best_provider[1].avg_latency_ms,
                    estimated_cost_usd=best_provider[1].cost_per_1k_tokens,
                    fallback_target="local" if self.local_capabilities.is_available else None
                )
        
        # Fallback to local
        if self.local_capabilities.is_available:
            self.stats['local_requests'] += 1
            return RoutingDecision(
                selected_target="local",
                target_type="local",
                reasoning="API-first mode: fallback to local",
                estimated_latency_ms=self.local_capabilities.avg_latency_ms,
                estimated_cost_usd=0.0
            )
        
        return RoutingDecision(
            selected_target="error",
            target_type="error",
            reasoning="No providers available",
            confidence_score=0.0
        )
    
    async def _route_automatic(self, request: Dict[str, Any]) -> RoutingDecision:
        """Automatically select best option based on multiple factors."""
        task_type = request.get('task_type', TaskType.GENERAL_CHAT.value if isinstance(request.get('task_type'), TaskType) else 'general_chat')
        privacy_level = request.get('privacy_level', 'normal')
        context_length = request.get('context_length', 0)
        latency_requirement = request.get('latency_requirement', 'normal')
        cost_sensitivity = request.get('cost_sensitivity', 'normal')
        
        candidates = []
        
        # Consider local model
        if self.local_capabilities.is_available and self._can_local_handle(request):
            local_score = self._calculate_score(
                self.local_capabilities, request, is_local=True
            )
            candidates.append(("local", self.local_capabilities, local_score))
        
        # Consider API providers
        for provider_id, provider in self.providers.items():
            if provider.is_available and self._can_provider_handle(provider, request):
                score = self._calculate_score(provider, request, is_local=False)
                candidates.append((provider_id, provider, score))
        
        if not candidates:
            return RoutingDecision(
                selected_target="error",
                target_type="error",
                reasoning="No suitable targets available",
                confidence_score=0.0
            )
        
        # Select highest scoring candidate
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        
        target_type = "local" if best[0] == "local" else "api"
        
        if target_type == "local":
            self.stats['local_requests'] += 1
        else:
            self.stats['api_requests'] += 1
        
        return RoutingDecision(
            selected_target=best[0],
            target_type=target_type,
            reasoning=self._generate_reasoning(best, request),
            fallback_target=candidates[1][0] if len(candidates) > 1 else None,
            estimated_latency_ms=best[1].avg_latency_ms,
            estimated_cost_usd=best[1].cost_per_1k_tokens if best[0] != "local" else 0.0,
            confidence_score=best[2]
        )
    
    def _can_local_handle(self, request: Dict[str, Any]) -> bool:
        """Check if local model can handle the request."""
        task_type = request.get('task_type')
        if isinstance(task_type, TaskType):
            task_type = task_type.value
        
        # Check task type support
        if task_type and TaskType(task_type) not in self.local_capabilities.supported_task_types:
            return False
        
        # Check context length
        context_length = request.get('context_length', 0)
        if context_length > self.local_capabilities.max_context_length:
            return False
        
        # Check modality
        if request.get('has_image') and not self.local_capabilities.supports_vision:
            return False
        if request.get('has_audio') and not self.local_capabilities.supports_audio:
            return False
        
        return True
    
    def _can_provider_handle(
        self,
        provider: ProviderCapabilities,
        request: Dict[str, Any]
    ) -> bool:
        """Check if provider can handle the request."""
        task_type = request.get('task_type')
        if isinstance(task_type, TaskType):
            task_type = task_type.value
        
        if task_type and TaskType(task_type) not in provider.supported_task_types:
            return False
        
        context_length = request.get('context_length', 0)
        if context_length > provider.max_context_length:
            return False
        
        return True
    
    def _calculate_score(
        self,
        capabilities: ProviderCapabilities,
        request: Dict[str, Any],
        is_local: bool
    ) -> float:
        """Calculate suitability score for a target."""
        score = 0.5  # Base score
        
        task_type = request.get('task_type', 'general_chat')
        if isinstance(task_type, TaskType):
            task_type = task_type.value
        privacy_level = request.get('privacy_level', 'normal')
        context_length = request.get('context_length', 0)
        cost_sensitivity = request.get('cost_sensitivity', 'normal')
        
        # Privacy boost for local
        if is_local and privacy_level == 'high':
            score += 0.3
        
        # Cost boost for local
        if is_local and cost_sensitivity == 'high':
            score += 0.2
        
        # Capability match
        if task_type in [t.value for t in capabilities.supported_task_types]:
            score += 0.1
        
        # Context length fit
        if capabilities.max_context_length >= context_length * 1.2:
            score += 0.1
        
        # Reliability factor
        score *= capabilities.reliability_score
        
        # Latency consideration
        if request.get('latency_requirement') == 'low':
            if capabilities.avg_latency_ms < 100:
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _select_best_provider(
        self,
        providers: List[Tuple[str, ProviderCapabilities]],
        request: Dict[str, Any]
    ) -> Optional[Tuple[str, ProviderCapabilities]]:
        """Select the best provider from available options."""
        if not providers:
            return None
        
        scored = [
            (pid, p, self._calculate_score(p, request, is_local=False))
            for pid, p in providers
        ]
        
        scored.sort(key=lambda x: x[2], reverse=True)
        return (scored[0][0], scored[0][1]) if scored else None
    
    def _get_best_fallback(self, request: Dict[str, Any]) -> Optional[str]:
        """Get best fallback target."""
        for provider_id, provider in self.providers.items():
            if provider.is_available and self._can_provider_handle(provider, request):
                return provider_id
        return None
    
    def _generate_reasoning(
        self,
        candidate: Tuple[str, Any, float],
        request: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for the decision."""
        target, capabilities, score = candidate
        reasons = []
        
        if target == "local":
            reasons.append("local processing")
            reasons.append(f"low latency ({capabilities.avg_latency_ms:.0f}ms)")
            reasons.append("no cost")
        else:
            reasons.append(f"{capabilities.provider_name}")
            reasons.append(f"reliability: {capabilities.reliability_score:.0%}")
        
        task_type = request.get('task_type', 'general')
        reasons.append(f"optimized for {task_type}")
        
        return f"Automatic routing: {'; '.join(reasons)}"
    
    def record_outcome(
        self,
        decision: RoutingDecision,
        actual_latency_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record the outcome of a routing decision for learning."""
        self.routing_history.append({
            'decision': {
                'target': decision.selected_target,
                'target_type': decision.target_type,
                'reasoning': decision.reasoning
            },
            'actual_latency_ms': actual_latency_ms,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update provider stats if API
        if decision.target_type == "api" and decision.selected_target in self.providers:
            provider = self.providers[decision.selected_target]
            # Exponential moving average for latency
            provider.avg_latency_ms = (
                0.9 * provider.avg_latency_ms + 0.1 * actual_latency_ms
            )
            if success:
                provider.rate_limit_remaining = max(0, provider.rate_limit_remaining - 1)
        
        if error:
            self.stats['fallbacks'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            **self.stats,
            'routing_mode': self.routing_mode.value,
            'registered_providers': len(self.providers),
            'active_rules': len([r for r in self.rules if r.is_enabled]),
            'local_available': self.local_capabilities.is_available
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert router state to dictionary."""
        return {
            'routing_mode': self.routing_mode.value,
            'manual_selection': self.manual_selection,
            'providers': {
                pid: {
                    'name': p.provider_name,
                    'is_available': p.is_available,
                    'avg_latency_ms': p.avg_latency_ms,
                    'cost_per_1k_tokens': p.cost_per_1k_tokens
                }
                for pid, p in self.providers.items()
            },
            'rules': [
                {
                    'id': r.id,
                    'name': r.name,
                    'priority': r.priority,
                    'target': r.target,
                    'is_enabled': r.is_enabled
                }
                for r in self.rules
            ],
            'stats': self.stats
        }
