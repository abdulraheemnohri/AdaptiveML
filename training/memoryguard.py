"""
MemoryGuard - Anti-Catastrophic Forgetting Engine

Protects previously learned capabilities during training.
Monitors: Reasoning, Math, Coding, General Knowledge, Languages, Vision, Audio, Speech, Safety
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from datetime import datetime


class CapabilityType(Enum):
    """Types of capabilities to protect."""
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    CODING = "coding"
    GENERAL_KNOWLEDGE = "general_knowledge"
    LANGUAGE_EN = "language_en"
    LANGUAGE_URDU = "language_urdu"
    LANGUAGE_MULTILINGUAL = "language_multilingual"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    SPEECH = "speech"
    SAFETY = "safety"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass
class CapabilityBaseline:
    """Baseline measurements for a capability."""
    capability: CapabilityType
    accuracy: float
    loss: float
    response_quality: float
    latency_ms: float
    confidence_score: float
    test_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    benchmark_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ForgettingMetrics:
    """Metrics measuring forgetting after training."""
    capability: CapabilityType
    baseline_accuracy: float
    new_accuracy: float
    accuracy_delta: float
    baseline_loss: float
    new_loss: float
    loss_delta: float
    forgetting_score: float  # 0 = no forgetting, 1 = complete forgetting
    is_acceptable: bool
    threshold_exceeded: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MemoryGuardConfig:
    """Configuration for MemoryGuard."""
    forgetting_threshold: float = 0.02  # 2% maximum acceptable forgetting
    regression_threshold: float = 0.01  # 1% maximum regression
    quality_gate_threshold: float = 0.90  # 90% minimum quality
    safety_gate_threshold: float = 0.95  # 95% minimum safety
    protected_capabilities: List[CapabilityType] = field(default_factory=list)
    replay_ratio: float = 0.2
    distillation_weight: float = 0.5
    ewc_strength: float = 1000.0
    
    def __post_init__(self):
        if not self.protected_capabilities:
            self.protected_capabilities = list(CapabilityType)


class MemoryGuard:
    """
    MemoryGuard - Protects previously learned capabilities during training.
    
    Implements anti-catastrophic forgetting through:
    1. Baseline creation before training
    2. Continuous monitoring during training
    3. Post-training evaluation against baselines
    4. Automatic rejection if forgetting exceeds thresholds
    """
    
    def __init__(self, config: MemoryGuardConfig):
        self.config = config
        self.baselines: Dict[CapabilityType, CapabilityBaseline] = {}
        self.post_training_metrics: Dict[CapabilityType, ForgettingMetrics] = {}
        self.protected_data: Dict[CapabilityType, List[Dict]] = {}
        self.history: List[Dict] = []
    
    def create_baseline(
        self,
        capability: CapabilityType,
        model: nn.Module,
        test_data: List[Dict],
        eval_fn
    ) -> CapabilityBaseline:
        """
        Create a baseline measurement for a capability.
        
        Args:
            capability: The capability to measure
            model: The model to evaluate
            test_data: Test data for this capability
            eval_fn: Evaluation function that returns metrics
        
        Returns:
            CapabilityBaseline with measured values
        """
        model.eval()
        
        # Run evaluation
        results = eval_fn(model, test_data)
        
        baseline = CapabilityBaseline(
            capability=capability,
            accuracy=results.get('accuracy', 0.0),
            loss=results.get('loss', float('inf')),
            response_quality=results.get('quality_score', 0.0),
            latency_ms=results.get('avg_latency_ms', 0.0),
            confidence_score=results.get('confidence', 0.0),
            test_samples=len(test_data),
            benchmark_scores=results.get('benchmark_scores', {})
        )
        
        self.baselines[capability] = baseline
        
        # Store protected data samples for this capability
        self.protected_data[capability] = self._select_protected_samples(test_data)
        
        return baseline
    
    def create_all_baselines(
        self,
        model: nn.Module,
        test_datasets: Dict[CapabilityType, List[Dict]],
        eval_fn
    ) -> Dict[CapabilityType, CapabilityBaseline]:
        """Create baselines for all protected capabilities."""
        baselines = {}
        
        for capability in self.config.protected_capabilities:
            if capability in test_datasets:
                baseline = self.create_baseline(
                    capability, model, test_datasets[capability], eval_fn
                )
                baselines[capability] = baseline
        
        return baselines
    
    def evaluate_after_training(
        self,
        model: nn.Module,
        eval_fn
    ) -> Dict[CapabilityType, ForgettingMetrics]:
        """
        Evaluate model after training and compute forgetting metrics.
        
        Args:
            model: The trained model to evaluate
            eval_fn: Evaluation function
        
        Returns:
            Dictionary of forgetting metrics per capability
        """
        metrics = {}
        
        for capability, baseline in self.baselines.items():
            # Get protected test data
            test_data = self.protected_data.get(capability, [])
            
            if not test_data:
                continue
            
            # Run evaluation on new model
            results = eval_fn(model, test_data)
            
            new_accuracy = results.get('accuracy', 0.0)
            new_loss = results.get('loss', float('inf'))
            
            # Compute deltas
            accuracy_delta = new_accuracy - baseline.accuracy
            loss_delta = new_loss - baseline.loss
            
            # Compute forgetting score (positive = forgetting occurred)
            # Negative accuracy delta means performance decreased
            forgetting_score = max(0, -accuracy_delta / baseline.accuracy) if baseline.accuracy > 0 else 0
            
            # Check if within acceptable thresholds
            threshold_exceeded = forgetting_score > self.config.forgetting_threshold
            is_acceptable = not threshold_exceeded
            
            metric = ForgettingMetrics(
                capability=capability,
                baseline_accuracy=baseline.accuracy,
                new_accuracy=new_accuracy,
                accuracy_delta=accuracy_delta,
                baseline_loss=baseline.loss,
                new_loss=new_loss,
                loss_delta=loss_delta,
                forgetting_score=forgetting_score,
                is_acceptable=is_acceptable,
                threshold_exceeded=threshold_exceeded
            )
            
            metrics[capability] = metric
            self.post_training_metrics[capability] = metric
        
        # Record in history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': {k: v.forgetting_score for k, v in metrics.items()},
            'all_acceptable': all(m.is_acceptable for m in metrics.values())
        })
        
        return metrics
    
    def check_candidate_approval(self) -> Tuple[bool, Dict]:
        """
        Check if candidate model passes all gates.
        
        Returns:
            Tuple of (is_approved, details)
        """
        if not self.post_training_metrics:
            return False, {'error': 'No evaluation metrics available'}
        
        # Check forgetting thresholds
        forgetting_violations = []
        for capability, metric in self.post_training_metrics.items():
            if metric.threshold_exceeded:
                forgetting_violations.append({
                    'capability': capability.value,
                    'forgetting_score': metric.forgetting_score,
                    'threshold': self.config.forgetting_threshold
                })
        
        # Check safety gate
        safety_metric = self.post_training_metrics.get(CapabilityType.SAFETY)
        safety_violated = (
            safety_metric is not None and 
            safety_metric.new_accuracy < self.config.safety_gate_threshold
        )
        
        # Check quality gate
        avg_quality = np.mean([
            m.new_accuracy for m in self.post_training_metrics.values()
        ])
        quality_violated = avg_quality < self.config.quality_gate_threshold
        
        is_approved = (
            len(forgetting_violations) == 0 and
            not safety_violated and
            not quality_violated
        )
        
        details = {
            'is_approved': is_approved,
            'forgetting_violations': forgetting_violations,
            'safety_violated': safety_violated,
            'quality_violated': quality_violated,
            'average_quality': avg_quality,
            'overall_forgetting_score': np.mean([
                m.forgetting_score for m in self.post_training_metrics.values()
            ])
        }
        
        return is_approved, details
    
    def get_recovery_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for recovery if forgetting is detected.
        
        Returns:
            Recommendations dictionary
        """
        recommendations = {
            'increase_replay': False,
            'increase_distillation': False,
            'increase_ewc': False,
            'add_protected_data': [],
            'retrain_required': False
        }
        
        for capability, metric in self.post_training_metrics.items():
            if metric.threshold_exceeded:
                recommendations['retrain_required'] = True
                
                # Recommend specific actions based on which capability forgot
                if capability in [CapabilityType.REASONING, CapabilityType.MATHEMATICS]:
                    recommendations['increase_replay'] = True
                    recommendations['add_protected_data'].append(capability.value)
                
                elif capability in [CapabilityType.CODING, CapabilityType.INSTRUCTION_FOLLOWING]:
                    recommendations['increase_distillation'] = True
                    recommendations['add_protected_data'].append(capability.value)
                
                elif capability == CapabilityType.SAFETY:
                    recommendations['increase_ewc'] = True
                    recommendations['add_protected_data'].append(capability.value)
                
                else:
                    recommendations['increase_replay'] = True
        
        # Calculate recommended parameter adjustments
        if recommendations['increase_replay']:
            recommendations['new_replay_ratio'] = min(
                0.5, self.config.replay_ratio * 1.5
            )
        
        if recommendations['increase_distillation']:
            recommendations['new_distillation_weight'] = min(
                0.8, self.config.distillation_weight * 1.3
            )
        
        if recommendations['increase_ewc']:
            recommendations['new_ewc_strength'] = min(
                5000.0, self.config.ewc_strength * 1.5
            )
        
        return recommendations
    
    def _select_protected_samples(
        self,
        data: List[Dict],
        n_samples: int = 100
    ) -> List[Dict]:
        """Select representative samples for protection."""
        if len(data) <= n_samples:
            return data
        
        # Stratified sampling based on difficulty/quality if available
        # Otherwise random sample
        indices = np.random.choice(len(data), n_samples, replace=False)
        return [data[i] for i in indices]
    
    def get_protected_capability_data(self) -> Dict[CapabilityType, List[Dict]]:
        """Get all protected capability data for training."""
        all_data = []
        
        for capability, samples in self.protected_data.items():
            for sample in samples:
                sample_copy = sample.copy()
                sample_copy['capability_type'] = capability.value
                sample_copy['is_protected'] = True
                all_data.append(sample_copy)
        
        return all_data
    
    def save_state(self, path: str):
        """Save MemoryGuard state to disk."""
        state = {
            'config': {
                'forgetting_threshold': self.config.forgetting_threshold,
                'regression_threshold': self.config.regression_threshold,
                'quality_gate_threshold': self.config.quality_gate_threshold,
                'safety_gate_threshold': self.config.safety_gate_threshold,
                'protected_capabilities': [c.value for c in self.config.protected_capabilities],
                'replay_ratio': self.config.replay_ratio,
                'distillation_weight': self.config.distillation_weight,
                'ewc_strength': self.config.ewc_strength
            },
            'baselines': {},
            'history': self.history
        }
        
        # Serialize baselines
        for cap, baseline in self.baselines.items():
            state['baselines'][cap.value] = {
                'accuracy': baseline.accuracy,
                'loss': baseline.loss,
                'response_quality': baseline.response_quality,
                'latency_ms': baseline.latency_ms,
                'confidence_score': baseline.confidence_score,
                'test_samples': baseline.test_samples,
                'timestamp': baseline.timestamp,
                'benchmark_scores': baseline.benchmark_scores
            }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str):
        """Load MemoryGuard state from disk."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        config_data = state.get('config', {})
        self.config = MemoryGuardConfig(
            forgetting_threshold=config_data.get('forgetting_threshold', 0.02),
            regression_threshold=config_data.get('regression_threshold', 0.01),
            quality_gate_threshold=config_data.get('quality_gate_threshold', 0.90),
            safety_gate_threshold=config_data.get('safety_gate_threshold', 0.95),
            protected_capabilities=[
                CapabilityType(c) for c in config_data.get('protected_capabilities', [])
            ],
            replay_ratio=config_data.get('replay_ratio', 0.2),
            distillation_weight=config_data.get('distillation_weight', 0.5),
            ewc_strength=config_data.get('ewc_strength', 1000.0)
        )
        
        # Load baselines
        for cap_str, baseline_data in state.get('baselines', {}).items():
            capability = CapabilityType(cap_str)
            self.baselines[capability] = CapabilityBaseline(
                capability=capability,
                accuracy=baseline_data['accuracy'],
                loss=baseline_data['loss'],
                response_quality=baseline_data['response_quality'],
                latency_ms=baseline_data['latency_ms'],
                confidence_score=baseline_data['confidence_score'],
                test_samples=baseline_data['test_samples'],
                timestamp=baseline_data['timestamp'],
                benchmark_scores=baseline_data.get('benchmark_scores', {})
            )
        
        self.history = state.get('history', [])
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of MemoryGuard status."""
        if not self.baselines:
            return {'status': 'no_baselines'}
        
        report = {
            'status': 'active',
            'total_capabilities': len(self.baselines),
            'capabilities': {}
        }
        
        for capability, baseline in self.baselines.items():
            metric = self.post_training_metrics.get(capability)
            
            cap_report = {
                'baseline_accuracy': baseline.accuracy,
                'baseline_loss': baseline.loss,
                'test_samples': baseline.test_samples
            }
            
            if metric:
                cap_report.update({
                    'current_accuracy': metric.new_accuracy,
                    'accuracy_delta': metric.accuracy_delta,
                    'forgetting_score': metric.forgetting_score,
                    'is_acceptable': metric.is_acceptable
                })
            
            report['capabilities'][capability.value] = cap_report
        
        # Overall status
        if self.post_training_metrics:
            all_acceptable = all(
                m.is_acceptable for m in self.post_training_metrics.values()
            )
            report['overall_status'] = 'passed' if all_acceptable else 'failed'
            report['average_forgetting'] = np.mean([
                m.forgetting_score for m in self.post_training_metrics.values()
            ])
        
        return report
