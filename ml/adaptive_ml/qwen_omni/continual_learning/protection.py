"""
Parameter Protection for Qwen2.5-Omni-3B.
Implements EWC, MAS, SI, and Fisher Information methods to protect important parameters.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    ProtectionStats,
    TrainingStats,
)

logger = logging.getLogger(__name__)


@dataclass
class ProtectionConfig:
    """Configuration for parameter protection."""
    use_ewc: bool = True
    ewc_lambda: float = 0.1
    
    use_mas: bool = True
    mas_lambda: float = 0.1
    
    use_si: bool = True
    si_lambda: float = 0.1
    
    use_fisher: bool = False
    fisher_lambda: float = 0.1
    
    # Modality-specific settings
    modality_lambdas: Dict[ModalityType, Dict[str, float]] = field(default_factory=dict)
    
    # Layer-specific settings
    layer_importance: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize default modality settings
        if not self.modality_lambdas:
            self.modality_lambdas = {
                ModalityType.TEXT: {"ewc": 0.1, "mas": 0.1, "si": 0.1},
                ModalityType.VISION: {"ewc": 0.15, "mas": 0.15, "si": 0.15},
                ModalityType.AUDIO: {"ewc": 0.1, "mas": 0.1, "si": 0.1},
                ModalityType.VIDEO: {"ewc": 0.2, "mas": 0.2, "si": 0.2},
                ModalityType.SPEECH: {"ewc": 0.1, "mas": 0.1, "si": 0.1},
            }


class ParameterProtection:
    """
    Base class for parameter protection methods.
    Provides common functionality for EWC, MAS, SI, etc.
    """
    
    def __init__(self, config: Optional[ProtectionConfig] = None):
        self.config = config or ProtectionConfig()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Statistics
        self._stats = ProtectionStats()
        
    def compute_protection_loss(
        self,
        model: PreTrainedModel,
        modality: Optional[ModalityType] = None,
    ) -> Tuple[torch.Tensor, ProtectionStats]:
        """
        Compute parameter protection loss.
        
        Args:
            model: The model being trained
            modality: Optional modality for modality-specific settings
            
        Returns:
            Tuple of (protection_loss, stats)
        """
        raise NotImplementedError("Subclasses must implement compute_protection_loss")
    
    def update_importance(
        self,
        model: PreTrainedModel,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Update parameter importance estimates.
        
        Args:
            model: The model being trained
            gradients: Optional gradients for importance calculation
            **kwargs: Additional arguments
        """
        raise NotImplementedError("Subclasses must implement update_importance")
    
    def get_importance(self, param_name: str) -> float:
        """Get importance for a specific parameter."""
        raise NotImplementedError("Subclasses must implement get_importance")
    
    def get_stats(self) -> ProtectionStats:
        """Get current protection statistics."""
        return self._stats
    
    def update_stats(self, stats: ProtectionStats) -> None:
        """Update protection statistics."""
        self._stats = stats


class EWCTrainer(ParameterProtection):
    """
    Elastic Weight Consolidation (EWC) for parameter protection.
    Uses Fisher Information Matrix to estimate parameter importance.
    """
    
    def __init__(self, config: Optional[ProtectionConfig] = None):
        super().__init__(config)
        
        # Fisher Information Matrix (diagonal approximation)
        self._fisher: Dict[str, torch.Tensor] = {}
        self._optimal_params: Dict[str, torch.Tensor] = {}
        
        # Store previous model state
        self._previous_state: Dict[str, torch.Tensor] = {}
        
    def update_importance(
        self,
        model: PreTrainedModel,
        dataloader: Optional[Any] = None,
        num_samples: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Update Fisher Information Matrix.
        
        Args:
            model: The model to compute Fisher for
            dataloader: Optional dataloader for Fisher estimation
            num_samples: Number of samples for Fisher estimation
            **kwargs: Additional arguments
        """
        # Store current parameters as optimal
        self._optimal_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._optimal_params[name] = param.detach().clone()
        
        # Compute Fisher Information (diagonal approximation)
        if dataloader is not None:
            self._compute_fisher_diagonal(model, dataloader, num_samples)
        else:
            # Initialize Fisher with zeros if no dataloader provided
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self._fisher[name] = torch.zeros_like(param)
        
        # Update stats
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        protected_params = sum(1 for name in self._fisher if self._fisher[name].sum() > 0)
        
        self._stats = ProtectionStats(
            ewc_lambda=self.config.ewc_lambda,
            mas_lambda=0.0,
            si_lambda=0.0,
            protected_parameters=protected_params,
            total_parameters=total_params,
            protection_ratio=protected_params / max(total_params, 1),
        )
        
        logger.info(f"Updated EWC importance. Protected {protected_params}/{total_params} parameters")
    
    def _compute_fisher_diagonal(
        self,
        model: PreTrainedModel,
        dataloader: Any,
        num_samples: int,
    ) -> None:
        """
        Compute diagonal Fisher Information Matrix.
        
        Args:
            model: The model
            dataloader: Dataloader with representative data
            num_samples: Number of samples to use
        """
        model.eval()
        model.to(self._device)
        
        # Initialize Fisher
        self._fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._fisher[name] = torch.zeros_like(param)
        
        # Compute Fisher using samples
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            # Move batch to device
            batch = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Compute Hessian diagonal (Fisher) for each parameter
            for name, param in model.named_parameters():
                if param.requires_grad and name in self._fisher:
                    # Compute gradient of log probability w.r.t. parameter
                    grad_outputs = torch.autograd.grad(
                        log_probs.sum(),
                        param,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    
                    # Square the gradient to get Fisher diagonal
                    fisher_update = grad_outputs ** 2
                    self._fisher[name] += fisher_update
            
            sample_count += 1
        
        # Average over samples
        if sample_count > 0:
            for name in self._fisher:
                self._fisher[name] /= sample_count
        
        model.train()
    
    def compute_protection_loss(
        self,
        model: PreTrainedModel,
        modality: Optional[ModalityType] = None,
    ) -> Tuple[torch.Tensor, ProtectionStats]:
        """
        Compute EWC protection loss.
        
        Args:
            model: The model being trained
            modality: Optional modality for modality-specific settings
            
        Returns:
            Tuple of (protection_loss, stats)
        """
        if not self._fisher or not self._optimal_params:
            return torch.tensor(0.0, device=self._device), self._stats
        
        # Get modality-specific lambda
        if modality and modality in self.config.modality_lambdas:
            lambda_ = self.config.modality_lambdas[modality].get("ewc", self.config.ewc_lambda)
        else:
            lambda_ = self.config.ewc_lambda
        
        # Compute penalty
        penalty = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._fisher and name in self._optimal_params:
                # Difference from optimal parameters
                diff = param - self._optimal_params[name]
                
                # Weighted by Fisher
                fisher_weighted_diff = self._fisher[name] * (diff ** 2)
                penalty += fisher_weighted_diff.sum()
        
        protection_loss = lambda_ * penalty
        
        # Update stats
        self._stats.ewc_lambda = lambda_
        
        return protection_loss, self._stats


class MASTrainer(ParameterProtection):
    """
    Memory Aware Synapses (MAS) for parameter protection.
    Uses gradient magnitude to estimate parameter importance.
    """
    
    def __init__(self, config: Optional[ProtectionConfig] = None):
        super().__init__(config)
        
        # Parameter importance
        self._importance: Dict[str, torch.Tensor] = {}
        self._optimal_params: Dict[str, torch.Tensor] = {}
        
    def update_importance(
        self,
        model: PreTrainedModel,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Update parameter importance using gradient magnitudes.
        
        Args:
            model: The model being trained
            gradients: Optional pre-computed gradients
            **kwargs: Additional arguments
        """
        # Store current parameters as optimal
        self._optimal_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._optimal_params[name] = param.detach().clone()
        
        # Compute importance as |gradient * parameter|
        if gradients is None:
            # Use current gradients if available
            gradients = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradients[name] = param.grad.detach().clone()
        
        self._importance = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in gradients:
                    # MAS importance: |∇L/∇θ| * |θ|
                    importance = torch.abs(gradients[name]) * torch.abs(param)
                else:
                    # Initialize with parameter magnitude
                    importance = torch.abs(param)
                
                self._importance[name] = importance
        
        # Update stats
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        protected_params = sum(1 for name in self._importance if self._importance[name].sum() > 0)
        
        self._stats = ProtectionStats(
            ewc_lambda=0.0,
            mas_lambda=self.config.mas_lambda,
            si_lambda=0.0,
            protected_parameters=protected_params,
            total_parameters=total_params,
            protection_ratio=protected_params / max(total_params, 1),
        )
        
        logger.info(f"Updated MAS importance. Protected {protected_params}/{total_params} parameters")
    
    def compute_protection_loss(
        self,
        model: PreTrainedModel,
        modality: Optional[ModalityType] = None,
    ) -> Tuple[torch.Tensor, ProtectionStats]:
        """
        Compute MAS protection loss.
        
        Args:
            model: The model being trained
            modality: Optional modality for modality-specific settings
            
        Returns:
            Tuple of (protection_loss, stats)
        """
        if not self._importance or not self._optimal_params:
            return torch.tensor(0.0, device=self._device), self._stats
        
        # Get modality-specific lambda
        if modality and modality in self.config.modality_lambdas:
            lambda_ = self.config.modality_lambdas[modality].get("mas", self.config.mas_lambda)
        else:
            lambda_ = self.config.mas_lambda
        
        # Compute penalty
        penalty = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._importance and name in self._optimal_params:
                # Difference from optimal parameters
                diff = param - self._optimal_params[name]
                
                # Weighted by importance
                importance_weighted_diff = self._importance[name] * (diff ** 2)
                penalty += importance_weighted_diff.sum()
        
        protection_loss = lambda_ * penalty
        
        # Update stats
        self._stats.mas_lambda = lambda_
        
        return protection_loss, self._stats


class SITrainer(ParameterProtection):
    """
    Synaptic Intelligence (SI) for parameter protection.
    Uses change in gradient direction to estimate parameter importance.
    """
    
    def __init__(self, config: Optional[ProtectionConfig] = None):
        super().__init__(config)
        
        # Parameter importance (accumulated gradient changes)
        self._importance: Dict[str, torch.Tensor] = {}
        self._previous_gradients: Dict[str, torch.Tensor] = {}
        self._optimal_params: Dict[str, torch.Tensor] = {}
        
    def update_importance(
        self,
        model: PreTrainedModel,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Update parameter importance using gradient changes.
        
        Args:
            model: The model being trained
            gradients: Current gradients
            **kwargs: Additional arguments
        """
        # Store current parameters as optimal
        self._optimal_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._optimal_params[name] = param.detach().clone()
        
        # Get current gradients
        if gradients is None:
            gradients = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradients[name] = param.grad.detach().clone()
        
        # Update importance based on gradient changes
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in gradients and name in self._previous_gradients:
                    # SI importance: Σ (∇L_new/∇θ - ∇L_old/∇θ) * θ
                    grad_change = gradients[name] - self._previous_gradients[name]
                    importance_update = grad_change * param
                    
                    if name in self._importance:
                        self._importance[name] += importance_update
                    else:
                        self._importance[name] = importance_update
                elif name in gradients:
                    # First time seeing this parameter
                    if name not in self._importance:
                        self._importance[name] = torch.zeros_like(param)
        
        # Store current gradients as previous
        self._previous_gradients = {}
        for name, grad in gradients.items():
            self._previous_gradients[name] = grad.clone()
        
        # Update stats
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        protected_params = sum(1 for name in self._importance if self._importance[name].abs().sum() > 0)
        
        self._stats = ProtectionStats(
            ewc_lambda=0.0,
            mas_lambda=0.0,
            si_lambda=self.config.si_lambda,
            protected_parameters=protected_params,
            total_parameters=total_params,
            protection_ratio=protected_params / max(total_params, 1),
        )
        
        logger.info(f"Updated SI importance. Protected {protected_params}/{total_params} parameters")
    
    def compute_protection_loss(
        self,
        model: PreTrainedModel,
        modality: Optional[ModalityType] = None,
    ) -> Tuple[torch.Tensor, ProtectionStats]:
        """
        Compute SI protection loss.
        
        Args:
            model: The model being trained
            modality: Optional modality for modality-specific settings
            
        Returns:
            Tuple of (protection_loss, stats)
        """
        if not self._importance or not self._optimal_params:
            return torch.tensor(0.0, device=self._device), self._stats
        
        # Get modality-specific lambda
        if modality and modality in self.config.modality_lambdas:
            lambda_ = self.config.modality_lambdas[modality].get("si", self.config.si_lambda)
        else:
            lambda_ = self.config.si_lambda
        
        # Compute penalty
        penalty = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._importance and name in self._optimal_params:
                # Difference from optimal parameters
                diff = param - self._optimal_params[name]
                
                # Weighted by importance (absolute value)
                importance_weighted_diff = torch.abs(self._importance[name]) * (diff ** 2)
                penalty += importance_weighted_diff.sum()
        
        protection_loss = lambda_ * penalty
        
        # Update stats
        self._stats.si_lambda = lambda_
        
        return protection_loss, self._stats


class CombinedParameterProtection:
    """
    Combined parameter protection using multiple methods.
    """
    
    def __init__(
        self,
        use_ewc: bool = True,
        use_mas: bool = True,
        use_si: bool = True,
        config: Optional[ProtectionConfig] = None,
    ):
        self.use_ewc = use_ewc
        self.use_mas = use_mas
        self.use_si = use_si
        self.config = config or ProtectionConfig()
        
        # Initialize individual methods
        self.ewc = EWCTrainer(config) if use_ewc else None
        self.mas = MASTrainer(config) if use_mas else None
        self.si = SITrainer(config) if use_si else None
        
        # Combined statistics
        self._stats = ProtectionStats()
        
    def update_importance(
        self,
        model: PreTrainedModel,
        dataloader: Optional[Any] = None,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        """Update importance for all enabled methods."""
        if self.ewc:
            self.ewc.update_importance(model, dataloader, **kwargs)
        if self.mas:
            self.mas.update_importance(model, gradients, **kwargs)
        if self.si:
            self.si.update_importance(model, gradients, **kwargs)
        
        # Update combined stats
        self._update_combined_stats()
    
    def compute_protection_loss(
        self,
        model: PreTrainedModel,
        modality: Optional[ModalityType] = None,
    ) -> Tuple[torch.Tensor, ProtectionStats]:
        """
        Compute combined protection loss.
        
        Args:
            model: The model being trained
            modality: Optional modality for modality-specific settings
            
        Returns:
            Tuple of (protection_loss, stats)
        """
        total_loss = 0.0
        
        if self.ewc:
            ewc_loss, _ = self.ewc.compute_protection_loss(model, modality)
            total_loss += ewc_loss
        
        if self.mas:
            mas_loss, _ = self.mas.compute_protection_loss(model, modality)
            total_loss += mas_loss
        
        if self.si:
            si_loss, _ = self.si.compute_protection_loss(model, modality)
            total_loss += si_loss
        
        # Update combined stats
        self._update_combined_stats()
        
        return total_loss, self._stats
    
    def _update_combined_stats(self) -> None:
        """Update combined statistics."""
        total_params = 0
        protected_params = 0
        
        if self.ewc:
            total_params += self.ewc._stats.total_parameters
            protected_params += self.ewc._stats.protected_parameters
        if self.mas:
            total_params += self.mas._stats.total_parameters
            protected_params += self.mas._stats.protected_parameters
        if self.si:
            total_params += self.si._stats.total_parameters
            protected_params += self.si._stats.protected_parameters
        
        # Calculate lambda values
        ewc_lambda = self.ewc._stats.ewc_lambda if self.ewc else 0.0
        mas_lambda = self.mas._stats.mas_lambda if self.mas else 0.0
        si_lambda = self.si._stats.si_lambda if self.si else 0.0
        
        self._stats = ProtectionStats(
            ewc_lambda=ewc_lambda,
            mas_lambda=mas_lambda,
            si_lambda=si_lambda,
            protected_parameters=protected_params,
            total_parameters=total_params,
            protection_ratio=protected_params / max(total_params, 1),
        )
    
    def get_stats(self) -> ProtectionStats:
        """Get combined protection statistics."""
        return self._stats
