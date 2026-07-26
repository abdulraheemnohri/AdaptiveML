"""
Synaptic Intelligence (SI) for Adaptive ML Framework.
Protects parameters based on their contribution to the change in loss across tasks.

SI computes parameter importance as:
    Ω = Σ (∇L_new/∇θ - ∇L_old/∇θ) * θ
    
This measures the cumulative change in gradients across tasks, weighted by
parameter values. Parameters that consistently change their gradients across
tasks are considered more important.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import ParameterImportanceMethod


@dataclass
class SIStats:
    """Statistics for SI."""

    num_parameters: int = 0
    num_important_parameters: int = 0
    mean_importance: float = 0.0
    max_importance: float = 0.0
    importance_distribution: Dict[str, float] = field(default_factory=dict)


class SI:
    """
    Synaptic Intelligence (SI) for continual learning.
    
    SI protects parameters based on their contribution to the change in loss
    across tasks. The importance of a parameter θ_i is given by:
        Ω_i = Σ (∇L_new/∇θ_i - ∇L_old/∇θ_i) * θ_i
    
    This captures the cumulative change in gradients across tasks, weighted
    by parameter values. Parameters that consistently change their gradients
    across tasks are considered more important.
    
    Reference:
        Zenke et al., "Continual Learning with Deep Neural Networks without
        Forgetting", ICML 2017.
        https://arxiv.org/abs/1701.08691
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_si: float = 100.0,
        config: Optional[AdaptiveMLConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize SI.
        
        Args:
            model: The neural network model
            lambda_si: SI regularization strength
            config: AdaptiveMLConfig instance
            device: Device to run on (defaults to model's device)
        """
        self.model = model
        self.lambda_si = lambda_si
        self.config = config or AdaptiveMLConfig()
        self.device = device or next(model.parameters()).device
        self.method = ParameterImportanceMethod.SI
        
        # Parameter importance dictionary: name -> importance tensor
        self.importance: Dict[str, torch.Tensor] = {}
        
        # Saved parameters (before update)
        self.saved_params: Dict[str, torch.Tensor] = {}
        
        # Previous gradients (for SI computation)
        self.prev_grads: Dict[str, torch.Tensor] = {}
        
        # Initialize importance
        self._init_importance()

    def _init_importance(self) -> None:
        """Initialize parameter importance to zero."""
        self.importance = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.importance[name] = torch.zeros_like(param.data).to(self.device)

    def update_importance(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
    ) -> None:
        """
        Update parameter importance using data from dataloader.
        
        SI importance: Ω += (∇L/∇θ - ∇L_prev/∇θ) * θ
        
        Args:
            dataloader: DataLoader with training data
            num_batches: Number of batches to use (None = all)
        """
        # Set model to training mode
        self.model.train()
        
        # Enable gradients
        was_training = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        # Number of samples
        N = 0
        
        # Store current gradients as previous for SI computation
        self._store_current_gradients()
        
        for batch_idx, batch in enumerate(dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            # Get batch
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            else:
                inputs = batch.get("input_ids", batch.get("inputs"))
                targets = batch.get("labels", batch.get("targets"))
            
            # Move to device and ensure inputs require grad
            inputs = inputs.to(self.device)
            if inputs.requires_grad is False:
                inputs = inputs.detach().requires_grad_(True)
            
            if targets is not None:
                targets = targets.to(self.device)
            
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Get logits
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            
            # Compute loss
            if targets is not None:
                if isinstance(logits, torch.Tensor) and isinstance(targets, torch.Tensor):
                    # Classification loss
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                else:
                    # For other cases, use MSE
                    loss = torch.nn.functional.mse_loss(logits, targets)
            else:
                # Unsupervised: use negative log likelihood
                probs = torch.softmax(logits, dim=-1)
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Update importance: Ω += (∇L/∇θ - ∇L_prev/∇θ) * θ
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.importance:
                    if param.grad is not None:
                        # SI importance: (current_grad - prev_grad) * parameter
                        grad_diff = param.grad
                        if name in self.prev_grads:
                            grad_diff = param.grad - self.prev_grads[name]
                        
                        # SI update: Ω += grad_diff * θ
                        self.importance[name] += grad_diff * param.data
            
            N += inputs.shape[0]
        
        # Average over samples
        if N > 0:
            for name in self.importance:
                self.importance[name] /= N
        
        # Save current parameters and gradients
        self._save_parameters()
        self._store_current_gradients()
        
        # Restore gradient state
        torch.set_grad_enabled(was_training)

    def _store_current_gradients(self) -> None:
        """Store current gradients for next SI update."""
        self.prev_grads = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.prev_grads[name] = param.grad.clone().detach()

    def _save_parameters(self) -> None:
        """Save current parameter values."""
        self.saved_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.saved_params[name] = param.data.clone().detach()

    def get_si_loss(self) -> torch.Tensor:
        """
        Compute SI regularization loss.
        
        SI loss: L_SI = λ * Σ Ω_i * (θ_i - θ_i^old)^2
        
        Returns:
            SI regularization loss
        """
        if not self.importance or not self.saved_params:
            return torch.tensor(0.0, device=self.device)
        
        si_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.importance and name in self.saved_params:
                # SI penalty: Ω * (θ - θ_old)^2
                penalty = self.importance[name] * (param - self.saved_params[name]).pow(2)
                si_loss += torch.sum(penalty)
        
        return self.lambda_si * si_loss

    def get_penalty(self, param_name: str) -> torch.Tensor:
        """
        Get SI penalty for a specific parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            SI penalty tensor for the parameter
        """
        if param_name not in self.importance or param_name not in self.saved_params:
            return torch.tensor(0.0, device=self.device)
        
        param = dict(self.model.named_parameters())[param_name]
        return self.importance[param_name] * (param - self.saved_params[param_name]).pow(2)

    def get_stats(self) -> SIStats:
        """
        Get statistics about SI importance.
        
        Returns:
            SIStats object with statistics
        """
        if not self.importance:
            return SIStats()
        
        importance_values = []
        for name, importance in self.importance.items():
            importance_values.extend(importance.flatten().cpu().numpy())
        
        importance_array = np.array(importance_values)
        
        return SIStats(
            num_parameters=len(self.importance),
            num_important_parameters=len(self.importance),
            mean_importance=float(np.mean(importance_array)) if importance_array.size > 0 else 0.0,
            max_importance=float(np.max(importance_array)) if importance_array.size > 0 else 0.0,
            importance_distribution={
                "min": float(np.min(importance_array)) if importance_array.size > 0 else 0.0,
                "q25": float(np.percentile(importance_array, 25)) if importance_array.size > 0 else 0.0,
                "q50": float(np.percentile(importance_array, 50)) if importance_array.size > 0 else 0.0,
                "q75": float(np.percentile(importance_array, 75)) if importance_array.size > 0 else 0.0,
                "max": float(np.max(importance_array)) if importance_array.size > 0 else 0.0,
            },
        )

    def get_importance_dict(self) -> Dict[str, torch.Tensor]:
        """Get the parameter importance dictionary."""
        return self.importance.copy()

    def get_param_values(self) -> Dict[str, torch.Tensor]:
        """Get the saved parameter values."""
        return self.saved_params.copy()

    def reset(self) -> None:
        """Reset SI importance, saved parameters, and gradients."""
        self._init_importance()
        self.saved_params = {}
        self.prev_grads = {}

    def update_lambda(self, lambda_si: float) -> None:
        """Update SI regularization strength."""
        self.lambda_si = lambda_si

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"SI(lambda={self.lambda_si:.2f}, "
            f"params={stats.num_parameters}, "
            f"mean_importance={stats.mean_importance:.4f}, "
            f"max_importance={stats.max_importance:.4f})"
        )
