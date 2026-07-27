"""
Memory Aware Synapses (MAS) for Adaptive ML Framework.
Protects parameters based on their sensitivity to changes in the loss function.

MAS computes parameter importance as:
    Ω = |∇L/∇θ| * |θ|
    
This measures how much the loss changes with respect to parameter changes,
weighted by the parameter magnitude.
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
class MASStats:
    """Statistics for MAS."""

    num_parameters: int = 0
    num_important_parameters: int = 0
    mean_importance: float = 0.0
    max_importance: float = 0.0
    importance_distribution: Dict[str, float] = field(default_factory=dict)


class MAS:
    """
    Memory Aware Synapses (MAS) for continual learning.
    
    MAS protects parameters that are most sensitive to changes in the loss function.
    The importance of a parameter θ_i is given by:
        Ω_i = |∂L/∂θ_i| * |θ_i|
    
    This captures both the gradient magnitude and the parameter magnitude,
    giving higher importance to parameters that significantly affect the loss.
    
    Reference:
        Aljundi et al., "Memory Aware Synapses: Learning what (not) to Forget
        in Continual Learning", ECCV 2018.
        https://arxiv.org/abs/1711.09601
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_mas: float = 100.0,
        config: Optional[AdaptiveMLConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize MAS.
        
        Args:
            model: The neural network model
            lambda_mas: MAS regularization strength
            config: AdaptiveMLConfig instance
            device: Device to run on (defaults to model's device)
        """
        self.model = model
        self.lambda_mas = lambda_mas
        self.config = config or AdaptiveMLConfig()
        self.device = device or next(model.parameters()).device
        self.method = ParameterImportanceMethod.MAS
        
        # Parameter importance dictionary: name -> importance tensor
        self.importance: Dict[str, torch.Tensor] = {}
        
        # Saved parameters (before update)
        self.saved_params: Dict[str, torch.Tensor] = {}
        
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
        
        MAS importance: Ω = |∇L/∇θ| * |θ|
        
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
            
            # Update importance: Ω = |∇L/∇θ| * |θ|
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.importance:
                    if param.grad is not None:
                        # MAS importance: |gradient| * |parameter|
                        self.importance[name] += torch.abs(param.grad) * torch.abs(param.data)
            
            N += inputs.shape[0]
        
        # Average over samples
        if N > 0:
            for name in self.importance:
                self.importance[name] /= N
        
        # Save current parameters
        self._save_parameters()
        
        # Restore gradient state
        torch.set_grad_enabled(was_training)

    def _save_parameters(self) -> None:
        """Save current parameter values."""
        self.saved_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.saved_params[name] = param.data.clone().detach()

    def get_mas_loss(self) -> torch.Tensor:
        """
        Compute MAS regularization loss.
        
        MAS loss: L_MAS = λ * Σ Ω_i * (θ_i - θ_i^old)^2
        
        Returns:
            MAS regularization loss
        """
        if not self.importance or not self.saved_params:
            return torch.tensor(0.0, device=self.device)
        
        mas_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.importance and name in self.saved_params:
                # MAS penalty: Ω * (θ - θ_old)^2
                penalty = self.importance[name] * (param - self.saved_params[name]).pow(2)
                mas_loss += torch.sum(penalty)
        
        return self.lambda_mas * mas_loss

    def get_penalty(self, param_name: str) -> torch.Tensor:
        """
        Get MAS penalty for a specific parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            MAS penalty tensor for the parameter
        """
        if param_name not in self.importance or param_name not in self.saved_params:
            return torch.tensor(0.0, device=self.device)
        
        param = dict(self.model.named_parameters())[param_name]
        return self.importance[param_name] * (param - self.saved_params[param_name]).pow(2)

    def get_stats(self) -> MASStats:
        """
        Get statistics about MAS importance.
        
        Returns:
            MASStats object with statistics
        """
        if not self.importance:
            return MASStats()
        
        importance_values = []
        for name, importance in self.importance.items():
            importance_values.extend(importance.flatten().cpu().numpy())
        
        importance_array = np.array(importance_values)
        
        return MASStats(
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
        """Reset MAS importance and saved parameters."""
        self._init_importance()
        self.saved_params = {}

    def update_lambda(self, lambda_mas: float) -> None:
        """Update MAS regularization strength."""
        self.lambda_mas = lambda_mas

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MAS(lambda={self.lambda_mas:.2f}, "
            f"params={stats.num_parameters}, "
            f"mean_importance={stats.mean_importance:.4f}, "
            f"max_importance={stats.max_importance:.4f})"
        )
