"""
Elastic Weight Consolidation (EWC) for Adaptive ML Framework.
Protects important parameters from being overwritten during continual learning.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_ml.core.config import AdaptiveMLConfig


@dataclass
class EWCStats:
    """Statistics for EWC."""

    num_parameters: int = 0
    num_important_parameters: int = 0
    mean_importance: float = 0.0
    max_importance: float = 0.0
    min_importance: float = 0.0


class EWC:
    """
    Elastic Weight Consolidation (EWC) for continual learning.
    
    EWC protects parameters that are important for previous tasks by adding
    a quadratic penalty to the loss function based on the Fisher Information Matrix.
    
    The penalty term is:
        EWC_penalty = lambda * sum(F_i * (theta_i - theta_i^*)^2)
    
    where:
    - lambda: regularization strength (EWC lambda)
    - F_i: diagonal Fisher Information for parameter i
    - theta_i: current parameter value
    - theta_i^*: parameter value after previous task
    
    Usage:
        ewc = EWC(model, lambda_ewc=1000.0)
        
        # After training on a task, update Fisher
        ewc.update_fisher(train_loader)
        
        # During training on next task, add EWC penalty
        loss = task_loss + ewc.penalty(model)
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        fisher_diagonal: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize EWC.
        
        Args:
            model: The neural network model
            lambda_ewc: Regularization strength (higher = more protection)
            fisher_diagonal: Whether to use diagonal Fisher approximation
            device: Device to store Fisher matrix on
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_diagonal = fisher_diagonal
        self.device = device or next(model.parameters()).device
        
        # Store parameter values and Fisher information
        self.param_values: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        
        # Initialize with current parameters
        self._save_parameters()
        
        # Initialize Fisher (will be updated after first task)
        self._init_fisher()

    def _save_parameters(self) -> None:
        """Save current parameter values."""
        self.param_values = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_values[name] = param.data.clone().detach()

    def _init_fisher(self) -> None:
        """Initialize Fisher information to zero."""
        self.fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data).to(self.device)

    def update_fisher(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
        temperature: float = 1.0,
    ) -> None:
        """
        Update Fisher Information Matrix using the current data.
        
        The Fisher Information is computed as:
            F = E[(d log p(x|theta)/d theta)^2]
        
        For classification, we use the negative log-likelihood:
            log p(y|x, theta) = log softmax(z)
        
        Args:
            dataloader: DataLoader with training data
            num_batches: Number of batches to use (None = all)
            temperature: Temperature for softmax (default: 1.0)
        """
        # Initialize Fisher to zero
        self._init_fisher()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Number of samples
        N = 0
        
        # Enable gradients for Fisher computation
        was_training = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
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
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Get log probabilities
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                
                # Compute softmax probabilities
                probs = torch.softmax(logits / temperature, dim=-1)
                
                # Compute gradient of log probabilities
                # For classification: d log p(y|x)/d theta = p(y|x) - 1_{y=y_true}
                if targets is not None:
                    # Create one-hot targets
                    if len(targets.shape) == 1:
                        # Class indices
                        one_hot = torch.zeros_like(probs)
                        one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
                    else:
                        one_hot = targets
                    
                    # Gradient of log prob: probs - one_hot
                    grad_log_prob = probs - one_hot
                else:
                    # For unsupervised, use probs directly
                    grad_log_prob = probs
                
                # Compute Fisher: E[grad^2]
                # Compute gradients for all parameters at once
                grads = torch.autograd.grad(
                    outputs=logits,
                    inputs=[param for name, param in self.model.named_parameters() 
                           if param.requires_grad and name in self.fisher],
                    grad_outputs=grad_log_prob,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                
                # Update Fisher for each parameter
                for (name, param), grad in zip(
                    [(n, p) for n, p in self.model.named_parameters() 
                     if p.requires_grad and n in self.fisher],
                    grads
                ):
                    if grad is not None:
                        self.fisher[name] += grad.pow(2)
                
                N += inputs.shape[0]
        
        # Average over samples
        if N > 0:
            for name in self.fisher:
                self.fisher[name] /= N
        
        # Save current parameters
        self._save_parameters()
        
        # Restore gradient state
        torch.set_grad_enabled(was_training)
        
        # Set model back to training mode
        self.model.train()

    def penalty(self, model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Compute the EWC penalty for the current model parameters.
        
        Args:
            model: Model to compute penalty for (defaults to self.model)
        
        Returns:
            EWC penalty term (scalar tensor)
        """
        model = model or self.model
        
        penalty = torch.tensor(0.0, device=self.device)
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher and name in self.param_values:
                # Difference from saved parameters
                diff = param - self.param_values[name]
                
                # Fisher-weighted squared difference
                fisher_weighted = self.fisher[name] * diff.pow(2)
                
                # Sum over all parameters
                penalty += fisher_weighted.sum()
        
        # Scale by lambda
        return self.lambda_ewc * penalty

    def get_ewc_loss(
        self,
        task_loss: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute the total loss with EWC penalty.
        
        Args:
            task_loss: The task-specific loss (e.g., cross-entropy)
            model: Model to compute penalty for (defaults to self.model)
        
        Returns:
            Total loss = task_loss + EWC_penalty
        """
        ewc_penalty = self.penalty(model)
        return task_loss + ewc_penalty

    def update_lambda(self, lambda_ewc: float) -> None:
        """Update the EWC regularization strength."""
        self.lambda_ewc = lambda_ewc

    def reset(self) -> None:
        """Reset EWC (clear Fisher and saved parameters)."""
        self._init_fisher()
        self._save_parameters()

    def get_stats(self) -> EWCStats:
        """Get statistics about EWC."""
        num_parameters = 0
        num_important = 0
        importance_values = []
        
        for name, fisher in self.fisher.items():
            num_params = fisher.numel()
            num_parameters += num_params
            
            # Count important parameters (Fisher > 0)
            important = (fisher > 0).sum().item()
            num_important += important
            
            # Collect importance values
            importance_values.extend(fisher.cpu().numpy().flatten())
        
        if len(importance_values) > 0:
            importance_values = [v for v in importance_values if v > 0]
            mean_importance = float(np.mean(importance_values)) if importance_values else 0.0
            max_importance = float(np.max(importance_values)) if importance_values else 0.0
            min_importance = float(np.min(importance_values)) if importance_values else 0.0
        else:
            mean_importance = 0.0
            max_importance = 0.0
            min_importance = 0.0
        
        return EWCStats(
            num_parameters=num_parameters,
            num_important_parameters=num_important,
            mean_importance=mean_importance,
            max_importance=max_importance,
            min_importance=min_importance,
        )

    def get_fisher_dict(self) -> Dict[str, torch.Tensor]:
        """Get the Fisher Information Matrix."""
        return self.fisher.copy()

    def get_param_values(self) -> Dict[str, torch.Tensor]:
        """Get the saved parameter values."""
        return self.param_values.copy()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"EWC(lambda={self.lambda_ewc}, params={stats.num_parameters}, "
            f"important={stats.num_important_parameters}, mean_F={stats.mean_importance:.4f})"
        )
