"""
Continual Learning Engine for Adaptive Omni ML.

Implements:
- Experience Replay
- Reservoir Sampling
- Prioritized Replay
- Knowledge Distillation
- Learning Without Forgetting
- Elastic Weight Consolidation (EWC)
- Adapter Isolation/Fusion/Routing
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ReplayBufferConfig:
    """Configuration for replay buffer."""
    max_size: int = 10000
    replay_ratio: float = 0.2
    prioritized: bool = True
    alpha: float = 0.6  # Priority exponent
    beta: float = 0.4  # Importance sampling exponent


class ReplayBuffer:
    """
    Experience replay buffer with reservoir sampling and prioritized replay.
    """
    
    def __init__(self, config: ReplayBufferConfig):
        self.max_size = config.max_size
        self.replay_ratio = config.replay_ratio
        self.prioritized = config.prioritized
        self.alpha = config.alpha
        self.beta = config.beta
        
        # Storage
        self.buffer: deque = deque(maxlen=self.max_size)
        self.priorities: deque = deque(maxlen=self.max_size)
        
        # Statistics
        self.total_added = 0
        self.total_sampled = 0
    
    def add(self, sample: Dict[str, Any], priority: float = 1.0):
        """Add a sample to the replay buffer."""
        self.buffer.append(sample)
        self.priorities.append(priority)
        self.total_added += 1
    
    def add_batch(self, samples: List[Dict[str, Any]], priorities: Optional[List[float]] = None):
        """Add multiple samples to the buffer."""
        if priorities is None:
            priorities = [1.0] * len(samples)
        
        for sample, priority in zip(samples, priorities):
            self.add(sample, priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], List[float], List[int]]:
        """
        Sample from the replay buffer.
        
        Returns:
            samples: List of sampled data
            weights: Importance sampling weights
            indices: Indices of sampled items
        """
        if len(self.buffer) == 0:
            return [], [], []
        
        n_samples = min(batch_size, len(self.buffer))
        
        if self.prioritized and len(self.buffer) > 1:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            indices = np.random.choice(
                len(self.buffer),
                size=n_samples,
                replace=False,
                p=probabilities
            )
            
            # Calculate importance sampling weights
            weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize
        else:
            # Uniform sampling
            indices = random.sample(range(len(self.buffer)), n_samples)
            weights = [1.0] * n_samples
        
        samples = [self.buffer[i] for i in indices]
        self.total_sampled += n_samples
        
        return samples, weights.tolist(), indices.tolist()
    
    def update_priorities(self, indices: List[int], new_priorities: List[float]):
        """Update priorities for specific samples."""
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def get_replay_samples(self, new_data_size: int) -> List[Dict]:
        """Get replay samples based on replay ratio."""
        replay_size = int(new_data_size * self.replay_ratio)
        samples, _, _ = self.sample(replay_size)
        return samples
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()
    
    def save(self, path: str):
        """Save buffer to disk."""
        data = {
            'buffer': list(self.buffer),
            'priorities': list(self.priorities),
            'config': {
                'max_size': self.max_size,
                'replay_ratio': self.replay_ratio,
                'prioritized': self.prioritized,
                'alpha': self.alpha,
                'beta': self.beta
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load buffer from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.buffer = deque(data['buffer'], maxlen=self.max_size)
        self.priorities = deque(data['priorities'], maxlen=self.max_size)
        
        if 'config' in data:
            config = data['config']
            self.max_size = config.get('max_size', self.max_size)
            self.replay_ratio = config.get('replay_ratio', self.replay_ratio)
            self.prioritized = config.get('prioritized', self.prioritized)
            self.alpha = config.get('alpha', self.alpha)
            self.beta = config.get('beta', self.beta)


class KnowledgeDistillation:
    """
    Knowledge Distillation for preventing catastrophic forgetting.
    
    Uses soft targets from the old model to guide training of the new model.
    """
    
    def __init__(self, temperature: float = 2.0, weight: float = 0.5):
        self.temperature = temperature
        self.weight = weight
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Logits from the current (student) model
            teacher_logits: Logits from the previous (teacher) model
        
        Returns:
            Distillation loss tensor
        """
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        
        loss = self.kl_div(student_log_probs, teacher_probs)
        return loss * (self.temperature ** 2) * self.weight
    
    def compute_combined_loss(
        self,
        task_loss: torch.Tensor,
        distillation_loss: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Combine task loss with distillation loss.
        
        Args:
            task_loss: Primary task loss
            distillation_loss: Knowledge distillation loss
            alpha: Balancing factor (higher = more focus on old knowledge)
        
        Returns:
            Combined loss
        """
        return (1 - alpha) * task_loss + alpha * distillation_loss


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.
    
    Penalizes changes to important weights identified from previous tasks.
    """
    
    def __init__(self, model: nn.Module, fisher_estimate_batches: int = 100):
        self.model = model
        self.fisher_estimate_batches = fisher_estimate_batches
        
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_weights: Dict[str, torch.Tensor] = {}
        self.lambda_ewc: float = 1000.0
    
    def compute_fisher_information(self, dataloader, loss_fn):
        """
        Compute Fisher Information Matrix diagonal approximation.
        
        Args:
            dataloader: Data loader for previous task
            loss_fn: Loss function
        """
        # Initialize fisher information
        self.fisher_information = {}
        for name, param in self.model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param)
        
        # Store optimal weights
        self.optimal_weights = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        # Compute gradient squared as Fisher approximation
        self.model.train()
        batch_count = 0
        
        for batch in dataloader:
            if batch_count >= self.fisher_estimate_batches:
                break
            
            # Forward pass
            outputs = self.model(batch)
            loss = loss_fn(outputs, batch['targets'])
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data.pow(2)
            
            batch_count += 1
        
        # Average over batches
        if batch_count > 0:
            for name in self.fisher_information:
                self.fisher_information[name] /= batch_count
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        Returns:
            EWC loss tensor
        """
        if not self.fisher_information:
            return torch.tensor(0.0)
        
        ewc_loss = torch.tensor(0.0)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_weights:
                diff = param - self.optimal_weights[name]
                ewc_loss += (self.fisher_information[name] * diff.pow(2)).sum()
        
        return 0.5 * self.lambda_ewc * ewc_loss
    
    def set_lambda(self, lambda_value: float):
        """Set the EWC regularization strength."""
        self.lambda_ewc = lambda_value


class LearningWithoutForgetting:
    """
    Learning Without Forgetting (LwF) implementation.
    
    Uses the old model's responses on new data to preserve old capabilities.
    """
    
    def __init__(self, old_model: nn.Module, weight: float = 1.0):
        self.old_model = old_model
        self.weight = weight
        self.old_model.eval()
        
        # Freeze old model
        for param in self.old_model.parameters():
            param.requires_grad = False
    
    def compute_lwf_loss(
        self,
        new_model_outputs: torch.Tensor,
        old_model_outputs: torch.Tensor,
        temperature: float = 2.0
    ) -> torch.Tensor:
        """
        Compute Learning Without Forgetting loss.
        
        Args:
            new_model_outputs: Outputs from the model being trained
            old_model_outputs: Stored outputs from the old model
            temperature: Temperature for softening outputs
        
        Returns:
            LwF loss tensor
        """
        kl_div = nn.KLDivLoss(reduction='batchmean')
        
        new_log_probs = torch.log_softmax(new_model_outputs / temperature, dim=-1)
        old_probs = torch.softmax(old_model_outputs / temperature, dim=-1)
        
        lwf_loss = kl_div(new_log_probs, old_probs)
        return self.weight * lwf_loss * (temperature ** 2)


class AdapterManager:
    """
    Manager for LoRA/QLoRA adapters with isolation and fusion capabilities.
    """
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.adapters: Dict[str, Any] = {}
        self.active_adapters: List[str] = []
        self.adapter_router: Optional[nn.Module] = None
    
    def register_adapter(self, name: str, adapter_config: Dict):
        """Register a new adapter."""
        self.adapters[name] = {
            'config': adapter_config,
            'path': adapter_config.get('path'),
            'task_type': adapter_config.get('task_type', 'general'),
            'is_active': False
        }
    
    def activate_adapter(self, name: str):
        """Activate a specific adapter."""
        if name not in self.adapters:
            raise ValueError(f"Adapter {name} not found")
        
        # Deactivate all others
        for adapter_name in self.adapters:
            self.adapters[adapter_name]['is_active'] = False
        
        self.adapters[name]['is_active'] = True
        self.active_adapters = [name]
    
    def activate_adapters(self, names: List[str], weights: Optional[List[float]] = None):
        """Activate multiple adapters for fusion."""
        if weights is None:
            weights = [1.0 / len(names)] * len(names)
        
        for name, weight in zip(names, weights):
            if name in self.adapters:
                self.adapters[name]['is_active'] = True
        
        self.active_adapters = names
    
    def route_to_adapter(self, input_features: torch.Tensor) -> str:
        """
        Route input to the most appropriate adapter.
        
        Args:
            input_features: Input representation for routing decision
        
        Returns:
            Name of selected adapter
        """
        if self.adapter_router is None:
            # Default to first active adapter or general
            return self.active_adapters[0] if self.active_adapters else 'general'
        
        # Use router to select adapter
        adapter_scores = self.adapter_router(input_features)
        adapter_idx = torch.argmax(adapter_scores, dim=-1)
        
        if adapter_idx < len(self.active_adapters):
            return self.active_adapters[adapter_idx]
        
        return self.active_adapters[0]
    
    def fuse_adapters(self, adapter_names: List[str], weights: List[float]):
        """
        Fuse multiple adapters into a combined adapter.
        
        Args:
            adapter_names: Names of adapters to fuse
            weights: Weights for each adapter
        """
        if len(adapter_names) != len(weights):
            raise ValueError("Names and weights must have same length")
        
        fused_config = {
            'type': 'fused',
            'components': list(zip(adapter_names, weights)),
            'created_at': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
        }
        
        fused_name = f"fused_{'_'.join(adapter_names)}"
        self.register_adapter(fused_name, fused_config)
        
        return fused_name


class ContinualLearningTrainer:
    """
    Main trainer class combining all continual learning strategies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        replay_config: ReplayBufferConfig,
        use_distillation: bool = True,
        use_ewc: bool = True,
        use_lwf: bool = False,
        distillation_weight: float = 0.5,
        ewc_strength: float = 1000.0
    ):
        self.model = model
        self.replay_buffer = ReplayBuffer(replay_config)
        
        self.use_distillation = use_distillation
        self.use_ewc = use_ewc
        self.use_lwf = use_lwf
        
        self.distillation = KnowledgeDistillation(weight=distillation_weight) if use_distillation else None
        self.ewc = ElasticWeightConsolidation(model) if use_ewc else None
        self.lwf = None
        
        self.old_model_state = None
        self.adapter_manager = AdapterManager(model)
    
    def store_old_model_state(self):
        """Store the current model state before training."""
        self.old_model_state = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
    
    def prepare_training_batch(
        self,
        new_data: List[Dict],
        batch_size: int
    ) -> Tuple[List[Dict], List[float]]:
        """
        Prepare a training batch combining new data with replay data.
        
        Args:
            new_data: New training samples
            batch_size: Desired batch size
        
        Returns:
            Combined batch and sample weights
        """
        # Get replay samples
        replay_samples = self.replay_buffer.get_replay_samples(len(new_data))
        
        # Combine new and replay data
        combined = new_data + replay_samples
        
        # Create weights (new data gets higher weight)
        weights = [1.0] * len(new_data) + [0.5] * len(replay_samples)
        
        return combined, weights
    
    def compute_total_loss(
        self,
        task_loss: torch.Tensor,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        include_replay: bool = True
    ) -> torch.Tensor:
        """
        Compute total loss including all regularization terms.
        
        Args:
            task_loss: Primary task loss
            outputs: Model outputs
            targets: Ground truth targets
            include_replay: Whether to include replay-based losses
        
        Returns:
            Total loss
        """
        total_loss = task_loss
        
        # Add EWC loss
        if self.use_ewc and include_replay:
            ewc_loss = self.ewc.compute_ewc_loss()
            total_loss = total_loss + ewc_loss
        
        # Add distillation loss if we have old model outputs
        if self.use_distillation and self.old_model_state is not None and include_replay:
            # Would need to compute old model outputs here
            pass
        
        return total_loss
    
    def after_training(self, dataloader, loss_fn):
        """Call after training to update fisher information."""
        if self.use_ewc:
            self.ewc.compute_fisher_information(dataloader, loss_fn)
    
    def add_to_replay(self, samples: List[Dict], priorities: Optional[List[float]] = None):
        """Add samples to replay buffer."""
        self.replay_buffer.add_batch(samples, priorities)
    
    def save_state(self, path: str):
        """Save trainer state."""
        state = {
            'replay_buffer_path': str(Path(path) / 'replay_buffer.json'),
            'use_distillation': self.use_distillation,
            'use_ewc': self.use_ewc,
            'use_lwf': self.use_lwf
        }
        
        # Save replay buffer
        self.replay_buffer.save(state['replay_buffer_path'])
        
        with open(str(Path(path) / 'trainer_config.json'), 'w') as f:
            json.dump(state, f)
    
    def load_state(self, path: str):
        """Load trainer state."""
        with open(str(Path(path) / 'trainer_config.json'), 'r') as f:
            state = json.load(f)
        
        replay_path = state.get('replay_buffer_path', str(Path(path) / 'replay_buffer.json'))
        if Path(replay_path).exists():
            self.replay_buffer.load(replay_path)
