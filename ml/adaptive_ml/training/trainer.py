"""
Continual Trainer for Adaptive ML Framework.
Main training engine that combines EWC, Knowledge Distillation, and Experience Replay.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import SamplingStrategy, Task
from adaptive_ml.data.dataset import ContinualDataset, DatasetEntry
from adaptive_ml.memory.replay import ReplayBuffer
from adaptive_ml.models.adapters import AdapterManager
from adaptive_ml.training.ewc import EWC
from adaptive_ml.training.distillation import KnowledgeDistillation


@dataclass
class TrainingState:
    """State of the training process."""

    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")
    best_accuracy: float = 0.0
    current_loss: float = 0.0
    current_accuracy: float = 0.0


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    task_id: str
    epoch: int
    step: int
    loss: float
    accuracy: float
    ewc_penalty: float = 0.0
    distill_loss: float = 0.0
    replay_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContinualTrainer:
    """
    Main training engine for continual learning.
    
    Combines three anti-catastrophic forgetting techniques:
    1. Experience Replay: Replay buffer with various sampling strategies
    2. Elastic Weight Consolidation (EWC): Protect important parameters
    3. Knowledge Distillation: Preserve old model behavior
    
    Features:
    - Task-aware training
    - Dynamic mixing of new data and replay samples
    - Automatic Fisher Information updates
    - Teacher model management for distillation
    - Adapter support for parameter-efficient fine-tuning
    - Comprehensive logging and metrics
    
    Usage:
        trainer = ContinualTrainer(model, tokenizer, config)
        
        # Train on first task
        metrics = trainer.train_task(
            task_id="task_a",
            train_data=train_data_a,
            val_data=val_data_a,
        )
        
        # Train on second task (with anti-forgetting)
        metrics = trainer.train_task(
            task_id="task_b",
            train_data=train_data_b,
            val_data=val_data_b,
        )
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[AdaptiveMLConfig] = None,
        adapter_manager: Optional[AdapterManager] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        ewc: Optional[EWC] = None,
        distillation: Optional[KnowledgeDistillation] = None,
    ):
        """
        Initialize ContinualTrainer.
        
        Args:
            model: The base pre-trained model
            tokenizer: Optional tokenizer for the model
            config: AdaptiveMLConfig instance
            adapter_manager: Optional AdapterManager for parameter-efficient fine-tuning
            replay_buffer: Optional ReplayBuffer for experience replay
            ewc: Optional EWC instance for parameter importance
            distillation: Optional KnowledgeDistillation instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or AdaptiveMLConfig()
        
        # Initialize components if not provided
        self.adapter_manager = adapter_manager or AdapterManager(model, config)
        self.replay_buffer = replay_buffer or ReplayBuffer(config)
        self.ewc = ewc
        self.distillation = distillation
        
        # Training state
        self.state = TrainingState()
        self.metrics: List[TrainingMetrics] = []
        
        # Task tracking
        self.tasks: Dict[str, Task] = {}
        self.current_task: Optional[str] = None
        
        # Teacher model for distillation
        self.teacher_model: Optional[nn.Module] = None
        
        # Device
        self.device = self.config.training.device
        self.model.to(self.device)
        
        # Initialize components
        self._init_ewc()
        self._init_distillation()

    def _init_ewc(self) -> None:
        """Initialize EWC if not provided."""
        if self.ewc is None:
            self.ewc = EWC(
                self.model,
                lambda_ewc=self.config.training.ewc_lambda,
                device=self.device,
            )

    def _init_distillation(self) -> None:
        """Initialize Knowledge Distillation if not provided."""
        if self.distillation is None:
            # Create a copy of the model as teacher
            self.teacher_model = self._copy_model(self.model)
            self.distillation = KnowledgeDistillation(
                self.teacher_model,
                alpha=self.config.training.distill_alpha,
                temperature=2.0,
            )

    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of a model."""
        # Create a new instance of the same class
        model_copy = type(model)(model.config)
        model_copy.load_state_dict(model.state_dict())
        model_copy.to(self.device)
        model_copy.eval()
        return model_copy

    def train_task(
        self,
        task_id: str,
        train_data: List[DatasetEntry],
        val_data: Optional[List[DatasetEntry]] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        use_replay: bool = True,
        use_ewc: bool = True,
        use_distillation: bool = True,
        sampling_strategy: Optional[SamplingStrategy] = None,
        replay_ratio: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Train on a new task with continual learning techniques.
        
        Args:
            task_id: Unique identifier for the task
            train_data: Training data for the task
            val_data: Optional validation data
            num_epochs: Number of epochs (defaults to config)
            batch_size: Batch size (defaults to config)
            learning_rate: Learning rate (defaults to config)
            use_replay: Whether to use experience replay
            use_ewc: Whether to use EWC
            use_distillation: Whether to use knowledge distillation
            sampling_strategy: Sampling strategy for replay buffer
            replay_ratio: Ratio of replay samples in batch
            callbacks: Optional list of callback functions
        
        Returns:
            Dictionary with training metrics and results
        """
        # Update task tracking
        self.current_task = task_id
        if task_id not in self.tasks:
            self.tasks[task_id] = Task(id=task_id, name=task_id)
        
        # Use config defaults if not specified
        num_epochs = num_epochs or self.config.training.num_epochs
        batch_size = batch_size or self.config.training.batch_size
        learning_rate = learning_rate or self.config.training.learning_rate
        sampling_strategy = sampling_strategy or self.config.memory.sampling_strategy
        replay_ratio = replay_ratio or self.config.training.replay_ratio
        
        # Add new data to replay buffer
        if use_replay:
            for entry in train_data:
                self.replay_buffer.add(
                    task_id=task_id,
                    data=entry.input,
                    label=entry.label,
                    metadata=entry.metadata,
                )
        
        # Create continual dataset
        dataset = ContinualDataset(
            new_data=train_data,
            replay_buffer=list(self.replay_buffer) if use_replay else [],
            replay_ratio=replay_ratio,
            sampling_strategy=sampling_strategy,
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # For simplicity
        )
        
        # Setup optimizer
        optimizer = self._create_optimizer(learning_rate)
        
        # Setup scheduler
        scheduler = self._create_scheduler(optimizer)
        
        # Setup loss function
        loss_fn = self._create_loss_fn()
        
        # Training loop
        self.model.train()
        
        for epoch in range(num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            for step, batch in enumerate(dataloader):
                self.state.step = step
                
                # Get batch
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
                    metadata = batch[2] if len(batch) > 2 else {}
                else:
                    inputs = batch.get("input", batch.get("input_ids"))
                    labels = batch.get("label", batch.get("labels"))
                    metadata = batch.get("metadata", {})
                
                # Move to device
                inputs = inputs.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)
                
                # Forward pass
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                
                # Get logits
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                
                # Compute task loss
                if labels is not None:
                    task_loss = loss_fn(logits, labels)
                else:
                    task_loss = torch.tensor(0.0, device=self.device)
                
                # Compute EWC penalty
                ewc_penalty = torch.tensor(0.0, device=self.device)
                if use_ewc and self.ewc is not None:
                    ewc_penalty = self.ewc.penalty(self.model)
                
                # Compute distillation loss
                distill_loss = torch.tensor(0.0, device=self.device)
                if use_distillation and self.distillation is not None:
                    # Get teacher outputs
                    with torch.no_grad():
                        teacher_logits = self.teacher_model(inputs)
                        if isinstance(teacher_logits, torch.Tensor):
                            pass
                        else:
                            teacher_logits = teacher_logits.logits if hasattr(teacher_logits, "logits") else teacher_logits[0]
                    
                    # Compute distillation loss
                    distill_loss = self.distillation._compute_distillation_loss(
                        logits, teacher_logits
                    )
                
                # Total loss
                total_loss = task_loss + ewc_penalty + distill_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Compute accuracy
                if labels is not None:
                    preds = torch.argmax(logits, dim=-1)
                    correct = (preds == labels).float().sum()
                    accuracy = correct / labels.shape[0]
                else:
                    accuracy = 0.0
                
                # Update epoch stats
                epoch_loss += total_loss.item()
                epoch_accuracy += accuracy.item()
                
                # Collect metrics
                metrics = TrainingMetrics(
                    task_id=task_id,
                    epoch=epoch,
                    step=step,
                    loss=total_loss.item(),
                    accuracy=accuracy.item(),
                    ewc_penalty=ewc_penalty.item(),
                    distill_loss=distill_loss.item(),
                    replay_ratio=replay_ratio,
                    metadata=metadata,
                )
                self.metrics.append(metrics)
                
                # Call callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(metrics, self)
            
            # Update state
            self.state.current_loss = epoch_loss / (step + 1)
            self.state.current_accuracy = epoch_accuracy / (step + 1)
            
            # Update best metrics
            if self.state.current_loss < self.state.best_loss:
                self.state.best_loss = self.state.current_loss
            if self.state.current_accuracy > self.state.best_accuracy:
                self.state.best_accuracy = self.state.current_accuracy
            
            # Validation
            if val_data is not None:
                val_metrics = self.evaluate(val_data)
                print(f"Epoch {epoch}: Val Loss = {val_metrics['loss']:.4f}, Val Acc = {val_metrics['accuracy']:.4f}")
        
        # Update EWC after training on this task
        if use_ewc and self.ewc is not None:
            # Create a dataloader for the current task
            task_dataset = ContinualDataset(
                new_data=train_data,
                replay_buffer=[],
            )
            task_loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True)
            self.ewc.update_fisher(task_loader)
        
        # Update teacher model for next task
        if use_distillation and self.distillation is not None:
            self.teacher_model = self._copy_model(self.model)
            self.distillation.teacher = self.teacher_model
        
        # Update task status
        self.tasks[task_id].status = "completed"
        self.tasks[task_id].metrics = {
            "final_loss": self.state.current_loss,
            "final_accuracy": self.state.current_accuracy,
            "best_loss": self.state.best_loss,
            "best_accuracy": self.state.best_accuracy,
        }
        
        return {
            "task_id": task_id,
            "final_loss": self.state.current_loss,
            "final_accuracy": self.state.current_accuracy,
            "best_loss": self.state.best_loss,
            "best_accuracy": self.state.best_accuracy,
            "metrics": self.metrics[-len(train_data) :],
        }

    def evaluate(
        self,
        data: List[DatasetEntry],
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data: List of dataset entries to evaluate on
            batch_size: Batch size (defaults to config)
        
        Returns:
            Dictionary with evaluation metrics
        """
        batch_size = batch_size or self.config.training.batch_size
        
        dataset = ContinualDataset(new_data=data, replay_buffer=[])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch.get("input", batch.get("input_ids"))
                    labels = batch.get("label", batch.get("labels"))
                
                inputs = inputs.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                
                # Compute loss
                if labels is not None:
                    loss = F.cross_entropy(logits, labels)
                    total_loss += loss.item() * inputs.shape[0]
                    
                    # Compute accuracy
                    preds = torch.argmax(logits, dim=-1)
                    correct = (preds == labels).float().sum()
                    total_accuracy += correct.item()
                
                total_samples += inputs.shape[0]
        
        self.model.train()
        
        return {
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "accuracy": total_accuracy / total_samples if total_samples > 0 else 0.0,
            "samples": total_samples,
        }

    def _create_optimizer(self, learning_rate: float) -> Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config.training.optimizer.lower()
        
        # Get parameters to optimize
        if self.adapter_manager.active_adapter:
            # Only optimize adapter parameters
            params = list(self.adapter_manager.get_adapter(self.adapter_manager.active_adapter).parameters())
        else:
            # Optimize all parameters
            params = list(self.model.parameters())
        
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif optimizer_name == "adam":
            return torch.optim.Adam(
                params,
                lr=learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler based on config."""
        scheduler_name = self.config.training.scheduler.lower()
        
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=0.0,
            )
        elif scheduler_name == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.training.num_epochs,
            )
        elif scheduler_name == "constant":
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        else:
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def _create_loss_fn(self) -> Callable:
        """Create loss function based on config."""
        # Default to cross-entropy
        return F.cross_entropy

    def save_checkpoint(
        self,
        path: Union[str, Path],
        include_optimizer: bool = True,
        include_scheduler: bool = True,
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Directory to save checkpoint
            include_optimizer: Whether to save optimizer state
            include_scheduler: Whether to save scheduler state
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model": self.model.state_dict(),
            "config": self.config.get_dict(),
            "state": self.state.__dict__,
            "tasks": {tid: task.__dict__ for tid, task in self.tasks.items()},
            "metrics": [m.__dict__ for m in self.metrics],
        }
        
        # Save adapter manager state
        checkpoint["adapters"] = {
            "active": self.adapter_manager.active_adapter,
            "info": {aid: info.__dict__ for aid, info in self.adapter_manager.adapter_info.items()},
        }
        
        # Save replay buffer state
        checkpoint["replay_buffer"] = {
            "size": len(self.replay_buffer),
            "capacity": self.replay_buffer.capacity,
        }
        
        # Save EWC state
        if self.ewc is not None:
            checkpoint["ewc"] = {
                "lambda": self.ewc.lambda_ewc,
                "fisher": {k: v.cpu() for k, v in self.ewc.fisher.items()},
                "param_values": {k: v.cpu() for k, v in self.ewc.param_values.items()},
            }
        
        torch.save(checkpoint, path / "checkpoint.pt")

    def load_checkpoint(
        self,
        path: Union[str, Path],
    ) -> None:
        """
        Load training checkpoint.
        
        Args:
            path: Directory containing checkpoint
        """
        path = Path(path)
        checkpoint = torch.load(path / "checkpoint.pt")
        
        # Load model
        self.model.load_state_dict(checkpoint["model"])
        
        # Load state
        self.state = TrainingState(**checkpoint["state"])
        
        # Load tasks
        self.tasks = {tid: Task(**task_dict) for tid, task_dict in checkpoint["tasks"].items()}
        
        # Load metrics
        self.metrics = [TrainingMetrics(**m_dict) for m_dict in checkpoint["metrics"]]
        
        # Load adapters
        if "adapters" in checkpoint:
            self.adapter_manager.active_adapter = checkpoint["adapters"]["active"]
            # Note: Actual adapter loading would need to be implemented separately
        
        # Load EWC
        if "ewc" in checkpoint and self.ewc is not None:
            self.ewc.lambda_ewc = checkpoint["ewc"]["lambda"]
            self.ewc.fisher = {k: v.to(self.device) for k, v in checkpoint["ewc"]["fisher"].items()}
            self.ewc.param_values = {k: v.to(self.device) for k, v in checkpoint["ewc"]["param_values"].items()}

    def get_state(self) -> TrainingState:
        """Get current training state."""
        return self.state

    def get_metrics(self) -> List[TrainingMetrics]:
        """Get all collected metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset collected metrics."""
        self.metrics = []

    def __repr__(self) -> str:
        return (
            f"ContinualTrainer(model={type(self.model).__name__}, "
            f"tasks={len(self.tasks)}, current_task={self.current_task})"
        )
