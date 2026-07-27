"""
Checkpoint management for Adaptive Qwen Omni training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 500
    max_checkpoints: int = 5
    save_best: bool = True
    save_latest: bool = True
    metric_for_best: str = "loss"
    mode: str = "min"  # min or max


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    path: str
    step: int
    epoch: int
    metrics: Dict[str, float] = field(default_factory=dict)
    is_best: bool = False
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "step": self.step,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "is_best": self.is_best,
            "timestamp": self.timestamp,
        }


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Features:
    - Save checkpoints at regular intervals
    - Keep best checkpoints
    - Manage checkpoint rotation
    - Load from checkpoints
    """

    def __init__(self, config: Optional[CheckpointConfig] = None):
        """
        Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration
        """
        self.config = config or CheckpointConfig()
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints: List[CheckpointInfo] = []
        self.best_metric: Optional[float] = None
        self.best_checkpoint: Optional[CheckpointInfo] = None

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: Optional[bool] = None,
    ) -> CheckpointInfo:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Scheduler to save
            step: Training step
            epoch: Training epoch
            metrics: Training metrics
            is_best: Whether this is the best checkpoint

        Returns:
            CheckpointInfo for the saved checkpoint
        """
        import time

        metrics = metrics or {}
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Determine if best
        if is_best is None:
            is_best = self._is_best(metrics)

        # Create checkpoint path
        checkpoint_name = f"step_{step}_epoch_{epoch}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"

        # Save checkpoint
        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else {},
        }

        torch.save(checkpoint_data, str(checkpoint_path))

        # Create info
        info = CheckpointInfo(
            path=str(checkpoint_path),
            step=step,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best,
            timestamp=timestamp,
        )

        self.checkpoints.append(info)

        if is_best:
            self.best_checkpoint = info
            self.best_metric = self._get_metric_value(metrics)

            # Save best checkpoint separately
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint_data, str(best_path))

        # Rotate checkpoints if needed
        if len(self.checkpoints) > self.config.max_checkpoints:
            self._rotate_checkpoints()

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return info

    def _is_best(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are the best."""
        if self.best_metric is None:
            return True

        current_value = self._get_metric_value(metrics)
        if current_value is None:
            return False

        if self.config.mode == "min":
            return current_value < self.best_metric
        else:
            return current_value > self.best_metric

    def _get_metric_value(self, metrics: Dict[str, float]) -> Optional[float]:
        """Get the value of the metric to track."""
        return metrics.get(self.config.metric_for_best)

    def _rotate_checkpoints(self) -> None:
        """Remove oldest checkpoints to stay within limit."""
        while len(self.checkpoints) > self.config.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            checkpoint_path = Path(oldest.path)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_path}")

    def load_checkpoint(
        self,
        path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            path: Path to checkpoint
            model: Model to load
            optimizer: Optimizer to load
            scheduler: Scheduler to load

        Returns:
            Dictionary with loaded data
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint_data = torch.load(str(checkpoint_path), map_location="cpu")

        # Load model
        model.load_state_dict(checkpoint_data["model"])

        # Load optimizer
        if optimizer and checkpoint_data.get("optimizer"):
            optimizer.load_state_dict(checkpoint_data["optimizer"])

        # Load scheduler
        if scheduler and checkpoint_data.get("scheduler"):
            scheduler.load_state_dict(checkpoint_data["scheduler"])

        logger.info(f"Loaded checkpoint: {path}")
        return checkpoint_data

    def load_latest(self, model: torch.nn.Module) -> Optional[CheckpointInfo]:
        """Load the latest checkpoint."""
        if not self.checkpoints:
            return None

        latest = self.checkpoints[-1]
        self.load_checkpoint(latest.path, model)
        return latest

    def load_best(self, model: torch.nn.Module) -> Optional[CheckpointInfo]:
        """Load the best checkpoint."""
        if self.best_checkpoint is None:
            return None

        self.load_checkpoint(self.best_checkpoint.path, model)
        return self.best_checkpoint

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all checkpoints."""
        return self.checkpoints

    def get_checkpoint_info(self, path: Union[str, Path]) -> Optional[CheckpointInfo]:
        """Get info for a specific checkpoint."""
        for info in self.checkpoints:
            if info.path == str(path):
                return info
        return None

    def delete_checkpoint(self, path: Union[str, Path]) -> bool:
        """Delete a checkpoint."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            return False

        checkpoint_path.unlink()

        # Remove from list
        self.checkpoints = [
            info for info in self.checkpoints
            if info.path != str(path)
        ]

        if self.best_checkpoint and self.best_checkpoint.path == str(path):
            self.best_checkpoint = None
            self.best_metric = None

        logger.info(f"Deleted checkpoint: {path}")
        return True

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        for info in self.checkpoints:
            checkpoint_path = Path(info.path)
            if checkpoint_path.exists():
                checkpoint_path.unlink()

        self.checkpoints.clear()
        self.best_checkpoint = None
        self.best_metric = None

        logger.info("Cleared all checkpoints")
