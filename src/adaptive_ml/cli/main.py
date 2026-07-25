"""
CLI for Adaptive ML Framework.
Provides command-line interface for training, evaluation, and model management.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from adaptive_ml.core.config import AdaptiveMLConfig, get_config
from adaptive_ml.core.types import Task, TaskStatus
from adaptive_ml.data.drift import DriftDetector
from adaptive_ml.memory.replay import ReplayBuffer
from adaptive_ml.models.adapters import AdapterManager
from adaptive_ml.serving.registry import ModelRegistry
from adaptive_ml.training.trainer import ContinualTrainer
from adaptive_ml.evaluation.promoter import PromotionController

# Create Typer app
app = typer.Typer(name="adaptive-ml", help="Adaptive ML Framework CLI")
console = Console()

# Global state
_state: Dict[str, Any] = {
    "config": None,
    "trainer": None,
    "registry": None,
    "promoter": None,
}


def get_state() -> Dict[str, Any]:
    """Get the global CLI state."""
    return _state


# Configuration commands
@app.command()
def init(
    config_path: str = "config/default.yaml",
    project_name: str = "adaptive_ml",
):
    """Initialize a new Adaptive ML project."""
    config_path = Path(config_path)
    
    # Create config directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create default config if it doesn't exist
    if not config_path.exists():
        default_config = AdaptiveMLConfig()
        default_config.project_name = project_name
        default_config.to_yaml(config_path)
        console.print(f"[green]Created default configuration at {config_path}[/green]")
    else:
        console.print(f"[yellow]Configuration already exists at {config_path}[/yellow]")
    
    # Create necessary directories
    directories = [
        "model_registry",
        "logs",
        "mlruns",
        "adapters",
        "data",
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Created directory: {dir_name}[/green]")
    
    console.print("[green]Project initialized successfully![/green]")


@app.command()
def config(
    show: bool = False,
    path: str = "config/default.yaml",
):
    """Show or validate configuration."""
    config = get_config(path)
    
    if show:
        table = Table(title="Adaptive ML Configuration")
        table.add_column("Section", style="cyan")
        table.add_column("Key", style="magenta")
        table.add_column("Value", style="green")
        
        config_dict = config.get_dict()
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    table.add_row(section, str(key), str(value))
            else:
                table.add_row("global", str(section), str(values))
        
        console.print(table)
    else:
        console.print("[green]Configuration is valid[/green]")
    
    _state["config"] = config


# Training commands
@app.command()
def train(
    task_id: str,
    data_path: str,
    config_path: str = "config/default.yaml",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
):
    """Train on a new task."""
    # Load config
    config = get_config(config_path)
    _state["config"] = config
    
    # Load data
    data_path = Path(data_path)
    if not data_path.exists():
        console.print(f"[red]Error: Data file not found at {data_path}[/red]")
        raise typer.Exit(1)
    
    # Load data (simplified for CLI)
    # In practice, you'd implement proper data loading
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1)
    
    # For demo purposes, create dummy data
    from adaptive_ml.data.dataset import DatasetEntry
    train_data = [
        DatasetEntry(input=d.get("text", ""), label=d.get("label", 0), task_id=task_id)
        for d in data[:100]  # Use first 100 examples
    ]
    
    console.print(f"[green]Loaded {len(train_data)} training examples[/green]")
    
    # Initialize trainer
    # For demo, we'll use a simple model
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Use a small model for demo
        model_name = config.model.base_model
        console.print(f"[blue]Loading model: {model_name}[/blue]")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Update config with actual model info
        config.model.tokenizer = model_name
        
        # Create trainer
        trainer = ContinualTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        
        _state["trainer"] = trainer
        _state["model"] = model
        _state["tokenizer"] = tokenizer
        
        # Train
        console.print(f"[blue]Training on task: {task_id}[/blue]")
        
        result = trainer.train_task(
            task_id=task_id,
            train_data=train_data,
            num_epochs=epochs or config.training.num_epochs,
            batch_size=batch_size or config.training.batch_size,
            learning_rate=learning_rate or config.training.learning_rate,
        )
        
        console.print(f"[green]Training completed![/green]")
        console.print(f"Final loss: {result['final_loss']:.4f}")
        console.print(f"Final accuracy: {result['final_accuracy']:.4f}")
        console.print(f"Best loss: {result['best_loss']:.4f}")
        console.print(f"Best accuracy: {result['best_accuracy']:.4f}")
        
        # Save checkpoint
        checkpoint_path = Path(f"checkpoints/{task_id}")
        trainer.save_checkpoint(checkpoint_path)
        console.print(f"[green]Checkpoint saved to {checkpoint_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    task_id: str,
    data_path: str,
    config_path: str = "config/default.yaml",
):
    """Evaluate on a task."""
    # Load config
    config = get_config(config_path)
    
    # Load data
    data_path = Path(data_path)
    if not data_path.exists():
        console.print(f"[red]Error: Data file not found at {data_path}[/red]")
        raise typer.Exit(1)
    
    # Load data
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1)
    
    # Create dummy data
    from adaptive_ml.data.dataset import DatasetEntry
    eval_data = [
        DatasetEntry(input=d.get("text", ""), label=d.get("label", 0), task_id=task_id)
        for d in data[:100]
    ]
    
    # Get trainer from state
    trainer = _state.get("trainer")
    if not trainer:
        console.print("[red]Error: No trainer initialized. Run 'train' first.[/red]")
        raise typer.Exit(1)
    
    # Evaluate
    console.print(f"[blue]Evaluating on task: {task_id}[/blue]")
    result = trainer.evaluate(eval_data)
    
    console.print(f"[green]Evaluation completed![/green]")
    console.print(f"Loss: {result['loss']:.4f}")
    console.print(f"Accuracy: {result['accuracy']:.4f}")
    console.print(f"Samples: {result['samples']}")


# Model registry commands
@app.command()
def save(
    version: str,
    config_path: str = "config/default.yaml",
):
    """Save the current model as a version."""
    # Get model from state
    model = _state.get("model")
    if not model:
        console.print("[red]Error: No model loaded. Run 'train' first.[/red]")
        raise typer.Exit(1)
    
    # Load config
    config = get_config(config_path)
    
    # Create registry
    registry = ModelRegistry(config)
    _state["registry"] = registry
    
    # Save version
    console.print(f"[blue]Saving version: {version}[/blue]")
    model_version = registry.save_version(
        version=version,
        model=model,
        metadata={
            "task": _state.get("trainer", {}).current_task,
            "timestamp": datetime.now().isoformat(),
        },
    )
    
    console.print(f"[green]Version saved successfully![/green]")
    console.print(f"Path: {model_version.model_path}")
    console.print(f"Parameters: {model_version.parameters}")
    console.print(f"Size: {model_version.size_bytes / 1024 / 1024:.2f} MB")


@app.command()
def promote(
    version: str,
    config_path: str = "config/default.yaml",
):
    """Promote a model version to production."""
    # Load config
    config = get_config(config_path)
    
    # Create registry
    registry = _state.get("registry") or ModelRegistry(config)
    _state["registry"] = registry
    
    # Promote version
    console.print(f"[blue]Promoting version: {version}[/blue]")
    success = registry.promote(version)
    
    if success:
        console.print(f"[green]Version {version} promoted successfully![/green]")
        console.print(f"Current version: {registry.get_current_version()}")
    else:
        console.print(f"[red]Error: Failed to promote version {version}[/red]")
        raise typer.Exit(1)


@app.command()
def rollback(
    version: Optional[str] = None,
    config_path: str = "config/default.yaml",
):
    """Rollback to a previous version."""
    # Load config
    config = get_config(config_path)
    
    # Create registry
    registry = _state.get("registry") or ModelRegistry(config)
    _state["registry"] = registry
    
    # Rollback
    if version:
        console.print(f"[blue]Rolling back to version: {version}[/blue]")
    else:
        console.print("[blue]Rolling back to previous version[/blue]")
    
    success = registry.rollback(version)
    
    if success:
        console.print(f"[green]Rollback successful![/green]")
        console.print(f"Current version: {registry.get_current_version()}")
    else:
        console.print("[red]Error: Failed to rollback[/red]")
        raise typer.Exit(1)


@app.command()
def list_versions(
    config_path: str = "config/default.yaml",
):
    """List all model versions."""
    # Load config
    config = get_config(config_path)
    
    # Create registry
    registry = _state.get("registry") or ModelRegistry(config)
    
    # Get versions
    versions = registry.list_versions()
    
    if not versions:
        console.print("[yellow]No versions found[/yellow]")
        return
    
    table = Table(title="Model Versions")
    table.add_column("Version", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Created", style="green")
    table.add_column("Size (MB)", style="blue")
    table.add_column("Parameters", style="blue")
    
    for version in versions:
        v = registry.get_version(version)
        if v:
            status = v.status
            size_mb = v.size_bytes / 1024 / 1024
            table.add_row(
                version,
                status,
                v.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                f"{size_mb:.2f}",
                f"{v.parameters:,}",
            )
    
    console.print(table)
    console.print(f"[green]Current version: {registry.get_current_version()}[/green]")


# Registry status command
@app.command()
def registry_status(
    config_path: str = "config/default.yaml",
):
    """Show registry status."""
    # Load config
    config = get_config(config_path)
    
    # Create registry
    registry = _state.get("registry") or ModelRegistry(config)
    
    stats = registry.get_stats()
    
    table = Table(title="Registry Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Number of versions", str(stats.num_versions))
    table.add_row("Current version", stats.current_version or "None")
    table.add_row("Total size", f"{stats.total_size_bytes / 1024 / 1024:.2f} MB")
    table.add_row("Max versions", str(stats.max_versions))
    table.add_row("Auto archive", str(stats.auto_archive))
    
    console.print(table)


# Drift detection command
@app.command()
def detect_drift(
    data_path: str,
    reference_path: str,
    config_path: str = "config/default.yaml",
):
    """Detect drift in data."""
    # Load config
    config = get_config(config_path)
    
    # Load data
    data_path = Path(data_path)
    reference_path = Path(reference_path)
    
    if not data_path.exists() or not reference_path.exists():
        console.print("[red]Error: Data or reference file not found[/red]")
        raise typer.Exit(1)
    
    # Load data
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        with open(reference_path, "r") as f:
            reference = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1)
    
    # Create drift detector
    detector = DriftDetector(config)
    
    # Add reference data
    for item in reference[:100]:  # Use first 100 items
        # For demo, use a simple numeric feature
        # In practice, you'd extract proper features
        feature = item.get("feature", 0.0)
        detector.add_reference(feature)
    
    # Check drift for new data
    console.print(f"[blue]Checking drift for {len(data)} items[/blue]")
    
    drift_detected = False
    for i, item in enumerate(data[:100]):
        feature = item.get("feature", 0.0)
        result = detector.check_drift(feature)
        
        if result.is_drift:
            drift_detected = True
            console.print(
                f"[red]Drift detected at item {i}: {result.drift_type.value} "
                f"(score: {result.score:.4f}, threshold: {result.threshold:.4f})[/red]"
            )
    
    if not drift_detected:
        console.print("[green]No drift detected[/green]")


# Main command
@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    config_path: str = "config/default.yaml",
):
    """Start the inference server."""
    # Load config
    config = get_config(config_path)
    
    # Get model from state
    model = _state.get("model")
    if not model:
        console.print("[red]Error: No model loaded. Run 'train' first.[/red]")
        raise typer.Exit(1)
    
    # Create adapter manager
    adapter_manager = AdapterManager(model, config)
    
    # Create server
    from adaptive_ml.serving.inference import ModelServer, ServerConfig
    
    server_config = ServerConfig(host=host, port=port)
    server = ModelServer(
        model=model,
        adapter_manager=adapter_manager,
        config=config,
        server_config=server_config,
    )
    
    console.print(f"[green]Starting server on {host}:{port}[/green]")
    console.print("[blue]Press Ctrl+C to stop[/blue]")
    
    try:
        server.start()
    except KeyboardInterrupt:
        console.print("[yellow]Server stopped[/yellow]")


if __name__ == "__main__":
    app()
