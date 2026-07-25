#!/usr/bin/env python3
"""
Demo script for Adaptive ML Framework.
Demonstrates the core continual learning capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import DatasetEntry, SamplingStrategy
from adaptive_ml.data.dataset import ContinualDataset
from adaptive_ml.data.drift import DriftDetector
from adaptive_ml.memory.replay import ReplayBuffer
from adaptive_ml.training.ewc import EWC
from adaptive_ml.training.distillation import KnowledgeDistillation
from adaptive_ml.training.trainer import ContinualTrainer
from adaptive_ml.evaluation.metrics import ContinualEvaluator
from adaptive_ml.evaluation.promoter import PromotionController


def create_simple_model(input_size=10, hidden_size=20, output_size=2):
    """Create a simple neural network for demonstration."""
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    return model


def create_simple_data(num_samples=100, input_size=10, num_classes=2):
    """Create simple synthetic data."""
    X = torch.randn(num_samples, input_size)
    # Create a simple decision boundary
    y = (X[:, 0] + X[:, 1] > 0).long()
    return X, y


def demo_replay_buffer():
    """Demonstrate the replay buffer functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Replay Buffer")
    print("=" * 60)
    
    config = AdaptiveMLConfig()
    config.memory.buffer_size = 100
    config.memory.sampling_strategy = SamplingStrategy.BALANCED
    
    # Create replay buffer
    buffer = ReplayBuffer(config)
    
    # Add some data
    print("\nAdding data to replay buffer...")
    for i in range(50):
        buffer.add(
            task_id="task_a",
            data=f"data_{i}",
            label=i % 2,
            importance=1.0 + i * 0.01,
            uncertainty=0.1 * (i % 5),
        )
    
    for i in range(30):
        buffer.add(
            task_id="task_b",
            data=f"data_{i + 50}",
            label=(i % 2) + 1,
        )
    
    # Show stats
    stats = buffer.get_stats()
    print(f"\nBuffer stats:")
    print(f"  Size: {stats.size}")
    print(f"  Capacity: {stats.capacity}")
    print(f"  Tasks: {stats.num_tasks}")
    print(f"  Task distribution: {stats.task_distribution}")
    
    # Sample from buffer
    print("\nSampling from buffer...")
    samples = buffer.sample(batch_size=10, strategy=SamplingStrategy.BALANCED)
    print(f"  Sampled {len(samples[0])} items")
    print(f"  Task IDs: {samples[2]['task_ids']}")
    
    # Test different sampling strategies
    for strategy in [SamplingStrategy.UNIFORM, SamplingStrategy.IMPORTANCE, SamplingStrategy.HARD_EXAMPLE]:
        samples = buffer.sample(batch_size=5, strategy=strategy)
        print(f"  {strategy.value}: {samples[2]['task_ids']}")


def demo_ewc():
    """Demonstrate EWC functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Elastic Weight Consolidation (EWC)")
    print("=" * 60)
    
    # Create a simple model
    model = create_simple_model()
    
    # Create EWC
    ewc = EWC(model, lambda_ewc=1000.0)
    
    # Create some data
    X, y = create_simple_data(num_samples=100)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Update Fisher Information
    print("\nUpdating Fisher Information...")
    ewc.update_fisher(dataloader)
    
    # Show stats
    stats = ewc.get_stats()
    print(f"\nEWC stats:")
    print(f"  Parameters: {stats.num_parameters}")
    print(f"  Important parameters: {stats.num_important_parameters}")
    print(f"  Mean importance: {stats.mean_importance:.4f}")
    print(f"  Max importance: {stats.max_importance:.4f}")
    
    # Compute penalty
    penalty = ewc.penalty()
    print(f"\nEWC penalty: {penalty.item():.4f}")


def demo_knowledge_distillation():
    """Demonstrate Knowledge Distillation functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Knowledge Distillation")
    print("=" * 60)
    
    # Create teacher and student models
    teacher = create_simple_model()
    student = create_simple_model()
    
    # Create Knowledge Distillation
    kd = KnowledgeDistillation(
        teacher=teacher,
        alpha=0.5,
        temperature=2.0,
    )
    
    # Create some data
    X, y = create_simple_data(num_samples=10)
    
    # Get teacher outputs
    with torch.no_grad():
        teacher_outputs = teacher(X)
    
    # Get student outputs
    student_outputs = student(X)
    
    # Compute distillation loss
    loss = kd._compute_distillation_loss(student_outputs, teacher_outputs)
    print(f"\nDistillation loss: {loss.item():.4f}")
    
    # Show stats
    print(f"\nKD stats:")
    print(f"  Alpha: {kd.alpha}")
    print(f"  Temperature: {kd.temperature}")


def demo_continual_dataset():
    """Demonstrate Continual Dataset functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Continual Dataset")
    print("=" * 60)
    
    # Create new data
    new_data = [
        DatasetEntry(data=f"new_{i}", label=i % 2, task_id="task_a")
        for i in range(20)
    ]
    
    # Create replay data
    replay_data = [
        DatasetEntry(data=f"old_{i}", label=(i % 2) + 1, task_id="task_b")
        for i in range(10)
    ]
    
    # Create continual dataset
    dataset = ContinualDataset(
        new_data=new_data,
        replay_buffer=replay_data,
        replay_ratio=0.3,
        sampling_strategy=SamplingStrategy.BALANCED,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Task distribution: {dataset.get_task_distribution()}")
    
    # Sample a batch
    inputs, labels, metadata = dataset.sample_batch(batch_size=8)
    print(f"\nSampled batch:")
    print(f"  Inputs: {inputs}")
    print(f"  Labels: {labels}")
    print(f"  Task IDs: {metadata['task_ids']}")
    print(f"  Is replay: {metadata['is_replay']}")


def demo_drift_detection():
    """Demonstrate Drift Detection functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Drift Detection")
    print("=" * 60)
    
    config = AdaptiveMLConfig()
    config.drift.statistical_test = "ks"
    config.drift.statistical_threshold = 0.05
    
    # Create drift detector
    detector = DriftDetector(config)
    
    # Add reference data (normal distribution)
    print("\nAdding reference data (normal distribution)...")
    reference_data = np.random.normal(0, 1, 100)
    for x in reference_data:
        detector.add_reference(x)
    
    # Check drift with similar data (no drift)
    print("\nChecking drift with similar data...")
    similar_data = np.random.normal(0, 1, 50)
    for x in similar_data:
        result = detector.check_drift(x)
    
    print(f"  Final result: {result.drift_type.value}, score: {result.score:.4f}, is_drift: {result.is_drift}")
    
    # Check drift with different data (drift)
    print("\nChecking drift with different data (shifted mean)...")
    detector.reset()
    for x in reference_data:
        detector.add_reference(x)
    
    shifted_data = np.random.normal(2, 1, 50)  # Shifted mean
    for x in shifted_data:
        result = detector.check_drift(x)
    
    print(f"  Final result: {result.drift_type.value}, score: {result.score:.4f}, is_drift: {result.is_drift}")


def demo_evaluation():
    """Demonstrate Evaluation functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Evaluation Metrics")
    print("=" * 60)
    
    # Create a simple model
    model = create_simple_model()
    
    # Create evaluator
    evaluator = ContinualEvaluator(model)
    
    # Create some data
    X, y = create_simple_data(num_samples=50)
    data = [DatasetEntry(data=X[i], label=y[i].item(), task_id="task_a") for i in range(len(X))]
    
    # Evaluate
    print("\nEvaluating on task_a...")
    result = evaluator.evaluate_task("task_a", data)
    print(f"  Loss: {result.loss:.4f}")
    print(f"  Accuracy: {result.accuracy:.4f}")
    print(f"  F1 Score: {result.f1_score:.4f}")
    print(f"  Samples: {result.num_samples}")


def demo_promotion():
    """Demonstrate Promotion Controller functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Promotion Controller")
    print("=" * 60)
    
    config = AdaptiveMLConfig()
    config.evaluation.promotion_strategy = "balanced"
    config.evaluation.retention_threshold = 0.8
    
    # Create promotion controller
    promoter = PromotionController(config)
    
    # Create dummy models
    model1 = create_simple_model()
    model2 = create_simple_model()
    
    # Create dummy data
    X, y = create_simple_data(num_samples=20)
    new_data = [DatasetEntry(data=X[i], label=y[i].item(), task_id="new_task") for i in range(len(X))]
    
    old_data = {
        "task_a": [DatasetEntry(data=X[i], label=y[i].item(), task_id="task_a") for i in range(10)],
        "task_b": [DatasetEntry(data=X[i], label=(y[i] + 1).item() % 2, task_id="task_b") for i in range(10)],
    }
    
    # Evaluate candidate
    print("\nEvaluating candidate model...")
    result = promoter.evaluate_candidate(
        candidate_model=model2,
        baseline_model=model1,
        new_task_data=new_data,
        old_task_data=old_data,
        new_task_id="new_task",
        old_task_ids=["task_a", "task_b"],
    )
    
    print(f"  Decision: {result.decision.value}")
    print(f"  Retention score: {result.retention_score:.4f}")
    print(f"  Old task score: {result.old_task_score:.4f}")
    print(f"  New task score: {result.new_task_score:.4f}")
    print(f"  Forgetting penalty: {result.forgetting_penalty:.4f}")
    print(f"  Passed: {result.passed}")
    print(f"  Message: {result.message}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("ADAPTIVE ML FRAMEWORK - DEMO")
    print("=" * 60)
    
    # Run demos
    demo_replay_buffer()
    demo_ewc()
    demo_knowledge_distillation()
    demo_continual_dataset()
    demo_drift_detection()
    demo_evaluation()
    demo_promotion()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    main()
