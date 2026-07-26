#!/usr/bin/env python3
"""
Advanced Demo for Adaptive ML Framework with Qwen2.5-Omni-3B.
Demonstrates multi-modal capabilities and new features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adaptive_ml import (
    AdaptiveMLConfig,
    ParameterImportanceMethod,
    ModalityType,
    MAS,
    SI,
    EWC,
    DynamicLoRAManager,
    LoRARankAdapter,
    CLIPDriftDetector,
    MemoryCompressor,
    CompressedReplayBuffer,
    AdvancedEvaluator,
    PerplexityCalculator,
    ReplayBuffer,
    DriftDetector,
)


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


def demo_config():
    """Demo: Configuration with Qwen2.5-Omni-3B."""
    print("\n" + "=" * 60)
    print("DEMO: Advanced Configuration with Qwen2.5-Omni-3B")
    print("=" * 60)
    
    config = AdaptiveMLConfig()
    
    print(f"\nBase Model: {config.model.base_model}")
    print(f"Modality: {config.model.modality}")
    print(f"Data Type: {config.model.dtype}")
    print(f"Quantization: {config.model.quantize} ({config.model.quantization_bits} bits)")
    
    print(f"\nTraining:")
    print(f"  Importance Method: {config.training.importance_method}")
    print(f"  EWC Lambda: {config.training.ewc_lambda}")
    print(f"  MAS Lambda: {config.training.mas_lambda}")
    print(f"  SI Lambda: {config.training.si_lambda}")
    print(f"  Dynamic LoRA: {config.training.dynamic_lora}")
    print(f"  LoRA Rank Range: {config.training.lora_rank_range}")
    
    print(f"\nAdapters:")
    print(f"  Adapter Type: {config.adapters.adapter_type}")
    print(f"  Dynamic Rank: {config.adapters.dynamic_rank}")
    print(f"  Rank Range: {config.adapters.min_rank}-{config.adapters.max_rank}")
    
    print(f"\nMemory:")
    print(f"  Buffer Size: {config.memory.buffer_size}")
    print(f"  Compression: {config.memory.use_compression}")
    print(f"  Method: {config.memory.compression_method}")
    
    print(f"\nDrift Detection:")
    print(f"  Use CLIP: {config.drift.use_clip}")
    print(f"  CLIP Model: {config.drift.clip_model}")
    print(f"  Ensemble: {config.drift.use_ensemble}")
    
    print(f"\nEvaluation:")
    print(f"  Use Perplexity: {config.evaluation.use_perplexity}")
    print(f"  Use BLEU: {config.evaluation.use_bleu}")
    print(f"  Use ROUGE: {config.evaluation.use_rouge}")


def demo_mas():
    """Demo: Memory Aware Synapses (MAS)."""
    print("\n" + "=" * 60)
    print("DEMO: Memory Aware Synapses (MAS)")
    print("=" * 60)
    
    # Create a simple model
    model = create_simple_model()
    
    # Create MAS
    mas = MAS(model, lambda_mas=100.0)
    
    # Create some data
    X, y = create_simple_data(num_samples=100)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Update importance
    print("\nUpdating MAS importance...")
    mas.update_importance(dataloader, num_batches=5)
    
    # Show stats
    stats = mas.get_stats()
    print(f"\nMAS stats:")
    print(f"  Parameters: {stats.num_parameters}")
    print(f"  Mean importance: {stats.mean_importance:.4f}")
    print(f"  Max importance: {stats.max_importance:.4f}")
    
    # Compute MAS loss
    mas_loss = mas.get_mas_loss()
    print(f"\nMAS loss: {mas_loss.item():.4f}")


def demo_si():
    """Demo: Synaptic Intelligence (SI)."""
    print("\n" + "=" * 60)
    print("DEMO: Synaptic Intelligence (SI)")
    print("=" * 60)
    
    # Create a simple model
    model = create_simple_model()
    
    # Create SI
    si = SI(model, lambda_si=100.0)
    
    # Create some data
    X, y = create_simple_data(num_samples=100)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Update importance
    print("\nUpdating SI importance...")
    si.update_importance(dataloader, num_batches=5)
    
    # Show stats
    stats = si.get_stats()
    print(f"\nSI stats:")
    print(f"  Parameters: {stats.num_parameters}")
    print(f"  Mean importance: {stats.mean_importance:.4f}")
    print(f"  Max importance: {stats.max_importance:.4f}")
    
    # Compute SI loss
    si_loss = si.get_si_loss()
    print(f"\nSI loss: {si_loss.item():.4f}")


def demo_dynamic_lora():
    """Demo: Dynamic LoRA Rank Adaptation."""
    print("\n" + "=" * 60)
    print("DEMO: Dynamic LoRA Rank Adaptation")
    print("=" * 60)
    
    # Create a simple model
    model = create_simple_model()
    
    # Create DynamicLoRAManager
    dynamic_manager = DynamicLoRAManager(model)
    
    print(f"\nInitial rank: {dynamic_manager.current_rank}")
    print(f"Rank range: {dynamic_manager.dynamic_config.min_rank}-{dynamic_manager.dynamic_config.max_rank}")
    
    # Simulate task performance
    tasks = ["task_a", "task_b", "task_c"]
    performances = [0.95, 0.85, 0.75]  # Decreasing performance
    
    for task_id, performance in zip(tasks, performances):
        new_rank = dynamic_manager.update_rank(task_id, performance)
        print(f"\nTask {task_id} (performance={performance:.2f}):")
        print(f"  New rank: {new_rank}")
        print(f"  Growth rate: {dynamic_manager.dynamic_config.growth_rate}")
    
    # Show stats
    stats = dynamic_manager.get_stats()
    print(f"\nRank adaptation stats:")
    print(f"  Current rank: {stats.current_rank}")
    print(f"  Rank changes: {stats.rank_changes}")
    print(f"  Growth count: {stats.growth_count}")
    print(f"  Shrink count: {stats.shrink_count}")
    print(f"  Last change: {stats.last_change_reason}")


def demo_memory_compression():
    """Demo: Memory Compression with FAISS."""
    print("\n" + "=" * 60)
    print("DEMO: Memory Compression with FAISS")
    print("=" * 60)
    
    # Create compressor with flat index (simpler for demo)
    compressor = MemoryCompressor(
        dimension=10,
        method="flat",  # Use flat index for demo
        nlist=5,
        nprobe=5,
    )
    
    # Generate random vectors
    num_vectors = 1000
    vectors = [np.random.randn(10).astype(np.float32) for _ in range(num_vectors)]
    
    # Add vectors
    print(f"\nAdding {num_vectors} vectors...")
    compressor.add_vectors(vectors[:500])
    print(f"Index size: {len(compressor)}")
    
    # Add more vectors
    compressor.add_vectors(vectors[500:])
    print(f"Index size after adding more: {len(compressor)}")
    
    # Search
    query_vector = np.random.randn(10).astype(np.float32)
    indices, distances = compressor.search([query_vector], k=5)
    
    print(f"\nSearch results (k=5):")
    print(f"  Indices: {indices[0]}")
    print(f"  Distances: {[f'{d:.4f}' for d in distances[0]]}")
    
    # Show stats
    stats = compressor.get_stats()
    print(f"\nCompression stats:")
    print(f"  Original size: {stats.original_size}")
    print(f"  Index size: {stats.index_size}")
    print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
    print(f"  Memory savings: {stats.memory_savings:.1f}%")


def demo_compressed_replay_buffer():
    """Demo: Compressed Replay Buffer."""
    print("\n" + "=" * 60)
    print("DEMO: Compressed Replay Buffer")
    print("=" * 60)
    
    # Create compressed replay buffer with custom config
    buffer = CompressedReplayBuffer(
        embedding_dim=10,
        method="flat",  # Use flat index for demo
        nlist=5,
        nprobe=5,
    )
    
    # Add embeddings
    print("\nAdding embeddings...")
    for i in range(100):
        embedding = np.random.randn(10).astype(np.float32)
        metadata = {"task_id": f"task_{i % 3}", "importance": i / 100.0}
        buffer.add_embedding(embedding, metadata)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Search similar
    query_embedding = np.random.randn(10).astype(np.float32)
    indices, distances, metadata = buffer.search_similar(query_embedding, k=5)
    
    print(f"\nSimilar embeddings (k=5):")
    for i, (idx, dist, meta) in enumerate(zip(indices, distances, metadata)):
        print(f"  {i+1}. Index={idx}, Distance={dist:.4f}, Task={meta.get('task_id')}")
    
    # Show stats
    stats = buffer.get_stats()
    print(f"\nCompression stats:")
    print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
    print(f"  Memory savings: {stats.memory_savings:.1f}%")


def demo_advanced_evaluator():
    """Demo: Advanced Evaluator with Perplexity."""
    print("\n" + "=" * 60)
    print("DEMO: Advanced Evaluator")
    print("=" * 60)
    
    # Create a simple model
    model = create_simple_model()
    
    # Create advanced evaluator
    evaluator = AdvancedEvaluator(model)
    
    # Create some data
    X, y = create_simple_data(num_samples=50)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Evaluate
    print("\nEvaluating model...")
    result = evaluator.evaluate(
        dataloader,
        compute_perplexity=False,  # Skip perplexity for simple model
        compute_bleu=False,
        compute_rouge=False,
        compute_f1=True,
    )
    
    print(f"\nEvaluation results:")
    print(f"  Accuracy: {result.accuracy:.4f}")
    print(f"  F1 Score: {result.f1_score:.4f}")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall: {result.recall:.4f}")


def demo_parameter_importance_methods():
    """Demo: All Parameter Importance Methods."""
    print("\n" + "=" * 60)
    print("DEMO: Parameter Importance Methods")
    print("=" * 60)
    
    # Create a simple model
    model = create_simple_model()
    
    # Create data
    X, y = create_simple_data(num_samples=100)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Test EWC
    print("\n1. Elastic Weight Consolidation (EWC)")
    ewc = EWC(model, lambda_ewc=1000.0)
    ewc.update_fisher(dataloader, num_batches=3)
    stats = ewc.get_stats()
    print(f"   Mean importance: {stats.mean_importance:.4f}")
    print(f"   Max importance: {stats.max_importance:.4f}")
    
    # Test MAS
    print("\n2. Memory Aware Synapses (MAS)")
    mas = MAS(model, lambda_mas=100.0)
    mas.update_importance(dataloader, num_batches=3)
    stats = mas.get_stats()
    print(f"   Mean importance: {stats.mean_importance:.4f}")
    print(f"   Max importance: {stats.max_importance:.4f}")
    
    # Test SI
    print("\n3. Synaptic Intelligence (SI)")
    si = SI(model, lambda_si=100.0)
    si.update_importance(dataloader, num_batches=3)
    stats = si.get_stats()
    print(f"   Mean importance: {stats.mean_importance:.4f}")
    print(f"   Max importance: {stats.max_importance:.4f}")


def main():
    """Run all advanced demos."""
    print("\n" + "=" * 60)
    print("ADAPTIVE ML FRAMEWORK - ADVANCED DEMO")
    print("=" * 60)
    print(f"\nUsing device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    
    # Run demos
    demo_config()
    demo_parameter_importance_methods()
    demo_mas()
    demo_si()
    demo_dynamic_lora()
    demo_memory_compression()
    demo_compressed_replay_buffer()
    demo_advanced_evaluator()
    
    print("\n" + "=" * 60)
    print("ADVANCED DEMO COMPLETED")
    print("=" * 60)
    
    print("\nNew Features Added:")
    print("  ✓ Qwen/Qwen2.5-Omni-3B as default base model")
    print("  ✓ Multi-modal support (text, vision, audio)")
    print("  ✓ CLIP-based semantic drift detection")
    print("  ✓ MAS (Memory Aware Synapses) parameter importance")
    print("  ✓ SI (Synaptic Intelligence) parameter importance")
    print("  ✓ Dynamic LoRA rank adaptation")
    print("  ✓ FAISS-based memory compression (IVF, PQ, IVF_PQ, HNSW)")
    print("  ✓ Advanced evaluation metrics (perplexity, BLEU, ROUGE)")
    print("  ✓ Compressed replay buffer for efficient similarity search")


if __name__ == "__main__":
    main()
