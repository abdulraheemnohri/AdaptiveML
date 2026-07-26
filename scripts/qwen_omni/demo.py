#!/usr/bin/env python3
"""
Adaptive Qwen Omni - Complete Demo
Demonstrates the full continual learning system for Qwen2.5-Omni-3B.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_environment() -> None:
    """Setup environment and check dependencies."""
    logger.info("Setting up Adaptive Qwen Omni environment...")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 9):
        logger.error("Python 3.9+ required")
        sys.exit(1)
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, using CPU")
    except ImportError:
        logger.error("PyTorch not installed")
        sys.exit(1)
    
    # Check Transformers
    try:
        from transformers import __version__ as transformers_version
        logger.info(f"Transformers version: {transformers_version}")
    except ImportError:
        logger.error("Transformers not installed")
        sys.exit(1)
    
    # Check PEFT
    try:
        from peft import __version__ as peft_version
        logger.info(f"PEFT version: {peft_version}")
    except ImportError:
        logger.error("PEFT not installed")
        sys.exit(1)
    
    logger.info("Environment setup complete!")


def demo_core_components() -> None:
    """Demonstrate core components."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Core Components")
    logger.info("="*60)
    
    from adaptive_ml.qwen_omni.core import (
        ModalityType,
        AdapterType,
        TaskType,
        DomainType,
        MultimodalData,
        MultimodalEntry,
        QwenOmniConfig,
    )
    
    # Create multimodal data
    data = MultimodalData(
        text="What is in this image?",
        image="path/to/image.jpg",
    )
    logger.info(f"Created MultimodalData with modalities: {[m.value for m in data.modalities]}")
    
    # Create entry
    entry = MultimodalEntry(
        id="test-001",
        data=data,
        instruction="Describe the image",
        expected_output="The image shows a beautiful sunset over the ocean.",
        domain=DomainType.VISION,
        language="en",
        importance=0.9,
        novelty=0.8,
    )
    logger.info(f"Created entry with priority score: {entry.get_priority_score():.2f}")
    
    # Load configuration
    config = QwenOmniConfig()
    logger.info(f"Loaded configuration with base model: {config.model.base_model}")
    logger.info(f"Adaptation level: {config.training.adaptation_level}")
    logger.info(f"Replay ratio: {config.training.replay_ratio}")


def demo_adaptive_learning() -> None:
    """Demonstrate adaptive learning components."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Adaptive Learning")
    logger.info("="*60)
    
    from adaptive_ml.qwen_omni.core import (
        ModalityType,
        TaskType,
        DomainType,
        MultimodalData,
    )
    from adaptive_ml.qwen_omni.adaptive import (
        TaskDetector,
        DomainDetector,
        NoveltyDetector,
        AdaptiveRouter,
        LearningController,
        AdaptiveLearningOS,
    )
    
    # Create detectors
    task_detector = TaskDetector()
    domain_detector = DomainDetector()
    novelty_detector = NoveltyDetector()
    
    # Test data
    test_data = [
        ("Write a Python function to sort a list", "python", "coding"),
        ("What is the capital of France?", "general", "general"),
        ("Explain the concept of machine learning", "education", "general"),
        ("Translate this to Urdu", "urdu", "urdu"),
        ("Describe this image", "image", "vision"),
    ]
    
    for text, expected_domain, expected_task in test_data:
        data = MultimodalData(text=text)
        
        # Detect task
        task_result = task_detector.classify(data, text)
        logger.info(f"Text: {text[:50]}...")
        logger.info(f"  Detected task: {task_result.predicted_task.value if task_result.predicted_task else 'unknown'}")
        logger.info(f"  Expected task: {expected_task}")
        
        # Detect domain
        domain_result = domain_detector.classify(data, text)
        logger.info(f"  Detected domain: {domain_result.predicted_domain.value if domain_result.predicted_domain else 'unknown'}")
        logger.info(f"  Expected domain: {expected_domain}")
        logger.info()
    
    # Test router
    router = AdaptiveRouter()
    data = MultimodalData(text="Write a Python function")
    routing = router.route(data, "Write a Python function")
    logger.info(f"Routing decision: {routing.to_dict()}")
    
    # Test learning controller
    controller = LearningController()
    strategy = controller.determine_strategy(
        data=data,
        instruction="Write a Python function",
    )
    logger.info(f"Learning strategy: {strategy.decision.value}")
    logger.info(f"Adapters to use: {[a.value for a in strategy.adapters_to_use]}")
    
    # Test Adaptive Learning OS
    learning_os = AdaptiveLearningOS()
    decision = learning_os.process(
        data=data,
        instruction="Write a Python function",
    )
    logger.info(f"Learning OS decision: {decision.to_dict()}")


def demo_continual_learning() -> None:
    """Demonstrate continual learning components."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Continual Learning")
    logger.info("="*60)
    
    from adaptive_ml.qwen_omni.core import (
        ModalityType,
        DomainType,
        MultimodalData,
        MultimodalEntry,
    )
    from adaptive_ml.qwen_omni.continual_learning import (
        MultimodalReplayBuffer,
        ReplayMemory,
        ForgettingDetector,
        AntiForgettingEngine,
    )
    
    # Create replay buffer
    buffer = MultimodalReplayBuffer(
        max_size=1000,
        sampling_strategy="priority_based",
    )
    
    # Add some entries
    entries = [
        MultimodalEntry(
            id=f"entry-{i}",
            data=MultimodalData(text=f"Sample text {i}"),
            domain=DomainType.GENERAL,
            importance=0.5 + i * 0.1,
            novelty=0.8 - i * 0.05,
        )
        for i in range(10)
    ]
    
    for entry in entries:
        buffer.add(entry)
    
    logger.info(f"Added {len(entries)} entries to replay buffer")
    logger.info(f"Buffer size: {len(buffer)}")
    
    # Sample from buffer
    sampled = buffer.sample(3)
    logger.info(f"Sampled {len(sampled)} entries")
    
    # Test replay memory
    memory = ReplayMemory(
        general_buffer_size=1000,
        domain_buffer_sizes={DomainType.GENERAL: 500, DomainType.CODING: 300},
    )
    
    for entry in entries:
        memory.add(entry)
    
    logger.info(f"Replay memory stats: {memory.get_stats().to_dict()}")
    
    # Test forgetting detector
    detector = ForgettingDetector(
        strategy="modality_specific",
        modality_thresholds={ModalityType.TEXT: 0.05},
    )
    
    # Simulate some metrics
    metrics = {
        ModalityType.TEXT: {"accuracy": 0.95, "previous_accuracy": 0.98},
        ModalityType.VISION: {"accuracy": 0.90, "previous_accuracy": 0.92},
    }
    
    result = detector.detect(metrics)
    logger.info(f"Forgetting detection result: {result.to_dict()}")
    
    # Test anti-forgetting engine
    engine = AntiForgettingEngine(
        initial_replay_ratio=0.3,
        initial_distillation_weight=0.5,
        initial_protection_lambda=0.1,
    )
    
    response = engine.respond(result)
    logger.info(f"Anti-forgetting response: {response.to_dict()}")


def demo_training() -> None:
    """Demonstrate training components."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Training Components")
    logger.info("="*60)
    
    from adaptive_ml.qwen_omni.core import (
        QwenOmniModelConfig,
        QwenOmniTrainingConfig,
    )
    from adaptive_ml.qwen_omni.training import (
        QwenOmniTrainer,
        LoRATrainer,
        QLoRATrainer,
        MultimodalTrainer,
    )
    
    # Create configs
    model_config = QwenOmniModelConfig(
        base_model="Qwen/Qwen2.5-Omni-3B",
        use_flash_attention=True,
        use_bfloat16=True,
    )
    
    training_config = QwenOmniTrainingConfig(
        learning_rate=2e-5,
        batch_size=4,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16.0,
        use_ewc=True,
        ewc_lambda=0.1,
        use_replay=True,
        replay_ratio=0.3,
    )
    
    logger.info(f"Model config: {model_config.model_dump()}")
    logger.info(f"Training config: {training_config.model_dump()}")
    
    # Note: Actual training would require the model to be downloaded
    # This is just demonstrating the configuration
    logger.info("Note: Actual training requires Qwen2.5-Omni-3B to be downloaded")
    logger.info("Use: huggingface-cli download Qwen/Qwen2.5-Omni-3B")


def demo_evaluation() -> None:
    """Demonstrate evaluation components."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Evaluation Components")
    logger.info("="*60)
    
    from adaptive_ml.qwen_omni.core import (
        ModalityType,
        DomainType,
        MultimodalData,
    )
    from adaptive_ml.qwen_omni.evaluation import (
        RetentionScorer,
        ForgettingScore,
        RetentionScore,
        PromotionGate,
    )
    
    # Test retention scorer
    scorer = RetentionScorer(
        modality_weights={
            ModalityType.TEXT: 0.3,
            ModalityType.VISION: 0.2,
            ModalityType.AUDIO: 0.2,
            ModalityType.VIDEO: 0.2,
            ModalityType.SPEECH: 0.1,
        }
    )
    
    # Simulate scores
    new_scores = {
        ModalityType.TEXT: 0.95,
        ModalityType.VISION: 0.90,
    }
    old_scores = {
        ModalityType.TEXT: 0.92,
        ModalityType.VISION: 0.88,
    }
    
    retention_score = scorer.compute_retention(new_scores, old_scores)
    logger.info(f"Retention score: {retention_score.retention:.4f}")
    logger.info(f"Forgetting scores: {retention_score.forgetting_scores}")
    
    # Test promotion gate
    gate = PromotionGate(
        min_improvement=0.05,
        max_forgetting=0.03,
        min_retention=0.98,
    )
    
    decision = gate.decide(
        retention_score=retention_score.retention,
        new_scores=new_scores,
        old_scores=old_scores,
    )
    logger.info(f"Promotion decision: {decision.decision.value}")
    logger.info(f"Reasons: {decision.reasons}")


def demo_inference() -> None:
    """Demonstrate inference components."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Inference Components")
    logger.info("="*60)
    
    from adaptive_ml.qwen_omni.core import (
        ModalityType,
        TaskType,
        DomainType,
        AdapterType,
        MultimodalData,
    )
    from adaptive_ml.qwen_omni.inference import (
        MultimodalRouter,
        RoutingDecision,
    )
    
    # Create router
    router = MultimodalRouter(
        available_adapters=[
            AdapterType.GENERAL,
            AdapterType.CODING,
            AdapterType.VISION,
            AdapterType.URDU,
        ],
        routing_strategy="hierarchical",
    )
    
    # Test routing
    test_cases = [
        ("Write a Python function", "coding", "CODE_GENERATION"),
        ("Describe this image", "vision", "IMAGE_UNDERSTANDING"),
        ("Translate to Urdu", "urdu", "TRANSLATION"),
        ("What is machine learning?", "general", "QUESTION_ANSWERING"),
    ]
    
    for text, expected_domain, expected_task in test_cases:
        data = MultimodalData(text=text)
        routing = router.route(data, text)
        
        logger.info(f"Text: {text}")
        logger.info(f"  Detected modality: {routing.modality.value}")
        logger.info(f"  Detected task: {routing.task_type.value if routing.task_type else 'unknown'}")
        logger.info(f"  Detected domain: {routing.domain.value}")
        logger.info(f"  Primary adapter: {routing.primary_adapter.value if routing.primary_adapter else 'unknown'}")
        logger.info(f"  Confidence: {routing.confidence:.2f}")
        logger.info()


def demo_datasets() -> None:
    """Demonstrate dataset components."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Dataset Components")
    logger.info("="*60)
    
    from adaptive_ml.qwen_omni.core import (
        DomainType,
        MultimodalData,
        MultimodalEntry,
    )
    from adaptive_ml.qwen_omni.datasets import (
        DatasetConfig,
        MultimodalDataset,
        DataCleaning,
        CleaningConfig,
        Deduplication,
        DeduplicationConfig,
        QualityFilter,
        QualityConfig,
    )
    
    # Create sample entries
    entries = [
        MultimodalEntry(
            id=f"sample-{i}",
            data=MultimodalData(text=f"This is sample text {i}"),
            instruction="Answer the question",
            expected_output=f"Sample output {i}",
            domain=DomainType.GENERAL,
            language="en",
        )
        for i in range(5)
    ]
    
    # Test cleaning
    cleaner = DataCleaning(
        config=CleaningConfig(
            remove_html=True,
            remove_urls=True,
            min_text_length=10,
        )
    )
    
    cleaned_entries = cleaner.batch_clean_entries(entries)
    logger.info(f"Cleaned {len(cleaned_entries)} entries")
    
    # Test deduplication
    dedup = Deduplication(
        config=DeduplicationConfig(
            use_exact=True,
            use_fuzzy=True,
            fuzzy_threshold=90,
        )
    )
    
    unique, duplicates = dedup.deduplicate(entries)
    logger.info(f"Deduplication: {len(unique)} unique, {len(duplicates)} duplicates")
    
    # Test quality filter
    quality_filter = QualityFilter(
        config=QualityConfig(
            min_text_length=10,
            min_word_count=3,
        )
    )
    
    valid, invalid = quality_filter.filter_entries(entries)
    logger.info(f"Quality filter: {len(valid)} valid, {len(invalid)} invalid")


def demo_complete_system() -> None:
    """Demonstrate the complete Adaptive Qwen Omni system."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Complete Adaptive Qwen Omni System")
    logger.info("="*60)
    
    from adaptive_ml.qwen_omni.core import (
        QwenOmniConfig,
        ModalityType,
        DomainType,
        MultimodalData,
        MultimodalEntry,
    )
    from adaptive_ml.qwen_omni.adaptive import AdaptiveLearningOS
    from adaptive_ml.qwen_omni.continual_learning import ReplayMemory
    from adaptive_ml.qwen_omni.inference import MultimodalRouter
    
    # Load configuration
    config = QwenOmniConfig()
    logger.info(f"Loaded complete configuration")
    logger.info(f"  Base model: {config.model.base_model}")
    logger.info(f"  Adaptation level: {config.training.adaptation_level}")
    logger.info(f"  Replay ratio: {config.training.replay_ratio}")
    
    # Initialize components
    learning_os = AdaptiveLearningOS()
    replay_memory = ReplayMemory(
        general_buffer_size=1000,
        domain_buffer_sizes={d: 500 for d in DomainType},
    )
    router = MultimodalRouter(
        available_adapters=config.adapters.default_adapters,
    )
    
    logger.info(f"Initialized Adaptive Learning OS")
    logger.info(f"Initialized Replay Memory")
    logger.info(f"Initialized Multimodal Router")
    
    # Simulate a learning cycle
    data = MultimodalData(
        text="Write a Python function to calculate factorial",
    )
    
    # Step 1: Process through Learning OS
    decision = learning_os.process(data, "Write a Python function")
    logger.info(f"Learning decision: {decision.decision.value}")
    
    # Step 2: Route to appropriate adapter
    routing = router.route(data, "Write a Python function")
    logger.info(f"Routing: primary={routing.primary_adapter.value if routing.primary_adapter else 'none'}")
    
    # Step 3: Add to replay memory
    entry = MultimodalEntry(
        id="demo-001",
        data=data,
        instruction="Write a Python function",
        domain=DomainType.CODING,
    )
    replay_memory.add(entry)
    logger.info(f"Added entry to replay memory")
    
    # Get stats
    stats = replay_memory.get_stats()
    logger.info(f"Replay memory stats: {stats.to_dict()}")
    
    logger.info("\nComplete system demo finished!")


def main() -> None:
    """Run all demos."""
    setup_environment()
    
    logger.info("\n" + "="*60)
    logger.info("ADAPTIVE QWEN OMNI - COMPLETE DEMO")
    logger.info("="*60)
    
    # Run individual demos
    demo_core_components()
    demo_adaptive_learning()
    demo_continual_learning()
    demo_training()
    demo_evaluation()
    demo_inference()
    demo_datasets()
    demo_complete_system()
    
    logger.info("\n" + "="*60)
    logger.info("ALL DEMOS COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    logger.info("\nNext steps:")
    logger.info("1. Install Qwen2.5-Omni-3B: huggingface-cli download Qwen/Qwen2.5-Omni-3B")
    logger.info("2. Run training with your own data")
    logger.info("3. Deploy the inference engine")
    logger.info("4. Monitor continual learning performance")


if __name__ == "__main__":
    main()
