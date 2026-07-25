# Adaptive ML Framework

**Learn new knowledge without destroying old knowledge.**

A production-ready **Continual Learning Framework** for building adaptive ML systems that prevent catastrophic forgetting. The framework combines **Experience Replay**, **Elastic Weight Consolidation (EWC)**, **Knowledge Distillation**, and **Dynamic Adapters** to enable models to continuously learn from new data while preserving existing knowledge.

## 🧠 Core Features

- **Experience Replay**: Maintain a buffer of representative examples from previous tasks with multiple sampling strategies (uniform, balanced, importance-weighted, diversity-based, hard examples)
- **Elastic Weight Consolidation (EWC)**: Protect important parameters using Fisher Information Matrix
- **Knowledge Distillation**: Preserve old model behavior by matching soft predictions
- **Dynamic Adapters**: Use LoRA/QLoRA adapters for parameter-efficient fine-tuning with task routing
- **Drift Detection**: Statistical (KS-test, PSI, Wasserstein) and semantic drift detection
- **Model Registry**: Versioned model storage with atomic promotion and rollback
- **Evaluation Suite**: Comprehensive metrics for retention, forgetting, and performance

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/abdulraheemnohri/AdaptiveML.git
cd AdaptiveML

# Install dependencies
pip install -e ".[dev]"

# Or with Poetry
poetry install
```

## 🚀 Quick Start

### 1. Initialize Project

```bash
adaptive-ml init --name my_project
```

This creates:
- `config/default.yaml` - Configuration file
- `model_registry/` - Model version storage
- `logs/` - Training logs
- `mlruns/` - MLflow tracking
- `adapters/` - Adapter storage

### 2. Train on First Task

```bash
# Create training data (JSON format)
echo '[{"text": "sample text", "label": 0}, ...]' > data/task_a.json

# Train
adaptive-ml train --task-id task_a --data data/task_a.json
```

### 3. Train on Second Task (with Anti-Forgetting)

```bash
# Create new task data
echo '[{"text": "new text", "label": 1}, ...]' > data/task_b.json

# Train with continual learning
adaptive-ml train --task-id task_b --data data/task_b.json
```

### 4. Evaluate and Promote

```bash
# Evaluate on test data
adaptive-ml evaluate --task-id task_b --data data/test_b.json

# Save model version
adaptive-ml save --version v1.0.0

# Promote to production
adaptive-ml promote --version v1.0.0
```

### 5. Start Inference Server

```bash
adaptive-ml serve --host 0.0.0.0 --port 8000
```

Then send requests:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "task_id": "task_a"}'
```

## 🏗️ Architecture

```
┌──────────────────────┐
│     Data Sources     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Ingestion Pipeline   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Quality + Dedup      │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Drift Detection      │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Adaptive Controller  │
└──────────┬───────────┘
           │
   ┌───────┼───────┐
   ▼       ▼       ▼
Replay   Adapter   Expert
Memory   Manager   Manager
   │       │       │
   └───────┼───────┘
           ▼
┌──────────────────────┐
│ Continual Trainer    │
│ - EWC               │
│ - Distillation      │
│ - Replay            │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Candidate Model      │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Evaluation Engine    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Promotion Controller │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Production Model     │
└──────────────────────┘
```

## 📚 Python API

### Basic Usage

```python
from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.training.trainer import ContinualTrainer
from adaptive_ml.data.dataset import DatasetEntry
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load configuration
config = AdaptiveMLConfig.from_yaml("config/default.yaml")

# Load model
model = AutoModelForCausalLM.from_pretrained(config.model.base_model)
tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)

# Create trainer
trainer = ContinualTrainer(model, tokenizer, config)

# Prepare data
train_data = [
    DatasetEntry(input="text 1", label=0, task_id="task_a"),
    DatasetEntry(input="text 2", label=1, task_id="task_a"),
    # ...
]

# Train on task A
metrics = trainer.train_task(
    task_id="task_a",
    train_data=train_data,
    num_epochs=3,
    batch_size=32,
)

# Train on task B (with anti-forgetting)
metrics = trainer.train_task(
    task_id="task_b",
    train_data=new_data,
    use_replay=True,
    use_ewc=True,
    use_distillation=True,
)
```

### Using Adapters

```python
from adaptive_ml.models.adapters import AdapterManager, AdapterRouter

# Create adapter manager
adapter_manager = AdapterManager(model, config)

# Create adapter for a new task
adapter_manager.create_adapter("task_a")

# Activate adapter
model_with_adapter = adapter_manager.activate_adapter("task_a")

# Create router
router = AdapterRouter(adapter_manager, config)

# Route input to appropriate adapter
adapter_id = router.route("What is the capital of France?")
```

### Using Model Registry

```python
from adaptive_ml.serving.registry import ModelRegistry

# Create registry
registry = ModelRegistry(config)

# Save model version
registry.save_version("v1.0.0", model, metadata={"accuracy": 0.95})

# Load model version
model = registry.load_version("v1.0.0")

# Promote version
registry.promote("v1.0.0")

# Rollback
registry.rollback()
```

### Using Promotion Controller

```python
from adaptive_ml.evaluation.promoter import PromotionController

# Create promoter
promoter = PromotionController(config)

# Evaluate candidate
result = promoter.evaluate_candidate(
    candidate_model=new_model,
    baseline_model=old_model,
    new_task_data=new_data,
    old_task_data=old_data,
)

# Make decision
if result.passed:
    promoter.promote_candidate("v1.1.0")
else:
    promoter.rollback()
```

## 🔧 Configuration

The framework uses a YAML-based configuration system. See `config/default.yaml` for all available options.

### Key Configuration Sections

```yaml
# Model configuration
model:
  base_model: "sshleifer/tiny-gpt2"
  tokenizer: null
  dtype: "float32"

# Training configuration
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 3
  ewc_lambda: 1000.0
  distill_alpha: 0.5
  replay_ratio: 0.3

# Adapter configuration
adapters:
  adapter_type: "lora"
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]

# Memory configuration
memory:
  buffer_size: 10000
  sampling_strategy: "balanced"

# Drift detection
drift:
  window_size: 1000
  statistical_test: "ks"
  statistical_threshold: 0.05

# Evaluation
evaluation:
  promotion_strategy: "strict"
  retention_threshold: 0.95
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=adaptive_ml --cov-report=html

# Run specific test
pytest tests/test_replay.py -v
```

## 📊 Demo

Run the demo script to see all components in action:

```bash
python scripts/demo.py
```

This demonstrates:
- Replay Buffer with different sampling strategies
- Elastic Weight Consolidation (EWC)
- Knowledge Distillation
- Continual Dataset
- Drift Detection
- Evaluation Metrics
- Promotion Controller

## 📁 Project Structure

```
AdaptiveML/
├── pyproject.toml          # Dependencies and project config
├── README.md               # This file
├── config/
│   └── default.yaml        # Default configuration
├── src/
│   └── adaptive_ml/
│       ├── __init__.py
│       ├── core/           # Config, types, logging
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── types.py
│       ├── data/           # Data pipelines
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── drift.py
│       ├── memory/         # Replay buffer
│       │   ├── __init__.py
│       │   └── replay.py
│       ├── models/         # Model adapters & routing
│       │   ├── __init__.py
│       │   ├── adapters.py
│       │   └── router.py
│       ├── training/       # Training engine
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── ewc.py
│       │   └── distillation.py
│       ├── evaluation/     # Evaluation & promotion
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── promoter.py
│       ├── serving/        # Model registry & inference
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   └── inference.py
│       └── cli/            # Command-line interface
│           ├── __init__.py
│           └── main.py
├── tests/                 # Pytest suite
│   ├── __init__.py
│   ├── test_replay.py
│   ├── test_ewc.py
│   └── test_promoter.py
├── scripts/               # Utility scripts
│   ├── __init__.py
│   └── demo.py
└── docs/                  # Documentation
    └── ...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by research in **Continual Learning** and **Catastrophic Forgetting**
- Built on top of **PyTorch**, **Transformers**, and **PEFT**
- Special thanks to the open-source ML community
