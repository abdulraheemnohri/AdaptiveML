# Adaptive Omni ML

> One Adaptive Brain. Continuous Learning. Zero Catastrophic Forgetting. Local Intelligence + Global AI.

## Project Vision

A complete, production-grade Adaptive Machine Learning and AI Orchestration Platform based on **Qwen/Qwen2.5-Omni-3B**.

The platform supports multimodal AI capabilities including:
- Text
- Image
- Audio
- Video
- Speech
- Multimodal understanding

## Two Operating Modes

### 🧠 Training Mode
Collect → Clean → Learn → Test → Anti-Forget → Approve → Save

### 🚀 Serving Mode
- Option 1: Trained Local Model
- Option 2: Multiple AI Provider APIs

## Core Architecture

```
ADAPTIVE OMNI ML
                │
      ┌─────────┴─────────┐
      │                   │
TRAINING MODE        SERVING MODE
      │                   │
Qwen2.5-Omni-3B     AI Engine Selector
      │                   │
┌─────┴─────┐         ┌───┴────┐
│           │         │        │
Data      Continual  LOCAL    API
Acquisition Learning  Model   Providers
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# Setup
python scripts/setup.py

# Run backend
python backend/app/main.py

# Run frontend
cd frontend && npm run dev
```

## Version Roadmap

### V1 — Foundation
- Qwen2.5-Omni-3B base model
- Training & Serving modes
- Local model serving
- AI API serving
- Dataset manager
- LoRA/QLoRA training
- Basic continual learning
- Experience replay
- Model registry
- Local RAG

### V2 — Advanced Adaptive ML
- Automatic data collection
- Knowledge-gap detection
- Research agents
- Advanced continual learning (EWC, distillation)
- Automatic evaluation
- Model promotion gates
- Advanced RAG
- Multi-AI routing

### V3 — Autonomous Adaptive AI Platform
- Autonomous research & learning
- Self-discovering knowledge gaps
- Dynamic adapter ecosystem
- Advanced multi-agent system
- Distributed training
- Full plugin ecosystem

## License

MIT License
