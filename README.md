# Adaptive Omni ML Platform

## A Self-Evolving Multimodal AI Learning Platform

**Tagline:** An AI That Learns Continuously Without Forgetting What It Already Knows

---

## Project Summary

**Adaptive Omni ML** is a complete **Self-Evolving Multimodal AI Learning Platform** built on **Qwen/Qwen2.5-Omni-3B**.

### Core Features:
- Multimodal Learning - Text, Image, Audio, Video
- Continual Learning - Learn new things without forgetting old knowledge
- Multi-Source Data Acquisition - Automatically collect data from various sources
- RAG + Knowledge Graph - Store and organize knowledge
- Anti-Catastrophic Forgetting - Protect existing knowledge
- Model Testing & Versioning - Test and version every model
- Complete Control Centre - Full control and monitoring
- Brain Evolution - Autonomous self-improvement

---

## Project Objectives

1. Start from Qwen2.5-Omni-3B as base model
2. Support multimodal learning (text, image, audio, video)
3. Automatically collect data from multiple sources
4. Clean and validate incoming data
5. Detect low-quality and poisoned data
6. Discover knowledge gaps
7. Decide learning strategy (RAG, Adapter, Fine-tuning, Continual)
8. Learn new capabilities without catastrophic forgetting
9. Maintain previous capabilities through continual evaluation
10. Automatically test every candidate model

---

## Project Structure

Backend (FastAPI):
- app/main.py - FastAPI application
- app/core/ - Configuration and database
- app/models/ - Database models
- app/api/ - API endpoints  
- app/services/ - Business services
- requirements.txt

Frontend (React + TypeScript):
- src/App.tsx - Main application
- src/components/ - React components
- src/layouts/ - Layout components
- src/pages/ - Page components
- package.json

---

## Versions

### V1 - Adaptive ML Lite
For individual developers, students, researchers

### V2 - Adaptive Omni Pro
For production/research use

### V3 - Adaptive Omni Autonomous
Full autonomous AI system

---

## Tech Stack

- Frontend: React 18+, TypeScript, Tailwind CSS, shadcn/ui, Zustand
- Backend: FastAPI, Python 3.10+, SQLAlchemy
- Database: PostgreSQL, SQLite, Qdrant, Neo4j
- ML: Qwen2.5-Omni-3B, Transformers, PEFT, Accelerate

---

## Key Features

### Data Acquisition
- Web, RSS, YouTube, PDF, GitHub, Local folders, APIs
- Automatic scheduled collection
- Quality scoring and filtering

### Continual Learning
- Experience Replay
- Elastic Weight Consolidation (EWC)
- Learning Without Forgetting (LwF)
- Knowledge Distillation
- Adapter Isolation & Fusion

### Anti-Forgetting
- Baseline evaluation before training
- Regression testing after training
- Forgetting detection
- Automatic rollback on excessive forgetting

### Knowledge Graph
- Semantic knowledge storage
- Entity-relationship network
- Confidence tracking
- Source attribution

---

## Next Steps

1. Complete remaining backend services
2. Integrate Qwen2.5-Omni-3B
3. Implement data pipeline
4. Build RAG system
5. Create model registry
6. Develop testing framework
7. Connect frontend to backend

---

## Contribution

Pull requests welcome!

---

## License

MIT License

---

**GitHub:** [https://github.com/abdulraheemnohri/AdaptiveML](https://github.com/abdulraheemnohri/AdaptiveML)
**Base Model:** Qwen2.5-Omni-3B
**Status:** Under Development